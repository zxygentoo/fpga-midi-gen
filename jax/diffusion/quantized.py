"""The integer twin of the masked sheet: the arithmetic the board plays.

The float model of `diffusion/model.py` is what the trainer produced. This module is the
same model in the arithmetic the board can hold -- int8 weights with a power-of-two
exponent for each tensor, int16 activations, the norm folded into per-channel constants,
and the draw in integers -- and the circuit must equal it operation for operation, not
approximately. Nothing here approximates on purpose: every shift, every floor and every
table is a rule the RTL reads from this module rather than restates.

IT CARRIES THE FLOAT MODEL'S SKELETON: [QuantizedCoconet] is a `model.Trunk`, under the
same attribute names at every level, thus a reader can put `coconet.pairs[7].first` beside
`twin.pairs[7].first` and audit one layer against its twin.

THE ORDER OF OPERATIONS IS THE CONTRACT. A rewrite that is algebraically equal and
differently ordered is a different machine: the gates of `tests/test_rtl.py` hold the
circuit to these integers write for write.

THE RULES THAT ARE NOT THIS ERA'S COME FROM `fixed.py`: the exponent rule, the rounding,
the int16 rails, the temper, the shared exp2 table, the counted write and the integer pick
stand there, where every twin reads them. What stands here is era six's alone -- the
activation format, the norm fold, the module tree, the contract file and the walk.

The formats, and where each rule comes from, are `docs/diffusion_rtl.md`:

- Weights are int8 under the exponent rule of the eras, `quantize`.
- Activations are Q`ACTIVATION_Q` in int16, clamped and counted. Q6 IS MEASURED, not
  chosen: the trunk is a residual stack with no norm on the stream, thus activations grow
  with depth, and the golden candidate peaks at 313 on the openings the walk really
  visits. Q6 holds 512 with a 1.6 margin. The input planes enter exact -- a cell is 0
  or one.
- The accumulator is int32 and is exact up to `WIDEST_INPUTS` input channels, thus the sum
  is exact and the order of the taps cannot matter. `check_shape` refuses a wider layer.
- The norm folds at quantization: `gain = scale * rsqrt(variance + eps)` becomes a
  per-channel multiplier that also retires the weight exponent, and
  `bias = shift - mean * gain` becomes Q`ACTIVATION_Q` in int16. Then ReLU; the head keeps
  no ReLU, thus the logits carry the activation format.
- The draw is era four's pipeline: the logit differences shift up to the Q12 the exp2 unit
  reads -- exact, a left shift -- then temper against the peak under `log2e / T`, exp2
  over the shared table gives Q15 weights, and the pick takes a 24-bit uniform.
- The masks and the opening are integer rules already and stand in `diffusion/model.py`,
  where both walks read them; the twin consumes the same uniforms in the same places as
  `infer.gibbs`.

What the quantization costs is a measurement and not a promise: `drift` states it, on the
walk the board really takes.

THE CONTRACT FILE is what crosses the seam to the elaboration. `save` writes it and
`Model.of_int8_checkpoint` reads it; `load` reads it back and a round trip is
exact. It carries the quantized model and nothing else -- no population statistics and no
float scales, because the fold happens here, one time, and the file carries the result.

    tensor            dtype   shape                    value
    "5i + 0"          int32   [3, 3, inputs, outputs]  the kernel q, int8 in int32
    "5i + 1"          int32   []                       the kernel exponent e
    "5i + 2"          int32   [outputs]                the gain q_value
    "5i + 3"          int32   [outputs]                the gain q
    "5i + 4"          int32   [outputs]                the bias, Q6 int16 in int32
    "temper"          int32   [2]                      the temper: q_value, then q
    "activation_q"    int32   []                       the Q of the activation format

EVERY TENSOR IS INT32 AND TWO OF THEM ARE NOT LAYERS. The first is a fact of the OCaml
reader, which `fixed.py`'s "the contract file" section states once for the three eras;
the second is this era's layout, above. The metadata is written as well, for a reader
that has a Python tool in hand.
"""

from collections import deque
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import prng
from diffusion import model as sheet
from fixed import (
    EXP2_IN_Q,
    INT16_HIGH,
    INT16_LOW,
    Temper,
    apply_scale,
    exp2_q,
    largest_exponent,
    pick,
    quantize,
    read_contract,
    round_half_up,
    scalar_tensor,
    tallied_write,
    write_contract,
    write_tally,
)

# ---------------------------------------------------------------------
# the formats
# ---------------------------------------------------------------------

# THE ACTIVATION FORMAT IS Q6 IN INT16, AND IT IS MEASURED -- the module docstring holds
# the measurement and the margin. `activation_one` is the one of the format, and the fixed
# point of the biases.
ACTIVATION_Q = 6
ACTIVATION_ONE = 1 << ACTIVATION_Q

# THE WIDEST LAYER THE INT32 ACCUMULATOR IS EXACT FOR: 9 C products of int8 by int16 reach
# 9 * 57 * 127 * 32767, which stands under 2^31, and one channel more can pass it. The
# elected shapes stand far under; the rule stands so the prose cannot rot.
WIDEST_INPUTS = 57

# the planes the stem reads: one class plane and one mask plane for each seat
PLANES = 2 * sheet.VOICES

# ---------------------------------------------------------------------
# the quantization of a checkpoint
# ---------------------------------------------------------------------


def gain_scale(value, weight_exponent):
    """the 16-bit form of the exponent rule, as (q_value, q).

    The shift of the scale retires the weight exponent in the same move, thus the
    accumulator goes from Q(ACTIVATION_Q + e) to Q(ACTIVATION_Q) in one multiply. 30 caps
    the all-zero gain, as 14 caps the all-zero tensor."""
    e = largest_exponent(abs(value), opening=30, cap=32767)
    return int(round_half_up(np.ldexp(value, e))), e + weight_exponent


@jax.jit
def accumulate(x, kernel):
    """The nine taps of one 3 by 3 convolution over (step, row), zero at both edges,
    summed into the accumulator.

    THE ACCUMULATOR IS INT32 AND IT WRAPS: no sum reaches 2^31 below `WIDEST_INPUTS` input
    channels and `check_shape` refuses a wider layer, thus a wrap here is the finding it
    would be on the board and not an artefact of the host.

    Tap (dy, dx) reads the source at (step + dy - 1, row + dx - 1). The accumulator is
    exact below the bound, thus the tap order cannot matter and the circuit may take its
    own."""
    _, steps, rows, _ = x.shape
    padded = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))

    def tap(dy, dx):
        window = padded[:, dy : dy + steps, dx : dx + rows, :]
        return jnp.matmul(window, kernel[dy, dx])

    return sum(tap(dy, dx) for dy in range(3) for dx in range(3))


# ---------------------------------------------------------------------
# the module tree: model.Trunk in integers
# ---------------------------------------------------------------------


class QuantizedNormedConv(nnx.Module):
    """One layer as the machine holds it -- the twin of `model.NormedConv`.

    Five float tensors become three facts: the kernel, and the two per-channel rows the
    norm folded into. A gain's shift retires the weight exponent, thus the accumulator
    reaches the activation format in one multiply; the biases are int16 in that format."""

    def __init__(self, *, kernel, e, gain_q_value, gain_q, bias):
        # THE KERNEL LIVES ON THE DEVICE and the rest of the layer on the host. A walk
        # runs [accumulate] hundreds of times over one model, thus a host kernel would
        # cross to the device at every call; the gains, the bias and the clamp are numpy
        # int64 arithmetic and belong beside the tally.
        self.kernel = nnx.Variable(jnp.asarray(kernel, jnp.int32))
        self.e = int(e)
        self.gain_q_value = nnx.Variable(np.asarray(gain_q_value, np.int32))
        self.gain_q = nnx.Variable(np.asarray(gain_q, np.int32))
        self.bias = nnx.Variable(np.asarray(bias, np.int32))

    @property
    def inputs(self):
        return self.kernel.shape[2]

    @property
    def outputs(self):
        return self.kernel.shape[3]

    @classmethod
    def of(cls, layer):
        """one float [model.NormedConv] under the exponent rule, its norm folded.

        At inference batch normalization is the affine `a * gain + bias`, and the fold is
        that same affine — its rounding is part of what `drift` measures. It runs in
        float64 over the float32 tensors, IN THE ORDER WRITTEN HERE. A bias outside the
        activation format clamps, which the drift report would shout about."""
        q, e = quantize(layer.conv.kernel[...])
        scale = np.asarray(layer.norm.scale[...], np.float64)
        shift = np.asarray(layer.norm.shift[...], np.float64)
        mean = np.asarray(layer.norm.mean[...], np.float64)
        variance = np.asarray(layer.norm.variance[...], np.float64)
        gain = scale / np.sqrt(variance + sheet.NORM_EPSILON)
        scales = [gain_scale(float(value), e) for value in gain]
        bias = np.clip(
            round_half_up((shift - mean * gain) * ACTIVATION_ONE), INT16_LOW, INT16_HIGH
        )
        return cls(
            kernel=q,
            e=e,
            gain_q_value=np.array([q_value for q_value, _ in scales], np.int32),
            gain_q=np.array([shift_of for _, shift_of in scales], np.int32),
            bias=bias.astype(np.int32),
        )

    def __call__(self, x, relu, tally):
        """the convolution into the int32 accumulator, the folded norm, the optional ReLU,
        and the counted clamp of every write.

        The gain multiply rides int64 — an int32 accumulator by an int16 gain wants 47
        bits. The shift is arithmetic, toward minus infinity, as the circuit's is."""
        accumulated = np.asarray(accumulate(jnp.asarray(x), self.kernel[...]))
        gain, shift = self.gain_q_value[...], self.gain_q[...]
        value = ((accumulated.astype(np.int64) * gain) >> shift) + self.bias[...]
        return tallied_write(tally, np.maximum(value, 0) if relu else value)

    def tensors(self):
        """the five tensors of this layer in the order of the contract file"""
        return [
            np.asarray(self.kernel[...], np.int32),
            np.array(self.e, np.int32),
            self.gain_q_value[...],
            self.gain_q[...],
            self.bias[...],
        ]


class QuantizedResidualPair(nnx.Module):
    """Two layers and the skip past both -- the twin of `model.ResidualPair`.

    THE RELU STANDS IN A DIFFERENT PLACE HERE, and that is the contract and not a slip.
    The float pair activates the first layer's output before the second convolution reads
    it; the twin folds that ReLU into the first layer's own counted write, because the
    machine writes what it will read back. The arithmetic is the same and the WRITE STREAM
    is not, and the write stream is what `tests/test_rtl.py` holds the circuit to."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    @classmethod
    def of(cls, pair):
        return cls(
            QuantizedNormedConv.of(pair.first), QuantizedNormedConv.of(pair.second)
        )

    def __call__(self, x, tally):
        """the tensor the opening wrote and the JOINED tensor the close wrote. The
        residual add rides the same counted clamp, thus a reader of the write stream takes
        the closing tensor from here and never from the second layer alone."""
        first = self.first(x, True, tally)
        second = self.second(first, False, tally)
        return first, tallied_write(tally, np.maximum(x + second, 0))


class QuantizedCoconet(sheet.Trunk):
    """The paper's net in the arithmetic the board holds: `model.Coconet`, layer for
    layer.

    The temper stands beside the layers because the bitstream carries it: one quantization
    serves every seed of a batch, as one bitstream serves every seed of the board."""

    def __init__(self, *, stem, pairs, head, temper):
        self.stem = stem
        self.pairs = nnx.List(list(pairs))
        self.head = head
        self.temper = temper

    @classmethod
    def of(cls, coconet, temperature=1.0):
        """the float model in the arithmetic the board holds.

        This is the one quantization of the era -- the drift walk, the audition and the
        elaboration all take their model here, thus the pair under comparison cannot slip.
        """
        return cls(
            stem=QuantizedNormedConv.of(coconet.stem),
            pairs=[QuantizedResidualPair.of(pair) for pair in coconet.pairs],
            head=QuantizedNormedConv.of(coconet.head),
            temper=Temper.of(temperature),
        )

    def _writes(self, classes, hidden, tally, *, rows=sheet.ROWS):
        """the destination tensor of EVERY layer as written, in the layer order.

        IT IS A GENERATOR AND THE TRUNK IS NOT WALKED TWICE: `__call__` keeps the last of
        it and `layer_writes` keeps them all, thus a forward pass holds one destination
        tensor and never the trunk's 48.

        `rows` is P; it reaches the stem's decode alone, because every layer after it
        takes the shape it is handed."""
        x = self.stem(plane_activations(classes, hidden, rows=rows), True, tally)
        yield x
        for pair in self.pairs:
            first, x = pair(x, tally)
            yield first
            yield x
        yield self.head(x, False, tally)

    def __call__(self, classes, hidden, tally, *, rows=sheet.ROWS):
        """the logits of one pass over the batch: `[sheets, steps, rows, VOICES]` in the
        activation format, because the head takes no ReLU and keeps it"""
        # the head's write is the last of them, and the deque holds one tensor where a
        # list would hold the trunk's 48
        return deque(self._writes(classes, hidden, tally, rows=rows), maxlen=1)[0]

    def layer_writes(self, classes, hidden, tally, *, rows=sheet.ROWS):
        """the destination tensor of every layer AS WRITTEN, in the layer order — the
        stem's, then for each pair its opening's tensor and its close's JOINED tensor, and
        the head's logits last.

        IT IS FOR THE CIRCUIT'S STREAM GATE AND FOR NOTHING ELSE. `__call__` is the last
        of this list, thus the list is the arithmetic itself and never a second reading of
        it."""
        return list(self._writes(classes, hidden, tally, rows=rows))


def paired(layers):
    """The trunk's layers two at a time. The contract file is a FLAT list and the module
    is a tree; this and `model.Trunk.every_layer` are the two directions of that one seam.
    """
    return [
        QuantizedResidualPair(first, second)
        for first, second in zip(layers[::2], layers[1::2])
    ]


def check_shape(twin):
    """`Model.check_shape`: it raises when the model breaks a rule its consumers assume —
    the layers chain input to output, no layer reads more channels than the int32
    accumulator is exact for, the stem reads the planes and the head states the voices,
    and every constant row covers its output channels.

    A LAYER COUNT THAT IS ODD OR TOO SHORT IS NOT CHECKED HERE, because the tree cannot
    hold one. `load` is where a FILE of the wrong tensor count is refused."""
    layers = twin.every_layer()
    if layers[0].inputs != PLANES:
        raise ValueError(f"the stem reads {layers[0].inputs} planes, not {PLANES}")
    if layers[-1].outputs != sheet.VOICES:
        raise ValueError(
            f"the head states {layers[-1].outputs} channels, "
            f"not the {sheet.VOICES} voices"
        )
    for at, layer in enumerate(layers):
        if at and layer.inputs != layers[at - 1].outputs:
            raise ValueError(
                f"layer {at} reads {layer.inputs} channels and the layer before it "
                f"wrote {layers[at - 1].outputs}"
            )
        if layer.kernel.shape[:2] != (sheet.KERNEL, sheet.KERNEL):
            raise ValueError(f"the kernel of layer {at} is not {sheet.KERNEL} by 3")
        if layer.inputs > WIDEST_INPUTS:
            raise ValueError(
                f"layer {at} reads {layer.inputs} channels and the int32 accumulator "
                f"holds {WIDEST_INPUTS}"
            )
        rows = (layer.gain_q_value[...], layer.gain_q[...], layer.bias[...])
        if any(len(row) != layer.outputs for row in rows):
            raise ValueError(f"the constants of layer {at} do not cover its channels")


# ---------------------------------------------------------------------
# the contract file
# ---------------------------------------------------------------------

# the tensors one layer holds, in the order of the file
LAYER_TENSORS = 5
TEMPER = "temper"
ACTIVATION = "activation_q"
# the tensors the file carries beside its numbered layers
BESIDE_THE_LAYERS = (TEMPER, ACTIVATION)


def save(path, twin):
    """the contract file of `twin`: the module docstring holds the layout and the
    reasons"""
    check_shape(twin)
    tensors = {}
    for at, layer in enumerate(twin.every_layer()):
        base = LAYER_TENSORS * at
        for on, tensor in enumerate(layer.tensors()):
            tensors[str(base + on)] = tensor
    tensors[TEMPER] = twin.temper.tensor()
    tensors[ACTIVATION] = scalar_tensor(ACTIVATION_Q)
    write_contract(
        path,
        tensors,
        {
            "temper_q_value": str(twin.temper.q_value),
            "temper_q": str(twin.temper.q),
            "activation_q": str(ACTIVATION_Q),
            "temperature": repr(twin.temper.temperature),
        },
    )


def load(path):
    """the model of one contract file; a round trip through `save` is exact.

    The temperature is provenance and not arithmetic -- the temper is already folded --
    thus it travels in the metadata alone and only a reader with a Python tool sees it."""
    tensors, metadata = read_contract(path)
    count, spare = divmod(len(tensors) - len(BESIDE_THE_LAYERS), LAYER_TENSORS)
    if spare or count < 4 or count % 2:
        raise ValueError(f"{path}: {len(tensors)} tensors is no quantized sheet model")
    stated = int(tensors[ACTIVATION])
    if stated != ACTIVATION_Q:
        raise ValueError(
            f"{path} is quantized at Q{stated} and this twin reads Q{ACTIVATION_Q}"
        )

    def layer_at(at):
        base = LAYER_TENSORS * at
        return QuantizedNormedConv(
            kernel=tensors[str(base + 0)],
            e=int(tensors[str(base + 1)]),
            gain_q_value=tensors[str(base + 2)],
            gain_q=tensors[str(base + 3)],
            bias=tensors[str(base + 4)],
        )

    stem, *trunk = [layer_at(at) for at in range(count)]
    head = trunk.pop()
    twin = QuantizedCoconet(
        stem=stem,
        pairs=paired(trunk),
        head=head,
        temper=Temper.of_file(tensors, metadata, key=TEMPER),
    )
    check_shape(twin)
    return twin


# ---------------------------------------------------------------------
# the forward pass
# ---------------------------------------------------------------------


def plane_activations(classes, hidden, rows=sheet.ROWS):
    """the stem's input tensor, `[sheets, steps, rows, 2 * VOICES]` in the activation
    format.

    A cell of the masked roll is 0 or one, exact: a standing cell writes the one in its
    class row of plane `voice`, and a hidden cell writes it in EVERY row of plane `VOICES
    + voice`. It is `model.planes` in integers, plane for plane.

    `rows` IS P, THE CIRCUIT'S PARAMETER, and a walk never passes it: the seat registers
    of `model.opening_sheet` reach class 46 and fit no sheet narrower than `ROWS`. What
    passes a narrow P is the stream gate, whose input is data — see `tests/test_rtl.py`.
    """
    if int(classes.max()) >= rows:
        raise ValueError(f"a class of {int(classes.max())} in a column of {rows} rows")
    rows = np.arange(rows)[None, None, :, None]
    roll = rows == classes[:, :, None, :]
    masked = hidden[:, :, None, :]
    planes = np.concatenate(
        [np.where(masked, False, roll), np.broadcast_to(masked, roll.shape)], axis=-1
    )
    return ACTIVATION_ONE * planes.astype(np.int32)


# ---------------------------------------------------------------------
# the integer draw
# ---------------------------------------------------------------------


def tempered_weights(twin, raw):
    """the Q15 weight of every class of one cell, over the batch.

    The logits carry Q`ACTIVATION_Q` and the exp2 unit reads Q12, thus the difference
    shifts up by the gap FIRST. A difference read at the wrong Q is silently wrong music:
    unshifted, every weight stands within a fraction of a nat of the peak and the draw is
    uniform."""
    raw = np.asarray(raw, np.int64)
    peak = raw.max(axis=-1, keepdims=True)
    shifted = (raw - peak) << (EXP2_IN_Q - ACTIVATION_Q)
    return exp2_q(apply_scale(twin.temper.q_value, twin.temper.q, shifted))


# ---------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------


class Draw(NamedTuple):
    """One redraw of a pass, over the batch.

    `hidden` holds the walks the mask hid this cell for. THE OTHER WALKS STATE NOTHING
    HERE: they consumed no uniform, thus `word` and `drawn` carry whatever the batched
    arithmetic happened to compute for them, and no reader may look."""

    step: int
    voice: int
    hidden: np.ndarray  # [sheets]
    word: np.ndarray  # [sheets], the 24-bit uniform
    drawn: np.ndarray  # [sheets], the class


class Pass(NamedTuple):
    """What one pass states.

    `before` is the sheet the forward pass saw, which the drift report teacher-forces the
    float model on. `after` is the sheet its redraws left and `states` the generator
    behind them, thus a caller that wants only the walk reads the last pass."""

    before: np.ndarray  # [sheets, steps, VOICES]
    hidden: np.ndarray  # [sheets, steps, VOICES]
    said: np.ndarray  # [sheets, steps, ROWS, VOICES], the integer logits
    draws: list  # Draw, in the cell order
    after: np.ndarray  # [sheets, steps, VOICES]
    states: np.ndarray  # [sheets], the generator behind the redraws


def passes(twin, states, given, *, walk, tally):
    """One pass at a time: the masks, one integer forward, the redraws in the cell order.

    `given` is the opening, handed over rather than drawn here so that one walk cannot
    open on a sheet the other could not. `states` holds one generator for each sheet, thus
    the walk of seed 7 is the walk of seed 7 in any company, here and on the board.

    A CELL THE MASK LEFT STANDING TAKES NO UNIFORM: a walk that spends one on a standing
    cell states a different piece and no gate below says so. Over a batch that rule is
    `prng.uniform_word`'s `active`.

    THE TWO LOOPS HERE ARE NOT THE COST OF A WALK, and a round that scans them will find
    that out late. Profiled 2026-08-29 at T128 N512 on the golden shape: the walk is 60.3
    s for one sheet, of which the forward above is 81.6 percent and these loops are 18.4 —
    and they are batched numpy that barely grows with the batch, thus they fall to 1.7
    percent at sixteen sheets, which is how a sweep runs. The forward is a jitted int32
    convolution and not overhead. The build-log entry of that day holds the whole
    profile."""
    sheets, steps, _ = given.shape
    idle = np.zeros(sheets, np.int64)
    classes = given
    for at in range(walk):
        threshold = sheet.anneal_threshold(at, walk)
        states, hidden = sheet.hidden_cells(states, steps, threshold)
        said = twin(classes, hidden, tally)
        before, classes, draws = classes, classes.copy(), []
        for step, voice in sheet.cell_order(steps):
            active = hidden[:, step, voice]
            # A CELL NO SHEET HID TAKES NO UNIFORM: an inactive [uniform_word] would
            # leave every generator where it stood anyway. The record stands all the
            # same, because the cell order is what a reader of [draws] walks.
            word = drawn = idle
            if active.any():
                states, word = prng.uniform_word(states, active)
                drawn = pick(tempered_weights(twin, said[:, step, :, voice]), word)
                classes[active, step, voice] = drawn[active]
            draws.append(Draw(step, voice, active, word, drawn))
        yield Pass(before, hidden, said, draws, classes, states)


def gibbs(twin, states, given, *, walk, tally=None):
    """the whole walk: `infer.gibbs` in the arithmetic of the board.

    It gives the sheets and the generator behind them, as `infer.gibbs` does, thus a
    caller can hold the two walks side by side. A walk of no passes is the opening."""
    tally = write_tally() if tally is None else tally
    classes = given
    for taken in passes(twin, states, given, walk=walk, tally=tally):
        classes, states = taken.after, taken.states
    return classes, states


# ---------------------------------------------------------------------
# what the quantization costs
# ---------------------------------------------------------------------


class Drift(NamedTuple):
    """What the quantization costs, measured on the walk the board takes."""

    passes: int
    cells: int  # the redrawn cells: the comparisons of the report
    same_peak: int  # the cells where both models elect the same class
    same_draw: int  # the cells where both models pick the same class
    mean_cosine: float
    activations_clamped: float  # the share of activation writes that rode the clamp
    activation_peak: float  # the hottest write in real units; the format holds 512.0


def drift(coconet, states, given, *, walk, temperature=1.0):
    """the quantized walk, scored against the float model cell for cell.

    The engine walks; at every pass the float model is teacher-forced on the ENGINE'S
    sheet and the ENGINE'S mask, thus what stands between them is the arithmetic alone.
    The same-draw share reads the float draw ON THE VERY UNIFORM THE ENGINE TOOK, thus a
    difference there is never the generator's.

    The quantization happens here, from the float model handed in, thus the pair under
    comparison cannot slip."""
    twin = QuantizedCoconet.of(coconet, temperature)
    tally = write_tally()
    cells = same_peak = same_draw = 0
    cosine = 0.0
    for taken in passes(twin, states, given, walk=walk, tally=tally):
        said = np.asarray(
            sheet.logits(coconet, jnp.asarray(taken.before), jnp.asarray(taken.hidden)),
            dtype=np.float64,
        )
        for drawn in taken.draws:
            active = drawn.hidden
            if not active.any():
                continue
            here = taken.said[active, drawn.step, :, drawn.voice].astype(np.float64)
            there = said[active, drawn.step, :, drawn.voice]
            uniform = drawn.word[active] * 2.0**-prng.UNIFORM_BITS
            cells += int(active.sum())
            same_peak += int((here.argmax(axis=-1) == there.argmax(axis=-1)).sum())
            same_draw += int(
                (
                    sheet.tempered_pick(there, temperature, uniform)
                    == drawn.drawn[active]
                ).sum()
            )
            cosine += float(
                (
                    (here * there).sum(axis=-1)
                    / np.sqrt((here * here).sum(axis=-1) * (there * there).sum(axis=-1))
                ).sum()
            )
    return Drift(
        passes=walk,
        cells=cells,
        same_peak=same_peak,
        same_draw=same_draw,
        mean_cosine=1.0 if cells == 0 else cosine / cells,
        activations_clamped=(
            0.0 if tally["seen"] == 0 else tally["clamped"] / tally["seen"]
        ),
        activation_peak=tally["peak"] / ACTIVATION_ONE,
    )
