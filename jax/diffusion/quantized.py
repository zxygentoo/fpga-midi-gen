"""The integer twin of the masked sheet: the arithmetic the board plays.

The float model of `diffusion/model.py` is what the trainer produced; this is the same
model in the arithmetic the board can hold -- int8 weights with a power-of-two exponent
for each tensor, int16 activations, the norm folded into per-channel constants, and the
draw in integers. THE ORDER OF OPERATIONS IS THE CONTRACT: a rewrite that is algebraically
equal and differently ordered is a different machine, and `tests/test_rtl_diffusion.py`
holds the circuit to these integers write for write.

[Coconet] is a `model.Trunk` under the same attribute names at every level, thus
`coconet.pairs[7].first` and `twin.pairs[7].first` audit against each other. What is not
this era's comes from `quantized.py`; `ar_quantized.py` is the step-frame half and this
era reads none of it.

The formats, and where each rule comes from, are `docs/diffusion_rtl.md`:

- Weights are int8 under the exponent rule of the eras, `quantize`.
- Activations are Q`ACTIVATION_Q` in int16, clamped. Q6 IS MEASURED, not chosen: the
  trunk is a residual stack with no norm on the stream, thus activations grow with depth,
  and the golden candidate peaks at 313 where Q6 holds 512.
- The accumulator is int32 and is exact up to `WIDEST_INPUTS` input channels, thus the sum
  is exact and the order of the taps cannot matter. `check_shape` refuses a wider layer.
- The norm folds at quantization: `gain = scale * rsqrt(variance + eps)` becomes a
  per-channel multiplier that also retires the weight exponent, and
  `bias = shift - mean * gain` becomes Q`ACTIVATION_Q` in int16. Then ReLU; the head keeps
  no ReLU, thus the logits carry the activation format.
- The draw is era four's pipeline: the logit differences shift up to the Q12 the exp2 unit
  reads, temper against the peak under `log2e / T`, exp2 over the shared table, and the
  pick takes a 24-bit uniform.
- The masks and the opening stand in `diffusion/model.py`, where both walks read them.

What the quantization costs is a measurement and not a promise: `tests/test_drift.py`
states it, on the walk the board really takes.

THE CONTRACT FILE is what crosses the seam to the elaboration. `save` writes it and
`Model.of_int8_checkpoint` reads it; `load` reads it back and a round trip is exact. It
carries the quantized model and nothing else -- no population statistics and no float
scales, because the fold happens here, one time.

    tensor            dtype   shape                    value
    "5i + 0"          int32   [3, 3, inputs, outputs]  the kernel q, int8 in int32
    "5i + 1"          int32   []                       the kernel exponent e
    "5i + 2"          int32   [outputs]                the gain q_value
    "5i + 3"          int32   [outputs]                the gain q
    "5i + 4"          int32   [outputs]                the bias, Q6 int16 in int32
    "temper"          int32   [2]                      the temper: q_value, then q
    "activation_q"    int32   []                       the Q of the activation format

EVERY TENSOR IS INT32 -- a fact of the OCaml reader that `quantized.py` states once for
the three eras -- and two of them are not layers.
"""

from collections import deque
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file, save_file

import prng
import quantized as q
from diffusion import model, sample

# the formats


# THE ACTIVATION FORMAT IS Q6 IN INT16, AND IT IS MEASURED; the module docstring holds
# the measurement and the margin
ACTIVATION_Q = 6
ACTIVATION_ONE = 1 << ACTIVATION_Q

# THE WIDEST LAYER THE INT32 ACCUMULATOR IS EXACT FOR: 9 * 57 * 127 * 32767 stands under
# 2^31 and one channel more can pass it
WIDEST_INPUTS = 57

# THE DRAW OF THIS ERA, WHICH RE-ELECTS `quantized.ELECTED_*` AND SAYS SO HERE: the
# paper's sampler is the plain softmax and this era keeps it, with NO min-p floor where
# the step-frame eras hold one at 0.05. A floor here would not mean what it means there --
# a Gibbs redraw picks one cell against a sheet that is still wrong around it, and a floor
# that trimmed its tail would harden the sheet it opened on.
ELECTED_TEMPERATURE = 1.0


# the quantization of a checkpoint


def gain_scale(value, weight_exponent):
    """The 16-bit form of the exponent rule, as (q_value, q). The shift retires the weight
    exponent in the same move, thus the accumulator reaches Q(ACTIVATION_Q) in one
    multiply."""
    e = q.largest_exponent(abs(value), opening=30, cap=32767)
    return int(q.round_half_up(np.ldexp(value, e))), e + weight_exponent


@jax.jit
def accumulate(x, kernel):
    """The nine taps of one 3 by 3 convolution over (step, row), zero at both edges,
    summed into the accumulator; tap (dy, dx) reads (step + dy - 1, row + dx - 1).

    THE ACCUMULATOR IS INT32 AND IT WRAPS, and `check_shape` refuses a layer wide
    enough to reach 2^31 -- thus a wrap here is the finding it would be on the board.
    Below that bound the sum is exact and the circuit may take its own tap order."""
    _, steps, rows, _ = x.shape
    padded = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))

    def tap(dy, dx):
        window = padded[:, dy : dy + steps, dx : dx + rows, :]
        return jnp.matmul(window, kernel[dy, dx])

    return sum(tap(dy, dx) for dy in range(3) for dx in range(3))


# the module tree: model.Trunk in integers


class NormedConv(nnx.Module):
    """One layer as the machine holds it -- the twin of `model.NormedConv`. Five float
    tensors become three facts: the kernel, and the two per-channel rows the norm folded
    into."""

    def __init__(self, *, kernel, e, gain_q_value, gain_q, bias):
        # THE KERNEL LIVES ON THE DEVICE and the rest of the layer on the host: a walk
        # runs [accumulate] hundreds of times, and a host kernel would cross at every call
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
    def from_float(cls, layer):
        """One float [model.NormedConv] under the exponent rule, its norm folded. At
        inference batch normalization is the affine `a * gain + bias` and the fold is
        that same affine, in float64 and IN THE ORDER WRITTEN HERE; its rounding is
        part of what `drift` measures."""
        kernel, e = q.quantize(layer.conv.kernel[...])
        scale = np.asarray(layer.norm.scale[...], np.float64)
        shift = np.asarray(layer.norm.shift[...], np.float64)
        mean = np.asarray(layer.norm.mean[...], np.float64)
        variance = np.asarray(layer.norm.variance[...], np.float64)
        gain = scale / np.sqrt(variance + model.NORM_EPSILON)
        scales = [gain_scale(float(value), e) for value in gain]
        bias = np.clip(
            q.round_half_up((shift - mean * gain) * ACTIVATION_ONE),
            q.INT16_LOW,
            q.INT16_HIGH,
        )
        return cls(
            kernel=kernel,
            e=e,
            gain_q_value=np.array([q_value for q_value, _ in scales], np.int32),
            gain_q=np.array([shift_of for _, shift_of in scales], np.int32),
            bias=bias.astype(np.int32),
        )

    def __call__(self, x, relu):
        """The convolution into the int32 accumulator, the folded norm, the optional ReLU,
        and the clamp of every write. The gain multiply rides int64 -- 47 bits -- and the
        shift is arithmetic, toward minus infinity, as the circuit's is."""
        accumulated = np.asarray(accumulate(jnp.asarray(x), self.kernel[...]))
        gain, shift = self.gain_q_value[...], self.gain_q[...]
        value = ((accumulated.astype(np.int64) * gain) >> shift) + self.bias[...]
        return q.clamp16(np.maximum(value, 0) if relu else value).astype(np.int32)

    def tensors(self):
        """the five tensors of this layer in the order of the contract file"""
        return [
            np.asarray(self.kernel[...], np.int32),
            q.scalar_tensor(self.e),
            self.gain_q_value[...],
            self.gain_q[...],
            self.bias[...],
        ]


class ResidualPair(nnx.Module):
    """Two layers and the skip past both -- the twin of `model.ResidualPair`.

    THE RELU STANDS IN A DIFFERENT PLACE HERE, and that is the contract and not a slip:
    the twin folds it into the first layer's own write, because the machine writes what it
    will read back. The arithmetic is the same and the WRITE STREAM is not."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    @classmethod
    def from_float(cls, pair):
        return cls(
            NormedConv.from_float(pair.first), NormedConv.from_float(pair.second)
        )

    def __call__(self, x):
        """The tensor the opening wrote and the JOINED tensor the close wrote. The
        residual add rides the same clamp, thus the write stream takes the closing tensor
        from here and never from the second layer alone."""
        first = self.first(x, True)
        second = self.second(first, False)
        return first, q.clamp16(np.maximum(x + second, 0)).astype(np.int32)


class Coconet(model.Trunk):
    """The paper's net in the arithmetic the board holds: `model.Coconet`, layer for
    layer. The temper stands beside the layers because the bitstream carries it."""

    def __init__(self, *, stem, pairs, head, temper):
        self.stem = stem
        self.pairs = nnx.List(list(pairs))
        self.head = head
        self.temper = temper

    @classmethod
    def from_float(cls, coconet, temperature=ELECTED_TEMPERATURE):
        """The float model in the arithmetic the board holds. It is the ONE
        quantization of the era -- the drift walk, the audition and the elaboration all
        take their model here -- thus the pair under comparison cannot slip."""
        return cls(
            stem=NormedConv.from_float(coconet.stem),
            pairs=[ResidualPair.from_float(pair) for pair in coconet.pairs],
            head=NormedConv.from_float(coconet.head),
            temper=q.Temper.from_float(temperature),
        )

    def writes(self, classes, hidden, *, rows=model.ROWS):
        """The destination tensor of EVERY layer as written, in the layer order: the
        stem's, then for each pair its opening's tensor and its close's JOINED tensor,
        and the head's logits last.

        IT IS A GENERATOR AND THE TRUNK IS NOT WALKED TWICE: `__call__` keeps the last of
        it and the circuit's stream gate keeps them all, thus a forward pass holds one
        destination tensor and never the trunk's 48. `rows` is P and reaches the stem's
        decode alone."""
        x = self.stem(plane_activations(classes, hidden, rows=rows), True)
        yield x
        for pair in self.pairs:
            first, x = pair(x)
            yield first
            yield x
        yield self.head(x, False)

    def __call__(self, classes, hidden, *, rows=model.ROWS):
        """the logits of one pass over the batch: `[sheets, steps, rows, VOICES]` in the
        activation format, because the head takes no ReLU and keeps it"""
        # the head's write is the last of them, and the deque holds one tensor where a
        # list would hold the trunk's 48
        return deque(self.writes(classes, hidden, rows=rows), maxlen=1)[0]


def paired(layers):
    """The trunk's layers two at a time. The contract file is a FLAT list and the
    module is a tree; this and `model.Trunk.layers` are the two directions of that
    seam."""
    return [
        ResidualPair(first, second)
        for first, second in zip(layers[::2], layers[1::2])
    ]


def check_shape(twin):
    """`Model.check_shape`: the layers chain input to output, no layer reads more
    channels than the int32 accumulator is exact for, the stem reads the planes, the
    head states the voices, and every constant row covers its output channels. A LAYER
    COUNT THAT IS ODD OR TOO SHORT IS NOT CHECKED HERE, because the tree cannot hold
    one."""
    layers = twin.layers()
    if layers[0].inputs != model.PLANES:
        raise ValueError(
            f"the stem reads {layers[0].inputs} planes, not {model.PLANES}"
        )
    if layers[-1].outputs != model.VOICES:
        raise ValueError(
            f"the head states {layers[-1].outputs} channels, "
            f"not the {model.VOICES} voices"
        )
    for at, layer in enumerate(layers):
        if at and layer.inputs != layers[at - 1].outputs:
            raise ValueError(
                f"layer {at} reads {layer.inputs} channels and the layer before it "
                f"wrote {layers[at - 1].outputs}"
            )
        if layer.kernel.shape[:2] != (model.KERNEL, model.KERNEL):
            raise ValueError(f"the kernel of layer {at} is not {model.KERNEL} by 3")
        if layer.inputs > WIDEST_INPUTS:
            raise ValueError(
                f"layer {at} reads {layer.inputs} channels and the int32 accumulator "
                f"holds {WIDEST_INPUTS}"
            )
        rows = (layer.gain_q_value[...], layer.gain_q[...], layer.bias[...])
        if any(len(row) != layer.outputs for row in rows):
            raise ValueError(f"the constants of layer {at} do not cover its channels")


# the contract file


# the tensors one layer holds, in the order of the file
LAYER_TENSORS = 5
# `TEMPER` is `contract`'s; this era's own scalar name is the activation Q
ACTIVATION = "activation_q"
# the tensors the file carries beside its numbered layers
BESIDE_THE_LAYERS = (q.TEMPER, ACTIVATION)


def save(path, twin):
    """the contract file of `twin`: the module docstring holds the layout and the
    reasons"""
    check_shape(twin)
    tensors = {}
    for at, layer in enumerate(twin.layers()):
        base = LAYER_TENSORS * at
        for on, tensor in enumerate(layer.tensors()):
            tensors[str(base + on)] = tensor
    tensors[q.TEMPER] = twin.temper.tensor()
    tensors[ACTIVATION] = q.scalar_tensor(ACTIVATION_Q)
    save_file(tensors, str(path))


def load(path):
    """the model of one contract file; a round trip through `save` is exact"""
    tensors = load_file(str(path))
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
        return NormedConv(
            kernel=tensors[str(base + 0)],
            e=int(tensors[str(base + 1)]),
            gain_q_value=tensors[str(base + 2)],
            gain_q=tensors[str(base + 3)],
            bias=tensors[str(base + 4)],
        )

    stem, *trunk = [layer_at(at) for at in range(count)]
    head = trunk.pop()
    twin = Coconet(
        stem=stem,
        pairs=paired(trunk),
        head=head,
        temper=q.Temper.from_file(tensors, key=q.TEMPER),
    )
    check_shape(twin)
    return twin


# the forward pass


def plane_activations(classes, hidden, rows=model.ROWS):
    """`model.planes` in integers: the stem's input tensor, exact, because a cell of the
    masked roll is 0 or one.

    `rows` IS P, THE CIRCUIT'S PARAMETER, and a walk never passes it -- the seat registers
    of `model.opening_sheet` fit no sheet narrower than `ROWS`. What passes a narrow P is
    the stream gate, whose input is data."""
    if int(classes.max()) >= rows:
        raise ValueError(f"a class of {int(classes.max())} in a column of {rows} rows")
    row_index = np.arange(rows)[None, None, :, None]
    roll = row_index == classes[:, :, None, :]
    masked = hidden[:, :, None, :]
    planes = np.concatenate(
        [np.where(masked, False, roll), np.broadcast_to(masked, roll.shape)], axis=-1
    )
    return ACTIVATION_ONE * planes.astype(np.int32)


# the integer draw


def class_weights(twin, raw):
    """The Q15 weight of every class of one cell, over the batch. The logits carry
    Q`ACTIVATION_Q` and the exp2 unit reads Q12, thus the difference SHIFTS UP FIRST:
    unshifted, every weight stands within a fraction of a nat of the peak and the draw
    is uniform. It is not `ar_quantized.tempered_weights`, which reads Q12 already and
    holds a min-p floor."""
    raw = np.asarray(raw, np.int64)
    peak = raw.max(axis=-1, keepdims=True)
    shifted = (raw - peak) << (q.EXP2_IN_Q - ACTIVATION_Q)
    return q.exp2_q(q.apply_scale(twin.temper.q_value, twin.temper.q, shifted))


# the walk


class Draw(NamedTuple):
    """One redraw of a pass, over the batch. `hidden` holds the walks the mask hid this
    cell for; THE OTHER WALKS STATE NOTHING -- they consumed no uniform, thus `word` and
    `drawn` carry whatever the batched arithmetic happened to compute."""

    step: int
    voice: int
    hidden: np.ndarray  # [sheets]
    word: np.ndarray  # [sheets], the 24-bit uniform
    drawn: np.ndarray  # [sheets], the class


class Pass(NamedTuple):
    """`sample.Pass` with the `Draw` of every cell of the order beside it."""

    read: np.ndarray  # [sheets, steps, VOICES]
    hidden: np.ndarray  # [sheets, steps, VOICES]
    logits: np.ndarray  # [sheets, steps, ROWS, VOICES], the integer logits
    draws: list  # Draw, in the cell order
    redrawn: np.ndarray  # [sheets, steps, VOICES]
    states: np.ndarray  # [sheets], the generator behind the redraws


def passes(twin, states, given, *, walk):
    """The INTEGER walk of the era, one pass at a time: `sample.gibbs_passes` in the
    arithmetic of the board, with the record the drift report reads.

    The loop, the schedule and the order of the draws stand in `gibbs_passes`, once for
    both walks. What is here is this walk's arithmetic and the `Draw` of every cell of the
    ORDER: a cell nothing hid states an idle record. `given` is the opening, handed over
    rather than drawn here so that one walk cannot open on a sheet the other could not.

    THE CELL LOOPS ARE NOT THE COST OF A WALK, and a round that scans them will find
    that out late: profiled at T128 N512 on the golden shape, the forward is 81.6
    percent of a one-sheet walk and the loops are batched numpy that falls to 1.7
    percent at sixteen."""
    sheets, steps, _ = given.shape
    idle = np.zeros(sheets, np.int64)
    spent = {}

    def forward(classes, hidden):
        return twin(classes, hidden)

    def redraw(states, logits, step, voice, active):
        states, word = prng.uniform_word(states, active)
        drawn = q.pick(class_weights(twin, logits[:, step, :, voice]), word)
        spent[step, voice] = (word, drawn)
        return states, drawn

    for taken in sample.gibbs_passes(
        states, given, passes=walk, forward=forward, redraw=redraw
    ):
        draws = [
            Draw(
                step,
                voice,
                taken.hidden[:, step, voice],
                *spent.get((step, voice), (idle, idle)),
            )
            for step, voice in model.cell_order(steps)
        ]
        spent.clear()
        yield Pass(
            taken.read, taken.hidden, taken.logits, draws, taken.redrawn, taken.states
        )


def gibbs(twin, states, given, *, walk):
    """The whole walk: `infer.gibbs` in the arithmetic of the board, giving the sheets and
    the generator behind them so a caller can hold the two side by side. A walk of no
    passes is the opening."""
    classes = given
    for taken in passes(twin, states, given, walk=walk):
        classes, states = taken.redrawn, taken.states
    return classes, states


