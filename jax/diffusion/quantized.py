"""The integer twin of the masked sheet: the arithmetic the board plays.

The float model of `diffusion/model.py` is what the trainer produced. This module is the
same model in the arithmetic the board can hold -- int8 weights with a power-of-two
exponent for each tensor, int16 activations, the norm folded into per-channel constants,
and the draw in integers -- and the circuit must equal it operation for operation, not
approximately. Nothing here approximates on purpose: every shift, every floor and every
table is a rule the RTL reads from this module rather than restates.

IT CARRIES THE FLOAT MODEL'S SKELETON. [QuantizedCoconet] is a `model.Trunk`, as
`model.Coconet` is: a stem, the residual pairs, a head, under the same attribute names at
every level. A reader can put `coconet.pairs[7].first` beside `twin.pairs[7].first` and
audit one layer against its twin; nothing has to be aligned by hand from two lists.

THE ORDER OF OPERATIONS IS THE CONTRACT. Every function below mirrors one function of the
OCaml twin this module replaced, and each names it. A rewrite that is algebraically equal
and differently ordered is a different machine: the gates of `tests/test_rtl.py` hold the
circuit to these integers write for write.

A CITATION NAMES WHAT A READER CAN FIND. `Model.*` and `Nn_quantized.*` are live OCaml and
still stand; `Quantized.*` names the twin AS IT WAS, and the cut of 2026-08-28 (commit
5ed90f8) deleted it -- those names are history, and `git show 5ed90f8^:lib/diffusion/
quantized.ml` is where a reader finds them.

The formats, and where each rule comes from, are `docs/diffusion_rtl.md`:

- Weights are int8 under the exponent rule of the eras, `quantize`.
- Activations are Q`ACTIVATION_Q` in int16, clamped and counted -- Q6, AND THE NUMBER IS
  MEASURED, not chosen: the trunk is a residual stack with no norm on the stream, thus a
  trained model's activations grow with depth, and the golden candidate peaks at 184 on
  half-masked corpus sheets and at 313 on the seeded openings the walk really visits.
  Q6 holds 512 with a 1.6 margin. The input planes enter exact -- a cell is 0 or one.
- The accumulator is int32 and is exact up to `WIDEST_INPUTS` input channels -- 9 C
  products of int8 by int16 reach under 2^31 there and one channel more can pass it --
  thus the sum is exact and the order of the taps cannot matter. `check_shape` refuses a
  wider layer, thus the bound is a rule and not a comment.
- The norm folds at quantization: `gain = scale * rsqrt(variance + eps)` becomes a
  per-channel multiplier that also retires the weight exponent, and
  `bias = shift - mean * gain` becomes Q`ACTIVATION_Q` in int16. Then ReLU; the head keeps
  no ReLU, thus the logits carry the activation format.
- The draw is era four's pipeline: the logit differences shift up to the Q12 the exp2 unit
  reads -- exact, a left shift -- then temper against the peak under `log2e / T`, exp2 over
  the shared table gives Q15 weights, and the pick takes a 24-bit uniform.
- The masks and the opening are the integer rules of the walk already, and the twin
  consumes the same uniforms in the same places as `infer.gibbs`.

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

EVERY TENSOR IS INT32 AND TWO OF THEM ARE NOT LAYERS. Both are facts of the reader and not
tastes: `Nx_io.load_safetensors` reads F32, F64, I32, F16 and BF16 and skips every other
dtype with a warning, thus an int8 or int16 tensor would arrive at the elaboration as a
hole; and it gives no access to `__metadata__`, thus the two numbers the elaboration needs
-- the temper and the Q it was quantized at -- travel as named tensors beside the numbered
layers. The metadata is written as well, for a reader that has a Python tool in hand.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

import nn
import prng
from diffusion import model as sheet

# ---------------------------------------------------------------------
# the formats
# ---------------------------------------------------------------------

# THE ACTIVATION FORMAT IS Q6 IN INT16, AND IT IS MEASURED -- the module docstring holds
# the measurement and the margin. `activation_one` is the one of the format, and the fixed
# point of the biases.
ACTIVATION_Q = 6
ACTIVATION_ONE = 1 << ACTIVATION_Q
ACTIVATION_BITS = 16
ACCUMULATOR_BITS = 32

INT16_LOW = -(1 << (ACTIVATION_BITS - 1))
INT16_HIGH = (1 << (ACTIVATION_BITS - 1)) - 1

# THE WIDEST LAYER THE INT32 ACCUMULATOR IS EXACT FOR: 9 C products of int8 by int16 reach
# 9 * 57 * 127 * 32767, which stands under 2^31, and one channel more can pass it. The
# elected shapes stand far under; the rule stands so the prose cannot rot.
WIDEST_INPUTS = 57

# the planes the stem reads: one class plane and one mask plane for each seat
PLANES = 2 * sheet.VOICES

# the Q of log2(e), and the Q the temper takes: one below it. The extra bit is headroom for
# the temperature -- the circuits carry this constant on an 18-bit signed port, thus the Q
# of log2(e) would overflow that port under a temperature of about 0.36.
LOG2E_Q = 15
TEMPER_Q = LOG2E_Q - 1

# the Q the exp2 unit reads its magnitudes at, and the Q of its answer
EXP2_IN_Q = 12
EXP2_OUT_Q = 15


def round_half_up(x):
    """Base's `Float.iround_nearest_exn`: floor(x + 0.5).

    A TIE GOES TOWARD PLUS INFINITY, thus -2.5 is -2 and 2.5 is 3. Python's `round` and
    `numpy.rint` are half-to-even and state a different number at every tie; every rounding
    of this module goes through here so that no reading of it can drift."""
    return np.floor(np.asarray(x, np.float64) + 0.5)


# ---------------------------------------------------------------------
# the quantization of a checkpoint
# ---------------------------------------------------------------------


def max_exponent(peak):
    """`Nn_quantized.max_exponent`: the largest exponent, from 14 down, that keeps
    round(peak * 2^e) at 127 or less.

    14 caps the all-zero tensor, where every exponent fits. The predicate falls
    monotonically in e, thus the first e that fits is the largest."""
    if peak <= 0.0:
        return 14
    e = 14
    while round_half_up(np.ldexp(peak, e)) > 127:
        e -= 1
    return e


def quantize(weights):
    """`Nn_quantized.quantize`: the int8 form of one tensor, and the exponent that reads it.

    The byte is two's complement and the negative end is not used: the clamp is -127 and
    not -128, thus the image is symmetric and a negated weight is a negated byte."""
    weights = np.asarray(weights, np.float64)
    e = max_exponent(float(np.abs(weights).max(initial=0.0)))
    return np.clip(round_half_up(np.ldexp(weights, e)), -127, 127).astype(np.int32), e


def gain_scale(value, weight_exponent):
    """`Quantized.Model.gain_scale`: the 16-bit form of the exponent rule, as (q_value, q).

    The shift of the scale retires the weight exponent in the same move, thus the
    accumulator goes from Q(ACTIVATION_Q + e) to Q(ACTIVATION_Q) in one multiply. 30 caps
    the all-zero gain, as 14 caps the all-zero tensor."""
    magnitude = abs(value)
    e = 30
    if magnitude > 0.0:
        while round_half_up(np.ldexp(magnitude, e)) > 32767:
            e -= 1
    return int(round_half_up(np.ldexp(value, e))), e + weight_exponent


def temper_of(temperature):
    """`Nn_quantized.policy`: the sampling temper, log2(e) / T, as (q_value, q)."""
    if temperature <= 0.0:
        raise ValueError("the temperature is positive")
    return int(round_half_up(np.ldexp(1.0 / math.log(2.0) / temperature, TEMPER_Q))), (
        TEMPER_Q
    )


class Temper(NamedTuple):
    """The sampling temper as the bitstream carries it: log2(e) / T at [q].

    The temperature is PROVENANCE and not arithmetic -- the temper is already folded -- thus
    it travels in the metadata of the contract file alone, and a file written by an older
    tool can read back with no temperature at all."""

    q_value: int
    q: int
    temperature: float

    @classmethod
    def of(cls, temperature):
        q_value, q = temper_of(temperature)
        return cls(q_value, q, temperature)


# ---------------------------------------------------------------------
# the counted activation write
# ---------------------------------------------------------------------


def counters():
    """`Quantized.Clamps` as a running tally: the activation writes, the writes that rode
    the clamp, and the hottest write BEFORE it.

    The formats are chosen with margin and not metered on a trained checkpoint; a clamp that
    fires is the finding that says which format is wrong, thus it is counted and never
    assumed away. The peak reads before the clamp, thus it answers the format question
    directly."""
    return {"seen": 0, "clamped": 0, "peak": 0}


def write(tally, value):
    """`Quantized.write`: every activation write goes through here -- the clamp is counted
    and the peak is kept, never assumed away.

    The peak is `max(|v|)` and the clamp fires outside int16, thus a peak inside the format
    proves that nothing clamped and the clip is skipped: the walk writes millions of these
    and the short circuit is the whole of the difference."""
    high, low = int(value.max()), int(value.min())
    tally["seen"] += value.size
    tally["peak"] = max(tally["peak"], high, -low)
    if high <= INT16_HIGH and low >= INT16_LOW:
        return value.astype(np.int32)
    tally["clamped"] += int(np.count_nonzero(value > INT16_HIGH))
    tally["clamped"] += int(np.count_nonzero(value < INT16_LOW))
    return np.clip(value, INT16_LOW, INT16_HIGH).astype(np.int32)


@jax.jit
def accumulate(x, kernel):
    """The nine taps of one 3 by 3 convolution over (step, row), zero at both edges, summed
    into the accumulator -- the inner loops of `Quantized.layer_forward`.

    THE ACCUMULATOR IS INT32 AND IT WRAPS. The twin's claim is that no sum reaches 2^31
    below `WIDEST_INPUTS` input channels, and `check_shape` refuses a wider layer, thus a
    wrap here is the finding it would be on the board and not an artefact of the host.

    The tap (dy, dx) reads the source at (step + dy - 1, row + dx - 1), and the kernel reads
    as [dy, dx, input, output]; the accumulator is exact below the bound, thus the tap order
    cannot matter and the circuit may take its own."""
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

    The five float tensors of that layer become three facts: the kernel, and the two
    per-channel constant rows the norm folded into. The gains are scales whose shift
    retires the weight exponent, thus the accumulator reaches the activation format in one
    multiply; the biases are int16 in that same format."""

    def __init__(self, *, kernel, e, gain_q_value, gain_q, bias):
        # THE KERNEL LIVES ON THE DEVICE and the rest of the layer on the host. A walk runs
        # [accumulate] hundreds of times over one model, thus a host kernel would cross to
        # the device at every call; the gains, the bias and the clamp are numpy int64
        # arithmetic and belong beside the tally.
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
        """`Quantized.Model.fold_layer`: one float [model.NormedConv] under the exponent
        rule, its norm folded.

        At inference batch normalization is the affine `a * gain + bias` with
        `gain = scale * rsqrt(variance + eps)` and `bias = shift - mean * gain`. The fold is
        the same affine -- its rounding is part of what `drift` measures -- and it runs in
        float64 over the float32 tensors of the checkpoint, in the order written here. A
        bias outside the activation format clamps; a trained norm that puts one there is a
        format fault the drift report would shout about."""
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
        """`Quantized.layer_forward`: the convolution into the int32 accumulator, the folded
        norm, the optional ReLU, and the counted clamp of every write.

        The gain multiply rides int64 -- an int32 accumulator by an int16 gain wants 47 bits,
        and the RTL sizes its own product. The shift is arithmetic, toward minus infinity, as
        the circuits' is."""
        accumulated = np.asarray(accumulate(jnp.asarray(x), self.kernel[...]))
        gain, shift = self.gain_q_value[...], self.gain_q[...]
        value = ((accumulated.astype(np.int64) * gain) >> shift) + self.bias[...]
        return write(tally, np.maximum(value, 0) if relu else value)

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

    THE RELU STANDS IN A DIFFERENT PLACE HERE, and that is the contract and not a slip. The
    float pair activates the first layer's output before the second convolution reads it;
    the twin folds that ReLU into the first layer's own counted write, because the machine
    writes what it will read back. The arithmetic is the same and the WRITE STREAM is not,
    and the write stream is what `tests/test_rtl.py` holds the circuit to."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    @classmethod
    def of(cls, pair):
        return cls(
            QuantizedNormedConv.of(pair.first), QuantizedNormedConv.of(pair.second)
        )

    def __call__(self, x, tally):
        """the tensor the opening wrote and the JOINED tensor the close wrote.

        THE PAIR-CLOSING WRITE IS THE JOINED TENSOR AND NOT THE CONVOLUTION'S: the residual
        add rides the same counted clamp, thus a reader of the write stream takes it from
        here and never from the second layer alone."""
        first = self.first(x, True, tally)
        second = self.second(first, False, tally)
        return first, write(tally, np.maximum(x + second, 0))


def _collect(held, written):
    return held + [written]


def _hold_the_last(held, written):
    del held
    return written


class QuantizedCoconet(sheet.Trunk):
    """The paper's net in the arithmetic the board holds: `model.Coconet`, layer for layer.

    The temper stands beside the layers because the bitstream carries it: one quantization
    serves every seed of a batch, as one bitstream serves every seed of the board."""

    def __init__(self, *, stem, pairs, head, temper):
        self.stem = stem
        self.pairs = nnx.List(list(pairs))
        self.head = head
        self.temper = temper

    @classmethod
    def of(cls, coconet, temperature=1.0):
        """`Quantized.Model.of_params`: the float model in the arithmetic the board holds.

        This is the one quantization of the era -- the drift walk, the audition and the
        elaboration all take their model here, thus the pair under comparison cannot slip."""
        return cls(
            stem=QuantizedNormedConv.of(coconet.stem),
            pairs=[QuantizedResidualPair.of(pair) for pair in coconet.pairs],
            head=QuantizedNormedConv.of(coconet.head),
            temper=Temper.of(temperature),
        )

    def _fold_writes(self, classes, hidden, tally, taken, keep):
        """`Quantized.fold_layer_writes`: the trunk -- the stem, the residual pairs, the head
        -- with `keep` folded over the destination tensor of EVERY layer as written, in the
        layer order.

        THE COLLECTOR IS WHAT PARTS `__call__` FROM `layer_writes`, AND THE TRUNK IS NOT
        WALKED TWICE: a forward that read the last of a list would hold every layer's
        destination tensor alive to give one back -- 48 of them at the elected shape."""
        x = self.stem(plane_activations(classes, hidden), True, tally)
        taken = keep(taken, x)
        for pair in self.pairs:
            first, x = pair(x, tally)
            taken = keep(keep(taken, first), x)
        return keep(taken, self.head(x, False, tally))

    def __call__(self, classes, hidden, tally):
        """the logits of one pass over the batch: `[sheets, steps, ROWS, VOICES]` in the
        activation format, because the head takes no ReLU and keeps it"""
        return self._fold_writes(classes, hidden, tally, None, _hold_the_last)

    def layer_writes(self, classes, hidden, tally):
        """`Quantized.layer_writes`: the destination tensor of every layer AS WRITTEN, in the
        layer order -- the stem's, then for each pair the tensor its opening wrote and the
        JOINED tensor its close wrote, and the head's logits last. Each one reads as
        `[sheets, steps, ROWS, that layer's output channels]`.

        IT IS FOR THE CIRCUIT'S STREAM GATE AND FOR NOTHING ELSE. `__call__` is the last of
        this list, thus the list is the arithmetic itself and never a second reading of it."""
        return self._fold_writes(classes, hidden, tally, [], _collect)


def paired(layers):
    """The trunk's layers two at a time, as [QuantizedResidualPair]s.

    The contract file is a FLAT list and the module is a tree; this and
    `model.Trunk.every_layer` are the two directions of that one seam, and nothing else
    knows both forms."""
    return [
        QuantizedResidualPair(first, second)
        for first, second in zip(layers[::2], layers[1::2])
    ]


def check_shape(twin):
    """`Model.check_shape`: it raises when the model breaks a rule its consumers
    assume -- the layers chain input to output, no layer reads more channels than the int32
    accumulator is exact for, the stem reads the planes and the head states the voices, and
    every constant row holds one entry for each output channel.

    A LAYER COUNT THAT IS ODD OR TOO SHORT IS NOT CHECKED HERE, because the tree cannot hold
    one: a [QuantizedCoconet] is a stem, whole pairs and a head by construction. `load` is
    where a FILE of the wrong tensor count is refused."""
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
    """the contract file of `twin`: the module docstring holds the layout and the reasons"""
    check_shape(twin)
    tensors = {}
    for at, layer in enumerate(twin.every_layer()):
        base = LAYER_TENSORS * at
        for on, tensor in enumerate(layer.tensors()):
            tensors[str(base + on)] = tensor
    tensors[TEMPER] = np.array([twin.temper.q_value, twin.temper.q], np.int32)
    tensors[ACTIVATION] = np.array(ACTIVATION_Q, np.int32)
    save_file(
        tensors,
        str(path),
        metadata={
            "temper_q_value": str(twin.temper.q_value),
            "temper_q": str(twin.temper.q),
            "activation_q": str(ACTIVATION_Q),
            "temperature": repr(twin.temper.temperature),
        },
    )


def load(path):
    """the model of one contract file; a round trip through `save` is exact.

    The temperature is provenance and not arithmetic -- the temper is already folded -- thus
    it travels in the metadata alone and only a reader with a Python tool sees it."""
    tensors = load_file(str(path))
    with safe_open(str(path), framework="numpy") as opened:
        metadata = opened.metadata() or {}
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

    q_value, q = (int(value) for value in tensors[TEMPER])
    stem, *trunk = [layer_at(at) for at in range(count)]
    head = trunk.pop()
    twin = QuantizedCoconet(
        stem=stem,
        pairs=paired(trunk),
        head=head,
        temper=Temper(q_value, q, float(metadata.get("temperature", math.nan))),
    )
    check_shape(twin)
    return twin


# ---------------------------------------------------------------------
# the forward pass
# ---------------------------------------------------------------------


def plane_activations(classes, hidden):
    """`Quantized.plane_activations`: the stem's input tensor, `[sheets, steps, ROWS,
    2 * VOICES]` in the activation format.

    A cell of the masked roll is 0 or one, exact: a standing cell writes the one in its
    class row of plane `voice`, and a hidden cell writes it in EVERY row of plane
    `VOICES + voice`. It is `model.planes` in integers, plane for plane."""
    rows = np.arange(sheet.ROWS)[None, None, :, None]
    roll = rows == classes[:, :, None, :]
    masked = hidden[:, :, None, :]
    planes = np.concatenate(
        [np.where(masked, False, roll), np.broadcast_to(masked, roll.shape)], axis=-1
    )
    return ACTIVATION_ONE * planes.astype(np.int32)


# ---------------------------------------------------------------------
# the integer draw
# ---------------------------------------------------------------------

# the quantized exponential: exp2 of -j/256 in Q15 -- `Nn_quantized.Constants.exp2_table`,
# the one table the samplers of every era read
EXP2_TABLE = np.array(
    [
        int(round_half_up(float(1 << EXP2_OUT_Q) * 2.0 ** (-j / 256.0)))
        for j in range(256)
    ],
    np.int64,
)


def exp2_of_magnitude(magnitude):
    """`Nn_quantized.exp2_of_magnitude`: 2^-m in Q15 over a nonnegative Q12 magnitude.

    The integer part shifts and the top eight bits of the fraction index the table; the
    peak -- a magnitude of 0 -- is 2^15, and a magnitude of 16 or more is 0. The shift is
    held under the width of the host word where the answer is 0 anyway, because a shift
    past the width states nothing in either language."""
    whole = magnitude >> EXP2_IN_Q
    entry = EXP2_TABLE[(magnitude >> (EXP2_IN_Q - 8)) & 255]
    return np.where(whole >= 16, 0, entry >> np.minimum(whole, 62))


def tempered_weights(twin, raw):
    """`Quantized.draw_cell`: the Q15 weight of every class of one cell, over the batch.

    `raw` is `[sheets, ROWS]`. The logits carry Q`ACTIVATION_Q` and the exp2 unit reads
    Q12, thus the difference shifts up by the gap FIRST -- exact, because a left shift of an
    integer is. A difference read at the wrong Q is silently wrong music: unshifted, every
    weight stands within a fraction of a nat of the peak and the draw is uniform."""
    raw = np.asarray(raw, np.int64)
    peak = raw.max(axis=-1, keepdims=True)
    shifted = (raw - peak) << (EXP2_IN_Q - ACTIVATION_Q)
    return exp2_of_magnitude(-((shifted * twin.temper.q_value) >> twin.temper.q))


def pick(weights, word):
    """`Nn_quantized.draw`: the class a 24-bit uniform word lands, over the batch.

    The total is the last running total and never a second sum of the same weights; the
    threshold is `(word * total) >> 24` and the class is the first whose running total
    passes it, or the last. THE PICK ALWAYS LANDS: the peak weighs 2^15, thus the total is
    2^15 or more, and the word falls under 2^24, thus the threshold stands strictly under
    the total. No fallback is necessary and none is written."""
    running = np.cumsum(weights, axis=-1)
    threshold = (np.asarray(word, np.int64) * running[..., -1]) >> prng.UNIFORM_BITS
    return (running > threshold[..., None]).argmax(axis=-1)


# ---------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------


def engine_states(seeds):
    """the generator of each walk under `prng.create`: THE SEED AS IT STANDS, which is the
    rule of the board's SEED cell -- thus seed 0 is the walk that stands still, as the
    circuit stands still on it.

    `prng.states` folds instead, which is the float walk's rule: a seed inside 32 bits
    names itself under both, and 0 is the one seed where the two walks are not one walk."""
    return np.array([prng.create(int(seed)) for seed in seeds], dtype=np.uint32)


def anneal_threshold(step, walk):
    """`Model.anneal_threshold`: the masking threshold of pass `step` of `walk`, on the
    24-bit grid of the generator. A cell hides exactly when its word falls under it."""
    return math.floor(sheet.anneal(step, walk) * 2.0**prng.UNIFORM_BITS)


def hidden_cells(states, steps, threshold, everyone):
    """`Model.hidden_cells`: the mask of one pass -- one uniform for each cell in the
    cell order, step-major and seat-minor, hidden exactly when its word falls under the
    threshold.

    The word compare and the float compare `u * 2^24 < threshold` are one test: the product
    is the word, exactly, on the grid."""
    hidden = np.zeros((len(states), steps, sheet.VOICES), dtype=bool)
    for step in range(steps):
        for voice in range(sheet.VOICES):
            states, word = prng.uniform_word(states, everyone)
            hidden[:, step, voice] = word < threshold
    return states, hidden


class Draw(NamedTuple):
    """One redraw of a pass -- `Quantized.Engine.draw`, over the batch.

    `hidden` holds the walks the mask hid this cell for. THE OTHER WALKS STATE NOTHING
    HERE: they consumed no uniform, thus `word` and `drawn` carry whatever the batched
    arithmetic happened to compute for them and no reader may look."""

    step: int
    voice: int
    hidden: np.ndarray  # [sheets]
    word: np.ndarray  # [sheets], the 24-bit uniform
    drawn: np.ndarray  # [sheets], the class


class Pass(NamedTuple):
    """What one pass states -- `Quantized.Engine.pass`.

    `before` is the sheet the forward pass saw; the drift report teacher-forces the float
    model on exactly it. `after` is the sheet its redraws left, and `states` the generator
    behind them, thus a caller that wants only the walk reads the last pass and no other."""

    before: np.ndarray  # [sheets, steps, VOICES]
    hidden: np.ndarray  # [sheets, steps, VOICES]
    said: np.ndarray  # [sheets, steps, ROWS, VOICES], the integer logits
    draws: list  # Draw, in the cell order
    after: np.ndarray  # [sheets, steps, VOICES]
    states: np.ndarray  # [sheets], the generator behind the redraws


def passes(twin, states, given, *, walk, tally):
    """`Quantized.Engine.next_pass`, one pass at a time: the masks, one integer forward, the
    redraws in the cell order.

    `given` is the opening -- `infer.opening_sheet` draws it, and it is handed over rather
    than drawn here so that the twin never reaches back into the audition. `states` holds
    one generator for each sheet, thus every sheet of a batch is one reproducible piece:
    the walk of seed 7 is the walk of seed 7 in any company, here and on the board.

    A CELL THE MASK LEFT STANDING TAKES NO UNIFORM. The opening and the mask draw for every
    cell, but a redraw draws only where the mask hid; a walk that spends a uniform on a
    standing cell states a different piece and no gate below says so. Over a batch that rule
    is `prng.uniform_word`'s `active`: a sheet whose cell stands keeps the state it came in
    with, and a sheet whose cell hides advances."""
    sheets, steps, voices = given.shape
    everyone = np.ones(sheets, dtype=bool)
    classes = given
    for at in range(walk):
        states, hidden = hidden_cells(states, steps, anneal_threshold(at, walk), everyone)
        said = twin(classes, hidden, tally)
        before, classes, draws = classes, classes.copy(), []
        for step in range(steps):
            for voice in range(voices):
                active = hidden[:, step, voice]
                states, word = prng.uniform_word(states, active)
                drawn = np.zeros(sheets, np.int64)
                if active.any():
                    drawn = pick(tempered_weights(twin, said[:, step, :, voice]), word)
                    classes[active, step, voice] = drawn[active]
                draws.append(Draw(step, voice, active, word, drawn))
        yield Pass(before, hidden, said, draws, classes, states)


def gibbs(twin, states, given, *, walk, tally=None):
    """the whole walk: `infer.gibbs` in the arithmetic of the board.

    It gives the sheets and the generator behind them, as `infer.gibbs` does, thus a
    caller can hold the two walks side by side. A walk of no passes is the opening."""
    tally = counters() if tally is None else tally
    classes = given
    for taken in passes(twin, states, given, walk=walk, tally=tally):
        classes, states = taken.after, taken.states
    return classes, states


# ---------------------------------------------------------------------
# what the quantization costs
# ---------------------------------------------------------------------


class Drift(NamedTuple):
    """What the quantization costs, measured on the walk the board takes --
    `Quantized.Drift.stats`."""

    passes: int
    cells: int  # the redrawn cells: the comparisons of the report
    same_peak: int  # the cells where both models elect the same class
    same_draw: int  # the cells where both models pick the same class
    mean_cosine: float
    activations_clamped: float  # the share of activation writes that rode the clamp
    activation_peak: float  # the hottest write in real units; the format holds 512.0


@nnx.jit
def _float_logits(coconet, classes, hidden):
    """the float model's logits over the batch. It is `infer.forward`'s pass, called here
    from the model's own home so that the twin never imports the audition."""
    said, _ = coconet(sheet.planes(classes, hidden))
    return said


def drift(coconet, states, given, *, walk, temperature=1.0):
    """`Quantized.Drift.walk`: the quantized walk, scored against the float model cell for
    cell.

    The engine walks; at every pass the float model is teacher-forced on the ENGINE'S sheet
    and the ENGINE'S mask, thus the two read one context and what stands between them is the
    arithmetic alone. The same-draw share reads the float draw ON THE VERY UNIFORM THE
    ENGINE TOOK -- `infer.tempered_pick`'s two calls, from `nn`'s own home -- thus a
    difference there is the arithmetic and never the generator.

    The quantization happens here, from the float model handed in, thus the pair under
    comparison cannot slip."""
    twin = QuantizedCoconet.of(coconet, temperature)
    tally = counters()
    cells = same_peak = same_draw = 0
    cosine = 0.0
    for taken in passes(twin, states, given, walk=walk, tally=tally):
        said = np.asarray(
            _float_logits(coconet, jnp.asarray(taken.before), jnp.asarray(taken.hidden)),
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
                    nn.pick(nn.temper(there, temperature, 0.0), uniform)
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
