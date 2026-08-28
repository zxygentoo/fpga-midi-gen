"""The integer twin of the state-space model: the arithmetic the board plays.

The float model of `mamba/model.py` is what the ear elected. This module is the same
network in the arithmetic the board can hold -- int8 weights with a power-of-two exponent
for each tensor, int16 activations, the per-head numbers folded into the constants the ops
carry, and the draw in integers -- and the circuit of era five must equal it operation for
operation, not approximately.

THE ORDER OF OPERATIONS IS THE CONTRACT. A rewrite that is algebraically equal and
differently ordered is a different machine.

The rules that are not this era's come from `nn.py`: the exponent rule, the rounding, the
int16 rails, the temper, the min-p floor, the Q12 port clamp, the shared exp2 table and
the integer pick stand there, where every twin reads them.

THREE THINGS OF A BLOCK ARE NOT TENSORS OF THE IMAGE. `a_log`, `dt_bias` and `d_skip` hold
one value for each head, and an int8 tensor cannot carry them: the bias enters a softplus,
where a step of one part in 127 of its range moves `dt` by more than a small `dt` is, and
the decay would follow it. They quantize here into the numbers the ops carry -- `a *
log2(e)` folds into one Q constant for each head, exactly as the temperature folds into
the temper -- thus the run time never sees them as tensors.

ONE ULP DECIDES A ROM BYTE. `decay` reads the exponential of `a_log`, and the OCaml
quantizer read it through the C library's `exp`. `math.exp` is the same library on the
same machine, thus this side takes it in float64 and never `np.exp`, whose vectorized path
may differ by one ulp -- and one ulp there moves a `q_value` by one, which moves a byte,
which moves the netlist md5.

THE CONTRACT FILE is what crosses the seam to the elaboration. `save` writes it and
`Model.of_int8_checkpoint` reads it; `load` reads it back exactly. IT CARRIES THE ROM
ORDER AND NOT THE CHECKPOINT ORDER: a block holds six tensors in a checkpoint and three of
them never reach the ROM, thus the two orders are two structures and neither is implied by
the other.

    tensor            dtype   shape             value
    "0"               int32   [4, 48, d]        the seat tables, int8 in int32
    "1"               int32   [16, d]           the bar-phase table
    "2" ..            int32   see below         the image tensors of each layer, in order
    "exponents"       int32   [tensors]         the exponent of each tensor, in that order
    "span"            int32   []                the ALiBi span of the attention layers
    "ring"            int32   []                the keys and values a layer holds behind
    "temper"          int32   [2]               the temper: q_value, then q
    "min_weight"      int32   []                the min-p share of the peak weight 2^15
    "decay_q_value"   int32   [blocks, heads]   a * log2(e), the Decay op's constant
    "decay_q"         int32   [blocks, heads]   the Q of it
    "dt_bias"         int32   [blocks, heads]   Q12, clamped to the int16 sum it joins
    "d_skip"          int32   [blocks, heads]   Q12, clamped to the 18-bit operand port

A layer's image is: a block, `w_in` TRANSPOSED at [projection, d], `conv` at
[channels, taps] and `w_out` at [d_in, d]; an attention layer, `wq` and `wk` at [2d, d]
and `wv` and `wo` at [d, d]; the feed-forward, `w1` at [d, 4d] and `w2` at [4d, d]. The
four per-block rows are indexed by the ordinal of the block among the BLOCKS and not among
the layers, which is what indexes a memory of the circuit as well.

W_in IS STORED TRANSPOSED, and the reason is the address. The circuit reaches a weight by
CONCATENATING the two walk counters, which costs nothing and is the row-major address only
when the dimension under the outer counter is a power of two. `d` is one; the projection --
2 d_in + 2 N + H, 292 at the baseline -- is not. Storing the tensor the other way round
puts `d` under the outer counter and the concatenation is right again.

THE PLAN IS IN THE SHAPES and no tensor states it: the first tensor of a group names its
kind, thus the reader walks the image sequentially and reads the kind before the count.
`wq` is [2d, d] and `w1` is [d, 4d]; a transposed `w_in` is [projection, d] and the
projection is never 2d, thus no block can be read as anything else. A SQUARE first tensor
is era four's plain attention, which this model does not hold, and it is refused.

EVERY TENSOR IS INT32 AND THE SCALARS ARE TENSORS. Both are facts of the reader:
`Nx_io.load_safetensors` skips every dtype it does not hold, and it gives no access to
`__metadata__`. The metadata is written as well, for a reader that has a Python tool in
hand: the temperature, the min-p and the checkpoint are PROVENANCE, because the temper and
the floor are already folded.
"""

import math
from typing import NamedTuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

import data
import nn
import prng
from mamba import model as recurrence

EXPONENTS = "exponents"
SPAN = "span"
RING = "ring"
TEMPER = "temper"
MIN_WEIGHT = "min_weight"
DECAY_Q_VALUE = "decay_q_value"
DECAY_Q = "decay_q"
DT_BIAS = "dt_bias"
D_SKIP = "d_skip"
# the tensors the file carries beside its numbered weights
BESIDE_THE_WEIGHTS = (
    EXPONENTS,
    SPAN,
    RING,
    TEMPER,
    MIN_WEIGHT,
    DECAY_Q_VALUE,
    DECAY_Q,
    DT_BIAS,
    D_SKIP,
)

# one letter for each kind, which is how `docs/mamba.md`, the checkpoint names and the
# `--plan` flag of the trainer all spell a plan. The elected model is MMMMMMZF.
LETTERS = {recurrence.MAMBA: "M", recurrence.ZATTN: "Z", recurrence.MLP: "F"}

# the tensors each kind of layer puts in the ROM image, in the order it puts them
IMAGE_TENSORS = {
    recurrence.MAMBA: ("w_in", "conv", "w_out"),
    recurrence.ZATTN: ("wq", "wk", "wv", "wo"),
    recurrence.MLP: ("w1", "w2"),
}

# The Q the Decay op's constant carries. The constant rides the 25-bit port and `dt` the
# 18-bit one, which is the way round that costs nothing -- `dt` is int16 -- and it leaves
# the constant three million units of room where the other order would clamp a decay rate
# above 22.
DECAY_Q_BITS = 12
DECAY_HIGH = (1 << 24) - 1

# the ports the two per-head Q12 numbers ride: the bias joins an int16 sum and the skip
# rides the 18-bit operand port
DT_BIAS_BOUND = 32767
D_SKIP_BOUND = 131071

# the policy the ear elected, `Mgen_nn.Policy`: the draw the bitstream commits to
ELECTED_TEMPERATURE = 1.0
ELECTED_MIN_P = 0.05
# The depth of the ring at INFERENCE, which is a choice of the player and not a fact of
# the training run. It is era four's training window, thus a window of the loss reads
# exactly the attention the trainer computed.
ELECTED_RING = recurrence.ATTN_CONTEXT


def decay_scale(a_log):
    """`Constants.decay_scale`: a * log2(e) as the constant the Decay op carries.

    It takes the exponential through `math.exp` in float64 -- the C library's own, which is
    what the OCaml quantizer read -- because one ulp here moves a ROM byte."""
    a = math.exp(float(a_log))
    q_value = int(nn.round_half_up(math.ldexp(a / math.log(2.0), DECAY_Q_BITS)))
    return min(max(q_value, 0), DECAY_HIGH)


class Quantized(NamedTuple):
    """The model as the bitstream carries it.

    `tensors` holds the image in THE ORDER OF THE ROM -- the two tables, then what each
    layer puts in it -- and the four per-block rows stand beside it, indexed by the ordinal
    of the block among the blocks."""

    plan: tuple
    tensors: list  # (q, e) in the order of the image
    decay: np.ndarray  # [blocks, heads], the Decay op's constant
    dt_bias: np.ndarray  # [blocks, heads], Q12
    d_skip: np.ndarray  # [blocks, heads], Q12
    span: int
    ring: int
    temper: nn.Temper
    min_weight: int

    @property
    def d(self):
        """the width of the residual stream: the seat tensor sizes it"""
        return self.tensors[0][0].size // (data.SEATS * data.CLASSES)

    @property
    def blocks(self):
        return sum(1 for kind in self.plan if kind == recurrence.MAMBA)

    @property
    def heads(self):
        return self.dt_bias.shape[1]

    @classmethod
    def of(
        cls,
        params,
        *,
        ring=ELECTED_RING,
        temperature=ELECTED_TEMPERATURE,
        min_p=ELECTED_MIN_P,
    ):
        """the float params under the exponent rule of the eras, and the per-head numbers
        folded into the constants the ops carry.

        THE SEAT AND PHASE TABLES SHARE ONE EXPONENT and take it from the larger peak:
        their rows ADD, thus a difference of exponents would be a difference of formats
        inside one sum. Every other tensor takes its own."""
        shape = recurrence.shape_of(params)
        tables = [np.asarray(params[name], np.float64) for name in nn.TABLES]
        shared = nn.max_exponent(max(float(np.abs(t).max(initial=0.0)) for t in tables))
        tensors = [nn.quantize(t, e=shared) for t in tables]
        decay, dt_bias, d_skip = [], [], []
        for layer in params["layers"]:
            kind = recurrence.kind_of(layer)
            if kind == recurrence.ATTN:
                raise ValueError(
                    "a square query is era four's attention and no layer of this model"
                )
            tensors += [
                nn.quantize(image_tensor(kind, name, layer, shape.d))
                for name in IMAGE_TENSORS[kind]
            ]
            if kind == recurrence.MAMBA:
                decay.append([decay_scale(a) for a in np.asarray(layer["a_log"])])
                dt_bias.append(nn.fixed_q12(layer["dt_bias"], DT_BIAS_BOUND))
                d_skip.append(nn.fixed_q12(layer["d_skip"], D_SKIP_BOUND))
        return cls(
            plan=shape.plan,
            tensors=tensors,
            decay=np.array(decay, np.int32).reshape(len(decay), shape.heads),
            dt_bias=np.array(dt_bias, np.int32).reshape(len(dt_bias), shape.heads),
            d_skip=np.array(d_skip, np.int32).reshape(len(d_skip), shape.heads),
            span=int(params.get(recurrence.SPAN_KEY, nn.SLOPE_SPAN)),
            ring=ring,
            temper=nn.Temper.of(temperature),
            min_weight=nn.min_weight_of(min_p),
        )


def image_tensor(kind, name, layer, d):
    """one float tensor as the image holds it: `w_in` transposed, every other as it stands"""
    value = np.asarray(layer[name], np.float64)
    if kind == recurrence.MAMBA and name == "w_in":
        return np.ascontiguousarray(value.T)
    return value


def check_shape(twin):
    """the rules the consumers assume, refused loudly here rather than inside a walk.

    The arithmetic of the circuit is shifts and address concatenations, thus every field of
    an address is a power of two: the two widths, the heads, the head widths, the state,
    the taps and the ring. The ring is the one of them that is not a fact of the training
    run, and a player that asks for a depth the mask cannot wrap refuses here."""
    d, heads = twin.d, twin.heads
    if not twin.blocks:
        raise ValueError("a plan of attention alone is not this model")
    if len(twin.tensors) != len(nn.TABLES) + sum(
        len(IMAGE_TENSORS[kind]) for kind in twin.plan
    ):
        raise ValueError(f"{len(twin.tensors)} tensors do not fill the plan")
    d_in = twin.tensors[image_at(twin, 0) + 2][0].shape[0]
    channels, taps = twin.tensors[image_at(twin, 0) + 1][0].shape
    state = (channels - d_in) // 2
    for name, value in [
        ("d", d),
        ("d_in", d_in),
        ("heads", heads),
        ("the state", state),
        ("the taps", taps),
        ("the ring", twin.ring),
        ("the block head", d_in // heads),
        ("the attention head", d // heads),
    ]:
        if value < 1 or value & (value - 1):
            raise ValueError(f"{name} is {value} and must be a power of two")
    if twin.tensors[0][0].size != data.SEATS * data.CLASSES * d:
        raise ValueError("the seat table holds no row for each seat and class")
    if twin.tensors[0][1] != twin.tensors[1][1]:
        raise ValueError("the seat and phase tables must share one exponent")
    for row in (twin.decay, twin.dt_bias, twin.d_skip):
        if row.shape != (twin.blocks, heads):
            raise ValueError(f"a per-head row is {row.shape}, not {(twin.blocks, heads)}")


def image_at(twin, block):
    """where the image tensors of block [block] open, counting from the whole image"""
    at = len(nn.TABLES)
    seen = 0
    for kind in twin.plan:
        if kind == recurrence.MAMBA:
            if seen == block:
                return at
            seen += 1
        at += len(IMAGE_TENSORS[kind])
    raise ValueError(f"the plan holds no block {block}")


def save(path, twin):
    """the contract file of `twin`: the module docstring holds the layout and the reasons"""
    check_shape(twin)
    tensors = {str(at): q for at, (q, _) in enumerate(twin.tensors)}
    tensors[EXPONENTS] = np.array([e for _, e in twin.tensors], np.int32)
    tensors[SPAN] = np.array(twin.span, np.int32)
    tensors[RING] = np.array(twin.ring, np.int32)
    tensors[TEMPER] = np.array([twin.temper.q_value, twin.temper.q], np.int32)
    tensors[MIN_WEIGHT] = np.array(twin.min_weight, np.int32)
    tensors[DECAY_Q_VALUE] = twin.decay
    tensors[DECAY_Q] = np.full_like(twin.decay, DECAY_Q_BITS)
    tensors[DT_BIAS] = twin.dt_bias
    tensors[D_SKIP] = twin.d_skip
    save_file(
        tensors,
        str(path),
        metadata={
            "plan": "".join(LETTERS[kind] for kind in twin.plan),
            "temper_q_value": str(twin.temper.q_value),
            "temper_q": str(twin.temper.q),
            "temperature": repr(twin.temper.temperature),
            "min_weight": str(twin.min_weight),
        },
    )


def load(path):
    """the model of one contract file; a round trip through `save` is exact.

    THE PLAN COMES BACK OUT OF THE SHAPES, by the rule the module docstring states, thus
    the reader of this side and the reader of the elaboration walk the image alike."""
    tensors = load_file(str(path))
    with safe_open(str(path), framework="numpy") as opened:
        metadata = opened.metadata() or {}
    count = len(tensors) - len(BESIDE_THE_WEIGHTS)
    if count < len(nn.TABLES) + 1:
        raise ValueError(f"{path}: {len(tensors)} tensors is no quantized state model")
    exponents = tensors[EXPONENTS]
    image = [(tensors[str(at)], int(exponents[at])) for at in range(count)]
    d = image[0][0].size // (data.SEATS * data.CLASSES)
    plan, at = [], len(nn.TABLES)
    while at < count:
        kind = kind_of_shape(image[at][0].shape, d, path)
        plan.append(kind)
        at += len(IMAGE_TENSORS[kind])
    if at != count:
        raise ValueError(f"{path}: {count} image tensors do not fill whole layer groups")
    q_value, q = (int(value) for value in tensors[TEMPER])
    twin = Quantized(
        plan=tuple(plan),
        tensors=image,
        decay=tensors[DECAY_Q_VALUE],
        dt_bias=tensors[DT_BIAS],
        d_skip=tensors[D_SKIP],
        span=int(tensors[SPAN]),
        ring=int(tensors[RING]),
        temper=nn.Temper(q_value, q, float(metadata.get("temperature", np.nan))),
        min_weight=int(tensors[MIN_WEIGHT]),
    )
    check_shape(twin)
    return twin


def kind_of_shape(shape, d, path):
    """THE FIRST TENSOR OF A GROUP NAMES ITS KIND, thus the walk is sequential and it reads
    the kind before it reads the count."""
    if shape == (2 * d, d):
        return recurrence.ZATTN
    if shape == (d, 4 * d):
        return recurrence.MLP
    if shape == (d, d):
        raise ValueError(
            f"{path} opens a layer with a square query: era four's attention is not a "
            "layer of this model"
        )
    if len(shape) != 2 or shape[1] != d:
        raise ValueError(f"{path} opens a layer with {shape}, which is no image tensor")
    return recurrence.MAMBA


# ---------------------------------------------------------------------
# the integer engine: one running inference over a batch of seeds
# ---------------------------------------------------------------------

# The formats of the machine, `Model.Constants`. A Q number holds value * 2^-q.
H_Q = 16  # the residual stream, in int32
Y_Q = 12  # the normed vector, and the score of attention
V_Q = 12  # the value rows of a block and of the attention rings
S_Q = 12  # the state of the recurrence
ALPHA_Q = 15  # the decay of one step
BETA_Q = 15  # the input coefficient
HID_Q = 10  # the feed-forward hidden vector after its ReLU
# the gate product, in an int32: two Q12 values multiply and nothing truncates them before
# the norm that reads them
GATE_Q = 2 * V_Q
EPS_Q = int(nn.round_half_up(math.ldexp(1e-6, 2 * Y_Q)))
LOG2E = nn.Temper(int(nn.round_half_up(math.ldexp(1.0 / math.log(2.0), 15))), 15, 1.0)

# the silent lead-in of the boot, in steps: one bar, as the float sampler plays it
LEAD = data.BAR_STEPS


def rescale(value, *, at, to):
    """value * 2^-at as value * 2^-to; the arithmetic shift floors, as the circuit's"""
    if to >= at:
        return value << (to - at)
    return value >> (at - to)


def apply_scale(q_value, q, value):
    """`Constants.apply`: value times a fixed-point multiplier, toward negative infinity"""
    return (value * q_value) >> q


def truncated(numerator, denominator):
    """OCaml's `/` on integers, which goes TOWARD ZERO where numpy's `//` floors.

    Every division of the circuit truncates, thus a floor here would part from it on the
    negative half of a vector and nowhere else."""
    numerator = np.asarray(numerator, np.int64)
    denominator = np.asarray(denominator, np.int64)
    sign = np.sign(numerator) * np.sign(denominator)
    return sign * (np.abs(numerator) // np.abs(denominator))


def isqrt(values):
    """floor of the square root, over an array: the one answer the [Isqrt] unit gives"""
    values = np.asarray(values, np.int64)
    guess = np.where(values <= 0, 0, np.sqrt(np.maximum(values, 0)).astype(np.int64))
    while True:
        low = np.maximum(guess - ((guess * guess > values) & (guess > 0)), 0)
        high = low + ((low + 1) * (low + 1) <= values)
        if np.array_equal(high, guess):
            return guess
        guess = high


def rms_norm(v, *, at, width):
    """rms_norm over [width] elements of a Q[at] vector, giving Q12.

    The sum squares a Q12 copy -- one DSP-sized product -- then one isqrt, and one
    truncating division for each element. The stream enters at Q16 and the gate of a block
    at Q24, thus the shift of the NUMERATOR is the one thing that moves between callers."""
    copy = rescale(v, at=at, to=Y_Q)
    total = (copy * copy).sum(axis=-1, keepdims=True)
    mean = (total >> (width.bit_length() - 1)) + EPS_Q
    return nn.clamp16(truncated(v * (1 << ((2 * Y_Q) - at)), isqrt(mean)))


def tensor_at(twin, at):
    """one image tensor as int64, flat in the row-major order the ROM holds"""
    return np.asarray(twin.tensors[at][0], np.int64).reshape(-1)


def matvec(y, weight, *, outer_major, inner, outer, at, to):
    """One matvec column: [inner] terms of a Q[at] vector against a row of the weight.

    [outer_major] states which axis the tensor's rows are, and the circuit reads the same
    order: it is true for W_in, which the image stores transposed, and for the seat
    readout, which the checkpoint already stores that way."""
    values, exponent = weight
    matrix = values.reshape(outer, inner).T if outer_major else values.reshape(inner, outer)
    return nn.clamp16(rescale(y @ matrix, at=at + exponent, to=to))


def join(h, weight, *, values, d, at):
    """a residual join: [values] times the weight lands on the stream; the exponent of the
    weight folds into the shift with [at], the format of [values]"""
    matrix, exponent = weight
    return h + rescale(values @ matrix.reshape(-1, d), at=at + exponent, to=H_Q)


class Clamps(NamedTuple):
    """The clamps of the walk, and the chances each one had.

    The formats of this era are chosen with margin and not metered on a trained
    checkpoint, thus a clamp that fires is the finding that says which one is wrong -- and
    where era four could let a hot signal die with its window, an error in the state
    carries forward."""

    dt: int = 0
    dt_seen: int = 0
    beta: int = 0
    beta_seen: int = 0
    state: int = 0
    state_seen: int = 0


class Engine(NamedTuple):
    """One running inference over a batch of walks. Everything is frozen: a step gives the
    engine after it, as `Quantized.Engine` does.

    THE STATE AND THE TAPS ARE THE MEMORY OF THE WALK and the only things that survive a
    step; the key and value rings are era four's, and they die with their window."""

    twin: Quantized
    state: np.ndarray  # [walks, blocks * d_in * n], Q12
    taps: np.ndarray  # [walks, blocks * channels * taps], Q12
    kc: np.ndarray  # [walks, attentions, ring, d], Q12 in a coarse byte
    vc: np.ndarray
    h: np.ndarray  # [walks, d], Q16
    position: int
    states: np.ndarray  # [walks], the generator of each walk
    clamps: Clamps


def widths(twin):
    """every width of the model, out of its own image"""
    at = image_at(twin, 0)
    projection, d = twin.tensors[at][0].shape
    channels, taps = twin.tensors[at + 1][0].shape
    d_in = twin.tensors[at + 2][0].shape[0]
    state = (channels - d_in) // 2
    return d, d_in, twin.heads, state, taps, channels, projection


def ordinals(twin):
    """the ordinal of each layer among the layers of ITS OWN KIND, which is what indexes a
    memory: the state RAM and the tap ring hold one region for each block, and the key and
    value rings one for each attention layer"""
    seen, out = {}, []
    for kind in twin.plan:
        out.append(seen.get(kind, 0))
        seen[kind] = out[-1] + 1
    return out


def engine(twin, seeds):
    """the origin of a batch of walks: a zero state, an empty tap ring, an empty key and
    value ring, and no residual"""
    check_shape(twin)
    d, d_in, _, state, taps, channels, _ = widths(twin)
    walks = len(seeds)
    blocks = twin.blocks
    rings = sum(1 for kind in twin.plan if kind == recurrence.ZATTN)
    return Engine(
        twin=twin,
        state=np.zeros((walks, blocks * d_in * state), np.int64),
        taps=np.zeros((walks, blocks * channels * taps), np.int64),
        kc=np.zeros((walks, max(1, rings), twin.ring, d), np.int64),
        vc=np.zeros((walks, max(1, rings), twin.ring, d), np.int64),
        h=np.zeros((walks, d), np.int64),
        position=0,
        states=nn.engine_states(seeds),
        clamps=Clamps(),
    )


def embed(twin, classes, phase):
    """the embedding: the four seat rows and the phase row add in the shared exponent, then
    shift to Q16"""
    d, e = twin.d, twin.tensors[0][1]
    seats = tensor_at(twin, 0).reshape(data.SEATS, data.CLASSES, d)
    table = tensor_at(twin, 1).reshape(-1, d)
    value = np.broadcast_to(table[phase], (len(classes), d)).copy()
    for seat in range(data.SEATS):
        value = value + seats[seat, classes[:, seat]]
    return rescale(value, at=e, to=H_Q)


def image(twin, at, kind):
    """the image tensors of one layer, by name, as (flat values, exponent)"""
    return {
        name: (tensor_at(twin, at + on), twin.tensors[at + on][1])
        for on, name in enumerate(IMAGE_TENSORS[kind])
    }


def block(e, at, ordinal, h, state, taps):
    """One block of the trunk: the stream after the residual join, and the clamps it met.

    It writes the state and the taps of its own region IN PLACE -- the two arrays are
    copies the caller made for this step -- as the state RAM of the circuit is written in
    place."""
    twin = e.twin
    d, d_in, heads, n, width, channels, projection = widths(twin)
    head = d_in // heads
    w = image(twin, at, recurrence.MAMBA)
    position = e.position
    y = rms_norm(h, at=H_Q, width=d)
    zxbcdt = matvec(
        y, w["w_in"], outer_major=True, inner=d, outer=projection, at=Y_Q, to=V_Q
    )
    # the convolution: the step's input enters the ring, then a row of [width] terms for
    # each channel, then the SiLU chain over the sums
    tap_base = ordinal * channels * width
    slot = tap_base + (np.arange(channels) * width) + (position & (width - 1))
    taps[:, slot] = zxbcdt[:, d_in : d_in + channels]
    kernel, kernel_e = w["conv"]
    kernel = kernel.reshape(channels, width)
    accumulated = np.zeros((len(h), channels), np.int64)
    for k in range(width):
        # TAP k READS THE STEP k BACK, and it reads ZERO while the walk has not run k
        # steps -- thus the origin needs no clearing walk and the rule is a mux
        if position < k:
            continue
        back = tap_base + (np.arange(channels) * width) + ((position - k) & (width - 1))
        accumulated = accumulated + (taps[:, back] * kernel[:, k])
    xbc = nn.silu(nn.clamp16(rescale(accumulated, at=V_Q + kernel_e, to=V_Q)))
    x = xbc[:, :d_in]
    b = xbc[:, d_in : d_in + n]
    c = xbc[:, d_in + n :]
    # the decay of each head: softplus of the biased draw, then one exp2
    raw = zxbcdt[:, d_in + channels :]
    dt = nn.softplus(raw + twin.dt_bias[ordinal])
    tally = e.clamps._replace(
        dt=e.clamps.dt + int(((dt == nn.INT16_HIGH) | (dt == nn.INT16_LOW)).sum()),
        dt_seen=e.clamps.dt_seen + dt.size,
    )
    alpha = nn.exp2_of_magnitude(
        apply_scale(twin.decay[ordinal], DECAY_Q_BITS, dt)
    )
    # the state update and the readout, head by head. [beta] is the inject operand of the
    # head: [state] products of [dt] against B, written before the walk.
    base = ordinal * d_in * n
    read = np.zeros((len(h), d_in), np.int64)
    for hd in range(heads):
        wide = (dt[:, hd, None] * b) >> (V_Q + V_Q - BETA_Q)
        tally = tally._replace(
            beta=tally.beta + int(nn.clamps16(wide).sum()),
            beta_seen=tally.beta_seen + wide.size,
        )
        beta = nn.clamp16(wide)
        lanes = np.arange(hd * head, (hd + 1) * head)
        rows = base + (lanes[:, None] * n) + np.arange(n)
        updated = (
            (alpha[:, hd, None, None] * state[:, rows])
            + (x[:, lanes, None] * beta[:, None, :])
        ) >> ALPHA_Q
        tally = tally._replace(
            state=tally.state + int(nn.clamps16(updated).sum()),
            state_seen=tally.state_seen + updated.size,
        )
        state[:, rows] = nn.clamp16(updated)
        # the readout reads the state the update just wrote, and the skip folds into the
        # row as its last term
        read[:, lanes] = nn.clamp16(
            (
                (state[:, rows] * c[:, None, :]).sum(axis=-1)
                + (x[:, lanes] * twin.d_skip[ordinal][hd])
            )
            >> S_Q
        )
    # THE GATE PRODUCT STAYS WIDE. Both operands are Q12 values well under one -- the
    # readout of a small state and the SiLU of a gate -- thus a truncation back to Q12 here
    # would keep about five bits of a product that holds seventeen, and it would throw them
    # away immediately before the one operation that would have used them: the norm divides
    # by the size of the vector and does not care what scale it arrives in.
    gated = read * nn.silu(zxbcdt[:, :d_in])
    g = rms_norm(gated, at=GATE_Q, width=d_in)
    return join(h, w["w_out"], values=g, d=d, at=V_Q), tally


def attend(twin, kc, vc, *, ring, cur, filled, query):
    """Attention over the newest [filled] steps of the ring: the merged context of the
    query, head by head. Age a reads slot (cur - a) & (ring - 1), thus the ALiBi distance
    IS the age and the causal wall is the walk."""
    d, heads, slots = twin.d, twin.heads, twin.ring
    head_d = d // heads
    ages = np.arange(filled)
    rows = (cur - ages) & (slots - 1)
    keys, values = kc[:, ring, rows, :], vc[:, ring, rows, :]
    context = np.zeros((len(query), d), np.int64)
    shift = (2 * V_Q) - Y_Q + ((head_d.bit_length() - 1) // 2)
    for head in range(heads):
        band = slice(head * head_d, (head + 1) * head_d)
        slope = (twin.span * (head + 1)) // heads
        raw = (query[:, None, band] * keys[:, :, band]).sum(axis=-1)
        scores = (raw >> shift) - (ages << (Y_Q - slope))
        peak = scores.max(axis=-1, keepdims=True)
        # THE NEGATION STANDS OUTSIDE THE SCALE, as it stands outside the temper of the
        # draw: the circuit scales the score's distance BELOW the peak and negates the
        # shifted product, thus a scale that did not divide exactly would round the other
        # way if this side negated first.
        weight = nn.exp2_of_magnitude(
            -apply_scale(LOG2E.q_value, LOG2E.q, scores - peak)
        )
        total = weight.sum(axis=-1, keepdims=True)
        merged = (weight[:, :, None] * values[:, :, band]).sum(axis=1)
        context[:, band] = nn.clamp16(truncated(merged, total))
    return context


def attention(e, at, ordinal, h, embedding, kc, vc):
    """One attention layer, era four's with one addition: the query and the key read the
    JOINED vector -- the normed stream beside the normed embedding -- thus their walk is
    2d terms where the value's is d."""
    twin = e.twin
    d, slots = twin.d, twin.ring
    cur = e.position & (slots - 1)
    filled = min(e.position + 1, slots)
    w = image(twin, at, recurrence.ZATTN)
    y = rms_norm(h, at=H_Q, width=d)
    joined = np.concatenate([y, embedding], axis=-1)

    def project(name, source, inner):
        return matvec(
            source, w[name], outer_major=False, inner=inner, outer=d, at=Y_Q, to=V_Q
        )

    # THE RING KEEPS THE TOP BYTE of a Q12 row: the circuit stores eight bits and restores
    # eight zero low bits at the read. The query does not pass here.
    kc[:, ordinal, cur, :] = (project("wk", joined, 2 * d) >> 8) << 8
    vc[:, ordinal, cur, :] = (project("wv", y, d) >> 8) << 8
    context = attend(
        twin, kc, vc, ring=ordinal, cur=cur, filled=filled, query=project("wq", joined, 2 * d)
    )
    return join(h, w["wo"], values=context, d=d, at=V_Q)


def feed_forward(twin, at, h):
    """Era four's feed-forward as a layer of its own: one matvec and a ReLU, Q10.

    The ReLU stands after the clamp of the matvec and the circuit takes it before, which is
    the same integer: a value the clamp raised was negative and the ReLU makes it zero
    either way, and a value it lowered was above the ceiling and stays there."""
    d = twin.d
    w = image(twin, at, recurrence.MLP)
    y = rms_norm(h, at=H_Q, width=d)
    hidden = np.maximum(
        matvec(y, w["w1"], outer_major=False, inner=d, outer=4 * d, at=Y_Q, to=HID_Q), 0
    )
    return join(h, w["w2"], values=hidden, d=d, at=HID_Q)


def layer_streams(e, classes, phase):
    """the residual stream after the embed and then after each layer of the step the engine
    would take next -- one entry for each time the circuit writes the whole stream, in the
    order it writes them.

    A frame gate that fails says only THAT the circuit and the twin parted; this says
    where, and it is the instrument that found era five's four address faults."""
    twin = e.twin
    state, taps = e.state.copy(), e.taps.copy()
    kc, vc = e.kc.copy(), e.vc.copy()
    h = embed(twin, classes, phase)
    embedding = rms_norm(h, at=H_Q, width=twin.d)
    written = [h]
    at, tally = len(nn.TABLES), e.clamps
    for kind, ordinal in zip(twin.plan, ordinals(twin)):
        if kind == recurrence.MAMBA:
            h, tally = block(
                e._replace(clamps=tally), at, ordinal, h, state, taps
            )
        elif kind == recurrence.ZATTN:
            h = attention(e, at, ordinal, h, embedding, kc, vc)
        else:
            h = feed_forward(twin, at, h)
        written.append(h)
        at += len(IMAGE_TENSORS[kind])
    return written, (state, taps, kc, vc, tally)


def forward(e, classes, phase):
    """one step of the recurrence: the engine after it"""
    written, (state, taps, kc, vc, tally) = layer_streams(e, classes, phase)
    return e._replace(
        h=written[-1],
        state=state,
        taps=taps,
        kc=kc,
        vc=vc,
        position=e.position + 1,
        clamps=tally,
    )


def seat_logits(twin, stream, seat):
    """the tied head of one seat: rms_norm of the stream the chain has written so far, then
    that seat's table read backward; Q12 logits over the classes"""
    d, e = twin.d, twin.tensors[0][1]
    seats = tensor_at(twin, 0).reshape(data.SEATS, data.CLASSES, d)
    return (rms_norm(stream, at=H_Q, width=d) @ seats[seat].T) >> e


def add_row(twin, stream, seat, drawn):
    """what the chain adds after a seat draws: the drawn row, in the format of the stream"""
    d, e = twin.d, twin.tensors[0][1]
    seats = tensor_at(twin, 0).reshape(data.SEATS, data.CLASSES, d)
    return stream + rescale(seats[seat, drawn], at=e, to=H_Q)


def tempered_weights(twin, logits):
    """the Q15 weight of every class of one seat, and the min-p floor over it"""
    peak = logits.max(axis=-1, keepdims=True)
    weights = nn.exp2_of_magnitude(
        -apply_scale(twin.temper.q_value, twin.temper.q, logits - peak)
    )
    return np.where(weights >= twin.min_weight, weights, 0)


class Draw(NamedTuple):
    """one draw of the chain, over the batch"""

    seat: int
    logits: np.ndarray  # [walks, CLASSES], Q12 -- what the drift report compares
    word: np.ndarray  # [walks], the 24-bit uniform
    drawn: np.ndarray  # [walks], the class


def chain(e):
    """One frame, drawn in a chain from the soprano down: each seat reads the stream that
    the seats above it have written."""
    twin = e.twin
    stream, states, draws = e.h, e.states, []
    everyone = np.ones(len(stream), bool)
    for seat in reversed(range(data.SEATS)):
        logits = seat_logits(twin, stream, seat)
        states, word = prng.uniform_word(states, everyone)
        drawn = nn.pick(tempered_weights(twin, logits), word)
        if seat:
            stream = add_row(twin, stream, seat, drawn)
        draws.append(Draw(seat, logits, word, drawn))
    return e._replace(states=states), draws


def next_step(e):
    """one step of the walk: the engine after it, the classes of the frame, and the draws.

    THE BOOT IS A LEAD-IN OF SILENCE, one bar of it, drawing nothing and taking no number
    from the generator."""
    phase = e.position % data.BAR_STEPS
    if e.position < LEAD:
        classes = np.full((len(e.h), data.SEATS), data.SILENCE, np.int64)
        draws = []
    else:
        e, draws = chain(e)
        classes = np.stack([draw.drawn for draw in reversed(draws)], axis=-1)
    return forward(e, classes, phase), classes, draws


def walk(twin, seeds, steps):
    """the classes of each step of the walk, and the draws behind them"""
    e = engine(twin, seeds)
    played, taken = [], []
    for _ in range(steps):
        e, classes, draws = next_step(e)
        played.append(classes)
        taken.append(draws)
    return np.stack(played, axis=1), taken


def streams(twin, seeds, steps):
    """the stream writes of each step: the embed and then each layer, in the order the
    circuit writes them.

    It walks the model, thus the writes are the writes of a real walk and not of a step
    the engine would never take."""
    e = engine(twin, seeds)
    written = []
    for _ in range(steps):
        phase = e.position % data.BAR_STEPS
        if e.position < LEAD:
            classes = np.full((len(e.h), data.SEATS), data.SILENCE, np.int64)
        else:
            e, draws = chain(e)
            classes = np.stack([draw.drawn for draw in reversed(draws)], axis=-1)
        rows, _ = layer_streams(e, classes, phase)
        written.append(rows)
        e = forward(e, classes, phase)
    return written


# ---------------------------------------------------------------------
# what the quantization costs
# ---------------------------------------------------------------------


class Drift(NamedTuple):
    """What the quantization costs, measured on the walk the board takes."""

    steps: int  # the steps of the walk, the silent lead-in inside
    draws: int  # four for each drawn step: one for each seat of the chain
    same_peak: int  # the draws where both models elect the same class
    same_draw: int  # the draws where both models pick the same class
    mean_cosine: float
    clamps: Clamps


def cosine(here, there):
    """the cosine between an integer row and the float row of the same place"""
    here = np.asarray(here, np.float64)
    return float(np.dot(here, there) / np.sqrt(np.dot(here, here) * np.dot(there, there)))


class Counted(NamedTuple):
    """what the report has counted over the draws it has seen"""

    draws: int = 0
    same_peak: int = 0
    same_draw: int = 0
    cosine: float = 0.0


def count_draws(counted, floated, chain_draws):
    """one step's chain against the float logits of the same step, seat by seat, on the very
    uniform the engine took"""
    for taken in chain_draws:
        row = floated[taken.seat]
        mine = taken.logits[0]
        uniform = np.array([float(taken.word[0]) * 2.0**-prng.UNIFORM_BITS])
        weights = nn.temper(row[None], ELECTED_TEMPERATURE, ELECTED_MIN_P)
        counted = Counted(
            draws=counted.draws + 1,
            same_peak=counted.same_peak + int(np.argmax(mine) == np.argmax(row)),
            same_draw=counted.same_draw
            + int(nn.pick_share(weights, uniform)[0] == taken.drawn[0]),
            cosine=counted.cosine + cosine(mine, row),
        )
    return counted


def drift(params, *, steps, seed, ring=ELECTED_RING):
    """The quantized walk, scored against the float model draw for draw.

    ONE WEIGHTS SOURCE AND ONE POLICY: the walk quantizes `params` itself, thus the pair
    cannot slip. The float pass is TEACHER-FORCED on the quantized history and on the
    quantized chain -- it reads the classes the engine drew and conditions each seat on the
    classes the engine chose -- thus what the report measures is the quantization and never
    a walk that parted for another reason.

    BOTH MODELS TAKE ONE STEP FOR ONE STEP. Era four had to re-run a whole window at every
    step, which made a long comparison quadratic; here each carries its own memory, thus
    the walk can run past many decay lifetimes -- which it must, because a state error is
    cumulative in a way era four never had.

    The same-draw share reads the float draw on the very uniform the engine took, thus a
    difference there is the arithmetic and not the generator."""
    import jax.numpy as jnp

    twin = Quantized.of(params, ring=ring)
    shape = recurrence.shape_of(params)
    e = engine(twin, [seed])
    carry = recurrence.initial_carry(shape, 1, context=ring)
    counted = Counted()
    stream = None
    for at in range(steps):
        e, classes, chain_draws = next_step(e)
        # THE CHAIN OF A STEP READS THE STREAM OF THE STEP BEFORE IT, on both sides: the
        # twin draws from the stream its own last forward left, thus the float row must be
        # the row that same forward states and never the one this step's classes make.
        if chain_draws and stream is not None:
            floated = np.asarray(
                nn.seat_logits(params, stream[:, None, :], jnp.asarray(classes[None]))
            )[0, 0].astype(np.float64)
            counted = count_draws(counted, floated, chain_draws)
        carry, stream = recurrence.forward_step(
            params,
            carry,
            jnp.asarray(classes, np.int32),
            jnp.asarray([at % nn.PHASE_BUCKETS], np.int32),
            span=twin.span,
        )
    return Drift(
        steps=steps,
        draws=counted.draws,
        same_peak=counted.same_peak,
        same_draw=counted.same_draw,
        mean_cosine=counted.cosine / max(1, counted.draws),
        clamps=e.clamps,
    )
