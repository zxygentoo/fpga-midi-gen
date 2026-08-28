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
