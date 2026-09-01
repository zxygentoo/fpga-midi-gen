"""The integer twin of the state-space model: the arithmetic the board plays.

The float model of `mamba/model.py` is what the ear elected; this is the same network in
the arithmetic the board can hold -- int8 weights with a power-of-two exponent for each
tensor, int16 activations, the per-head numbers folded into the constants the ops carry,
and the draw in integers. The circuit of era five must equal it OPERATION FOR OPERATION:
a rewrite that is algebraically equal and differently ordered is a different machine. What
is not this era's comes from `quantized.py` and `ar_quantized.py`.

THREE THINGS OF A BLOCK ARE NOT TENSORS OF THE IMAGE. `a_log`, `dt_bias` and `d_skip` hold
one value for each head and an int8 tensor cannot carry them: the bias enters a softplus,
where a step of one part in 127 of its range moves `dt` by more than a small `dt` is. They
fold into the constants the ops carry, thus the run time never sees them as tensors.

ONE ULP DECIDES A ROM BYTE. `decay` reads the exponential of `a_log`, and the OCaml
quantizer read it through the C library's `exp`; this side takes `math.exp` in float64 and
never `np.exp`, whose vectorized path may differ by one ulp -- which moves a `q_value`,
which moves a byte, which moves the netlist md5.

THE CONTRACT FILE is what crosses the seam to the elaboration. `save` writes it and
`Model.of_int8_checkpoint` reads it; `load` reads it back exactly. IT CARRIES THE ROM
ORDER AND NOT THE CHECKPOINT ORDER: a block holds six tensors in a checkpoint and three
never reach the ROM.

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
per-block rows are indexed by the ordinal of the block among the BLOCKS, which is what
indexes a memory of the circuit.

W_in IS STORED TRANSPOSED, and the reason is the address: the circuit reaches a weight by
CONCATENATING the two walk counters, which is the row-major address only when the
dimension under the outer counter is a power of two. `d` is one and the projection is not.

THE PLAN IS IN THE SHAPES and no tensor states it: the first tensor of a group names its
kind. A transposed `w_in` is [projection, d] and the projection is never 2d, thus no block
can be read as anything else; a SQUARE first tensor is era four's plain attention, which
this model does not hold, and it is refused. `quantized.py` states why every tensor is
int32 and every scalar a tensor.
"""

import math
from typing import NamedTuple

import numpy as np
from flax import nnx

import ar_model
import ar_quantized
import corpus
import measure
from mamba import model as recurrence
from quantized import (
    ELECTED_MIN_P,
    ELECTED_TEMPERATURE,
    EXPONENTS,
    INT16_HIGH,
    INT16_LOW,
    MIN_WEIGHT,
    TEMPER,
    Temper,
    Weight,
    apply_scale,
    clamp16,
    clamps16,
    engine_states,
    exp2_of_magnitude,
    image_from_tensors,
    image_tensors,
    min_weight_of,
    read_contract,
    round_half_up,
    scalar_tensor,
    write_contract,
)

# the names of this era's own scalars; the shared ones are `contract`'s
SPAN = "span"
RING = "ring"
DECAY_Q_VALUE = "decay_q_value"
DECAY_Q = "decay_q"
DT_BIAS = "dt_bias"
D_SKIP = "d_skip"
# the tensors the file carries beside its numbered weights
BESIDE_THE_WEIGHTS = (
    EXPONENTS,
    TEMPER,
    MIN_WEIGHT,
    SPAN,
    RING,
    DECAY_Q_VALUE,
    DECAY_Q,
    DT_BIAS,
    D_SKIP,
)

# one letter for each kind, as `docs/mamba.md` and the `--plan` flag of the trainer spell
# a plan; the elected model is MMMMMMZF
LETTERS = {recurrence.MAMBA: "M", recurrence.ZATTN: "Z", recurrence.MLP: "F"}

# the tensors each kind of layer puts in the ROM image, in the order it puts them
IMAGE_TENSORS = {
    recurrence.MAMBA: ("w_in", "conv", "w_out"),
    recurrence.ZATTN: ("wq", "wk", "wv", "wo"),
    recurrence.MLP: ("w1", "w2"),
}

# The Q the Decay op's constant carries. The constant rides the 25-bit port and `dt` the
# 18-bit one -- `dt` is int16, thus this way costs nothing -- and the other order would
# clamp a decay rate above 22.
DECAY_Q_BITS = 12
DECAY_HIGH = (1 << 24) - 1

# the ports the two per-head Q12 numbers ride: the bias joins an int16 sum and the skip
# rides the 18-bit operand port
DT_BIAS_BOUND = 32767
D_SKIP_BOUND = 131071

# `ELECTED_TEMPERATURE` and `ELECTED_MIN_P` are imported above so era five's player can
# read them here; the policy has one home in `quantized.py`.
# The depth of the ring at INFERENCE, a choice of the player and no fact of the training
# run. It is era four's training window, thus a window of the loss reads exactly the
# attention the trainer computed.
ELECTED_RING = recurrence.ATTN_CONTEXT


def decay_scale(a_log):
    """`Constants.decay_scale`: a * log2(e) as the constant the Decay op carries.

    It takes the exponential through `math.exp` in float64 -- the C library's own, which
    is what the OCaml quantizer read -- because one ulp here moves a ROM byte."""
    a = math.exp(float(a_log))
    q_value = int(round_half_up(math.ldexp(a / math.log(2.0), DECAY_Q_BITS)))
    return min(max(q_value, 0), DECAY_HIGH)


class Block:
    """One block as the machine holds it -- the twin of `model.Block`.

    Six float tensors become three image tensors and three per-head ROWS: `dt_bias` and
    `d_skip` are Q12 numbers the ops carry, and `a_log` becomes the Decay op's constant,
    a * log2(e). The rows stand on the block that drew them, thus no index aligns them."""

    kind = recurrence.MAMBA

    def __init__(self, *, w_in, conv, w_out, decay, dt_bias, d_skip):
        self.w_in = w_in
        self.conv = conv
        self.w_out = w_out
        self.decay = np.asarray(decay, np.int32)
        self.dt_bias = np.asarray(dt_bias, np.int32)
        self.d_skip = np.asarray(d_skip, np.int32)

    @classmethod
    def from_float(cls, layer):
        """one float [model.Block] under the exponent rule, its per-head numbers folded
        into the constants the ops carry"""
        return cls(
            # THE IMAGE STORES W_IN TRANSPOSED, because the circuit walks the projection
            # as the outer axis; every other tensor stands as the checkpoint holds it.
            w_in=Weight.from_float(np.ascontiguousarray(np.asarray(layer.w_in[...]).T)),
            conv=Weight.from_float(layer.conv[...]),
            w_out=Weight.from_float(layer.w_out[...]),
            decay=[decay_scale(a) for a in np.asarray(layer.a_log[...])],
            dt_bias=ar_quantized.fixed_q12(layer.dt_bias[...], DT_BIAS_BOUND),
            d_skip=ar_quantized.fixed_q12(layer.d_skip[...], D_SKIP_BOUND),
        )

    @property
    def heads(self):
        return self.dt_bias.size

    @property
    def widths(self):
        """every width the model states, out of this block's own image"""
        d = self.w_in.values.shape[1]
        channels, taps = self.conv.values.shape
        d_in = self.w_out.values.shape[0]
        return recurrence.Shape(
            d=d,
            d_in=d_in,
            heads=self.heads,
            state=(channels - d_in) // 2,
            taps=taps,
            plan=(),
        )

    def tensors(self):
        """what this layer puts in the ROM image, in the order it puts it"""
        return [self.w_in, self.conv, self.w_out]

    def rows(self):
        """the three per-head rows, in the order the contract file carries them"""
        return [self.decay, self.dt_bias, self.d_skip]


class Attention(ar_quantized.Weights):
    """The Zamba attention layer as the machine holds it -- the twin of
    `model.Attention` at its widened kind.

    ERA FOUR'S SQUARE ATTENTION IS NOT A LAYER OF THIS MODEL and `of` refuses one: the
    circuit's query walk reads 2d terms and there is no narrow path."""

    kind = recurrence.ZATTN
    names = IMAGE_TENSORS[recurrence.ZATTN]

    @classmethod
    def from_float(cls, layer):
        if layer.kind == recurrence.ATTN:
            raise ValueError(
                "a square query is era four's attention and no layer of this model"
            )
        return super().from_float(layer)


class FeedForward(ar_quantized.Weights):
    """The feed-forward as the machine holds it -- the twin of `model.FeedForward`."""

    kind = recurrence.MLP
    names = IMAGE_TENSORS[recurrence.MLP]


TWIN_OF = {
    recurrence.MAMBA: Block,
    recurrence.ZATTN: Attention,
    recurrence.ATTN: Attention,
    recurrence.MLP: FeedForward,
}


class Mamba:
    """The model as the bitstream carries it. [tensors] walks it in THE ORDER OF THE
    ROM: the two tables, then what each layer puts in the image. The ring is an inference
    choice and travels in the file beside the span.

    IT IS NOT A `model.Trunk` -- `ar_quantized.Weights` states why no twin of this
    era is a Flax module -- thus [tensors] and [plan] are restated below. THE
    ATTRIBUTE NAMES ARE THE PARITY and not the base class."""

    def __init__(self, *, head, layers, span, ring, temper, min_weight):
        # the span reaches the circuit as `slope_exponent`, an integer shift; a
        # fractional one is REFUSED and not truncated, because `check_shape` sees only
        # the truncated twin and could not say the run drifted
        if int(span) != span:
            raise ValueError(f"the span {span} is fractional and no shift holds it")
        self.head = head
        self.layers = list(layers)
        self.span = int(span)
        self.ring = int(ring)
        self.temper = temper
        self.min_weight = int(min_weight)

    @property
    def d(self):
        return self.head.d

    def tensors(self):
        """Every tensor of the model in THE ONE ORDER -- the head's tables, then the
        tensors of each layer, which is `ar_model.Trunk`'s order and the ROM's."""
        return self.head.tensors() + [
            tensor for layer in self.layers for tensor in layer.tensors()
        ]

    @property
    def plan(self):
        """the kind of each layer, in order. No flag carries it on either side of the
        seam: a layer's own tensors say what it is."""
        return tuple(layer.kind for layer in self.layers)

    @property
    def blocks(self):
        return [layer for layer in self.layers if layer.kind == recurrence.MAMBA]

    @property
    def heads(self):
        return self.blocks[0].heads

    @property
    def widths(self):
        """the widths of the model, out of its first block, with the plan of the trunk"""
        return self.blocks[0].widths._replace(plan=self.plan)

    @classmethod
    def from_float(
        cls,
        model,
        *,
        ring=ELECTED_RING,
        temperature=ELECTED_TEMPERATURE,
        min_p=ELECTED_MIN_P,
    ):
        """the float model under the exponent rule of the eras, and the per-head numbers
        folded into the constants the ops carry"""
        return cls(
            head=ar_quantized.Head.from_float(model.head),
            layers=[TWIN_OF[layer.kind].from_float(layer) for layer in model.layers],
            span=model.span,
            ring=ring,
            temper=Temper.from_float(temperature),
            min_weight=min_weight_of(min_p),
        )

    def ordinals(self):
        """the ordinal of each layer among the layers of ITS OWN KIND, which is what
        indexes a memory: the state RAM and the tap ring hold one region for each block,
        and the key and value rings one for each attention layer"""
        seen, out = {}, []
        for layer in self.layers:
            out.append(seen.get(layer.kind, 0))
            seen[layer.kind] = out[-1] + 1
        return out


def check_shape(twin):
    """The rules the consumers assume, refused here rather than inside a walk.

    The arithmetic of the circuit is shifts and address concatenations, thus every field
    of an address is a power of two, and the attention head width obeys the stronger rule
    of `ar_quantized.score_shift`. A PLAN OF ATTENTION ALONE is refused first, because the
    widths come out of a block and the message should say so."""
    if not twin.blocks:
        raise ValueError("a plan of attention alone is not this model")
    shape = twin.widths
    for name, value in [
        ("d", shape.d),
        ("d_in", shape.d_in),
        ("heads", shape.heads),
        ("the state", shape.state),
        ("the taps", shape.taps),
        ("the ring", twin.ring),
        ("the block head", shape.head),
    ]:
        if value < 1 or value & (value - 1):
            raise ValueError(f"{name} is {value} and must be a power of two")
    if not ar_quantized.is_power_of_four(shape.head_d):
        raise ValueError(f"the head width {shape.head_d} must be a power of four")
    twin.head.check_tables(shape.d)
    for block in twin.blocks:
        for row in block.rows():
            if row.shape != (shape.heads,):
                raise ValueError(f"a per-head row is {row.shape}, not {(shape.heads,)}")


def save(path, twin):
    """the contract file of `twin`: the module docstring holds the layout and the
    reasons"""
    check_shape(twin)
    tensors = image_tensors(twin.tensors())
    tensors[SPAN] = scalar_tensor(twin.span)
    tensors[RING] = scalar_tensor(twin.ring)
    tensors[TEMPER] = twin.temper.tensor()
    tensors[MIN_WEIGHT] = scalar_tensor(twin.min_weight)
    # one tensor for each per-head row, a line for each BLOCK in the plan order: the
    # elaboration reads one image and not a tree
    decay, dt_bias, d_skip = (
        np.stack(rows, axis=0).astype(np.int32)
        for rows in zip(*[block.rows() for block in twin.blocks])
    )
    tensors[DECAY_Q_VALUE] = decay
    tensors[DECAY_Q] = np.full_like(decay, DECAY_Q_BITS)
    tensors[DT_BIAS] = dt_bias
    tensors[D_SKIP] = d_skip
    write_contract(
        path,
        tensors,
        {
            "plan": "".join(LETTERS[layer.kind] for layer in twin.layers),
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
    tensors, metadata = read_contract(path)
    count = len(tensors) - len(BESIDE_THE_WEIGHTS)
    if count < len(ar_model.TABLES) + 1:
        raise ValueError(f"{path}: {len(tensors)} tensors is no quantized state model")
    exponents = tensors[EXPONENTS]
    d = tensors["0"].size // (corpus.SEATS * corpus.CLASSES)
    plan, groups, at = [], [], len(ar_model.TABLES)
    while at < count:
        kind = kind_of_image(tensors[str(at)].shape, d, path)
        names = IMAGE_TENSORS[kind]
        plan.append(kind)
        groups.append(image_from_tensors(tensors, exponents, first=at, count=len(names)))
        at += len(names)
    if at != count:
        raise ValueError(f"{path}: {count} image tensors do not fill whole layer groups")
    rows = iter(
        zip(tensors[DECAY_Q_VALUE], tensors[DT_BIAS], tensors[D_SKIP])
    )

    def layer_of(kind, group):
        if kind != recurrence.MAMBA:
            return TWIN_OF[kind](group)
        decay, dt_bias, d_skip = next(rows)
        w_in, conv, w_out = group
        return Block(
            w_in=w_in,
            conv=conv,
            w_out=w_out,
            decay=decay,
            dt_bias=dt_bias,
            d_skip=d_skip,
        )

    twin = Mamba(
        head=ar_quantized.Head.from_file(tensors, exponents),
        layers=[layer_of(kind, group) for kind, group in zip(plan, groups)],
        span=int(tensors[SPAN]),
        ring=int(tensors[RING]),
        temper=Temper.from_file(tensors, metadata, key=TEMPER),
        min_weight=int(tensors[MIN_WEIGHT]),
    )
    check_shape(twin)
    return twin


def kind_of_image(shape, d, path):
    """THE FIRST TENSOR OF AN IMAGE GROUP NAMES ITS KIND, thus the walk is sequential and
    it reads the kind before it reads the count.

    It is not `model.kind_of_group`, which reads a CHECKPOINT: this image holds w_in
    transposed, and a square query is refused outright because the circuit's query walk
    reads 2d terms and there is no narrow path."""
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

# THE FORMATS THIS ERA NAMES OF ITS OWN; every other stands in `ar_quantized.py`.
# `V_Q` is the value rows of a block and of the attention rings, Q12 in int16. Era four's
# `KV_Q` names an attention ring's four tensors and nothing else, thus the two are one
# number and not one format.
V_Q = 12
S_Q = 12  # the state of the recurrence
ALPHA_Q = 15  # the decay of one step
BETA_Q = 15  # the input coefficient
# the gate product, in an int32: two Q12 values multiply and nothing truncates them before
# the norm that reads them
GATE_Q = 2 * V_Q


def matvec(y, weight, *, transposed, at, to):
    """One matvec column: the terms of a Q[at] vector against a row of the weight.
    [transposed] states that the image holds the tensor with its OUTER axis first, as it
    does for w_in, and the circuit reads that same order."""
    matrix = weight.values.T if transposed else weight.values
    return clamp16(ar_quantized.rescale(y @ matrix, at=at + weight.e, to=to))


class Clamps(NamedTuple):
    """The clamps of the walk, and the chances each one had. The formats of this era are
    chosen with margin and not metered on a trained checkpoint, thus a clamp that fires is
    the finding that says which is wrong -- and an error in the STATE carries forward,
    where era four could let a hot signal die with its window."""

    dt: int = 0
    dt_seen: int = 0
    beta: int = 0
    beta_seen: int = 0
    state: int = 0
    state_seen: int = 0

    def counting(self, name, hit, size):
        """[name] and its `_seen` twin, moved together: a counter that rose without its
        chances would read as a rate this walk never had"""
        seen = f"{name}_seen"
        return self._replace(
            **{
                name: getattr(self, name) + int(hit.sum()),
                seen: getattr(self, seen) + size,
            }
        )


class Engine(NamedTuple):
    """One running inference over a batch of walks. Everything is frozen: a step gives
    the engine after it, thus a walk is a fold and no state hides in a mutable field.
    THE STATE AND THE TAPS ARE THE MEMORY and the only things that survive a step."""

    twin: Mamba
    state: np.ndarray  # [walks, blocks * d_in * n], Q12
    taps: np.ndarray  # [walks, blocks * channels * taps], Q12
    kc: np.ndarray  # [walks, attentions, ring, d], Q12 in a coarse byte
    vc: np.ndarray
    h: np.ndarray  # [walks, d], Q16
    position: int
    states: np.ndarray  # [walks], the generator of each walk
    clamps: Clamps


def engine(twin, seeds):
    """the origin of a batch of walks: a zero state, an empty tap ring, an empty key and
    value ring, and no residual"""
    check_shape(twin)
    shape = twin.widths
    walks, blocks = len(seeds), len(twin.blocks)
    rings = sum(1 for layer in twin.layers if layer.kind == recurrence.ZATTN)
    return Engine(
        twin=twin,
        state=np.zeros((walks, blocks * shape.d_in * shape.state), np.int64),
        taps=np.zeros((walks, blocks * shape.channels * shape.taps), np.int64),
        kc=np.zeros((walks, max(1, rings), twin.ring, shape.d), np.int64),
        vc=np.zeros((walks, max(1, rings), twin.ring, shape.d), np.int64),
        h=np.zeros((walks, shape.d), np.int64),
        position=0,
        states=engine_states(seeds),
        clamps=Clamps(),
    )


def block(e, layer, ordinal, h, state, taps, tally):
    """One block of the trunk: the stream after the residual join, and the clamps it met.
    It writes the state and the taps of its own region IN PLACE, on copies the caller made
    for this step, as the state RAM of the circuit is written in place."""
    shape = e.twin.widths
    d, d_in, heads, n = shape.d, shape.d_in, shape.heads, shape.state
    width, channels, head = shape.taps, shape.channels, shape.head
    position = e.position
    y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)
    zxbcdt = matvec(y, layer.w_in, transposed=True, at=ar_quantized.Y_Q, to=V_Q)
    # the convolution: the step's input enters the ring, then a row of [width] terms for
    # each channel, then the SiLU chain over the sums
    tap_base = ordinal * channels * width
    slot = tap_base + (np.arange(channels) * width) + (position & (width - 1))
    taps[:, slot] = zxbcdt[:, d_in : d_in + channels]
    kernel, kernel_e = layer.conv.values, layer.conv.e
    accumulated = np.zeros((len(h), channels), np.int64)
    for k in range(width):
        # TAP k READS THE STEP k BACK, and it reads ZERO while the walk has not run k
        # steps -- thus the origin needs no clearing walk and the rule is a mux
        if position < k:
            continue
        back = tap_base + (np.arange(channels) * width) + ((position - k) & (width - 1))
        accumulated = accumulated + (taps[:, back] * kernel[:, k])
    scaled = ar_quantized.rescale(accumulated, at=V_Q + kernel_e, to=V_Q)
    xbc = ar_quantized.silu(clamp16(scaled))
    x = xbc[:, :d_in]
    b = xbc[:, d_in : d_in + n]
    c = xbc[:, d_in + n :]
    # the decay of each head: softplus of the biased draw, then one exp2
    raw = zxbcdt[:, d_in + channels :]
    dt = ar_quantized.softplus(raw + layer.dt_bias)
    rails = (dt == INT16_HIGH) | (dt == INT16_LOW)
    tally = tally.counting("dt", rails, dt.size)
    alpha = exp2_of_magnitude(apply_scale(layer.decay, DECAY_Q_BITS, dt))
    # the state update and the readout, head by head. [beta] is the inject operand of the
    # head: [state] products of [dt] against B, written before the walk.
    base = ordinal * d_in * n
    read = np.zeros((len(h), d_in), np.int64)
    for hd in range(heads):
        wide = (dt[:, hd, None] * b) >> (V_Q + V_Q - BETA_Q)
        tally = tally.counting("beta", clamps16(wide), wide.size)
        beta = clamp16(wide)
        lanes = np.arange(hd * head, (hd + 1) * head)
        rows = base + (lanes[:, None] * n) + np.arange(n)
        updated = (
            (alpha[:, hd, None, None] * state[:, rows])
            + (x[:, lanes, None] * beta[:, None, :])
        ) >> ALPHA_Q
        tally = tally.counting("state", clamps16(updated), updated.size)
        state[:, rows] = clamp16(updated)
        # the readout reads the state the update just wrote, and the skip folds into the
        # row as its last term
        read[:, lanes] = clamp16(
            (
                (state[:, rows] * c[:, None, :]).sum(axis=-1)
                + (x[:, lanes] * layer.d_skip[hd])
            )
            >> S_Q
        )
    # THE GATE PRODUCT STAYS WIDE. Both operands are Q12 values well under one, thus a
    # truncation back to Q12 would keep five bits of a product that holds seventeen --
    # right before the norm, which does not care what scale it arrives in.
    gated = read * ar_quantized.silu(zxbcdt[:, :d_in])
    g = ar_quantized.rms_norm_q(gated, at=GATE_Q, width=d_in)
    return ar_quantized.join(h, layer.w_out, values=g, at=V_Q), tally


def attention(e, layer, ordinal, h, embedding, kc, vc):
    """One attention layer, era four's with one addition: the query and the key read the
    JOINED vector -- the normed stream beside the normed embedding -- thus their walk is
    2d terms where the value's is d."""
    twin = e.twin
    d, slots = twin.d, twin.ring
    cur = e.position & (slots - 1)
    filled = min(e.position + 1, slots)
    y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)
    joined = np.concatenate([y, embedding], axis=-1)

    def project(name, source):
        return matvec(
            source, getattr(layer, name), transposed=False, at=ar_quantized.Y_Q, to=V_Q
        )

    kc[:, ordinal, cur, :] = ar_quantized.coarse_to_ring(project("wk", joined))
    vc[:, ordinal, cur, :] = ar_quantized.coarse_to_ring(project("wv", y))
    # the rings of THIS attention site: slicing the ordinal here lets `attend` name none
    context = ar_quantized.attend(
        kc[:, ordinal],
        vc[:, ordinal],
        query=project("wq", joined),
        cur=cur,
        filled=filled,
        heads=twin.heads,
        span=twin.span,
        row_q=V_Q,
    )
    return ar_quantized.join(h, layer.wo, values=context, at=V_Q)


def feed_forward(twin, layer, h):
    """Era four's feed-forward as a layer of its own: one matvec and a ReLU, Q10. The ReLU
    stands AFTER the clamp of the matvec where the circuit takes it before, which is the
    same integer either way."""
    y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=twin.d)
    hidden = np.maximum(
        matvec(
            y, layer.w1, transposed=False, at=ar_quantized.Y_Q, to=ar_quantized.HID_Q
        ),
        0,
    )
    return ar_quantized.join(h, layer.w2, values=hidden, at=ar_quantized.HID_Q)


def layer_streams(e, classes, phase):
    """The residual stream after the embed and after each layer of the step the engine
    would take next, in the order the circuit writes them. A frame gate says only THAT the
    circuit and the twin parted; this says where."""
    twin = e.twin
    state, taps = e.state.copy(), e.taps.copy()
    kc, vc = e.kc.copy(), e.vc.copy()
    h = twin.head.embed(classes, phase)
    embedding = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=twin.d)
    written = [h]
    tally = e.clamps
    for layer, ordinal in zip(twin.layers, twin.ordinals()):
        if layer.kind == recurrence.MAMBA:
            h, tally = block(e, layer, ordinal, h, state, taps, tally)
        elif layer.kind == recurrence.ZATTN:
            h = attention(e, layer, ordinal, h, embedding, kc, vc)
        else:
            h = feed_forward(twin, layer, h)
        written.append(h)
    return written, e._replace(
        h=written[-1],
        state=state,
        taps=taps,
        kc=kc,
        vc=vc,
        position=e.position + 1,
        clamps=tally,
    )


def forward(e, classes, phase):
    """one step of the recurrence: the engine after it"""
    return layer_streams(e, classes, phase)[1]


def next_step(e):
    """one step of the walk -- `ar_quantized.next_step` over era five's own recurrence"""
    return ar_quantized.next_step(e, forward)


def walk(twin, seeds, steps):
    """the classes of each step of the walk, and the draws behind them"""
    return ar_quantized.walk(engine(twin, seeds), steps, forward)


def streams(twin, seeds, steps):
    """the stream writes of each step, in the order the circuit writes them; it walks the
    model, thus they are the writes of a real walk"""
    e = engine(twin, seeds)
    written = []
    # `ar_quantized.next_step` states the lead-in and the chain, and the trunk pass it
    # takes is the one this gate reads: a step runs the recurrence once and not twice
    def recorded(e, classes, phase):
        rows, e = layer_streams(e, classes, phase)
        written.append(rows)
        return e

    for _ in range(steps):
        e, _, _ = ar_quantized.next_step(e, recorded)
    return written


# ---------------------------------------------------------------------
# what the quantization costs
# ---------------------------------------------------------------------


@nnx.jit
def float_step(held, carry, classes, phases):
    """One float step of the walk, for the drift report to score the twin against.

    It takes the model as an ARGUMENT and stands at the module level, thus its compiled
    form is keyed on the shapes and every step of a drift run reuses the first compile."""
    return held.forward_step(carry, classes, phases)


@nnx.jit
def float_row(held, stream, drawn):
    """the float logits of the seats of one step, on the stream the step before it left"""
    return held.head.logits(stream[:, None, :], drawn[None])[0, 0]


class Drift(NamedTuple):
    """What the quantization costs, measured on the walk the board takes."""

    steps: int  # the steps of the walk, the silent lead-in inside
    draws: int  # four for each drawn step: one for each seat of the chain
    same_peak: int  # the draws where both models elect the same class
    same_draw: int  # the draws where both models pick the same class
    mean_cosine: float
    clamps: Clamps


def drift(model, *, steps, seed, ring=ELECTED_RING):
    """The quantized walk, scored against the float model draw for draw.

    ONE WEIGHTS SOURCE AND ONE POLICY: the walk quantizes `model` itself, thus the pair
    cannot slip. The float pass is TEACHER-FORCED on the quantized history and on the
    quantized chain -- it reads the classes the engine drew and conditions each seat on
    the classes the engine chose -- thus what the report measures is the quantization and
    never a walk that parted for another reason.

    BOTH MODELS TAKE ONE STEP FOR ONE STEP. Era four had to re-run a whole window at every
    step, which made a long comparison quadratic; here each carries its own memory, thus
    the walk can run past many decay lifetimes -- which it must, because a state error is
    cumulative in a way era four never had.

    The same-draw share reads the float draw on the very uniform the engine took, thus a
    difference there is the arithmetic and not the generator."""
    e = engine(Mamba.from_float(model, ring=ring), [seed])
    carry = model.initial_carry(1, context=ring)
    counted = measure.Counted()
    stream = None
    for at in range(steps):
        e, classes, chain_draws = next_step(e)
        # THE CHAIN OF A STEP READS THE STREAM OF THE STEP BEFORE IT, on both sides: the
        # float row must be the row that same forward states and never the one this
        # step's classes make
        if chain_draws and stream is not None:
            floated = np.asarray(float_row(model, stream, classes)).astype(np.float64)
            counted = measure.count_chain_draws(
                counted,
                floated,
                chain_draws,
                temperature=ELECTED_TEMPERATURE,
                min_p=ELECTED_MIN_P,
            )
        carry, stream = float_step(
            model,
            carry,
            np.asarray(classes, np.int32),
            np.array([at % ar_model.PHASE_BUCKETS], np.int32),
        )
    return Drift(
        steps=steps,
        draws=counted.draws,
        same_peak=counted.same_peak,
        same_draw=counted.same_draw,
        mean_cosine=counted.cosine / max(1, counted.draws),
        clamps=e.clamps,
    )
