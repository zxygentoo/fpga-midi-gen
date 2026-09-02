"""The integer twin of the state-space model: the weights, the formats and the file.

The float model of `mamba/model.py` is what the ear elected; this is the same network in
the arithmetic the board can hold -- int8 weights with a power-of-two exponent for each
tensor, int16 activations, the per-head numbers folded into the constants the ops carry,
and the draw in integers. What RUNS it is `quantized/infer.py`, and the circuit of era
five must equal that walk OPERATION FOR OPERATION. What is not this era's comes from
`quantized.py` and `ar_quantized.py`.

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

import numpy as np
from safetensors.numpy import load_file, save_file

import ar_model
import ar_quantized
import corpus
import quantized as q
from mamba import model as recurrence

# the names of this era's own scalars; the shared ones are `contract`'s
SPAN = "span"
RING = "ring"
DECAY_Q_VALUE = "decay_q_value"
DECAY_Q = "decay_q"
DT_BIAS = "dt_bias"
D_SKIP = "d_skip"
# the tensors the file carries beside its numbered weights
BESIDE_THE_WEIGHTS = (
    q.EXPONENTS,
    q.TEMPER,
    q.MIN_WEIGHT,
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

# the policy has one home, `quantized.py`; era five's player reads it through this module
ELECTED_TEMPERATURE = q.ELECTED_TEMPERATURE
ELECTED_MIN_P = q.ELECTED_MIN_P

# The depth of the ring at INFERENCE, a choice of the player and no fact of the training
# run. It is era four's training window, thus a window of the loss reads exactly the
# attention the trainer computed.
ELECTED_RING = recurrence.ATTN_CONTEXT


def decay_scale(a_log):
    """`Constants.decay_scale`: a * log2(e) as the constant the Decay op carries.

    It takes the exponential through `math.exp` in float64 -- the C library's own, which
    is what the OCaml quantizer read -- because one ulp here moves a ROM byte."""
    a = math.exp(float(a_log))
    q_value = int(q.round_half_up(math.ldexp(a / math.log(2.0), DECAY_Q_BITS)))
    return min(max(q_value, 0), DECAY_HIGH)


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
            w_in=q.Weight.from_float(np.ascontiguousarray(np.asarray(layer.w_in[...]).T)),
            conv=q.Weight.from_float(layer.conv[...]),
            w_out=q.Weight.from_float(layer.w_out[...]),
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
            temper=q.Temper.from_float(temperature),
            min_weight=q.min_weight(min_p),
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

    def check_shape(self):
        """The rules the consumers assume, refused here rather than inside a walk.

        The arithmetic of the circuit is shifts and address concatenations, thus every
        field of an address is a power of two, and the attention head width obeys the
        stronger rule of `ar_quantized.score_shift`. A PLAN OF ATTENTION ALONE is refused
        first, because the widths come out of a block and the message should say so."""
        if not self.blocks:
            raise ValueError("a plan of attention alone is not this model")
        shape = self.widths
        for name, value in [
            ("d", shape.d),
            ("d_in", shape.d_in),
            ("heads", shape.heads),
            ("the state", shape.state),
            ("the taps", shape.taps),
            ("the ring", self.ring),
            ("the block head", shape.head),
        ]:
            if value < 1 or value & (value - 1):
                raise ValueError(f"{name} is {value} and must be a power of two")
        if not ar_quantized.is_power_of_four(shape.head_d):
            raise ValueError(f"the head width {shape.head_d} must be a power of four")
        self.head.check_tables(shape.d)
        for block in self.blocks:
            for row in block.rows():
                if row.shape != (shape.heads,):
                    raise ValueError(
                        f"a per-head row is {row.shape}, not {(shape.heads,)}"
                    )

    def save(self, path):
        """the contract file of this twin: the module docstring holds the layout and
        the reasons"""
        self.check_shape()
        tensors = q.image_tensors(self.tensors())
        tensors[SPAN] = q.scalar_tensor(self.span)
        tensors[RING] = q.scalar_tensor(self.ring)
        tensors[q.TEMPER] = self.temper.tensor()
        tensors[q.MIN_WEIGHT] = q.scalar_tensor(self.min_weight)
        # one tensor for each per-head row, a line for each BLOCK in the plan order: the
        # elaboration reads one image and not a tree
        decay, dt_bias, d_skip = (
            np.stack(rows, axis=0).astype(np.int32)
            for rows in zip(*[block.rows() for block in self.blocks])
        )
        tensors[DECAY_Q_VALUE] = decay
        tensors[DECAY_Q] = np.full_like(decay, DECAY_Q_BITS)
        tensors[DT_BIAS] = dt_bias
        tensors[D_SKIP] = d_skip
        save_file(tensors, str(path))

    @classmethod
    def load(cls, path):
        """the model of one contract file; a round trip through `save` is exact.

        THE PLAN COMES BACK OUT OF THE SHAPES, by the rule the module docstring states,
        thus the reader of this side and the reader of the elaboration walk the image
        alike."""
        tensors = load_file(str(path))
        count = len(tensors) - len(BESIDE_THE_WEIGHTS)
        if count < len(ar_model.TABLES) + 1:
            raise ValueError(
                f"{path}: {len(tensors)} tensors is no quantized state model"
            )
        exponents = tensors[q.EXPONENTS]
        d = tensors["0"].size // (corpus.SEATS * corpus.CLASSES)
        plan, groups, at = [], [], len(ar_model.TABLES)
        while at < count:
            kind = kind_of_image(tensors[str(at)].shape, d, path)
            names = IMAGE_TENSORS[kind]
            plan.append(kind)
            groups.append(
                q.image_from_tensors(tensors, exponents, first=at, count=len(names))
            )
            at += len(names)
        if at != count:
            raise ValueError(
                f"{path}: {count} image tensors do not fill whole layer groups"
            )
        rows = iter(zip(tensors[DECAY_Q_VALUE], tensors[DT_BIAS], tensors[D_SKIP]))

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

        twin = cls(
            head=ar_quantized.Head.from_file(tensors, exponents),
            layers=[layer_of(kind, group) for kind, group in zip(plan, groups)],
            span=int(tensors[SPAN]),
            ring=int(tensors[RING]),
            temper=q.Temper.from_file(tensors, key=q.TEMPER),
            min_weight=int(tensors[q.MIN_WEIGHT]),
        )
        twin.check_shape()
        return twin
