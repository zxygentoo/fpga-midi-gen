"""The integer rules of the STEP-FRAME twins: what eras four and five hold and era six
does not.

It stands to `quantized.py` as `ar_train.py` stands to each era's trainer. The two eras
are one machine in outline -- a residual stream of one frame's width, a norm on it, a
chained head over the four seats, and for era five's Zamba layer the same attention over
a ring -- and every format and rule of that machine is here, once, for both twins. What
is one thing across ALL THREE eras is `quantized.py`: the int16 rails, the exponent rule,
the temper, the exp2 table, the counted write, the integer pick and the contract file.

THE CUT RUNS ONE WAY. This module imports `quantized` and `quantized` imports nothing
back, thus era six can read the shared rules without reading a stream format it has no
stream for.

`lib/nn/quantized.ml` is the same pair of modules in OCaml, undivided: the OCaml side
carries one file for the three circuits and the elaborations take what each needs. What
holds the two sides together is `tests/test_quantized.py`, which states the numbers both
must give.

THE WALK AT THE FOOT OF THIS FILE READS AN ENGINE NOTHING HERE DECLARES. `chain`,
`next_step` and `walk` take an era's own `Engine`, a `NamedTuple` a `NamedTuple` cannot be
made to subclass; `walk`'s docstring names the fields it reads.
"""

import math
from typing import NamedTuple

import numpy as np

import corpus
import prng
from quantized import (
    LOG2E,
    Weight,
    apply_scale,
    clamp16,
    exp2_q,
    max_exponent,
    pick,
    quantize,
    round_half_up,
)

# THE FORMATS OF THE MACHINE, `Nn_quantized.Constants`. A Q number holds value * 2^-q. The
# OCaml side states these ONCE for all three circuits; this states them once for all the
# twins, and a twin that wrote a format of its own would part from its circuit in silence.
H_Q = 16  # the residual stream, in int32
Y_Q = 12  # the normed vector, and the score of attention: int16
HID_Q = 10  # the feed-forward hidden vector after its ReLU: int16


# the rms epsilon of the float models, in the Q of the squared stream
EPS_Q = int(round_half_up(math.ldexp(1e-6, 2 * Y_Q)))


# the silent lead-in of a boot, in steps: one bar, as the float samplers play it
LEAD = corpus.BAR_STEPS


def rescale(value, *, at, to):
    """value * 2^-at as value * 2^-to; the arithmetic shift floors, as the circuits'
    does"""
    if to >= at:
        return value << (to - at)
    return value >> (at - to)


def truncated(numerator, denominator):
    """OCaml's `/` on integers, which goes TOWARD ZERO where numpy's `//` floors.

    Every division of every circuit truncates, thus a floor here would part from it on the
    negative half of a stream and nowhere else -- which is the kind of difference that
    makes music and is still wrong."""
    numerator = np.asarray(numerator, np.int64)
    denominator = np.asarray(denominator, np.int64)
    sign = np.sign(numerator) * np.sign(denominator)
    return sign * (np.abs(numerator) // np.abs(denominator))


def isqrt(values):
    """floor of the square root, over an array: the one answer the [Isqrt] unit gives.

    The float root is correct to a unit at these widths and the two steps settle it; the
    loop is written all the same, because a silently wrong root is a silently wrong norm.
    """
    values = np.asarray(values, np.int64)
    guess = np.where(values <= 0, 0, np.sqrt(np.maximum(values, 0)).astype(np.int64))
    while True:
        low = np.maximum(guess - ((guess * guess > values) & (guess > 0)), 0)
        high = low + ((low + 1) * (low + 1) <= values)
        if np.array_equal(high, guess):
            return guess
        guess = high


def rms_norm_q(v, *, at, width):
    """rms_norm over [width] elements of a Q[at] vector, giving Q[Y_Q].

    The sum squares a Q[Y_Q] copy -- one DSP-sized product -- then one isqrt, and one
    truncating division for each element. The stream enters at [H_Q] and the gate of a
    Mamba block at 2 [Y_Q], thus the shift of the NUMERATOR is the one thing that moves
    between callers."""
    copy = rescale(v, at=at, to=Y_Q)
    total = (copy * copy).sum(axis=-1, keepdims=True)
    mean = (total >> (width.bit_length() - 1)) + EPS_Q
    return clamp16(truncated(v * (1 << ((2 * Y_Q) - at)), isqrt(mean)))


def score_shift(*, row_q, head_d):
    """`Constants.score_shift`: what carries a score walk's sum from Q(2 [row_q]) to
    Q[Y_Q] and applies the 1/sqrt([head_d]) in the same shift, thus [head_d] is a power of
    four"""
    return (2 * row_q) - Y_Q + ((head_d.bit_length() - 1) // 2)


def is_power_of_four(value):
    """Does `score_shift` divide by sqrt([value]) exactly?

    The shift is `(bit_length - 1) // 2`, thus a power of two that is not a power of four
    scales by the next power of two DOWN and the circuit agrees with the twin in silence.
    Every attention head width is checked against this."""
    return value >= 1 and not value & (value - 1) and not (value.bit_length() - 1) % 2


def slope_exponent(*, span, heads, head):
    """`Constants.slope_exponent`: the ALiBi exponent of one head -- its slope is
    2^-(this), thus the penalty of an age is a shift and never a multiply"""
    return (span * (head + 1)) // heads


def fixed_q12(values, bound):
    """a per-head number in Q12, clamped to the PORT that carries it.

    Era five's `dt_bias` joins an int16 sum and its `d_skip` rides an 18-bit operand port,
    thus the bound is a fact of the circuit and the caller states it."""
    values = np.ldexp(np.asarray(values, np.float64), 12)
    return np.clip(round_half_up(values), -bound, bound).astype(np.int32)


def join(h, weight, *, values, at):
    """a residual join: [values] times the weight lands on the stream; the exponent of the
    weight folds into the shift with [at], the format of [values].

    It stands here and not in a twin because it is the one arithmetic every residual write
    of every era does, and the shift it takes is the whole of what a residual write can
    get wrong."""
    return h + rescale(values @ weight.values, at=at + weight.e, to=H_Q)


class QuantizedImage:
    """A layer whose WHOLE image is named tensors: nothing stands beside the weights, thus
    the name list alone spells the layer.

    THE TWIN HOLDS THE FLOAT LAYER'S TENSORS UNDER THE SAME ATTRIBUTE NAMES, and that is
    the rule this class exists for: the float tree and the integer tree are one tree, and
    a reader can audit them layer for layer. `names` is the class attribute that states
    the order, and the order is the CHECKPOINT ORDER and the ROM order behind it, thus a
    subclass cannot carry one kind and read another's names.

    Each tensor takes its OWN exponent; nothing forces them together. A layer that is not
    all weights -- one whose float tensors become facts of another shape -- is not one of
    these and states itself.

    IT IS NOT A FLAX MODULE, and neither is any twin of the step-frame eras. A twin holds
    HOST NUMPY in int64 -- it is never traced, never split and never a pytree of device
    tensors -- thus `nnx.Module` would buy the tree walk and then every field would have
    to switch it off again with `nnx.data`. Era six's twin is the other case and is a
    module: it holds `nnx.Variable`s and `jax.jit` runs its forward."""

    def __init__(self, weights):
        for name, weight in zip(self.names, weights):
            setattr(self, name, weight)

    @classmethod
    def of(cls, layer):
        """one float layer under the exponent rule, tensor for tensor"""
        return cls([Weight.of(tensor) for tensor in layer.tensors()])

    def tensors(self):
        return [getattr(self, name) for name in self.names]


class QuantizedHead:
    """The two tables of `Head` as the machine holds them -- and ONE exponent over both.

    THE SEAT AND PHASE TABLES SHARE IT and take it from the larger peak: their rows ADD --
    the embedding sums them and the Embed op of a circuit walks them as one tensor -- thus
    a difference of exponents would be a difference of formats inside one sum. Here that
    rule is the shape of the module and cannot be broken by a caller; a FILE can still
    state two, and `of_file` refuses one that does.

    The four seat tables are ONE tensor, seat 0 first, and a circuit reaches a row of it
    with a shift and an add from the base: row (seat * CLASSES + class).

    Both frozen eras hold one of these, and the twin of each reads the stream at [H_Q]."""

    def __init__(self, *, seats, phase, e):
        self.seats = np.asarray(seats, np.int64)
        self.phase = np.asarray(phase, np.int64)
        self.e = int(e)

    @property
    def d(self):
        return self.seats.shape[-1]

    @classmethod
    def of_file(cls, tensors, exponents):
        """the head of a contract file: tensors "0" and "1", under the one exponent.

        THE REFUSAL STANDS HERE and not in an era's `load`, because the rule is this
        class's: a file that states two exponents states two formats inside one sum."""
        if exponents[0] != exponents[1]:
            raise ValueError("the seat and phase tables must share one exponent")
        return cls(seats=tensors["0"], phase=tensors["1"], e=int(exponents[0]))

    @classmethod
    def of(cls, head):
        """the float `Head` under the exponent rule, one exponent over both tables"""
        seats, phase = (np.asarray(t, np.float64) for t in head.tensors())
        shared = max_exponent(
            max(float(np.abs(t).max(initial=0.0)) for t in (seats, phase))
        )
        return cls(
            seats=quantize(seats, e=shared)[0],
            phase=quantize(phase, e=shared)[0],
            e=shared,
        )

    def embed(self, classes, phase):
        """the embedding: the four seat rows and the phase row add in the shared exponent,
        then shift to Q[H_Q]"""
        value = np.broadcast_to(self.phase[phase], (len(classes), self.d)).copy()
        for seat in range(corpus.SEATS):
            value = value + self.seats[seat, classes[:, seat]]
        return rescale(value, at=self.e, to=H_Q)

    def logits(self, stream, seat):
        """the tied head of one seat: rms_norm of the stream the chain has written so far,
        then that seat's table read backward; Q[Y_Q] logits over the classes"""
        return (rms_norm_q(stream, at=H_Q, width=self.d) @ self.seats[seat].T) >> self.e

    def add_row(self, stream, seat, drawn):
        """what the chain adds after a seat draws: the drawn row, in the stream's
        format"""
        return stream + rescale(self.seats[seat, drawn], at=self.e, to=H_Q)

    def tensors(self):
        """the two tables, in the order of the checkpoint and of the ROM"""
        return [Weight(self.seats, self.e), Weight(self.phase, self.e)]

    def check_tables(self, d):
        """the seat table holds one row for each seat and class, at width [d].

        THE RULE IS THIS CLASS'S, thus the refusal stands here and not in an era's
        `check_shape`: a circuit reaches row (seat * CLASSES + class) with a shift and an
        add, and a table of another height sends every seat but the first to another
        seat's rows."""
        if self.seats.size != corpus.SEATS * corpus.CLASSES * d:
            raise ValueError("the seat table holds no row for each seat and class")


# The sigmoid of a Q12 value, in Q15. The input is int16, thus its range is |v| < 8
# exactly and a clamp costs nothing: 256 buckets of 256 Q12 units cover it, and the index
# is the top eight bits with the sign flipped.
#
# The entry is the sigmoid at the CENTRE of its bucket and not at its left edge. The
# bucket is 1/16 wide and the slope peaks at 1/4, thus the left edge would bias every
# reading by up to 2^-10 of full scale; the centre halves the worst error and costs
# nothing at elaboration. The centres are symmetric about zero, thus the two halves of the
# table sum to 2^15 and sigmoid(-v) = 1 - sigmoid(v) survives the quantization.
SIGMOID_TABLE = np.array(
    [
        int(round_half_up(32768.0 / (1.0 + math.exp(-((j - 128) + 0.5) / 16.0))))
        for j in range(256)
    ],
    np.int64,
)


# The correction term of the softplus, ln(1 + exp(-|v|)), in Q12 over a Q12 magnitude:
# softplus(v) = relu(v) + this. The ramp is exact and carries the whole of a large input,
# thus the table only has to hold a quantity that falls to nothing: at |v| = 8, the
# largest magnitude an int16 Q12 value takes, it is one unit of Q12. 256 buckets of
# 128 units cover the range, and the entry is again the centre of its bucket.
SOFTPLUS_TABLE = np.array(
    [
        int(round_half_up(4096.0 * math.log(1.0 + math.exp(-(j + 0.5) / 32.0))))
        for j in range(256)
    ],
    np.int64,
)


def sigmoid_q(value):
    """`Nn_quantized.For_test.sigmoid_q`: the sigmoid of a Q12 value in Q15 -- the rule
    of the [Sigmoid] unit. The index is the top eight bits with the sign flipped, which is
    no arithmetic at all in a circuit."""
    return SIGMOID_TABLE[((np.asarray(value, np.int64) >> 8) + 128) & 255]


def silu(value):
    """the value times its sigmoid, shifted back to Q12 and clamped"""
    value = np.asarray(value, np.int64)
    return clamp16((value * sigmoid_q(value)) >> 15)


def softplus(value):
    """The ramp plus the correction the table holds -- the rule of the `Softplus` unit.

    The sum rides an int16, thus the input clamps before the table reads it and the result
    clamps after. The clamp of the index catches the one value -32768 whose magnitude does
    not fit the table."""
    value = clamp16(np.asarray(value, np.int64))
    index = np.minimum(255, np.abs(value) >> 7)
    return clamp16(np.maximum(value, 0) + SOFTPLUS_TABLE[index])


def coarse_to_ring(row):
    """what a KV ring keeps of a Q12 row: the top byte, with eight zero low bits restored
    at the read. The circuit stores eight bits, thus the granularity is 2^-4 and the
    format stays Q12. A query does not pass here -- only the stored rows coarsen."""
    return (np.asarray(row, np.int64) >> 8) << 8


def tempered_weights(twin, logits):
    """the Q15 weight of every class of one seat, and the min-p floor over it.

    The peak weighs 2^15, thus the floor is a plain share of it; a class the floor refuses
    weighs nothing and the pick cannot land on it."""
    peak = logits.max(axis=-1, keepdims=True)
    weights = exp2_q(apply_scale(twin.temper.q_value, twin.temper.q, logits - peak))
    return np.where(weights >= twin.min_weight, weights, 0)


def attend(keys, values, *, query, cur, filled, heads, span, row_q):
    """Attention of one site over the newest [filled] slots of its rings: the merged
    context of [query], head by head.

    [keys] and [values] are the rings of ONE site -- one layer of era four, one ring of
    era five -- as [walks, slots, d]; the caller slices its own axis, thus no era's name
    for that axis stands in here. [row_q] is the Q of a stored row, 12 under both eras,
    and each states its own because the two are one number and not one format.

    The ring is read NEWEST FIRST, which is what the ALiBi slope counts: the bias of a
    slot is its age times the slope of its head, and an unwritten slot is never read."""
    walks, slots, d = keys.shape
    head_d = d // heads
    ages = np.arange(filled)
    rows = (cur - ages) & (slots - 1)
    keys, values = keys[:, rows, :], values[:, rows, :]
    context = np.zeros((walks, d), np.int64)
    shift = score_shift(row_q=row_q, head_d=head_d)
    for head in range(heads):
        band = slice(head * head_d, (head + 1) * head_d)
        slope = slope_exponent(span=span, heads=heads, head=head)
        raw = (query[:, None, band] * keys[:, :, band]).sum(axis=-1)
        scores = (raw >> shift) - (ages << (Y_Q - slope))
        peak = scores.max(axis=-1, keepdims=True)
        # THE NEGATION STANDS OUTSIDE THE SCALE, as it stands outside the temper of the
        # draw: the circuit scales the score's distance BELOW the peak and negates the
        # shifted product, thus a scale that did not divide exactly would round the other
        # way if this side negated first. `exp2_q` is that order, named.
        weights = exp2_q(apply_scale(LOG2E.q_value, LOG2E.q, scores - peak))
        total = weights.sum(axis=-1, keepdims=True)
        merged = (weights[:, :, None] * values[:, :, band]).sum(axis=1)
        context[:, band] = clamp16(truncated(merged, total))
    return context


class Draw(NamedTuple):
    """one draw of the chain, over the batch"""

    seat: int
    logits: np.ndarray  # [walks, CLASSES], Q12 -- what the drift report compares
    word: np.ndarray  # [walks], the 24-bit uniform
    drawn: np.ndarray  # [walks], the class


def chain(e):
    """One frame, drawn in a chain from the soprano down: each seat reads the stream that
    the seats above it have written. The draws come back in the order they happened."""
    twin = e.twin
    stream, states, draws = e.h, e.states, []
    everyone = np.ones(len(stream), bool)
    for seat in reversed(range(corpus.SEATS)):
        logits = twin.head.logits(stream, seat)
        states, word = prng.uniform_word(states, everyone)
        drawn = pick(tempered_weights(twin, logits), word)
        if seat:
            stream = twin.head.add_row(stream, seat, drawn)
        draws.append(Draw(seat, logits, word, drawn))
    return e._replace(states=states), draws


def next_step(e, forward):
    """one step of the walk: the engine after it, the classes of the frame, and the draws.

    THE BOOT IS A LEAD-IN OF SILENCE, one bar of it, drawing nothing and taking no number
    from the generator. The model opens the music itself after it, thus the walk needs no
    pitch and no table to begin. [forward] is the era's own trunk, and it is the only part
    of a step that the two frozen eras do not share."""
    phase = e.position % corpus.BAR_STEPS
    if e.position < LEAD:
        classes = np.full((len(e.h), corpus.SEATS), corpus.SILENCE, np.int64)
        draws = []
    else:
        e, draws = chain(e)
        classes = np.stack([draw.drawn for draw in reversed(draws)], axis=-1)
    return forward(e, classes, phase), classes, draws


def walk(e, steps, forward):
    """the classes of each step of the walk, and the draws behind them.

    It is the integer twin of the float sampler, and the lead-in counts inside [steps] as
    it does there.

    [e] IS AN ENGINE AND NOTHING HERE DECLARES ONE. Each era's `Engine` is a `NamedTuple`
    of its own, which no class here can be the base of, thus the contract is this
    sentence: `chain`, `next_step` and `walk` read `e.twin` (the quantized model),
    `e.h` (the residual stream, Q`H_Q` int32), `e.position` (the step the walk stands at),
    `e.states` (one generator for each walk), and they rebuild the engine with
    `e._replace`. An era adds what its own arithmetic carries -- era five's block states,
    era four's rings -- and [forward] is what reads those."""
    played, taken = [], []
    for _ in range(steps):
        e, classes, draws = next_step(e, forward)
        played.append(classes)
        taken.append(draws)
    return np.stack(played, axis=1), taken
