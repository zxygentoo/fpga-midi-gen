"""The integer rules of the STEP-FRAME twins: what eras four and five hold and era six
does not.

The two eras are one machine in outline -- a residual stream of one frame's width, a norm
on it, a chained head over the four seats, and for era five's Zamba layer the same
attention over a ring -- and every format and rule of that machine is here, once, for both
twins. `quantized.py` holds what all three eras share.

THE CUT RUNS ONE WAY: this module imports `quantized` and `quantized` imports nothing
back. `lib/nn/quantized.ml` is the same pair undivided below the seam, and
`tests/test_quantized.py` states the numbers both sides must give.

THE WALK AT THE FOOT OF THIS FILE READS AN ENGINE NOTHING HERE DECLARES: `chain`,
`next_step` and `walk` take an era's own `Engine`, which no class here can be the base of.
`walk`'s docstring names the fields it reads.
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

# THE FORMATS OF THE MACHINE, `Nn_quantized.Constants`: a Q number holds value * 2^-q. A
# twin that wrote a format of its own would part from its circuit in silence.
H_Q = 16  # the residual stream, in int32
Y_Q = 12  # the normed vector, and the score of attention: int16
HID_Q = 10  # the feed-forward hidden vector after its ReLU: int16


# the rms epsilon of the float models, in the Q of the squared stream
EPS_Q = int(round_half_up(math.ldexp(1e-6, 2 * Y_Q)))


# the integer arithmetic both circuits share


def rescale(value, *, at, to):
    """value * 2^-at as value * 2^-to; the arithmetic shift floors, as the circuits'
    does"""
    if to >= at:
        return value << (to - at)
    return value >> (at - to)


def truncated(numerator, denominator):
    """OCaml's `/` on integers, which goes TOWARD ZERO where numpy's `//` floors. Every
    division of every circuit truncates, thus a floor here would part from it on the
    negative half of a stream and nowhere else."""
    numerator = np.asarray(numerator, np.int64)
    denominator = np.asarray(denominator, np.int64)
    sign = np.sign(numerator) * np.sign(denominator)
    return sign * (np.abs(numerator) // np.abs(denominator))


def isqrt(values):
    """Floor of the square root, the one answer the [Isqrt] unit gives. The float root is
    correct to a unit at these widths and two steps settle it; the loop is written all the
    same, because a silently wrong root is a silently wrong norm."""
    values = np.asarray(values, np.int64)
    guess = np.where(values <= 0, 0, np.sqrt(np.maximum(values, 0)).astype(np.int64))
    while True:
        low = np.maximum(guess - ((guess * guess > values) & (guess > 0)), 0)
        high = low + ((low + 1) * (low + 1) <= values)
        if np.array_equal(high, guess):
            return guess
        guess = high


def rms_norm_q(v, *, at, width):
    """rms_norm over [width] elements of a Q[at] vector, giving Q[Y_Q]: one squared
    Q[Y_Q] copy, one isqrt, one truncating division for each element. The stream enters at
    [H_Q] and a Mamba gate at 2 [Y_Q], thus [at] is what moves between callers."""
    copy = rescale(v, at=at, to=Y_Q)
    total = (copy * copy).sum(axis=-1, keepdims=True)
    mean = (total >> (width.bit_length() - 1)) + EPS_Q
    return clamp16(truncated(v * (1 << ((2 * Y_Q) - at)), isqrt(mean)))


def join(h, weight, *, values, at):
    """A residual join: [values] times the weight lands on the stream, and the weight's
    exponent folds into the shift with [at]. That shift is the whole of what a residual
    write can get wrong, thus every era does it here."""
    return h + rescale(values @ weight.values, at=at + weight.e, to=H_Q)


# the tables: what the arithmetic cannot reach


# The sigmoid of a Q12 value, in Q15. The input is int16, thus |v| < 8 exactly and 256
# buckets of 256 Q12 units cover it. THE ENTRY IS THE CENTRE OF ITS BUCKET and not its
# left edge: the centre halves the worst error, and the centres are symmetric about zero,
# thus sigmoid(-v) = 1 - sigmoid(v) survives the quantization.
SIGMOID_TABLE = np.array(
    [
        int(round_half_up(32768.0 / (1.0 + math.exp(-((j - 128) + 0.5) / 16.0))))
        for j in range(256)
    ],
    np.int64,
)


# The correction term of the softplus, ln(1 + exp(-|v|)) in Q12: softplus(v) = relu(v) +
# this. The ramp is exact and carries a large input whole, thus the table only holds a
# quantity that falls to nothing -- one Q12 unit at |v| = 8, the int16 maximum.
SOFTPLUS_TABLE = np.array(
    [
        int(round_half_up(4096.0 * math.log(1.0 + math.exp(-(j + 0.5) / 32.0))))
        for j in range(256)
    ],
    np.int64,
)


def sigmoid_q(value):
    """`Nn_quantized.For_test.sigmoid_q`: the sigmoid of a Q12 value in Q15. The index is
    the top eight bits with the sign flipped, which is no arithmetic in a circuit."""
    return SIGMOID_TABLE[((np.asarray(value, np.int64) >> 8) + 128) & 255]


def silu(value):
    """the value times its sigmoid, shifted back to Q12 and clamped"""
    value = np.asarray(value, np.int64)
    return clamp16((value * sigmoid_q(value)) >> 15)


def softplus(value):
    """The ramp plus the correction the table holds, the rule of the `Softplus` unit. The
    sum rides an int16, thus the input clamps before and the result after; the index clamp
    catches -32768, whose magnitude does not fit the table."""
    value = clamp16(np.asarray(value, np.int64))
    index = np.minimum(255, np.abs(value) >> 7)
    return clamp16(np.maximum(value, 0) + SOFTPLUS_TABLE[index])


# the shape rules the circuit forces


def score_shift(*, row_q, head_d):
    """`Constants.score_shift`: what carries a score walk's sum from Q(2 [row_q]) to
    Q[Y_Q] and applies the 1/sqrt([head_d]) in the same shift, thus [head_d] is a power of
    four"""
    return (2 * row_q) - Y_Q + ((head_d.bit_length() - 1) // 2)


def is_power_of_four(value):
    """Does `score_shift` divide by sqrt([value]) exactly? The shift is
    `(bit_length - 1) // 2`, thus a power of two that is NOT a power of four scales by the
    next power of two down and the circuit agrees with the twin in silence."""
    return value >= 1 and not value & (value - 1) and not (value.bit_length() - 1) % 2


def slope_exponent(*, span, heads, head):
    """`Constants.slope_exponent`: the ALiBi exponent of one head -- its slope is
    2^-(this), thus the penalty of an age is a shift and never a multiply"""
    return (span * (head + 1)) // heads


# a layer as the machine holds it


def fixed_q12(values, bound):
    """a per-head number in Q12, clamped to the PORT that carries it; the bound is a fact
    of the circuit, thus the caller states it"""
    values = np.ldexp(np.asarray(values, np.float64), 12)
    return np.clip(round_half_up(values), -bound, bound).astype(np.int32)


class Weights:
    """A layer that is nothing but named `Weight`s, thus the name list alone spells it.
    A layer with a number that is no tensor -- era five's block -- is not one of these.

    THE TWIN HOLDS THE FLOAT LAYER'S TENSORS UNDER THE SAME ATTRIBUTE NAMES, so a reader
    can audit the two trees layer for layer. `names` states the order, which is the
    checkpoint order and the ROM order behind it. Each tensor takes its own exponent.

    IT IS NOT A FLAX MODULE, and neither is any step-frame twin: a twin holds host numpy
    in int64, never traced and never a pytree, thus `nnx.Module` would buy a tree walk
    every field would then switch off. Era six's twin is the other case."""

    def __init__(self, weights):
        for name, weight in zip(self.names, weights):
            setattr(self, name, weight)

    @classmethod
    def from_float(cls, layer):
        """one float layer under the exponent rule, tensor for tensor"""
        return cls([Weight.from_float(tensor) for tensor in layer.tensors()])

    def tensors(self):
        return [getattr(self, name) for name in self.names]


class Head:
    """The two tables of `Head` as the machine holds them -- and ONE exponent over both.

    THE SEAT AND PHASE TABLES SHARE IT, from the larger peak: their rows ADD, thus two
    exponents would be two formats inside one sum. The module's shape holds the rule and
    `from_file` refuses a file that states two.

    The four seat tables are ONE tensor, seat 0 first: a circuit reaches row
    (seat * CLASSES + class) with a shift and an add from the base."""

    def __init__(self, *, seats, phase, e):
        self.seats = np.asarray(seats, np.int64)
        self.phase = np.asarray(phase, np.int64)
        self.e = int(e)

    @property
    def d(self):
        return self.seats.shape[-1]

    @classmethod
    def from_file(cls, tensors, exponents):
        """the head of a contract file: tensors "0" and "1", under the one exponent.
        The refusal stands here and not in an era's `load`, because the rule is this
        class's."""
        if exponents[0] != exponents[1]:
            raise ValueError("the seat and phase tables must share one exponent")
        return cls(seats=tensors["0"], phase=tensors["1"], e=int(exponents[0]))

    @classmethod
    def from_float(cls, head):
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
        """The seat table holds one row for each seat and class, at width [d]. A table of
        another height sends every seat but the first to another seat's rows."""
        if self.seats.size != corpus.SEATS * corpus.CLASSES * d:
            raise ValueError("the seat table holds no row for each seat and class")


# the attention over a ring


def coarse_to_ring(row):
    """what a KV ring keeps of a Q12 row: the top byte, with eight zero low bits restored
    at the read. The circuit stores eight bits, thus the granularity is 2^-4 and the
    format stays Q12. A query does not pass here -- only the stored rows coarsen."""
    return (np.asarray(row, np.int64) >> 8) << 8


def attend(keys, values, *, query, cur, filled, heads, span, row_q):
    """Attention of one site over the newest [filled] slots of its rings: the merged
    context of [query], head by head.

    [keys] and [values] are the rings of ONE site, as [walks, slots, d]; the caller slices
    its own axis. The ring is read NEWEST FIRST, which is what the ALiBi slope counts: the
    bias of a slot is its age times the slope of its head."""
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
        # THE NEGATION STANDS OUTSIDE THE SCALE, as in the temper of the draw: the
        # circuit scales the distance BELOW the peak and negates the shifted product, and
        # negating first would round the other way. `exp2_q` is that order, named.
        weights = exp2_q(apply_scale(LOG2E.q_value, LOG2E.q, scores - peak))
        total = weights.sum(axis=-1, keepdims=True)
        merged = (weights[:, :, None] * values[:, :, band]).sum(axis=1)
        context[:, band] = clamp16(truncated(merged, total))
    return context


# the chain and the walk over it


# the silent lead-in of a boot, in steps: one bar, as the float samplers play it
LEAD = corpus.BAR_STEPS


def tempered_weights(twin, logits):
    """the Q15 weight of every class of one seat, and the min-p floor over it; a class the
    floor refuses weighs nothing and the pick cannot land on it"""
    peak = logits.max(axis=-1, keepdims=True)
    weights = exp2_q(apply_scale(twin.temper.q_value, twin.temper.q, logits - peak))
    return np.where(weights >= twin.min_weight, weights, 0)


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
    """One step of the walk: the engine after it, the classes of the frame, and the draws.

    THE BOOT IS A LEAD-IN OF SILENCE, one bar of it, drawing nothing and taking no
    number from the generator -- the model opens the music itself after it. [forward]
    is the era's own trunk, the only part of a step the two frozen eras do not share."""
    phase = e.position % corpus.BAR_STEPS
    if e.position < LEAD:
        classes = np.full((len(e.h), corpus.SEATS), corpus.SILENCE, np.int64)
        draws = []
    else:
        e, draws = chain(e)
        classes = np.stack([draw.drawn for draw in reversed(draws)], axis=-1)
    return forward(e, classes, phase), classes, draws


def walk(e, steps, forward):
    """The classes of each step of the walk, and the draws behind them: the integer twin
    of the float sampler, the lead-in counted inside [steps] as it is there.

    [e] IS AN ENGINE AND NOTHING HERE DECLARES ONE. The contract is this sentence:
    `chain`, `next_step` and `walk` read `e.twin` (the quantized model), `e.h` (the
    residual stream, Q`H_Q` int32), `e.position` and `e.states` (one generator for each
    walk), and rebuild with `e._replace`. An era adds what its own arithmetic carries,
    and [forward] reads those."""
    played, taken = [], []
    for _ in range(steps):
        e, classes, draws = next_step(e, forward)
        played.append(classes)
        taken.append(draws)
    return np.stack(played, axis=1), taken
