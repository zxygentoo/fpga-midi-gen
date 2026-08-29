"""The integer rules of the twins: the fixed-point arithmetic every era's twin is built
on.

What stands here is the part of the integer arithmetic that is ONE THING ACROSS THE ERAS:
the fixed-point rails, the exponent rule of a checkpoint, the sampling policy, the shared
tables, the integer draw, and the walk of the step-frame eras around their trunks -- the
lead-in, the chain and the attention over a ring. Each era's twin keeps what is its own --
the parameter structures, the state formats of a recurrence, and the trunk pass of one
step. A rule written here is read by every twin and, through them, by every circuit.

`lib/nn/quantized.ml` is the same module in OCaml and the elaborations read it. The two
are TWO STATEMENTS OF ONE RULE and nothing in the types welds them; what holds them
together is `tests/test_fixed.py`, which states the numbers both must give. THE TWO
SIDES ARE LAID OUT ALIKE ON PURPOSE: this file stands to `nn.py` as `lib/nn/quantized.ml`
stands to the rest of `lib/nn`.

No float model reads this file and nothing here reads a float model: the seam between the
two is the exponent rule, and it runs one way -- `quantize` takes a trained tensor and
gives the machine's.
"""

import math
from typing import NamedTuple

import numpy as np
from flax import nnx

import data
import prng

# the rails of int16: a value that passes them saturates and never wraps. Every clamp of
# every twin reads them here, thus none can write a rail of its own and part from its
# circuit in silence.
INT16_BITS = 16
INT16_LOW = -(1 << (INT16_BITS - 1))
INT16_HIGH = (1 << (INT16_BITS - 1)) - 1

# the Q of log2(e), and the Q the temper takes: one below it. The extra bit is headroom
# for the temperature -- the circuits carry this constant on an 18-bit signed port, thus
# the Q of log2(e) would overflow that port under a temperature of about 0.36.
LOG2E_Q = 15
TEMPER_Q = LOG2E_Q - 1

# the Q the exp2 unit reads its magnitudes at, and the Q of its answer
EXP2_IN_Q = 12
EXP2_OUT_Q = 15

# THE FORMATS OF THE MACHINE, `Nn_quantized.Constants`. A Q number holds value * 2^-q. The
# OCaml side states these ONCE for all three circuits; this states them once for all the
# twins, and a twin that wrote a format of its own would part from its circuit in silence.
H_Q = 16  # the residual stream, in int32
Y_Q = 12  # the normed vector, and the score of attention: int16
HID_Q = 10  # the feed-forward hidden vector after its ReLU: int16


def round_half_up(x):
    """Base's `Float.iround_nearest_exn`: floor(x + 0.5).

    A TIE GOES TOWARD PLUS INFINITY, thus -2.5 is -2 and 2.5 is 3, where Python's `round`
    and `numpy.rint` are half-to-even. Every rounding of every twin goes through here."""
    return np.floor(np.asarray(x, np.float64) + 0.5)


# the rms epsilon of the float models, in the Q of the squared stream
EPS_Q = int(round_half_up(math.ldexp(1e-6, 2 * Y_Q)))

# the silent lead-in of a boot, in steps: one bar, as the float samplers play it
LEAD = data.BAR_STEPS

# The policy the ear elected on 2026-08-18: the draw the bitstreams commit to. The OCaml
# `Policy` went with the all-era cut, and this is the one home left -- an era that re-
# elects shadows these two in its own module and says so.
ELECTED_TEMPERATURE = 1.0
ELECTED_MIN_P = 0.05


def rescale(value, *, at, to):
    """value * 2^-at as value * 2^-to; the arithmetic shift floors, as the circuits'
    does"""
    if to >= at:
        return value << (to - at)
    return value >> (at - to)


def apply_scale(q_value, q, value):
    """`Constants.apply`: value times a fixed-point multiplier, toward negative infinity.

    The two halves of the scale travel together because they are one fact: a multiply that
    takes the wrong shift is silently wrong. [q_value] may be a per-head ROW, which is why
    this takes the two numbers and not a `Temper`."""
    return (value * q_value) >> q


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


def slope_exponent(*, span, heads, head):
    """`Constants.slope_exponent`: the ALiBi exponent of one head -- its slope is
    2^-(this), thus the penalty of an age is a shift and never a multiply"""
    return (span * (head + 1)) // heads


def largest_exponent(magnitude, *, opening, cap):
    """The largest exponent, from [opening] down, that keeps round(magnitude * 2^e) at
    [cap] or less.

    [opening] caps the all-zero value, where every exponent fits. The predicate falls
    monotonically in e, thus the first e that fits is the largest. Its readings differ
    only in where they open and what they must fit."""
    if magnitude <= 0.0:
        return opening
    e = opening
    while round_half_up(np.ldexp(magnitude, e)) > cap:
        e -= 1
    return e


def max_exponent(peak):
    """The exponent of one int8 tensor -- from 14 down, the largest that keeps
    round(peak * 2^e) inside the byte."""
    return largest_exponent(peak, opening=14, cap=127)


def quantize(weights, e=None):
    """The int8 form of one tensor, and the exponent that reads it.

    The byte is two's complement and the negative end is not used: the clamp is -127 and
    not -128, thus the image is symmetric and a negated weight is a negated byte. [e]
    overrides the exponent of the tensor's own peak, where tensors whose rows add share
    one."""
    weights = np.asarray(weights, np.float64)
    if e is None:
        e = max_exponent(float(np.abs(weights).max(initial=0.0)))
    return np.clip(round_half_up(np.ldexp(weights, e)), -127, 127).astype(np.int32), e


def fixed_q12(values, bound):
    """a per-head number in Q12, clamped to the PORT that carries it.

    Era five's `dt_bias` joins an int16 sum and its `d_skip` rides an 18-bit operand port,
    thus the bound is a fact of the circuit and the caller states it."""
    values = np.ldexp(np.asarray(values, np.float64), 12)
    return np.clip(round_half_up(values), -bound, bound).astype(np.int32)


def temper_of(temperature):
    """The sampling temper, log2(e) / T, as (q_value, q).

    `Nn_quantized.Constants.temper_at_one` is this rule at a temperature of one, which is
    the elected policy and the one reading a contract file carries."""
    if temperature <= 0.0:
        raise ValueError("the temperature is positive")
    return int(round_half_up(np.ldexp(1.0 / math.log(2.0) / temperature, TEMPER_Q))), (
        TEMPER_Q
    )


def min_weight_of(min_p):
    """The min-p floor as a share of the peak weight.

    The peak weighs 2^`EXP2_OUT_Q` after the temper, thus the floor is a plain share of it
    and the circuit compares two integers."""
    if not 0.0 <= min_p < 1.0:
        raise ValueError("min_p is 0 up to 1")
    return int(round_half_up(min_p * float(1 << EXP2_OUT_Q)))


class Temper(NamedTuple):
    """The sampling temper as the bitstream carries it: log2(e) / T at [q].

    The temperature is PROVENANCE and not arithmetic -- the temper is already folded --
    thus it travels in the metadata of a contract file alone, and a file written by an
    older tool can read back with no temperature at all."""

    q_value: int
    q: int
    temperature: float

    @classmethod
    def of(cls, temperature):
        q_value, q = temper_of(temperature)
        return cls(q_value, q, temperature)

    @classmethod
    def of_file(cls, tensors, metadata, *, key="temper"):
        """the temper a contract file carries: the pair from its named tensor and the
        temperature from the metadata, absent under a file an older tool wrote"""
        q_value, q = (int(value) for value in tensors[key])
        return cls(q_value, q, float(metadata.get("temperature", np.nan)))

    def tensor(self):
        """the pair as the contract file carries it -- `Nx_io` skips every dtype it does
        not hold, thus every scalar of a file travels as an int32 tensor"""
        return np.array([self.q_value, self.q], np.int32)


# log2(e): the exp2 form of an exponential, `Constants.log2e`. The temper is this
# constant divided by the temperature and it takes one Q less, which is `temper_of`.
LOG2E = Temper(int(round_half_up(math.ldexp(1.0 / math.log(2.0), LOG2E_Q))), LOG2E_Q, 1.0)


class Weight(NamedTuple):
    """One tensor of a twin's image: the int8 values in the shape the float tensor had,
    and the exponent that reads them.

    The values are int64 so that a product of two of them cannot wrap, and a contract file
    writes them back as int32."""

    values: np.ndarray
    e: int

    @classmethod
    def of(cls, tensor, e=None):
        """one float tensor under the exponent rule; [e] overrides the tensor's own
        peak"""
        q, e = quantize(np.asarray(tensor, np.float64), e=e)
        return cls(np.asarray(q, np.int64), e)


def join(h, weight, *, values, at):
    """a residual join: [values] times the weight lands on the stream; the exponent of the
    weight folds into the shift with [at], the format of [values].

    It stands here and not in a twin because it is the one arithmetic every residual write
    of every era does, and the shift it takes is the whole of what a residual write can
    get wrong."""
    return h + rescale(values @ weight.values, at=at + weight.e, to=H_Q)


class QuantizedHead(nnx.Module):
    """The two tables of `Head` as the machine holds them -- and ONE exponent over both.

    THE SEAT AND PHASE TABLES SHARE IT and take it from the larger peak: their rows ADD --
    the embedding sums them and the Embed op of a circuit walks them as one tensor -- thus
    a difference of exponents would be a difference of formats inside one sum. Here that
    rule is the shape of the module and cannot be broken by a caller; a FILE can still
    state two, and each era's `load` refuses one that does.

    The four seat tables are ONE tensor, seat 0 first, and a circuit reaches a row of it
    with a shift and an add from the base: row (seat * CLASSES + class).

    Both frozen eras hold one of these, and the twin of each reads the stream at [H_Q]."""

    def __init__(self, *, seats, phase, e):
        # `nnx.data`: these are a machine's own arrays and never a pytree of device
        # tensors -- the engines of the frozen eras are host numpy in int64
        self.seats = nnx.data(np.asarray(seats, np.int64))
        self.phase = nnx.data(np.asarray(phase, np.int64))
        self.e = int(e)

    @property
    def d(self):
        return self.seats.shape[-1]

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
        for seat in range(data.SEATS):
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


def write_tally():
    """a running tally: the activation writes, the writes that rode the clamp, and the
    hottest write BEFORE it.

    A clamp that fires is the finding that says which format is wrong, thus it is counted
    and never assumed away. The peak reads before the clamp, thus it answers the format
    question directly."""
    return {"seen": 0, "clamped": 0, "peak": 0}


def tallied_write(tally, value):
    """every activation write goes through here: the clamp is counted and the peak kept.

    A peak inside the format proves that nothing clamped, thus the clip is skipped — the
    walk writes millions of these and the short circuit is the whole of the difference."""
    high, low = int(value.max()), int(value.min())
    tally["seen"] += value.size
    tally["peak"] = max(tally["peak"], high, -low)
    if high <= INT16_HIGH and low >= INT16_LOW:
        return value.astype(np.int32)
    tally["clamped"] += int(np.count_nonzero(value > INT16_HIGH))
    tally["clamped"] += int(np.count_nonzero(value < INT16_LOW))
    return np.clip(value, INT16_LOW, INT16_HIGH).astype(np.int32)


# the quantized exponential: exp2 of -j/256 in Q15 -- the table `Nn_quantized.Constants`
# builds and `Constants.exp2_bits` hands the circuit, the one table the samplers of every
# era read
EXP2_TABLE = np.array(
    [
        int(round_half_up(float(1 << EXP2_OUT_Q) * 2.0 ** (-j / 256.0)))
        for j in range(256)
    ],
    np.int64,
)


def exp2_of_magnitude(magnitude):
    """2^-m in Q15 over a nonnegative Q12 magnitude -- the rule of the `Exp2` unit.

    The integer part shifts and the top eight bits of the fraction index the table; a
    magnitude of 16 or more is 0. The shift is held under the width of the host word where
    the answer is 0 anyway, because a shift past the width states nothing in either
    language."""
    whole = magnitude >> EXP2_IN_Q
    entry = EXP2_TABLE[(magnitude >> (EXP2_IN_Q - 8)) & 255]
    return np.where(whole >= 16, 0, entry >> np.minimum(whole, 62))


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


def clamp16(value):
    """the rails of int16: a value that passes them saturates and never wraps"""
    return np.clip(np.asarray(value, np.int64), INT16_LOW, INT16_HIGH)


def clamps16(value):
    """true where [clamp16] would clamp — the detector of the clamp counters"""
    value = np.asarray(value, np.int64)
    return (value > INT16_HIGH) | (value < INT16_LOW)


def pick(weights, word):
    """`Nn_quantized.draw`: the class a 24-bit uniform word lands, over the batch.

    The total is the last running total and never a second sum of the same weights. THE
    PICK ALWAYS LANDS: the peak weighs 2^15, thus the total is 2^15 or more, and the word
    falls under 2^24, thus the threshold stands strictly under it. No fallback is written.
    """
    running = np.cumsum(weights, axis=-1)
    threshold = (np.asarray(word, np.int64) * running[..., -1]) >> prng.UNIFORM_BITS
    return (running > threshold[..., None]).argmax(axis=-1)


def engine_states(seeds):
    """the generator of each walk: THE SEED AS IT STANDS, which is the board's SEED cell
    rule, thus seed 0 is the walk that stands still as the circuit stands still on it.

    `prng.states` folds instead, which is the float walk's rule: a seed inside 32 bits
    names itself under both, and 0 is the one seed where the two walks are not one walk.
    """
    return np.array([prng.create(int(seed)) for seed in seeds], dtype=np.uint32)


def exp2_q(value):
    """`Nn_quantized.For_test.exp2_q`: 2^value in Q15 over a Q12 value that is 0 or less.

    The eras exponentiate a nonpositive score and a decay that is a magnitude by
    construction, thus the negation stands here and the shared table takes the
    magnitude."""
    return exp2_of_magnitude(-np.asarray(value, np.int64))


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
    for seat in reversed(range(data.SEATS)):
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
    phase = e.position % data.BAR_STEPS
    if e.position < LEAD:
        classes = np.full((len(e.h), data.SEATS), data.SILENCE, np.int64)
        draws = []
    else:
        e, draws = chain(e)
        classes = np.stack([draw.drawn for draw in reversed(draws)], axis=-1)
    return forward(e, classes, phase), classes, draws


def walk(e, steps, forward):
    """the classes of each step of the walk, and the draws behind them.

    It is the integer twin of the float sampler, and the lead-in counts inside [steps] as
    it does there."""
    played, taken = [], []
    for _ in range(steps):
        e, classes, draws = next_step(e, forward)
        played.append(classes)
        taken.append(draws)
    return np.stack(played, axis=1), taken
