"""The common parts of the eras, above the seam and below it.

ABOVE THE SEAM stand the float models: the step-frame transformer and the state-space
model share `Head` -- the four tied voice tables, the bar-phase table and the chained
readout over them -- one position rule and one sampling chain, and each model module keeps
what is its own: the trunk, the layer layout and the checkpoint walk. BELOW IT stand the
integer rules every TWIN is built on: the fixed-point rails, the exponent rule of a
checkpoint, the sampling policy, the shared table and the integer draw, which are the twin
of `lib/nn/quantized.ml`. BETWEEN THEM stands the trainer's rule -- the rate curve and the
optax chain that every era updates under -- and not its loop, which is each era's own.

What is one thing across the eras stands here one time: a rule changed here changes every
model, or every twin, at once -- which is the point.

Matmul precision is pinned to true float32 here, no TF32; every model imports this
module, thus the pin holds everywhere.
"""

import math
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from safetensors.numpy import save_file

import data
import prng

jax.config.update("jax_default_matmul_precision", "float32")

# The slope of head k is 2^-(SLOPE_SPAN (k+1) / heads). Elected 2026-08-18 over spans 4,
# 8, 16, 24 and 64: the means of 4 and 8 are a dead heat, and the VARIANCE is the finding
# -- 5 to 7 times tighter over six seeds, replicated at two step budgets. Every head is
# then local, and seeds stop latching onto whatever distant structure their init favours.
SLOPE_SPAN = 4
# the phase table IS the bar -- one row for each step of it. Two names for one number let
# the corpus phase and the table part, and a phase outside the table gathers a clamped row
# in silence.
PHASE_BUCKETS = data.BAR_STEPS
TABLES = ("seats", "phase")
JAX_ROOT = Path(__file__).resolve().parent


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def attention_bias(heads, length, span=SLOPE_SPAN):
    """ALiBi plus the causal wall, [1, heads, length, length]."""
    pos = jnp.arange(length, dtype=jnp.float32)
    distance = pos[:, None] - pos[None, :]
    slopes = -(2.0 ** (-span * (jnp.arange(heads, dtype=jnp.float32) + 1.0) / heads))
    alibi = slopes[None, :, None, None] * distance[None, None, :, :]
    wall = jnp.triu(jnp.ones((length, length), dtype=jnp.float32), k=1) * -1e9
    return alibi + wall[None, None, :, :]


def dropout_masks(key, rate, shape):
    """the multiplier form of inverted dropout: 0 or 1/keep, one for each element"""
    keep = 1.0 - rate
    return jax.random.bernoulli(key, keep, shape) / keep


# The draw of every matrix of both frozen eras: a normal at this deviation. It is not a
# fan-in rule and it was measured against one -- `Mamba.drawn` records the reading on the
# convolution kernel, where 1/sqrt(K) read worse.
DRAW_SCALE = 0.02


def normal_at(key, shape, scale=DRAW_SCALE):
    """one drawn tensor of a frozen era: a normal, scaled"""
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


# ---------------------------------------------------------------------
# the host-side draw: numpy, float64, and the PRNG of the circuit
# ---------------------------------------------------------------------


def _host_rms_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def temper(raw, temperature, min_p):
    """the tempered weight of each class against the peak, then the min-p floor; the peak
    weighs one, thus min_p is a share of the peak"""
    weights = np.exp((raw - raw.max(axis=1, keepdims=True)) / temperature)
    if min_p > 0.0:
        weights = np.where(weights >= min_p, weights, 0.0)
    return weights


def pick_share(weights, share):
    """The class whose running total passes the draw.

    It takes the uniform and not a draw, thus one function owns both sums and the total is
    the last running total -- never a second sum of the same weights. numpy adds pairwise in
    sum() and left to right in cumsum(), thus two sums of one array differ in the last bits,
    and a draw made against the other sum can land above every running total, where no class
    passes at all.

    Against this total the draw is strictly below it, because the uniform falls under 1 by
    2**-24 at the least. Therefore the walk always ends on a class, and that class always
    holds weight the floor left standing: to reach the last index is to know that no earlier
    total passed, thus the weight there is the difference of two totals across the draw. No
    fallback is necessary, and none is written."""
    running = np.cumsum(weights, axis=1)
    return (running > (share * running[:, -1])[:, None]).argmax(axis=1)


# ---------------------------------------------------------------------
# the head: the tied voice tables, and the chain over them
# ---------------------------------------------------------------------


class Head(nnx.Module):
    """The four tied voice tables and the bar-phase table, and the chained head over them.

    IT IS THE INPUT AND THE READOUT AT ONCE, because the tables are tied: the table that
    reads a voice is the table that writes it. That is why one module carries both
    directions and why neither era holds a table of its own.

    A shared table with a voice tag cannot work here, and the reason is arithmetic and not
    capacity. Every step carries all four seats, thus the sum of the four tags is the same
    vector at every position -- a bias, which carries nothing -- and what remains is
    symmetric in the four codes. A soprano on 72 over a bass on 48 would give the vector of
    a soprano on 48 under a bass on 72, and the voices would be thrown away on the way in.
    Four tables break the symmetry, and no voice tag is then necessary anywhere.

    The two tensors stand FIRST in every checkpoint of both eras, in this order, thus
    [tensors] and [take] are the one statement of that layout."""

    def __init__(self, d, *, rngs):
        self.seats = nnx.Param(normal_at(rngs.params(), (data.SEATS, data.CLASSES, d)))
        self.phase = nnx.Param(normal_at(rngs.params(), (PHASE_BUCKETS, d)))

    @staticmethod
    def shapes(d):
        """the shape of each tensor of [tensors], for a draw that states no shape twice"""
        return [(data.SEATS, data.CLASSES, d), (PHASE_BUCKETS, d)]

    @property
    def d(self):
        """the width of the residual stream: the seat table sizes it"""
        return self.seats.shape[-1]

    def embed(self, classes, phases):
        """The input of one step: the four seat rows and the bar-phase row sum.

        A phase outside the table gathers a clamped row, in silence, which is why the
        table IS the bar and holds one row for each step of it."""
        seats = self.seats[...]
        rows = sum(seats[seat][classes[..., seat]] for seat in range(data.SEATS))
        return rows + self.phase[...][phases]

    def logits(self, h, drawn):
        """The chained head: [batch, length, d] -> [batch, length, SEATS, CLASSES].

        Each seat reads the stream that the seats above it have already written:

            h3 = h                   logits(seat 3) = E[3] . rms(h3)
            h2 = h3 + E[3][c3]       logits(seat 2) = E[2] . rms(h2)
            h1 = h2 + E[2][c2]       logits(seat 1) = E[1] . rms(h1)
            h0 = h1 + E[1][c1]       logits(seat 0) = E[0] . rms(h0)

        [drawn] holds the classes the chain conditions on -- the true frame in training,
        where all four heads then run in one pass with no sampling, and the drawn seats at
        the draw. Only seats 3, 2 and 1 are read.

        The chain runs from the soprano down, which keeps the one decision the ear
        accepted: the top voice is chosen first and conditions on no voice under it, as the
        music is written. Four heads that drew in parallel would make the voices
        conditionally independent, and a chord is a joint choice: measured on era four,
        that costs 0.3157 nats for each step -- 0.456 bits, sixteen times the seed spread.
        The chain removes the cost for no parameters at all -- parallel heads need the same
        four tables -- and three adds of a vector.

        What the chain adds is also what the next step reads: the input embedding of step
        t+1 is a3 + a2 + a1 + a0, and the chain assembles it one voice at a time."""
        seats = self.seats[...]
        stream = h
        logits = [None] * data.SEATS
        for seat in reversed(range(data.SEATS)):
            logits[seat] = rms_norm(stream) @ seats[seat].T
            if seat:
                stream = stream + seats[seat][drawn[..., seat]]
        return jnp.stack(logits, axis=-2)

    def nll(self, h, labels):
        """the negative log likelihood of every voice of every step, over a residual stream
        the era's own [hidden] computed; the caller reduces"""
        logp = jax.nn.log_softmax(self.logits(h, labels), axis=-1)
        return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]

    def draw_frame(self, h, state, temperature, min_p):
        """One step of the chain ON THE HOST, in numpy float64 and on the PRNG of the
        circuit: the soprano first, and each seat under it reading the stream the seats
        above have written.

        The chain is the reason a frame is a joint choice and not four independent ones.
        Seat 0 is the bass and seat 3 the soprano, thus the loop runs down.

        Every walk of the batch draws. A step is one frame and never a sentence of its own
        length, thus no walk of the batch finishes before another and none has to sit out a
        draw while the rest go on."""
        seats = np.asarray(self.seats[...])
        stream = h
        frame = np.zeros((len(h), data.SEATS), dtype=np.int32)
        for seat in reversed(range(data.SEATS)):
            raw = (_host_rms_norm(stream) @ seats[seat].T).astype(np.float64)
            weights = temper(raw, temperature, min_p)
            state, uniform = prng.uniform(state, True)
            frame[:, seat] = pick_share(weights, uniform)
            if seat:
                stream = stream + seats[seat][frame[:, seat]]
        return state, frame

    def tensors(self):
        """the two tables in the order every checkpoint of both eras carries them"""
        return [self.seats[...], self.phase[...]]

    def take(self, tensors):
        """the reverse of [tensors]. The two stand together so that the layout cannot drift
        apart."""
        seats, phase = tensors
        self.seats[...] = jnp.asarray(seats)
        self.phase[...] = jnp.asarray(phase)


# ---------------------------------------------------------------------
# the integer rules of the twins
# ---------------------------------------------------------------------

# WHAT STANDS HERE IS THE PART OF THE INTEGER ARITHMETIC THAT IS ONE THING ACROSS THE
# ERAS: the fixed-point rails, the exponent rule of a checkpoint, the sampling policy, the
# shared table and the integer draw. Each era's twin keeps what is its own -- the
# parameter structures, the state formats of a recurrence, and the engines themselves. A
# rule written here is read by every twin and, through them, by every circuit.
#
# `lib/nn/quantized.ml` is the same module in OCaml and the elaborations read it. The two
# are TWO STATEMENTS OF ONE RULE and nothing in the types welds them; what holds them
# together is `tests/test_quantized.py`, which states the numbers both must give.

# the rails of int16: a value that passes them saturates and never wraps. Every clamp of
# every twin reads them here, thus none can write a rail of its own and part from its
# circuit in silence.
INT16_BITS = 16
INT16_LOW = -(1 << (INT16_BITS - 1))
INT16_HIGH = (1 << (INT16_BITS - 1)) - 1

# the Q of log2(e), and the Q the temper takes: one below it. The extra bit is headroom for
# the temperature -- the circuits carry this constant on an 18-bit signed port, thus the Q
# of log2(e) would overflow that port under a temperature of about 0.36.
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


def rescale(value, *, at, to):
    """value * 2^-at as value * 2^-to; the arithmetic shift floors, as the circuits' does"""
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
    loop is written all the same, because a silently wrong root is a silently wrong norm."""
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
    """`Constants.score_shift`: what carries a score walk's sum from Q(2 [row_q]) to Q[Y_Q]
    and applies the 1/sqrt([head_d]) in the same shift, thus [head_d] is a power of four"""
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
    """`Nn_quantized.max_exponent`: the exponent of one int8 tensor -- from 14 down, the
    largest that keeps round(peak * 2^e) inside the byte."""
    return largest_exponent(peak, opening=14, cap=127)


def quantize(weights, e=None):
    """`Nn_quantized.quantize`: the int8 form of one tensor, and the exponent that reads it.

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
    """`Nn_quantized.policy`: the sampling temper, log2(e) / T, as (q_value, q)."""
    if temperature <= 0.0:
        raise ValueError("the temperature is positive")
    return int(round_half_up(np.ldexp(1.0 / math.log(2.0) / temperature, TEMPER_Q))), (
        TEMPER_Q
    )


def min_weight_of(min_p):
    """`Nn_quantized.policy`: the min-p floor as a share of the peak weight.

    The peak weighs 2^`EXP2_OUT_Q` after the temper, thus the floor is a plain share of it
    and the circuit compares two integers."""
    if not 0.0 <= min_p < 1.0:
        raise ValueError("min_p is 0 up to 1")
    return int(round_half_up(min_p * float(1 << EXP2_OUT_Q)))


class Temper(NamedTuple):
    """The sampling temper as the bitstream carries it: log2(e) / T at [q].

    The temperature is PROVENANCE and not arithmetic -- the temper is already folded -- thus
    it travels in the metadata of a contract file alone, and a file written by an older
    tool can read back with no temperature at all."""

    q_value: int
    q: int
    temperature: float

    @classmethod
    def of(cls, temperature):
        q_value, q = temper_of(temperature)
        return cls(q_value, q, temperature)


# log2(e): the exp2 form of an exponential, `Constants.log2e`. The temper is this
# constant divided by the temperature and it takes one Q less, which is `temper_of`.
LOG2E = Temper(int(round_half_up(math.ldexp(1.0 / math.log(2.0), LOG2E_Q))), LOG2E_Q, 1.0)


class Weight(NamedTuple):
    """One tensor of a twin's image: the int8 values in the shape the float tensor had, and
    the exponent that reads them.

    The values are int64 so that a product of two of them cannot wrap, and a contract file
    writes them back as int32."""

    values: np.ndarray
    e: int

    @classmethod
    def of(cls, tensor, e=None):
        """one float tensor under the exponent rule; [e] overrides the tensor's own peak"""
        q, e = quantize(np.asarray(tensor, np.float64), e=e)
        return cls(np.asarray(q, np.int64), e)


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
        """what the chain adds after a seat draws: the drawn row, in the stream's format"""
        return stream + rescale(self.seats[seat, drawn], at=self.e, to=H_Q)

    def tensors(self):
        """the two tables, in the order of the checkpoint and of the ROM"""
        return [Weight(self.seats, self.e), Weight(self.phase, self.e)]


def write_tally():
    """a running tally: the activation writes, the writes that rode the clamp, and the
    hottest write BEFORE it.

    A clamp that fires is the finding that says which format is wrong, thus it is counted and
    never assumed away. The peak reads before the clamp, thus it answers the format question
    directly."""
    return {"seen": 0, "clamped": 0, "peak": 0}


def tallied_write(tally, value):
    """every activation write goes through here: the clamp is counted and the peak kept.

    A peak inside the format proves that nothing clamped, thus the clip is skipped — the walk
    writes millions of these and the short circuit is the whole of the difference."""
    high, low = int(value.max()), int(value.min())
    tally["seen"] += value.size
    tally["peak"] = max(tally["peak"], high, -low)
    if high <= INT16_HIGH and low >= INT16_LOW:
        return value.astype(np.int32)
    tally["clamped"] += int(np.count_nonzero(value > INT16_HIGH))
    tally["clamped"] += int(np.count_nonzero(value < INT16_LOW))
    return np.clip(value, INT16_LOW, INT16_HIGH).astype(np.int32)


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
# reading by up to 2^-10 of full scale; the centre halves the worst error and costs nothing
# at elaboration. The centres are symmetric about zero, thus the two halves of the table
# sum to 2^15 and sigmoid(-v) = 1 - sigmoid(v) survives the quantization.
SIGMOID_TABLE = np.array(
    [
        int(round_half_up(32768.0 / (1.0 + math.exp(-((j - 128) + 0.5) / 16.0))))
        for j in range(256)
    ],
    np.int64,
)

# The correction term of the softplus, ln(1 + exp(-|v|)), in Q12 over a Q12 magnitude:
# softplus(v) = relu(v) + this. The ramp is exact and carries the whole of a large input,
# thus the table only has to hold a quantity that falls to nothing: at |v| = 8, the largest
# magnitude an int16 Q12 value takes, it is one unit of Q12. 256 buckets of 128 units cover
# the range, and the entry is again the centre of its bucket.
SOFTPLUS_TABLE = np.array(
    [
        int(round_half_up(4096.0 * math.log(1.0 + math.exp(-(j + 0.5) / 32.0))))
        for j in range(256)
    ],
    np.int64,
)


def sigmoid_q(value):
    """`Nn_quantized.sigmoid_q`: the sigmoid of a Q12 value in Q15 — the rule of the
    [Sigmoid] unit. The index is the top eight bits with the sign flipped, which is no
    arithmetic at all in a circuit."""
    return SIGMOID_TABLE[((np.asarray(value, np.int64) >> 8) + 128) & 255]


def silu(value):
    """`Nn_quantized.silu`: the value times its sigmoid, shifted back to Q12 and clamped"""
    value = np.asarray(value, np.int64)
    return clamp16((value * sigmoid_q(value)) >> 15)


def softplus(value):
    """`Nn_quantized.softplus`: the ramp plus the correction the table holds.

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

    The total is the last running total and never a second sum of the same weights. THE PICK
    ALWAYS LANDS: the peak weighs 2^15, thus the total is 2^15 or more, and the word falls
    under 2^24, thus the threshold stands strictly under it. No fallback is written."""
    running = np.cumsum(weights, axis=-1)
    threshold = (np.asarray(word, np.int64) * running[..., -1]) >> prng.UNIFORM_BITS
    return (running > threshold[..., None]).argmax(axis=-1)


def engine_states(seeds):
    """the generator of each walk: THE SEED AS IT STANDS, which is the board's SEED cell rule,
    thus seed 0 is the walk that stands still as the circuit stands still on it.

    `prng.states` folds instead, which is the float walk's rule: a seed inside 32 bits names
    itself under both, and 0 is the one seed where the two walks are not one walk."""
    return np.array([prng.create(int(seed)) for seed in seeds], dtype=np.uint32)


# ---------------------------------------------------------------------
# the trainer: the rate curve, the update rule, and the seam of a checkpoint
# ---------------------------------------------------------------------

# THE RULE STANDS HERE AND THE LOOP DOES NOT. Every era runs the same optax chain under
# the same curve, thus a rate read one step late would be read one step late by all three
# at once; but the shape of a step is the era's own -- the sheet folds a batch-norm
# population where the frozen eras draw a dropout mask -- and each `train.py` keeps its
# loop.


def learning_rates(peak, warmup, total):
    """The rate at every step of the run: linear from 0 to [peak] over [warmup] steps,
    then cosine from [peak] to 0 over the rest. A warmup of zero is a constant.

    THE SCHEDULE IS READ ONE STEP LATE OR NOT AT ALL. Optax hands a schedule its own update
    count, which is 0 at the first update where the loop's step is 1, thus a curve read at
    the raw count applies a rate of 0 to the first update and every later rate one step
    behind. The `+ 1` is that correction and `tests/test_train.py` holds it.

    THE TWO ENDS ARE THIS PROJECT'S RULES AND NOT OPTAX'S. A warmup of zero is a constant
    peak, where `warmup_cosine_decay_schedule` would be a bare cosine decay; and a run
    SHORTER THAN ITS OWN WARMUP -- which every short probe is -- never leaves the ramp,
    where optax refuses to build a cosine of a negative length at all."""
    if warmup == 0:
        curve = optax.constant_schedule(peak)
    elif total <= warmup:
        curve = optax.linear_schedule(0.0, peak, warmup)
    else:
        curve = optax.warmup_cosine_decay_schedule(0.0, peak, warmup, total, 0.0)
    return lambda count: curve(count + 1)


def update_rule(*, peak, warmup, total, clip, weight_decay):
    """The update of one step: the global-norm clip, then Adam with a decoupled weight
    decay under the schedule.

    A clip of zero or less is NO CLIP. It is not a clip at zero, which would zero every
    gradient of the run. A weight decay of zero makes AdamW Adam, by arithmetic and not by
    a second code path -- which is what era six's paper asks for."""
    adam = optax.adamw(
        learning_rate=learning_rates(peak, warmup, total),
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    if clip <= 0.0:
        return adam
    return optax.chain(optax.clip_by_global_norm(clip), adam)


def save_checkpoint(path, tensors, span=None):
    """The naming rule of the seam: the tensors named "0" upward, in construction order,
    then the ALiBi span last and alone where the model carries one -- an older file that
    does not still reads, because a reader takes whole layer groups and then one scalar
    if one is there. The era's trainer builds the flat list, because the layer layouts
    are its own."""
    if span is not None:
        tensors = list(tensors) + [np.asarray([span], dtype=np.float32)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({str(i): np.asarray(t) for i, t in enumerate(tensors)}, path)
