"""The integer rules of the twins: the fixed-point arithmetic every era's twin is built
on, and the contract file that carries the result across the seam.

What stands here is the part of the integer side that is ONE THING ACROSS ALL THREE ERAS:
the int16 rails and the clamp, the exponent rule of a checkpoint, the temper and the
bounds of the sampling policy, the shared exp2 table, the counted write, the integer draw,
and the ARCHIVE the three quantizers write and the elaboration reads. Each era's twin
keeps what is its own -- the parameter structures, the state formats of a recurrence, and
the trunk pass of one step. A rule written here is read by EVERY twin and, through them,
by every circuit.

THE STEP-FRAME HALF IS NOT HERE. `ar_quantized.py` holds what only eras four and five read
-- the stream formats, the norm, the attention over a ring, the chain and the walk -- as
`ar_train.py` holds their recipe. The cut runs ONE WAY: `ar_quantized` imports this module
and nothing here imports it back. A rule that only two of the three eras read is not a
rule of the eras, and a file that says otherwise makes era six read a format it does not
have.

THE OCAML SIDE PARTS THIS FILE IN TWO AND THIS SIDE DOES NOT, deliberately.
`lib/nn/quantized.ml` holds the arithmetic and `lib/nn/contract_file.ml` the archive, and
the reason for that boundary is a TYPE: `Contract_file` is a reader handle -- `type t`,
`open_ : string -> t` -- and a type wants a module of its own. This side has no handle.
The archive is four functions and three key names over a `Weight`, every reader of them
reads the arithmetic beside them, and a second module would only put the exponent rule and
the file that carries it on opposite sides of an import.

`tests/test_quantized.py` is what holds the two sides together: it states the numbers both
must give, and `tests/test_parity.py` states the netlist a build makes of them.

No float model reads this file and nothing here reads a float model: the seam between the
two is the exponent rule, and it runs one way -- `quantize` takes a trained tensor and
gives the machine's.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

import prng

# the rails of int16: a value that passes them saturates and never wraps. Every clamp of
# every twin reads them here, thus none can write a rail of its own and part from its
# circuit in silence.
INT16_BITS = 16
INT16_LOW = -(1 << (INT16_BITS - 1))
INT16_HIGH = (1 << (INT16_BITS - 1)) - 1


def clamp16(value):
    """the rails of int16: a value that passes them saturates and never wraps"""
    return np.clip(np.asarray(value, np.int64), INT16_LOW, INT16_HIGH)


def clamps16(value):
    """true where [clamp16] would clamp — the detector of the clamp counters"""
    value = np.asarray(value, np.int64)
    return (value > INT16_HIGH) | (value < INT16_LOW)


# the Q of log2(e), and the Q the temper takes: one below it. The extra bit is headroom
# for the temperature -- the circuits carry this constant on an 18-bit signed port, thus
# the Q of log2(e) would overflow that port under a temperature of about 0.36.
LOG2E_Q = 15
TEMPER_Q = LOG2E_Q - 1


# the Q the exp2 unit reads its magnitudes at, and the Q of its answer
EXP2_IN_Q = 12
EXP2_OUT_Q = 15


def round_half_up(x):
    """Base's `Float.iround_nearest_exn`: floor(x + 0.5).

    A TIE GOES TOWARD PLUS INFINITY, thus -2.5 is -2 and 2.5 is 3, where Python's `round`
    and `numpy.rint` are half-to-even. Every rounding of every twin goes through here."""
    return np.floor(np.asarray(x, np.float64) + 0.5)


# The policy the ear elected on 2026-08-18: the draw the bitstreams commit to. The OCaml
# `Policy` went with the all-era cut, and this is the one home left -- an era that re-
# elects shadows these two in its own module and says so.
ELECTED_TEMPERATURE = 1.0
ELECTED_MIN_P = 0.05


def apply_scale(q_value, q, value):
    """`Constants.apply`: value times a fixed-point multiplier, toward negative infinity.

    The two halves of the scale travel together because they are one fact: a multiply that
    takes the wrong shift is silently wrong. [q_value] may be a per-head ROW, which is why
    this takes the two numbers and not a `Temper`."""
    return (value * q_value) >> q


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
    def of_file(cls, tensors, metadata, *, key):
        """the temper a contract file carries: the pair from its named tensor and the
        temperature from the metadata, absent under a file an older tool wrote.

        [key] is `TEMPER` at all three callers. It stays an argument all the same: what
        a file names its tensors is the ERA'S layout, stated in the era's own module, and
        this reads whichever name the caller points at."""
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


# ---------------------------------------------------------------------
# the contract file: one writer and one reader
# ---------------------------------------------------------------------

# THE ARCHIVE IS THE SEAM, and two facts of the OCaml reader shape the whole of it. They
# are stated here and nowhere else; each era's module docstring holds its own LAYOUT --
# what stands beside the weights, and in what order -- and points here for the rules under
# it.
#
# - EVERY TENSOR IS INT32, the int8 image included, because `Nx_io.load_safetensors`
#   SKIPS every dtype it does not hold. An int8 tensor would arrive as a hole and the
#   model would refuse for the wrong reason. The values are the int8 image all the same,
#   and a twin holds them in int64 so that a product of two cannot wrap -- `Weight` keeps
#   that half and the writer casts.
# - EVERY SCALAR TRAVELS AS A NAMED TENSOR, because `Nx_io` gives no access to
#   `__metadata__`. The metadata is written all the same, for a reader with a Python tool,
#   and NOTHING IN IT IS REQUIRED: a file an older tool wrote has no temperature and reads
#   back with `nan`.
#
# `Mgen_nn.Contract_file` is the reader of this layout below the seam. A name, a dtype or
# a shape that moves here moves there, and `jax/tests/test_parity.py` fails first.

# The tensor names that are not weights and are not one era's. `EXPONENTS` stands in every
# file; `TEMPER` in all three eras and `MIN_WEIGHT` in the two that hold a min-p floor. An
# era's OWN names -- era five's span and ring, era six's activation Q -- stay in its
# module, because a reader of this file cannot say what they mean.
EXPONENTS = "exponents"
TEMPER = "temper"
MIN_WEIGHT = "min_weight"


def scalar_tensor(value):
    """one number as the archive carries it"""
    return np.array(value, np.int32)


def image_tensors(image):
    """the numbered tensors of an image of `Weight`s, and the `exponents` row beside
    them. The number is the position, thus the order of the list is the order of the ROM
    and a reader walks it with no index of its own."""
    tensors = {
        str(at): np.asarray(weight.values, np.int32) for at, weight in enumerate(image)
    }
    tensors[EXPONENTS] = np.array([weight.e for weight in image], np.int32)
    return tensors


def image_from_tensors(tensors, exponents, *, first, count):
    """[count] `Weight`s read out of the numbered tensors, from [first] -- the inverse of
    `image_tensors`, at int64"""
    return [
        Weight(np.asarray(tensors[str(at)], np.int64), int(exponents[at]))
        for at in range(first, first + count)
    ]


def write_contract(path, tensors, metadata):
    """the archive, written. The metadata values are strings and the caller makes them:
    what a file records of its provenance is the era's own.

    THE BYTES ARE NOT REPRODUCIBLE AND NEVER WERE. `safetensors` serialises
    `__metadata__` out of a Rust hash map, whose order is randomised for each process,
    thus two runs of one unchanged tree write two different files -- measured
    2026-08-29, three runs and three md5s. What IS stable is everything a reader reads:
    the tensor header, the metadata as a dict and the payload. Compare a contract file
    parsed and never by its md5; the netlist md5 of `test_parity.py` is a different
    number and it is stable, because the elaboration reads the tensors."""
    save_file(tensors, str(path), metadata=metadata)


def read_contract(path):
    """the tensors and the metadata of an archive.

    IT OPENS THE FILE TWICE because `safetensors` parts them: `load_file` gives the
    tensors and only `safe_open` reaches the metadata. A file an older tool wrote has
    none, and that reads back as an empty dict and not as a failure."""
    tensors = load_file(str(path))
    with safe_open(str(path), framework="numpy") as opened:
        metadata = opened.metadata() or {}
    return tensors, metadata


@dataclass
class Tally:
    """A running tally of a walk: the activation writes, the writes that rode the clamp,
    and the hottest write BEFORE it.

    A clamp that fires is the finding that says which format is wrong, thus it is counted
    and never assumed away. The peak reads before the clamp, thus it answers the format
    question directly.

    IT MUTATES, and that is the idiom: a walk makes millions of writes and every one of
    them updates the same three numbers. It is the one record of this module that is not a
    `NamedTuple` for exactly that reason."""

    seen: int = 0
    clamped: int = 0
    peak: int = 0

    @property
    def clamped_share(self):
        """the share of the writes that rode the clamp; a walk that wrote nothing rode
        nothing"""
        return 0.0 if self.seen == 0 else self.clamped / self.seen


def tallied_write(tally, value):
    """every activation write goes through here: the clamp is counted and the peak kept.

    A peak inside the format proves that nothing clamped, thus the clip is skipped — the
    walk writes millions of these and the short circuit is the whole of the difference."""
    high, low = int(value.max()), int(value.min())
    tally.seen += value.size
    tally.peak = max(tally.peak, high, -low)
    if high <= INT16_HIGH and low >= INT16_LOW:
        return value.astype(np.int32)
    tally.clamped += int(np.count_nonzero(value > INT16_HIGH))
    tally.clamped += int(np.count_nonzero(value < INT16_LOW))
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
