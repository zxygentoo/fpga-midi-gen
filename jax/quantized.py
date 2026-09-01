"""The integer rules of the twins: the fixed-point arithmetic every era's twin is built
on, and the contract file that carries the result across the seam.

What stands here is ONE THING ACROSS ALL THREE ERAS: the int16 rails, the exponent rule of
a checkpoint, the temper and the bounds of the sampling policy, the shared exp2 table, the
counted write, the integer draw, and the archive the quantizers write and the elaboration
reads. A rule written here is read by every twin and, through them, by every circuit.

THE STEP-FRAME HALF IS NOT HERE: `ar_quantized.py` holds the stream formats, the norm, the
attention over a ring, the chain and the walk. The cut runs ONE WAY -- `ar_quantized`
imports this module and nothing here imports it back -- thus era six cannot read a format
it has no stream for.

`lib/nn/quantized.ml` and `lib/nn/contract_file.ml` are the same rules below the seam,
parted in two because `Contract_file` owns a reader handle and a type wants a module; this
side has none. `tests/test_quantized.py` states the numbers both sides must give, and
`tests/test_parity.py` the netlist a build makes of them.

No float model reads this file and nothing here reads one: `quantize` runs one way.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

import prng

# the rails of int16; every clamp of every twin reads them here, thus none can write a
# rail of its own and part from its circuit in silence
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


# the Q of log2(e), and the Q the temper takes: one below it. The spare bit is headroom
# for the temperature -- on the circuits' 18-bit signed port, LOG2E_Q would overflow under
# a temperature of about 0.36.
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


# the policy the ear elected, and the draw the bitstreams commit to; an era that
# re-elects shadows these two in its own module and says so
ELECTED_TEMPERATURE = 1.0
ELECTED_MIN_P = 0.05


def apply_scale(q_value, q, value):
    """`Constants.apply`: value times a fixed-point multiplier, toward negative infinity.
    The two halves travel together because a multiply that takes the wrong shift is
    silently wrong; [q_value] may be a per-head ROW, thus two numbers and not a
    `Temper`."""
    return (value * q_value) >> q


def largest_exponent(magnitude, *, opening, cap):
    """The largest exponent, from [opening] down, that keeps round(magnitude * 2^e) at
    [cap] or less. [opening] caps the all-zero value, where every exponent fits; the
    predicate falls monotonically in e, thus the first that fits is the largest."""
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
    """The int8 form of one tensor, and the exponent that reads it. The clamp is -127 and
    NOT -128, thus the image is symmetric and a negated weight is a negated byte. [e]
    overrides the tensor's own peak, where tensors whose rows add share one."""
    weights = np.asarray(weights, np.float64)
    if e is None:
        e = max_exponent(float(np.abs(weights).max(initial=0.0)))
    return np.clip(round_half_up(np.ldexp(weights, e)), -127, 127).astype(np.int32), e


def temper_of(temperature):
    """the sampling temper, log2(e) / T, as (q_value, q); `Constants.temper_at_one` is
    this rule at the elected temperature of one"""
    if temperature <= 0.0:
        raise ValueError("the temperature is positive")
    return int(round_half_up(np.ldexp(1.0 / math.log(2.0) / temperature, TEMPER_Q))), (
        TEMPER_Q
    )


def min_weight_of(min_p):
    """The min-p floor as a share of the peak weight, which is 2^`EXP2_OUT_Q` after the
    temper -- thus the floor is a plain share and the circuit compares two integers."""
    if not 0.0 <= min_p < 1.0:
        raise ValueError("min_p is 0 up to 1")
    return int(round_half_up(min_p * float(1 << EXP2_OUT_Q)))


class Temper(NamedTuple):
    """The sampling temper as the bitstream carries it: log2(e) / T at [q].

    The temperature is PROVENANCE and not arithmetic, thus it travels in the metadata
    alone and a file an older tool wrote reads back with no temperature."""

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
        temperature from the metadata. [key] stays an argument because what a file names
        its tensors is the ERA'S layout."""
        q_value, q = (int(value) for value in tensors[key])
        return cls(q_value, q, float(metadata.get("temperature", np.nan)))

    def tensor(self):
        """the pair as the contract file carries it, an int32 tensor like every scalar"""
        return np.array([self.q_value, self.q], np.int32)


# log2(e), `Constants.log2e`: the exp2 form of an exponential
LOG2E = Temper(int(round_half_up(math.ldexp(1.0 / math.log(2.0), LOG2E_Q))), LOG2E_Q, 1.0)


class Weight(NamedTuple):
    """One tensor of a twin's image: the int8 values in the shape the float tensor had,
    and the exponent that reads them. They are held int64 so a product cannot wrap."""

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

# THE ARCHIVE IS THE SEAM, and two facts of the OCaml reader shape it. Each era's module
# docstring holds its own LAYOUT and points here for the rules under it.
#
# - EVERY TENSOR IS INT32, the int8 image included, because `Nx_io.load_safetensors` SKIPS
#   every dtype it does not hold: an int8 tensor would arrive as a hole.
# - EVERY SCALAR TRAVELS AS A NAMED TENSOR, because `Nx_io` cannot reach `__metadata__`.
#   The metadata is written all the same and nothing in it is required.
#
# `Mgen_nn.Contract_file` is the reader below the seam. A name, a dtype or a shape that
# moves here moves there, and `jax/tests/test_parity.py` fails first.

# The tensor names that are not weights and are not one era's. An era's OWN names -- era
# five's span and ring, era six's activation Q -- stay in its module, because a reader of
# this file cannot say what they mean.
EXPONENTS = "exponents"
TEMPER = "temper"
MIN_WEIGHT = "min_weight"


def scalar_tensor(value):
    """one number as the archive carries it"""
    return np.array(value, np.int32)


def image_tensors(image):
    """the numbered tensors of an image of `Weight`s, and the `exponents` row beside them;
    the number is the position, thus the list order is the order of the ROM"""
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
    """The archive, written; the metadata values are strings and the caller makes them.

    THE BYTES ARE NOT REPRODUCIBLE. `safetensors` serialises `__metadata__` out of a Rust
    hash map whose order is randomised per process, thus two runs of one unchanged tree
    write two different files. Compare a contract file PARSED and never by its md5."""
    save_file(tensors, str(path), metadata=metadata)


def read_contract(path):
    """The tensors and the metadata of an archive. IT OPENS THE FILE TWICE because
    `safetensors` parts them, and a file with no metadata reads back as an empty dict."""
    tensors = load_file(str(path))
    with safe_open(str(path), framework="numpy") as opened:
        metadata = opened.metadata() or {}
    return tensors, metadata


@dataclass
class Tally:
    """A running tally of a walk: the activation writes, the writes that rode the clamp,
    and the hottest write BEFORE it -- which answers the format question directly.

    IT MUTATES, and that is why it is the one record here that is not a `NamedTuple`: a
    walk makes millions of writes and each updates the same three numbers."""

    seen: int = 0
    clamped: int = 0
    peak: int = 0

    @property
    def clamped_share(self):
        """the share of the writes that rode the clamp; a walk that wrote nothing rode
        nothing"""
        return 0.0 if self.seen == 0 else self.clamped / self.seen


def tallied_write(tally, value):
    """Every activation write goes through here: the clamp is counted and the peak kept.
    A peak inside the format proves nothing clamped, thus the clip is skipped -- millions
    of writes make that short circuit the whole of the difference."""
    high, low = int(value.max()), int(value.min())
    tally.seen += value.size
    tally.peak = max(tally.peak, high, -low)
    if high <= INT16_HIGH and low >= INT16_LOW:
        return value.astype(np.int32)
    tally.clamped += int(np.count_nonzero(value > INT16_HIGH))
    tally.clamped += int(np.count_nonzero(value < INT16_LOW))
    return np.clip(value, INT16_LOW, INT16_HIGH).astype(np.int32)


# the quantized exponential, exp2 of -j/256 in Q15: the one table the samplers of every
# era read, and what `Constants.exp2_bits` hands the circuit
EXP2_TABLE = np.array(
    [
        int(round_half_up(float(1 << EXP2_OUT_Q) * 2.0 ** (-j / 256.0)))
        for j in range(256)
    ],
    np.int64,
)


def exp2_of_magnitude(magnitude):
    """2^-m in Q15 over a nonnegative Q12 magnitude, the rule of the `Exp2` unit: the
    integer part shifts and the top eight fraction bits index the table, and a
    magnitude of 16 or more is 0. The shift is held under the host word width, where a
    shift past the width states nothing in either language."""
    whole = magnitude >> EXP2_IN_Q
    entry = EXP2_TABLE[(magnitude >> (EXP2_IN_Q - 8)) & 255]
    return np.where(whole >= 16, 0, entry >> np.minimum(whole, 62))


def pick(weights, word):
    """`Nn_quantized.draw`: the class a 24-bit uniform word lands, over the batch.

    THE PICK ALWAYS LANDS: the peak weighs 2^15, thus the total is 2^15 or more, and the
    word falls under 2^24, thus the threshold stands strictly under it."""
    running = np.cumsum(weights, axis=-1)
    threshold = (np.asarray(word, np.int64) * running[..., -1]) >> prng.UNIFORM_BITS
    return (running > threshold[..., None]).argmax(axis=-1)


def engine_states(seeds):
    """The generator of each walk: THE SEED AS IT STANDS, which is the board's SEED cell
    rule, thus seed 0 is the walk that stands still. `prng.states` folds instead, and 0 is
    the one seed where the two walks are not one walk."""
    return np.array([prng.create(int(seed)) for seed in seeds], dtype=np.uint32)


def exp2_q(value):
    """`Nn_quantized.For_test.exp2_q`: 2^value in Q15 over a Q12 value that is 0 or less.
    The eras exponentiate a nonpositive score, thus the negation stands here and the
    shared table takes the magnitude."""
    return exp2_of_magnitude(-np.asarray(value, np.int64))
