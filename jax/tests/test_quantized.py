"""The integer rules every twin stands on.

`quantized.py` and `ar_quantized.py` on this side, `lib/nn/quantized.ml` undivided on the
other: TWO STATEMENTS OF ONE RULE, and nothing in the types welds them. What stands here
is the arithmetic, in numbers, that both must give.

THESE ARE THE SMALLEST GATES OF A TWIN -- the rounding, the exponent, the table entry that
`test_rtl_diffusion.py`, `test_drift.py` and `test_parity.py` would each break on FIRST --
so that a failure names the rule and not the walk.
"""

import numpy as np
import pytest
from safetensors import safe_open

import ar_quantized
import quantized
from tests import models
from transformer import quantized as step_twin


@pytest.mark.parametrize(
    "peak,exponent",
    [(0.0, 14), (0.02, 12), (0.08, 10), (127.0, 0), (127.49, 0), (127.5, -1), (1e9, -23)],
)
def test_the_exponent_rule_holds_at_its_boundaries(peak, exponent):
    """The largest e that keeps the peak at 127 or less. 14 caps the all-zero tensor,
    where every exponent fits, and 127.5 is the rounding boundary: it rounds to 128 and
    the exponent has to step down. A tie rounds UP and never away from zero, which is
    Base's `Float.iround_nearest_exn` and not Python's `round`."""
    assert quantized.max_exponent(peak) == exponent


def test_the_byte_is_symmetric_and_ties_round_up():
    """The byte is two's complement and the negative end is not used: the clamp is -127
    and not -128, thus the image is symmetric and a negated weight is a negated byte."""
    q, e = quantized.quantize(np.array([0.02, -0.01, 0.0]))
    assert (list(q), e) == ([82, -41, 0], 12)
    assert quantized.round_half_up([-5.5, -2.5, 2.5, 5.5]).tolist() == [-5, -2, 3, 6]


def test_a_stated_exponent_overrides_the_tensors_own_peak():
    """Tensors whose rows ADD share one exponent -- era four's seat and phase tables are
    one sum -- thus the caller takes the max over their peaks and states it here. The
    tensor's own peak would give 12; a stated 10 reads the same weights two bits coarser
    and the clamp still holds the byte."""
    assert list(quantized.quantize(np.array([0.02, -0.01, 0.0]), e=10)[0]) == [20, -10, 0]
    assert list(quantized.quantize(np.array([1.0, -1.0]), e=14)[0]) == [127, -127]


def test_the_counted_write_keeps_the_peak_and_counts_both_rails():
    """The two branches of every activation write. A write INSIDE the format keeps the
    peak, counts no clamp and skips the clip -- a walk makes millions of writes, and that
    short circuit is the whole of the difference. A write outside counts each rail it
    passed and clips to it.

    THE PEAK IS A MAGNITUDE AND THE FORMAT IS NOT SYMMETRIC: a write that lands on the low
    rail reads a peak of 32768, one above the high rail, with nothing clamped."""
    tally = quantized.Tally()
    assert tally.clamped_share == 0.0  # a walk that wrote nothing rode nothing
    inside = quantized.tallied_write(tally, np.array([[100, -32768, 32767]], np.int64))
    assert list(inside[0]) == [100, -32768, 32767] and inside.dtype == np.int32
    assert (tally.seen, tally.clamped, tally.peak) == (3, 0, 32768)
    outside = quantized.tallied_write(tally, np.array([[40000, -40000, 5]], np.int64))
    assert list(outside[0]) == [32767, -32768, 5]
    assert (tally.seen, tally.clamped, tally.peak) == (6, 2, 40000)
    assert tally.clamped_share == pytest.approx(2 / 6)


def test_the_exp2_table_is_the_shared_table():
    """exp2 of -j/256 in Q15, the one table the samplers of every era read: entry 0 is the
    peak 2^15, a full fractional step halves, and the last entry sits one table step above
    one half. It agrees with `Nn_quantized.Constants.exp2_bits` entry for entry."""
    table = quantized.EXP2_TABLE
    assert (table[0], table[128], table[255]) == (32768, 23170, 16428)
    assert quantized.exp2_of_magnitude(np.int64(4096)) == 16384
    # a magnitude of 16 or more is zero, and the shift may not run past the host word
    assert quantized.exp2_of_magnitude(np.int64(1 << 20)) == 0


def test_the_sigmoid_table_is_the_shared_table():
    """The sigmoid of a Q12 value in Q15, 256 buckets AT THEIR CENTRES: the left edge
    would bias every reading by up to 2^-10 of full scale. The centres are symmetric
    about zero, which is the property the table must have -- the two halves sum to
    2^15, thus sigmoid(-v) = 1 - sigmoid(v) survives the quantization."""
    table = ar_quantized.SIGMOID_TABLE
    assert (table[0], table[128], table[255]) == (11, 16640, 32757)
    for j in range(128):
        assert table[j] + table[255 - j] == 32768
    # the index is the top eight bits with the sign flipped, which is no arithmetic at all
    assert ar_quantized.sigmoid_q(np.int64(0)) == 16640
    assert ar_quantized.silu(np.int64(0)) == 0


def test_the_softplus_table_is_the_correction_alone():
    """softplus(v) = relu(v) + ln(1 + exp(-|v|)), and the table holds the correction. The
    ramp is exact and carries the whole of a large input, thus the table only has to
    hold a quantity that falls to nothing: at |v| = 8, the largest magnitude an int16
    Q12 value takes, it is one unit of Q12."""
    table = ar_quantized.SOFTPLUS_TABLE
    assert (table[0], table[128], table[255]) == (2807, 73, 1)
    # ln 2 in Q12 is the correction at zero, and the ramp adds nothing there
    assert ar_quantized.softplus(np.int64(0)) == 2807
    # the index is the magnitude shifted by seven, thus a value of 7.0 in Q12 reads bucket
    # 224 and the correction there is 4 units of Q12 -- the ramp carries the rest
    assert ar_quantized.softplus(np.int64(4096 * 7)) == (4096 * 7) + table[224]
    # the sum rides an int16, thus the result clamps
    assert ar_quantized.softplus(np.int64(32767)) == quantized.INT16_HIGH
    # the clamp of the index catches the one value whose magnitude does not fit the table
    assert ar_quantized.softplus(np.int64(-32768)) == table[255]


def test_the_division_goes_toward_zero_and_not_toward_the_floor():
    """OCaml's `/` on integers, which every division of every circuit is. numpy's `//`
    floors, and THE TWO PART ONLY WHERE THE QUOTIENT IS NEGATIVE. Where both signs are
    negative the quotient is positive and the floor agrees, thus a twin written with `//`
    passes on the positive half of a stream and drifts on the negative half alone."""
    assert int(ar_quantized.truncated(7, 2)) == 3 == 7 // 2
    assert int(ar_quantized.truncated(-7, 2)) == -3 and -7 // 2 == -4
    assert int(ar_quantized.truncated(7, -2)) == -3 and 7 // -2 == -4
    # both signs negative: the quotient is positive, and there the floor agrees
    assert int(ar_quantized.truncated(-7, -2)) == 3 == -7 // -2
    # an exact division leaves no remainder to send either way
    assert int(ar_quantized.truncated(-8, 2)) == -4 == -8 // 2
    assert list(ar_quantized.truncated([-7, 7], [2, 2])) == [-3, 3]


def test_the_root_is_the_floor_at_every_boundary():
    """Floor of the square root, the one answer the [Isqrt] unit gives. The boundaries are
    where a root moves: v*v - 1 is v - 1, v*v is v, and v*v + 1 is still v.

    THE FLOAT ROOT IS NOT THE ANSWER, which is why the settling loop is written at all: at
    (2^26 + 1)^2 - 1 it reads one unit high, and a silently wrong root is a silently wrong
    norm."""
    assert list(ar_quantized.isqrt([143, 144, 145])) == [11, 12, 12]
    assert int(ar_quantized.isqrt(0)) == 0
    big = (1 << 26) + 1
    assert int(np.sqrt(np.float64(big * big - 1))) == big  # the float root, one unit high
    assert list(ar_quantized.isqrt([big * big - 1, big * big, big * big + 1])) == [
        big - 1,
        big,
        big,
    ]


def test_the_norm_divides_by_the_root_of_the_mean_square():
    """rms_norm of a Q16 stream in Q12, at values known by hand: four ones have an rms of
    one and normalise to 1.0, which is 4096, and a vector carrying one of them alone has
    an rms of a half and normalises to 2.0.

    THE EPSILON IS WHAT HOLDS THE ALL-ZERO STREAM. Without it the root would be zero and
    the division would raise; with it the answer is the zero vector, as the circuit
    gives."""
    assert ar_quantized.EPS_Q == 17
    every = np.array([[1 << 16] * 4])
    assert list(ar_quantized.rms_norm_q(every, at=16, width=4)[0]) == [4096] * 4
    alone = np.array([[1 << 16, 0, 0, 0]])
    assert list(ar_quantized.rms_norm_q(alone, at=16, width=4)[0]) == [8192, 0, 0, 0]
    silent = np.zeros((1, 4), np.int64)
    assert list(ar_quantized.rms_norm_q(silent, at=16, width=4)[0]) == [0] * 4


def test_the_norm_of_a_negated_stream_is_the_negated_norm():
    """The property the toward-zero division buys, and the one a floor would break: a
    stream and its negation square the same, thus every element divides the same root and
    only the sign moves. Under `//` the negative half would fall one unit further and this
    symmetry would go -- which is what a norm written with the wrong division looks like
    from the outside."""
    odd = np.array([[3 << 16, -(1 << 16), 0, 0]])  # a division that leaves a remainder
    said = ar_quantized.rms_norm_q(odd, at=16, width=4)
    assert list(said[0]) == [7772, -2590, 0, 0]
    assert list(ar_quantized.rms_norm_q(-odd, at=16, width=4)[0]) == list(-said[0])


def test_a_stored_ring_row_keeps_its_top_byte():
    """What a KV ring keeps of a Q12 row: the circuit stores eight bits and restores eight
    zero low bits at the read, thus the granularity is 2^-4 and the format stays Q12.

    THE SHIFT IS ARITHMETIC AND IT FLOORS: a negative row coarsens DOWNWARD and never
    toward zero, thus -1 comes back as -256 and not as 0."""
    row = np.array([0x1234, 255, 0, -1, -256, -257])
    assert list(ar_quantized.coarse_to_ring(row)) == [0x1200, 0, 0, -256, -256, -512]


def test_the_temper_is_log2e_over_the_temperature():
    """log2(e) / T at a Q one below log2(e)'s own: the extra bit is headroom for the
    temperature, because the circuits carry this constant on an 18-bit signed port."""
    assert quantized.Temper.from_float(1.0) == (23637, 14)
    assert quantized.Temper.from_float(0.5) == (47274, 14)
    with pytest.raises(ValueError):
        quantized.Temper.from_float(0.0)


def test_a_saved_file_carries_no_map_beside_its_tensors(tmp_path):
    """AND IS THEREFORE REPRODUCIBLE BYTE FOR BYTE. The `__metadata__` map that once
    travelled was serialised out of a Rust hash map whose order is randomised PER PROCESS,
    thus two builds of one unchanged twin gave two md5s and no diff could say what moved.

    THE EMPTY HEADER IS WHAT THIS PINS, and the two saves below are its corollary: one
    process orders that map one way, thus two saves in one process would have agreed even
    under the old writer. What makes the file a function of the tensors alone is that no
    such map is written at all."""
    twin = models.transformer_twin()

    def saved(name):
        path = tmp_path / name
        step_twin.save(path, twin)
        return path

    first, second = saved("first.int8"), saved("second.int8")
    assert first.read_bytes() == second.read_bytes()
    with safe_open(str(first), framework="numpy") as opened:
        assert opened.metadata() is None


def test_the_min_p_floor_is_a_share_of_the_peak_weight():
    """The peak weighs 2^15 after the temper, thus the floor is a plain share of it and
    the circuit compares two integers. The elected 0.05 of the frozen eras is 1638."""
    assert quantized.min_weight(0.05) == 1638
    assert quantized.min_weight(0.0) == 0
    for outside in (-0.1, 1.0):
        with pytest.raises(ValueError):
            quantized.min_weight(outside)


def test_a_q12_number_clamps_to_the_port_that_carries_it():
    """Era five's per-head numbers: the `dt_bias` joins an int16 sum and the `d_skip`
    rides an 18-bit operand port, thus the bound is a fact of the CIRCUIT and the caller
    states it. A value the port cannot hold saturates and never wraps."""
    assert list(ar_quantized.fixed_q12([1.0, -0.5, 0.0], 32767)) == [4096, -2048, 0]
    assert list(ar_quantized.fixed_q12([100.0, -100.0], 32767)) == [32767, -32767]
    assert list(ar_quantized.fixed_q12([100.0, -100.0], 131071)) == [131071, -131071]


def test_the_pick_lands_on_a_class_that_holds_weight():
    """`Nn_quantized.draw`, at its ends: a word of 0 lands on the first class that holds
    weight, and the top of the 24-bit grid lands on the LAST class that holds weight and
    never past it. The threshold is the word times the total, shifted down by 24, thus it
    stands strictly under the total and no fallback is necessary."""
    weights = np.array([[0, 32768, 1, 0], [1, 0, 0, 32768]])
    top = np.array([(1 << 24) - 1, (1 << 24) - 1])
    assert list(quantized.pick(weights, np.array([0, 0]))) == [1, 0]
    assert list(quantized.pick(weights, top)) == [2, 3]
    assert list(quantized.pick(weights, np.array([1 << 23, 1 << 23]))) == [1, 3]


def test_the_walk_takes_the_seed_as_it_stands():
    """The board's SEED cell rule: the twin opens its generator on the seed itself, where
    the float walk folds it. A seed inside 32 bits names itself under both, and 0 is the
    one seed where the two walks are not one walk -- there the twin stands still, as the
    circuit stands still on it."""
    import prng

    assert list(quantized.engine_states([7, 11])) == [7, 11]
    assert list(prng.states([7, 11])) == [7, 11]
    # the one seed the two rules part on: the fold has no zero state and takes the top one
    assert list(quantized.engine_states([0])) == [0]
    assert list(prng.states([0])) == [0xFFFFFFFF]
