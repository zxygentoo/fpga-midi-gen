"""The integer rules every twin stands on.

`quantized.py` holds the part of the integer arithmetic that all three eras read and
`ar_quantized.py` the part the two step-frame eras read; `lib/nn/quantized.ml` holds both
in OCaml, undivided, where the elaborations take what each needs. The two sides are TWO
STATEMENTS OF ONE RULE and nothing in the types welds them: what stands here is the
arithmetic, stated in numbers, that both must give.

THESE ARE THE SMALLEST GATES OF A TWIN. `test_rtl_diffusion.py` holds a twin against its
circuit,
write for write; `test_drift.py` holds it against the float model it quantizes; and
`test_parity.py`'s G1 holds a quantizer through the netlist a build carries. What stands
here is the arithmetic each of those would break on FIRST -- a rounding, an exponent, a
table entry -- so that a failure names the rule and not the walk.
"""

import numpy as np
import pytest

import ar_quantized
import quantized


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


def test_the_exp2_table_is_the_shared_table():
    """exp2 of -j/256 in Q15, the one table the samplers of every era read. Entry 0 is the
    peak 2^15, a full fractional step halves, and the last entry sits one table step above
    one half. The whole table was compared entry for entry against
    `Nn_quantized.Constants.exp2_bits` when it was written, and no entry differed: the two
    libm implementations agree here."""
    table = quantized.EXP2_TABLE
    assert (table[0], table[128], table[255]) == (32768, 23170, 16428)
    assert quantized.exp2_of_magnitude(np.int64(4096)) == 16384
    # a magnitude of 16 or more is zero, and the shift may not run past the host word
    assert quantized.exp2_of_magnitude(np.int64(1 << 20)) == 0


def test_the_sigmoid_table_is_the_shared_table():
    """The sigmoid of a Q12 value in Q15, 256 buckets AT THEIR CENTRES. The bucket is 1/16
    wide and the slope peaks at 1/4, thus the left edge would bias every reading by up to
    2^-10 of full scale. The centres are symmetric about zero, and that is the property
    the table must have: the two halves sum to 2^15, thus sigmoid(-v) = 1 - sigmoid(v)
    survives the quantization. `Nn_quantized.Constants.sigmoid_table` states the same
    rule, and era five's RTL gate holds the two together through the circuit."""
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


def test_the_temper_is_log2e_over_the_temperature():
    """log2(e) / T at a Q one below log2(e)'s own: the extra bit is headroom for the
    temperature, because the circuits carry this constant on an 18-bit signed port."""
    assert quantized.temper_of(1.0) == (23637, 14)
    assert quantized.temper_of(0.5) == (47274, 14)
    with pytest.raises(ValueError):
        quantized.temper_of(0.0)


def test_the_min_p_floor_is_a_share_of_the_peak_weight():
    """The peak weighs 2^15 after the temper, thus the floor is a plain share of it and
    the circuit compares two integers. The elected 0.05 of the frozen eras is 1638."""
    assert quantized.min_weight_of(0.05) == 1638
    assert quantized.min_weight_of(0.0) == 0
    for outside in (-0.1, 1.0):
        with pytest.raises(ValueError):
            quantized.min_weight_of(outside)


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
