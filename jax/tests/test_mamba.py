"""The gate that the two forms of the recurrence are one recurrence.

jax/mamba/model.py holds the step form, which the sampler runs, and the window form, which
the trainer runs because a scan of the step form costs 203 ms for each step against era
four's 61. The window form is an UNROLLING and not an approximation, thus the two must
agree to float noise. The gate reads the whole forward and not the block alone.

THE CONTRACT FILE stands at the foot of this module: what crosses the seam to the
elaboration, and the rules the circuit cannot hold.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import ar_model
import ar_quantized
import corpus
import quantized as q
from mamba import model, train
from mamba.quantized import infer as qinfer
from mamba.quantized import model as qmodel
from tests.models import drawn_mamba, plan_of

# Six layers of float32 over a window of 64 steps, reduced in two different orders. A real
# disagreement -- a tap read backward, a decay off by one step, the readout taking the
# state before the update -- moves the stream far past this.
TOLERANCE = 2e-4


def drawn_window(rows, length):
    state = np.random.default_rng(11)
    shape = (rows, length, corpus.SEATS)
    classes = state.integers(0, corpus.CLASSES, shape).astype(np.int32)
    bar = np.arange(length) % ar_model.PHASE_BUCKETS
    phases = np.tile(bar, (rows, 1)).astype(np.int32)
    return jnp.asarray(classes), jnp.asarray(phases)


def assert_one_stream(stepped, windowed):
    """The two forms of the recurrence against each other, at the RELATIVE tolerance: the
    stream grows with the plan, thus a fixed epsilon would tighten on a wide model and
    slacken on a narrow one."""
    gap = float(jnp.max(jnp.abs(stepped - windowed)))
    scale = float(jnp.max(jnp.abs(windowed)))
    assert gap / scale < TOLERANCE, (
        f"the two forms part by {gap:.3e} on a scale of {scale:.3e}"
    )


def walk_by_steps(held, classes, phases):
    """the window form's answer, taken one step at a time through [forward_step]"""
    carry = held.initial_carry(classes.shape[0])
    rows = []
    for step in range(classes.shape[1]):
        carry, h = held.forward_step(carry, classes[:, step], phases[:, step])
        rows.append(h)
    return jnp.stack(rows, axis=1)


@pytest.mark.parametrize("taps", [4, 16])
def test_the_step_form_and_the_window_form_give_one_stream(taps):
    """K is a field of the checkpoint and not a constant, thus the gate runs at two
    widths: the tap ring of the step form and the pad of the window form both read it, and
    a width written into one of them alone lands here."""
    held = drawn_mamba(taps=taps)
    classes, phases = drawn_window(rows=2, length=64)
    stepped = walk_by_steps(held, classes, phases)
    windowed = held.hidden(classes, phases)
    assert stepped.shape == windowed.shape
    assert_one_stream(stepped, windowed)


def test_the_window_opens_on_a_zero_state():
    """A window is not a slice of a longer walk: it starts where the boot of the sampler
    starts. The first step of the window form must therefore equal one step of the step
    form from the origin, exactly, with no history behind either."""
    held = drawn_mamba()
    classes, phases = drawn_window(rows=2, length=8)
    carry = held.initial_carry(classes.shape[0])
    _, first = held.forward_step(carry, classes[:, 0], phases[:, 0])
    windowed = held.hidden(classes, phases)[:, 0]
    assert float(jnp.max(jnp.abs(first - windowed))) < 1e-5


@pytest.mark.parametrize("taps", [4, 16])
def test_the_convolution_reads_the_steps_behind_it(taps):
    """Tap k reads the step k back. The window form pads at the head, the step form holds
    a ring of K-1, and the gate is an impulse: a window that is zero but for step 0 must
    read the taps out in order down the channels. Nothing but [conv] states K."""
    block = drawn_mamba(taps=taps).layers[0]
    channels = block.conv.shape[0]
    conv = jnp.asarray(
        np.arange(channels * taps, dtype=np.float32).reshape(channels, taps)
    )
    block.conv[...] = conv
    u = jnp.zeros((1, taps + 3, channels)).at[0, 0].set(1.0)
    windowed = block.convolve_window(u)
    # step t reads the impulse through tap t, thus the row at t is the tap column t
    for t in range(taps):
        assert np.allclose(np.asarray(windowed[0, t]), np.asarray(conv[:, t]))
    assert np.allclose(np.asarray(windowed[0, taps:]), 0.0)


def test_the_sampler_walks_the_step_form():
    """The sampler carries a state and never a window, thus a walk of N steps costs N
    steps of work. The gate is the shape and the lead-in: one bar of silence at the head,
    drawn from nothing, and the same seed twice gives the same walk."""
    from mamba import infer

    held = drawn_mamba()
    walk = infer.draw(held, seeds=[7], steps=24, temperature=1.0, min_p=0.05)
    assert walk.shape == (1, 24, corpus.SEATS)
    assert np.all(walk[0, : corpus.BAR_STEPS] == 0), "the lead-in is silence"
    again = infer.draw(held, seeds=[7], steps=24, temperature=1.0, min_p=0.05)
    assert np.array_equal(walk, again), "the same seed must give the same walk"


@pytest.mark.parametrize("taps,length", [(4, 1), (4, 2), (4, 5), (16, 3), (16, 15)])
def test_a_window_shorter_than_the_taps_still_agrees(taps, length):
    """The tap rule reads zero for a step the window does not have. A window shorter than
    K is where a pad written the other way round would show."""
    held = drawn_mamba(taps=taps)
    classes, phases = drawn_window(rows=1, length=length)
    stepped = walk_by_steps(held, classes, phases)
    windowed = held.hidden(classes, phases)
    assert float(jnp.max(jnp.abs(stepped - windowed))) < 1e-4


@pytest.mark.parametrize("attention_at", [(1,), (0,), (2,), (0, 2)])
def test_a_hybrid_plan_agrees_step_for_step(attention_at):
    """The hybrid probe: era four's attention sublayer swapped in where a block stood.

    The attention layer is the one layer with a CONTEXT and its ring is the one place a
    hybrid can disagree with itself: a ring read one slot late, a distance counted from
    the wrong end, or a mask off by one all land here. The plan varies because a form
    that only works in the middle of the stack is not a form."""
    held = drawn_mamba(attention_at=attention_at)
    assert sum(1 for kind in held.plan if kind == model.ATTN) == len(attention_at)
    classes, phases = drawn_window(rows=2, length=64)
    stepped = walk_by_steps(held, classes, phases)
    windowed = held.hidden(classes, phases)
    assert_one_stream(stepped, windowed)


@pytest.mark.parametrize("spelt", ["MMZ", "MZM", "MMZF", "MZFM", "MMAF", "ZFMM"])
def test_a_spelt_plan_agrees_step_for_step(spelt):
    """The Zamba block and the feed-forward, held to the same gate as everything else.

    Z is the one layer whose KEY is built from something other than the residual
    stream, thus the step form must build it the way the window form did. F carries no
    state at all and its carry is None. The plans put Z first, last and in the middle,
    because a block that reads the embedding could accidentally work only where the
    stream still looks like it."""
    held = plan_of(spelt)
    classes, phases = drawn_window(rows=2, length=64)
    stepped = walk_by_steps(held, classes, phases)
    windowed = held.hidden(classes, phases)
    assert_one_stream(stepped, windowed)


def test_a_checkpoint_states_its_own_plan(tmp_path):
    """No flag carries the plan: the first tensor of a group names its kind, and a round
    trip through the file must give the plan back. The four kinds open with w_in
    [d, projection], wq [d, d], wq [2d, d] and w1 [d, 4d], and nothing else can."""
    spelt = "MZFAM"
    held = plan_of(spelt)
    path = tmp_path / "plan.ckpt"
    held.save(path)
    read = model.Mamba.load(path)
    assert read.plan == tuple(train.PLAN_LETTERS[c] for c in spelt.lower())


@pytest.mark.parametrize("span", [4.0, 8.0])
def test_the_span_rides_in_the_file_and_not_in_a_flag(tmp_path, span):
    """Era four carried the ALiBi span as a flag, and a span played back wrong is silently
    wrong music. This era writes it after the last layer and reads it back: the file gives
    the span back, a forward that reads it equals a forward told outright, and an older
    file with no span still reads at the elected 4."""
    held = plan_of("MMZF", span=span)
    classes, phases = drawn_window(rows=2, length=24)
    path = tmp_path / "span.ckpt"
    held.save(path)
    read = model.Mamba.load(path)
    assert read.span == span
    span = plan_of("MMZF").span
    assert span == ar_model.SLOPE_SPAN, "a model with no flag is era four's"
    from_file = read.hidden(classes, phases)
    assert float(jnp.max(jnp.abs(from_file - held.hidden(classes, phases)))) == 0.0
    # and the span must actually MOVE the model, or the round trip proves nothing
    other = plan_of("MMZF", span=32.0).hidden(classes, phases)
    assert float(jnp.max(jnp.abs(from_file - other))) > 1e-6


def test_the_attention_swap_is_smaller_than_the_block_it_replaces():
    """The probe must not win by capacity. Era four's attention sublayer is four
    square matrices; the block it replaces carries the projection, the kernel and the
    output matrix. If a swap ever grows the model, the result stops being a clean answer
    to the compute question and this says so before a run does."""
    swapped, whole = drawn_mamba(attention_at=(1,)), drawn_mamba()
    assert swapped.parameter_count() < whole.parameter_count()


def test_the_half_life_ladder_opens_each_head_on_its_rung():
    """C5's initialization: the ladder names a half-life for each head and solves for the
    dt that lands on it. The gate reads the arithmetic back -- ln 2 / (dt * a) -- because
    the ladder is stated in steps of music and stored as an inverse softplus, and nothing
    else in the trainer converts between the two."""
    heads, span = 4, (4.0, 256.0)
    decay = jnp.asarray([1.0, 2.0, 7.5, 16.0])
    dt = model.half_life_ladder(heads, span) / decay
    lives = np.asarray(jnp.log(2.0) / (dt * decay))
    assert np.allclose(lives, [4.0, 16.0, 64.0, 256.0], rtol=1e-5)


# The contract file: the seam to the elaboration


def quantized_plan(spelt="MZF", ring=8):
    """the twin of a drawn model of that plan, at the small test shape"""
    return qmodel.Mamba.from_float(plan_of(spelt), ring=ring)


def test_the_image_is_not_the_checkpoint_order():
    """A BLOCK HOLDS SIX TENSORS IN A CHECKPOINT AND THREE OF THEM NEVER REACH THE ROM.
    `a_log`, `dt_bias` and `d_skip` hold one value for each head, and an int8 tensor
    cannot carry them: they fold into the constants the ops carry instead. The two orders
    are two structures and neither is implied by the other."""
    twin = quantized_plan("MZF")
    # two tables, then three, four and two
    assert len(twin.tensors()) == len(ar_model.TABLES) + 3 + 4 + 2
    # the three per-head rows stand on the BLOCK that drew them, thus no index aligns them
    block = twin.blocks[0]
    assert block.decay.shape == block.dt_bias.shape == block.d_skip.shape
    assert block.decay.shape == (twin.heads,)


def test_w_in_is_stored_transposed():
    """The circuit reaches a weight by CONCATENATING the two walk counters, which is the
    row-major address only when the dimension under the outer counter is a power of two.
    `d` is one; the projection is not. Storing the tensor the other way round puts `d`
    under the outer counter."""
    held = plan_of("M")
    twin = qmodel.Mamba.from_float(held)
    rows, cols = twin.blocks[0].w_in.values.shape
    assert (rows, cols) == tuple(reversed(held.layers[0].w_in.shape))
    assert cols == twin.d


def test_the_decay_reads_the_libms_exponential():
    """ONE ULP DECIDES A ROM BYTE. The OCaml quantizer read the exponential through the C
    library's `exp`, and `math.exp` is that library; `np.exp`'s vectorized path may differ
    by one ulp, and one ulp there moves a `q_value` by one. The gate states the rule as
    arithmetic, and `test_parity.py`'s G1 states it through the netlist."""
    for a_log in (-1.5, 0.0, 0.5, 2.7):
        a = math.exp(a_log)
        want = int(q.round_half_up(math.ldexp(a / math.log(2.0), 12)))
        assert qmodel.decay_scale(a_log) == want
    # a decay rate the port cannot hold saturates and never wraps
    assert qmodel.decay_scale(20.0) == qmodel.DECAY_HIGH


def test_the_contract_file_round_trips_exactly(tmp_path):
    """`save` then `load` is the identity, THE PLAN INCLUDED: it comes back out of the
    shapes and no tensor states it, thus the reader of this side and the reader of the
    elaboration walk the image alike."""
    twin = quantized_plan("MZFM")
    path = tmp_path / "tiny.int8"
    twin.save(path)
    read = qmodel.Mamba.load(path)
    assert read.plan == twin.plan
    assert (read.span, read.ring) == (twin.span, twin.ring)
    assert (read.temper.q_value, read.temper.q) == (twin.temper.q_value, twin.temper.q)
    assert read.min_weight == twin.min_weight
    for here, there in zip(read.blocks, twin.blocks, strict=True):
        for mine, yours in zip(here.rows(), there.rows(), strict=True):
            assert np.array_equal(mine, yours)
    for here, there in zip(
        read.tensors(), twin.tensors(), strict=True
    ):
        assert np.array_equal(here.values, there.values) and here.e == there.e


def test_era_fours_attention_is_no_layer_of_this_model():
    """A square query is era four's plain attention, which measured null in this trunk
    three times. It is refused where a build fails loudly."""
    with pytest.raises(ValueError, match="square query"):
        qmodel.Mamba.from_float(plan_of("MA"))


def test_a_ring_the_mask_cannot_wrap_refuses_at_the_file():
    """The ring is the one number that is no fact of the training run: it is the depth at
    INFERENCE and a choice of the player. The circuit wraps it by a mask, thus a depth
    that is not a power of two has no circuit at all."""
    with pytest.raises(ValueError, match="ring"):
        quantized_plan("MZF", ring=12).check_shape()


def test_the_lead_in_draws_nothing_and_moves_no_generator():
    """One bar of silence opens the walk and the generator does not move through it: a
    twin that spent a uniform there would draw a different piece from the same seed,
    and every step of it would be legal music. IT IS THE TWIN AGAINST ITSELF and no
    circuit is in it, thus it stands here and not in `test_rtl_mamba.py`."""
    twin = quantized_plan("MZF")
    played, draws = qinfer.walk(twin, [1, 7], ar_quantized.LEAD + 2)
    assert (played[:, : ar_quantized.LEAD] == 0).all(), "the lead-in is not silent"
    assert all(not taken for taken in draws[: ar_quantized.LEAD]), "the lead-in drew"
    # the walks of a batch are independent: seed 7 draws what seed 7 draws alone
    alone, _ = qinfer.walk(twin, [7], ar_quantized.LEAD + 2)
    assert np.array_equal(alone[0], played[1])
