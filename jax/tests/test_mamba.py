"""The gate that the two forms of the recurrence are one recurrence.

jax/mamba/model.py holds the step form, which the sampler and the OCaml reference run, and
the window form, which the trainer runs because a scan of the step form takes 203 ms for
each training step against era four's 61. The window form is an unrolling and not an
approximation, thus the two must agree to float noise, and this is where that is stated.

The gate reads the WHOLE forward and not the block alone: the convolution, the state, the
gated norm and the residual joins all have a window form, and a difference in any of them
lands here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import data
from mamba import model, train

# Six layers of float32 over a window of 64 steps, reduced in two different orders: the
# window form sums a row of the decay matrix where the step form carries a state forward.
# A real disagreement -- a tap read backward, a decay off by one step, the readout taking
# the state before the update -- moves the stream far past this.
TOLERANCE = 2e-4

SHAPE = dict(
    d=32,
    layers=3,
    heads=2,
    state=8,
    expand=2,
    conv_scale=0.5,
    half_lives=None,
    attention_at=(),
)


def drawn_params(taps=model.CONV_TAPS, **over):
    return train.draw_params(jax.random.PRNGKey(3), taps=taps, **(SHAPE | over))


def plan_of(spelt):
    """a drawn model of the plan spelt out, at the small test shape"""
    letters = [train.PLAN_LETTERS[c] for c in spelt.lower()]
    return drawn_params(spelt=letters, layers=len(letters))


def drawn_window(rows, length):
    state = np.random.default_rng(11)
    classes = state.integers(0, data.CLASSES, (rows, length, data.SEATS)).astype(np.int32)
    phases = np.tile(np.arange(length) % model.PHASE_BUCKETS, (rows, 1)).astype(np.int32)
    return jnp.asarray(classes), jnp.asarray(phases)


def walk_by_steps(params, classes, phases):
    """the window form's answer, taken one step at a time through [forward_step]"""
    shape = model.shape_of(params)
    carry = model.initial_carry(shape, classes.shape[0])
    rows = []
    for step in range(classes.shape[1]):
        carry, h = model.forward_step(params, carry, classes[:, step], phases[:, step])
        rows.append(h)
    return jnp.stack(rows, axis=1)


@pytest.mark.parametrize("taps", [4, 16])
def test_the_step_form_and_the_window_form_give_one_stream(taps):
    """K is a field of the checkpoint and not a constant, thus the gate runs at two widths:
    the tap ring of the step form and the pad of the window form both read it, and a width
    written into one of them alone lands here."""
    params = drawn_params(taps=taps)
    classes, phases = drawn_window(rows=2, length=64)
    stepped = walk_by_steps(params, classes, phases)
    windowed = model.hidden(params, classes, phases)
    assert stepped.shape == windowed.shape
    gap = float(jnp.max(jnp.abs(stepped - windowed)))
    scale = float(jnp.max(jnp.abs(windowed)))
    assert gap / scale < TOLERANCE, f"the two forms part by {gap:.3e} on a scale of {scale:.3e}"


def test_the_window_opens_on_a_zero_state():
    """A window is not a slice of a longer walk: it starts where the boot of the sampler
    starts. The first step of the window form must therefore equal one step of the step
    form from the origin, exactly, with no history behind either."""
    params = drawn_params()
    classes, phases = drawn_window(rows=2, length=8)
    shape = model.shape_of(params)
    carry = model.initial_carry(shape, classes.shape[0])
    _, first = model.forward_step(params, carry, classes[:, 0], phases[:, 0])
    windowed = model.hidden(params, classes, phases)[:, 0]
    assert float(jnp.max(jnp.abs(first - windowed))) < 1e-5


@pytest.mark.parametrize("taps", [4, 16])
def test_the_convolution_reads_the_steps_behind_it(taps):
    """Tap k reads the step k back. The window form pads at the head, the step form holds
    a ring of K-1, and the gate is an impulse: a window that is zero but for step 0 must
    read the taps out in order down the channels. Nothing but [conv] states K."""
    channels = 5
    conv = jnp.asarray(np.arange(channels * taps, dtype=np.float32).reshape(channels, taps))
    u = jnp.zeros((1, taps + 3, channels)).at[0, 0].set(1.0)
    windowed = model.convolve_window(conv, u)
    # step t reads the impulse through tap t, thus the row at t is the tap column t
    for t in range(taps):
        assert np.allclose(np.asarray(windowed[0, t]), np.asarray(conv[:, t]))
    assert np.allclose(np.asarray(windowed[0, taps:]), 0.0)


def test_the_sampler_walks_the_step_form():
    """The sampler carries a state and never a window, thus a walk of N steps costs N
    steps of work. The gate is the shape and the lead-in: one bar of silence at the head,
    drawn from nothing, and the same seed twice gives the same walk."""
    from mamba import infer

    params = drawn_params()
    walk = infer.sample(params, seeds=[7], steps=24, temperature=1.0, min_p=0.05)
    assert walk.shape == (1, 24, data.SEATS)
    assert np.all(walk[0, : data.BAR_STEPS] == 0), "the lead-in is silence"
    again = infer.sample(params, seeds=[7], steps=24, temperature=1.0, min_p=0.05)
    assert np.array_equal(walk, again), "the same seed must give the same walk"


@pytest.mark.parametrize("taps,length", [(4, 1), (4, 2), (4, 5), (16, 3), (16, 15)])
def test_a_window_shorter_than_the_taps_still_agrees(taps, length):
    """The tap rule reads zero for a step the window does not have. A window shorter than
    K is where a pad written the other way round would show."""
    params = drawn_params(taps=taps)
    classes, phases = drawn_window(rows=1, length=length)
    stepped = walk_by_steps(params, classes, phases)
    windowed = model.hidden(params, classes, phases)
    assert float(jnp.max(jnp.abs(stepped - windowed))) < 1e-4


@pytest.mark.parametrize("attention_at", [(1,), (0,), (2,), (0, 2)])
def test_a_hybrid_plan_agrees_step_for_step(attention_at):
    """The hybrid probe: era four's attention sublayer swapped in where a block stood.

    The attention layer is the one layer of this model with a CONTEXT, and its ring is the
    one place a hybrid can disagree with itself. The window form attends over the whole
    window with ALiBi and the causal wall; the step form attends over a ring of the last
    keys and values, with a mask for the slots the walk has not written yet. They are one
    attention or they are a bug: a ring read one slot late, a distance counted from the
    wrong end, or a mask off by one all land here.

    The plan varies because the position of the attention layer is the probe's own second
    question, and a form that only works in the middle of the stack is not a form."""
    params = drawn_params(attention_at=attention_at)
    shape = model.shape_of(params)
    assert sum(1 for kind in shape.plan if kind == model.ATTN) == len(attention_at)
    classes, phases = drawn_window(rows=2, length=64)
    stepped = walk_by_steps(params, classes, phases)
    windowed = model.hidden(params, classes, phases)
    gap = float(jnp.max(jnp.abs(stepped - windowed)))
    scale = float(jnp.max(jnp.abs(windowed)))
    assert gap / scale < TOLERANCE, f"the two forms part by {gap:.3e} on {scale:.3e}"


@pytest.mark.parametrize("spelt", ["MMZ", "MZM", "MMZF", "MZFM", "MMAF", "ZFMM"])
def test_a_spelt_plan_agrees_step_for_step(spelt):
    """The Zamba block and the feed-forward, held to the same gate as everything else.

    Z is the one layer whose KEY is built from something other than the residual stream --
    the query and the key read the ORIGINAL EMBEDDING beside it -- thus the ring carries a
    key the step form must build the same way the window form did, or the two part. F
    carries no state at all and its carry is None, which the walk must not trip over.

    The plans put Z first, last and in the middle, because a block whose input is the
    embedding is exactly the one that could accidentally work only where the stream still
    looks like the embedding."""
    params = plan_of(spelt)
    classes, phases = drawn_window(rows=2, length=64)
    stepped = walk_by_steps(params, classes, phases)
    windowed = model.hidden(params, classes, phases)
    gap = float(jnp.max(jnp.abs(stepped - windowed)))
    scale = float(jnp.max(jnp.abs(windowed)))
    assert gap / scale < TOLERANCE, f"the two forms part by {gap:.3e} on {scale:.3e}"


def test_a_checkpoint_states_its_own_plan():
    """No flag carries the plan: the first tensor of a group names its kind, and a round
    trip through the file must give the plan back. The four kinds open with w_in
    [d, projection], wq [d, d], wq [2d, d] and w1 [d, 4d], and nothing else can."""
    import tempfile

    spelt = "MZFAM"
    params = plan_of(spelt)
    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        train.save_checkpoint(f.name, params)
        read = model.load_params(f.name)
    assert model.shape_of(read).plan == tuple(
        train.PLAN_LETTERS[c] for c in spelt.lower()
    )


@pytest.mark.parametrize("chunk", [8, 16, 32])
@pytest.mark.parametrize("length", [64, 48, 33, 7])
def test_the_chunked_window_is_the_quadratic_window(chunk, length):
    """The chunked semiseparable form against the quadratic form it replaces.

    [selective_window] is the oracle here and not a second opinion: it is already held to
    the STEP form by the gate above, thus holding the chunked form to it puts all three on
    one recurrence. What this catches is the seam -- a chunk carrying the state after its
    own add instead of before it, a cumulative sum taken from the window's head rather than
    the chunk's, a tail chunk the length does not fill.

    The lengths run over a multiple of every chunk, a length shorter than one chunk, and
    two that divide by none of them."""
    params = drawn_params()
    shape = model.shape_of(params)
    layer = params["layers"][0]
    rng = np.random.default_rng(5)

    def drawn(*tail):
        return jnp.asarray(rng.standard_normal((2, length, *tail)), jnp.float32)

    x = drawn(shape.d_in)
    b, c = drawn(shape.state), drawn(shape.state)
    dt = jnp.abs(drawn(shape.heads)) * 0.1
    a = jnp.exp(layer["a_log"])
    quadratic = model.selective_window(shape, layer, x, b, c, dt, a)
    chunked = model.selective_window_chunked(shape, layer, x, b, c, dt, a, chunk=chunk)
    assert chunked.shape == quadratic.shape
    gap = float(jnp.max(jnp.abs(chunked - quadratic)))
    scale = float(jnp.max(jnp.abs(quadratic)))
    assert gap / scale < TOLERANCE, f"the two windows part by {gap:.3e} on {scale:.3e}"


@pytest.mark.parametrize("span", [4.0, 8.0])
def test_the_span_rides_in_the_file_and_not_in_a_flag(span):
    """Era four carried the ALiBi span as a flag that had to match the training run, and a
    span played back wrong is silently wrong music -- the walk still sounds like music, it
    is just not the model's. This era writes it after the last layer and reads it back.

    The gate is a round trip and a consequence: the file gives the span back, and a forward
    that reads it from the file equals a forward told the span outright. It also holds the
    older files, which carry no span at all and must still read at era four's elected 4."""
    import tempfile

    params = plan_of("MMZF")
    classes, phases = drawn_window(rows=2, length=24)
    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        train.save_checkpoint(f.name, params, span=span)
        read = model.load_params(f.name)
    assert read[model.SPAN_KEY] == span
    assert model.span_of(read) == span
    assert model.span_of(params) == model.SLOPE_SPAN, "no span in the file is era four's"
    from_file = model.hidden(read, classes, phases)
    told = model.hidden(read, classes, phases, span=span)
    assert float(jnp.max(jnp.abs(from_file - told))) == 0.0
    # and the span must actually MOVE the model, or the round trip proves nothing
    other = model.hidden(read, classes, phases, span=32.0)
    assert float(jnp.max(jnp.abs(from_file - other))) > 1e-6


def test_the_attention_swap_is_smaller_than_the_block_it_replaces():
    """The probe must not win by capacity. Era four's attention sublayer is four
    square matrices; the block it replaces carries the projection, the kernel and the
    output matrix. If a swap ever grows the model, the result stops being a clean answer
    to the compute question and this says so before a run does."""
    trunk = drawn_params()
    hybrid = drawn_params(attention_at=(1,))

    def count(params):
        return sum(int(np.prod(t.shape)) for t in jax.tree.leaves(params))

    assert count(hybrid) < count(trunk)


def test_the_half_life_ladder_opens_each_head_on_its_rung():
    """C5's initialization: the ladder names a half-life for each head and solves for the
    dt that lands on it. The gate reads the arithmetic back -- ln 2 / (dt * a) -- because
    the ladder is stated in steps of music and stored as an inverse softplus, and nothing
    else in the trainer converts between the two."""
    heads, span = 4, (4.0, 256.0)
    decay = jnp.asarray([1.0, 2.0, 7.5, 16.0])
    dt = train.half_life_ladder(heads, span) / decay
    lives = np.asarray(jnp.log(2.0) / (dt * decay))
    assert np.allclose(lives, [4.0, 16.0, 64.0, 256.0], rtol=1e-5)
