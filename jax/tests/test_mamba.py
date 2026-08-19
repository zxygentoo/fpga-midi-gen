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

SHAPE = dict(d=32, layers=3, heads=2, state=8, expand=2, conv_scale=0.5)


def drawn_params():
    return train.draw_params(jax.random.PRNGKey(3), **SHAPE)


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


def test_the_step_form_and_the_window_form_give_one_stream():
    params = drawn_params()
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


def test_the_convolution_reads_the_steps_behind_it():
    """Tap k reads the step k back. The window form pads at the head, the step form holds
    a ring of K-1, and the gate is an impulse: a window that is zero but for step 0 must
    read the taps out in order down the channels."""
    channels, taps = 5, model.CONV_TAPS
    conv = jnp.asarray(np.arange(channels * taps, dtype=np.float32).reshape(channels, taps))
    u = jnp.zeros((1, 6, channels)).at[0, 0].set(1.0)
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


@pytest.mark.parametrize("length", [1, 2, 5])
def test_a_window_shorter_than_the_taps_still_agrees(length):
    """The tap rule reads zero for a step the window does not have. A window shorter than
    K is where a pad written the other way round would show."""
    params = drawn_params()
    classes, phases = drawn_window(rows=1, length=length)
    stepped = walk_by_steps(params, classes, phases)
    windowed = model.hidden(params, classes, phases)
    assert float(jnp.max(jnp.abs(stepped - windowed))) < 1e-4
