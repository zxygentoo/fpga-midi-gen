"""The trainer must train.

This file exists because of a bug every other check passed: a loop that fed the schedule
its own output collapsed the rate geometrically to zero, and the model still initialised
correctly, the shapes were still right, and a one-step run still reported the exact
baseline loss -- because a step-1 loss is measured before the first update. Nothing was
wrong except that no learning happened.

Therefore: run the loop and watch the loss fall. The schedule is inside the optimizer now
and the bug has no home left; the curve is held here all the same, because the rate the
loop applies still decides whether a run trains.
"""


import jax.numpy as jnp
import numpy as np
import pytest

import corpus
import train
from mamba import train as mamba_train
from tests import gate
from transformer import train as transformer_train

CORPUS = corpus.FRAMES


SHORT_RUN = [
    "--d", "8",
    "--heads", "2",
    "--context", "32",
    "--batch", "4",
    "--steps", "60",
    "--lr", "1e-2",
    "--warmup", "10",
    "--seed", "4",
    "--log-every", "20",
    "--eval-every", "1000",
]


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.exists(), reason="needs corpus_tool export")
@pytest.mark.parametrize(
    "era,command,flags",
    [
        ("four", transformer_train.main, ["--layers", "1"]),
        # era five's CLI threads twelve shape flags into `Mamba.drawn` and the plan
        # through a callback, thus its own run says what era four's cannot
        ("five", mamba_train.main, ["--plan", "MZF", "--state", "8"]),
    ],
)
def test_the_loss_falls_over_a_short_run(era, command, flags):
    """THE GUARD IS PER ERA, and `mamba/measure.py free` is why: it raised for anyone
    who ran it for weeks, because nothing ran it. Sixty steps at d 8 costs seconds and
    it is the only thing that reads a trainer's whole path -- the flags, the draw, the
    optimizer, the loop and the log."""
    losses, output = gate.losses_of(command, SHORT_RUN + flags)
    assert len(losses) >= 3, output
    assert losses[-1] < losses[0] - 0.2, f"era {era}: the loss did not fall: {losses}"


# ==================================================================== #
# The rate curve: optax says what this project's schedule says          #
# ==================================================================== #

# What is left is the CURVE, against the closed form written here: `train.learning_rates`
# is the only statement of it in the code, and a test that read it back from itself would
# state nothing.


def curve_at(step, peak, warmup, total):
    """The rate this project wants at [step] of the run, ONE BASED: linear to [peak] over
    [warmup] steps, then cosine to 0. THE TWO ENDS ARE THIS PROJECT'S AND NOT OPTAX'S -- a
    warmup of zero is a constant peak, and a run shorter than its warmup is the ramp
    alone."""
    if warmup == 0:
        return peak
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))


@pytest.mark.parametrize("warmup,total", [(1000, 30000), (0, 30000), (1000, 200)])
def test_the_rate_curve_is_the_one_the_recipe_states(warmup, total):
    """`train.learning_rates` at every step, and THE STEP IS ONE BASED: optax hands a
    schedule its own update count, which is 0 at the first update, thus a rule that
    dropped the correction would apply a rate of 0 there and every later rate one step
    behind."""
    peak = 3e-3
    theirs = np.array(
        [curve_at(step, peak, warmup, total) for step in range(1, total + 1)]
    )
    # a constant schedule states one scalar whatever it is handed, thus it broadcasts
    ours = np.broadcast_to(
        np.asarray(train.learning_rates(peak, warmup, total)(jnp.arange(total))),
        theirs.shape,
    )
    assert np.max(np.abs(ours - theirs)) < 1e-9
    # the peak is an input and it is never passed: a caller that overwrites it was the bug
    # this file guards, and this pins the half that lives in the schedule
    assert ours.max() <= peak * (1.0 + 1e-5)
    # the curve is float32 and the numbers here are small, thus the shape reads at the
    # relative tolerance of that format and the equality above carries the precision
    if warmup and total > warmup:
        assert ours[0] == pytest.approx(peak / warmup, rel=1e-5)
        assert ours[warmup - 1] == pytest.approx(peak, rel=1e-5)
        assert ours[-1] == pytest.approx(0.0, abs=1e-12)
    elif warmup:
        # the run ends inside its own warmup, thus the peak is never reached
        assert ours[-1] == pytest.approx(peak * total / warmup, rel=1e-5)
    else:
        assert np.allclose(ours, peak, rtol=1e-5)


def test_a_clip_of_zero_is_no_clip():
    """It is not a clip AT zero, which would zero every gradient of the run: the chain
    leaves the node out. ONLY the zero case is a property this can hold, because Adam is
    scale-invariant and a positive clip is unobservable through one update."""
    rule = train.update_rule(peak=1e-3, warmup=0, total=10, clip=0.0, weight_decay=0.0)
    tree = [jnp.full((3,), 1e3, jnp.float32)]
    state = rule.init(tree)
    updates, _ = rule.update(tree, state, tree)
    assert float(jnp.max(jnp.abs(updates[0]))) > 0.0
