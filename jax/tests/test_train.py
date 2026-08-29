"""The trainer must train.

This file exists because of a bug that every other check passed. Converting the CLI to
click renamed `args.lr` to `lr`, and the loop already wrote its rate back into a name of
its own -- so `lr = schedule(step, lr, ...)` fed the schedule its own output and the rate
collapsed geometrically to zero. The model initialised correctly, the shapes were right,
and a one-step run reported the exact baseline loss, because a step-1 loss is measured
before the first update. Nothing was wrong except that no learning happened.

Therefore: run the loop and watch the loss fall. A smoke test at d 8 costs seconds.

THE SCHEDULE IS INSIDE THE OPTIMIZER now and no loop can overwrite its peak, thus the bug
above has no home left; the curve is held here all the same, because the rate the loop
applies is still the thing that decides whether a run trains.

Era four and era five run ONE loop now -- `ar_train.train` -- thus this file holds the
shape of a training run and not of a trainer: each era's own CLI, its own draw, and the
loop they share.
"""

import re

import jax.numpy as jnp
import numpy as np
import pytest
from click.testing import CliRunner

import data
import nn
from mamba import train as mamba_train
from transformer import train as transformer_train

CORPUS = data.FRAMES


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


def losses_of(command, flags):
    """the losses one short run printed, or the failure that stopped it"""
    done = CliRunner().invoke(command, SHORT_RUN + flags)
    assert done.exit_code == 0, done.output
    return [float(m) for m in re.findall(r"loss (\d+\.\d+)", done.output)], done.output


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
    """THE GUARD IS PER ERA, and `mamba/measure.py free` is why: it called a click
    `Command` with a model for weeks and raised for anyone who ran it, because nothing ran
    it. A trainer that cannot learn still prints a correct step-1 loss -- the loss at step
    1 is measured before the first update -- so only a run of several steps can tell.

    Sixty steps at d 8 costs seconds, and it is the only thing that reads a trainer's
    whole path: the flags, the draw, the optimizer, the loop and the log."""
    losses, output = losses_of(command, flags)
    assert len(losses) >= 3, output
    assert losses[-1] < losses[0] - 0.2, f"era {era}: the loss did not fall: {losses}"


# ==================================================================== #
# The rate curve: optax says what this project's schedule says          #
# ==================================================================== #

# THE HAND-ROLLED `nn.adamw` AND `nn.schedule` ARE GONE, and the gates that held optax to
# them went with them: they had done their work, which was to prove that a checkpoint
# trained under the old rule is a checkpoint of the new one. What is left is the CURVE,
# against the closed form written here -- because `nn.learning_rates` is now the only
# statement of it in the code and a test that read it back from itself would state
# nothing.


def curve_at(step, peak, warmup, total):
    """The rate this project wants at [step] of the run, ONE BASED: linear from 0 to
    [peak] over [warmup] steps, then cosine from [peak] to 0 over the rest.

    THE TWO ENDS ARE THIS PROJECT'S AND NOT OPTAX'S. A warmup of zero is a constant peak,
    where `warmup_cosine_decay_schedule` would be a bare cosine decay; and a run SHORTER
    THAN ITS OWN WARMUP -- every short probe -- is the ramp alone, where optax refuses to
    build a cosine of a negative length at all."""
    if warmup == 0:
        return peak
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))


@pytest.mark.parametrize("warmup,total", [(1000, 30000), (0, 30000), (1000, 200)])
def test_the_rate_curve_is_the_one_the_recipe_states(warmup, total):
    """`nn.learning_rates` at every step of the run, and THE STEP IS ONE BASED.

    optax hands a schedule its own update count, which is 0 at the first update; the
    correction inside `learning_rates` is what puts the curve on the loop's step, and a
    rule that dropped it would apply a rate of 0 to the first update and every later rate
    one step behind -- which trains a slightly different model and says nothing."""
    peak = 3e-3
    theirs = np.array(
        [curve_at(step, peak, warmup, total) for step in range(1, total + 1)]
    )
    # a constant schedule states one scalar whatever it is handed, thus it broadcasts
    ours = np.broadcast_to(
        np.asarray(nn.learning_rates(peak, warmup, total)(jnp.arange(total))),
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


@pytest.mark.parametrize("clip", [0.0, 1.0])
def test_a_clip_of_zero_or_less_is_no_clip(clip):
    """It is not a clip AT zero, which would zero every gradient of the run. The chain
    leaves the node out entirely."""
    rule = nn.update_rule(peak=1e-3, warmup=0, total=10, clip=clip, weight_decay=0.0)
    tree = [jnp.full((3,), 1e3, jnp.float32)]
    state = rule.init(tree)
    updates, _ = rule.update(tree, state, tree)
    assert float(jnp.max(jnp.abs(updates[0]))) > 0.0
