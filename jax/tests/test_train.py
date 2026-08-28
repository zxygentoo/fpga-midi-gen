"""The trainer must train.

This file exists because of a bug that every other check passed. Converting the CLI to
click renamed `args.lr` to `lr`, and the loop already wrote its rate back into a name of
its own -- so `lr = schedule(step, lr, ...)` fed the schedule its own output and the rate
collapsed geometrically to zero. The model initialised correctly, the shapes were right,
and a one-step run reported the exact baseline loss, because a step-1 loss is measured
before the first update. Nothing was wrong except that no learning happened.

Therefore: run the loop and watch the loss fall. A smoke test at d 8 costs seconds.
"""

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from click.testing import CliRunner

import nn
from diffusion import train as sheet
from transformer import train

JAX_ROOT = Path(__file__).resolve().parent.parent
CORPUS = JAX_ROOT / "_data" / "frames.safetensors"


def test_the_schedule_holds_its_peak():
    """the peak is an input and never moves; a caller that overwrites it is the bug this
    file guards, and this pins the half that lives in the schedule"""
    peak = 1e-3
    rates = [nn.schedule(step, peak, 300, 48000) for step in range(1, 1000)]
    assert max(rates) == pytest.approx(peak)
    assert rates[299] == pytest.approx(peak)  # the warmup ends exactly on the peak
    assert rates[-1] > peak * 0.9  # and the cosine has barely begun to fall


def test_the_skipped_key_keeps_the_layers_of_the_elected_runs():
    """The window-position table left the design and its key stays skipped. Without the
    skip a seed draws different layers, and the runs that elected this model stop
    reproducing from their seeds."""
    import jax

    key = jax.random.PRNGKey(6)
    params = train.draw_params(key, 8, 1)
    keys = iter(jax.random.split(key, 3 + 6))
    for _ in range(3):  # seats, phase, and the skipped row of the dropped table
        next(keys)
    first = jax.random.normal(next(keys), (8, 8), dtype="float32") * 0.02
    assert (params["layers"][0]["wq"] == first).all()


@pytest.mark.skipif(not CORPUS.exists(), reason="needs corpus_tool export")
def test_the_loss_falls_over_a_short_run():
    """the guard the click conversion needed: a trainer that cannot learn still prints a
    correct step-1 loss, so only a run of several steps can tell"""
    done = CliRunner().invoke(
        train.main,
        [
            "--d",
            "8",
            "--layers",
            "1",
            "--heads",
            "2",
            "--context",
            "32",
            "--batch",
            "4",
            "--steps",
            "60",
            "--lr",
            "1e-2",
            "--warmup",
            "10",
            "--seed",
            "4",
            "--log-every",
            "20",
            "--eval-every",
            "1000",
        ],
    )
    assert done.exit_code == 0, done.output
    losses = [float(m) for m in re.findall(r"loss (\d+\.\d+)", done.output)]
    assert len(losses) >= 3, done.output
    assert losses[-1] < losses[0] - 0.2, f"the loss did not fall: {losses}"


# ==================================================================== #
# Era six: optax says the same thing the hand-rolled rule said          #
# ==================================================================== #

# THE SHEET TRAINER MOVED TO OPTAX AND OWES NO RETRAIN, and these two gates are why. The
# frozen eras keep `nn.adamw` and `nn.schedule`; the sheet trainer calls neither, and a
# checkpoint trained under the old rule is still a checkpoint of the new one -- because
# the two rules are ONE rule, held here leaf for leaf and step for step.
#
# They run on whatever platform the suite has, and they are equalities of the SAME
# arithmetic in the same order, not of two reductions: the trees are small and every
# operation is elementwise.

# the leaves of the toy tree: three shapes, so that a rule that folded a tree wrongly
# cannot pass by accident
LEAF_SHAPES = ((3, 4), (5,), (2, 2, 2))
RULE_STEPS = 1000


def toy_run(steps, seed=4):
    """the opening tree and one gradient tree for each step, drawn once and shared by both
    rules, with the gradient norm growing over the run so that a clip has something to bite"""
    rng = np.random.default_rng(seed)
    opening = [
        jnp.asarray(rng.normal(0.0, 0.3, shape), jnp.float32) for shape in LEAF_SHAPES
    ]
    grads = [
        [
            jnp.asarray(rng.normal(0.0, spread, shape), jnp.float32)
            for shape in LEAF_SHAPES
        ]
        for spread in np.linspace(0.05, 2.0, steps)
    ]
    return opening, grads


def both_rules(steps, *, lr, warmup, clip, weight_decay):
    """The same opening tree and the same gradients through `nn.adamw` and through the
    optax chain of the sheet trainer, step for step: the two parameter trees at each step.

    THE RATE COMES FROM THE SCHEDULE ON BOTH SIDES, AND THAT IS WHAT HOLDS THE STEP COUNT
    ITSELF. `nn.adamw` is handed `nn.schedule` at the LOOP's step, which is 1 at the first
    update; optax reads its OWN count inside the chain, which is 0 there, and
    `learning_rates` corrects it. Nothing else in this file runs that correction through
    the real optimizer: the curve gate below reads the schedule directly and would pass a
    trainer whose optimizer never used it.

    A warmup of zero is the constant rate on both sides -- `nn.schedule` short-circuits to
    the peak -- thus one loop reads the update rule alone and the same loop reads it under
    a moving rate."""
    opening, grads = toy_run(steps)
    rule = sheet.update_rule(
        peak=lr, warmup=warmup, total=steps, clip=clip, weight_decay=weight_decay
    )
    here, state = opening, nn.optimizer_init(opening)
    there, opt_state = opening, rule.init(opening)
    walked = []
    for t, gradient in enumerate(grads, start=1):
        here, state = nn.adamw(
            state,
            here,
            gradient,
            jnp.float32(t),
            nn.schedule(t, lr, warmup, steps),
            clip=clip,
            weight_decay=weight_decay,
        )
        updates, opt_state = rule.update(gradient, opt_state, there)
        there = optax.apply_updates(there, updates)
        walked.append((here, there))
    return walked


def furthest_apart(here, there):
    """the largest relative difference over the leaves of two parameter trees"""
    return max(
        float(
            np.max(
                np.abs(np.asarray(a) - np.asarray(b)) / (np.abs(np.asarray(a)) + 1e-12)
            )
        )
        for a, b in zip(here, there)
    )


# The warmup of the moving-rate rows ends INSIDE the run, thus both legs of the schedule
# are crossed: the ramp, where the rate is smallest and a step count out by one is loudest,
# and the cosine behind it.
RULE_WARMUP = 200

# What the two rules stand apart by under a moving rate, MEASURED 2026-08-28 over the rows
# below: 7.5e-8 at t 1 and at worst 1.5e-6 at t 1000. It is the FORMAT and not the rule --
# optax states the rate in float32 and `nn.schedule` in float64, and 1,000 updates
# accumulate the last bits. Dropping the `+ 1` of `learning_rates` reads 3.2e-3 at t 1 and
# 5.0e-3 at t 1000 on the same rows, thus this floor stands 6 times over the noise and 300
# times under the fault.
MOVING_APART = 1e-5


@pytest.mark.parametrize(
    "warmup,clip,weight_decay",
    [
        # a constant rate, thus the update rule alone:
        # no clip at all -- `nn.adamw`'s guard and the chain that leaves the node out
        (0, 0.0, 0.0),
        # a clip that never bites, and one that bites at every step
        (0, 100.0, 0.0),
        (0, 0.01, 0.0),
        # AdamW proper, where the decoupled decay is not a no-op
        (0, 1.0, 0.01),
        # and the same rule under a MOVING rate, which is the only thing here that runs
        # the schedule through the real optimizer
        (RULE_WARMUP, 1.0, 0.0),
        (RULE_WARMUP, 0.01, 0.01),
    ],
)
def test_the_optax_rule_is_the_hand_rolled_rule(warmup, clip, weight_decay):
    """`optax.adamw` is `nn.adamw`'s line, term for term: `scale_by_adam` bias-corrected
    with eps AFTER the square root and `eps_root` 0, then the decoupled decay, then the
    rate -- behind the same global-norm clip. The first update and the thousandth are both
    read, because the bias correction is what a wrong step count would move and it is
    loudest at t 1 and quietest at t 1000.

    THE MOVING-RATE ROWS HOLD THE STEP COUNT THROUGH THE OPTIMIZER. `learning_rates` reads
    optax's own update count at `count + 1` because that count is 0 at the first update;
    the curve gate below cannot see whether the optimizer really uses it, and these rows
    can. Dropping the `+ 1` applies a rate of 0 to the first update and every later rate
    one step behind, and the two trees then part at step 1 and never meet again.

    The moving rows read at [MOVING_APART] and the constant rows at `1e-6`; the reason is
    the FORMAT and not the rule, and it is measured where that constant stands."""
    walked = both_rules(
        RULE_STEPS, lr=1e-3, warmup=warmup, clip=clip, weight_decay=weight_decay
    )
    apart = MOVING_APART if warmup else 1e-6
    for t in (1, RULE_STEPS):
        here, there = walked[t - 1]
        assert furthest_apart(here, there) < apart, (
            f"warmup {warmup}, clip {clip}, wd {weight_decay}: "
            f"the two rules part at step {t}"
        )


@pytest.mark.parametrize("warmup,total", [(1000, 30000), (0, 30000), (1000, 200)])
def test_the_two_schedules_are_one_curve(warmup, total):
    """`learning_rates` is `nn.schedule` at every step of the run, and THE STEP IS ONE
    BASED. optax hands a schedule its own update count, which is 0 at the first update;
    the correction inside `learning_rates` is what makes the two curves one, and a rule
    that dropped it would apply a rate of 0 to the first update and every later rate one
    step behind -- which trains a slightly different model and says nothing.

    THE TWO ENDS ARE `nn.schedule`'S RULES AND NOT OPTAX'S, thus both are read here. A
    warmup of zero is a constant, where `warmup_cosine_decay_schedule` with no warmup is a
    bare cosine decay; and a run shorter than its own warmup -- every short probe of this
    trainer -- is the ramp alone, where optax refuses to build a cosine of a negative
    length and the trainer would not start at all."""
    peak = 3e-3
    theirs = np.array(
        [nn.schedule(step, peak, warmup, total) for step in range(1, total + 1)]
    )
    # a constant schedule states one scalar whatever it is handed, thus it broadcasts
    ours = np.broadcast_to(
        np.asarray(sheet.learning_rates(peak, warmup, total)(jnp.arange(total))),
        theirs.shape,
    )
    assert np.max(np.abs(ours - theirs)) < 1e-9
    # and the curve is the one the trainer wants: the peak is never passed
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
