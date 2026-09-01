"""What the quantization costs: each integer twin against the float model it quantizes.

Era six holds the file and the frozen eras stand at its foot. Every era takes the same
shape of gate and what differs is its FEEDBACK AXIS -- era four's KV ring, era five's
state, era six's sheet -- because that is what decides whether an arithmetic error dies
with its step or compounds.

TWO PARTS, and the drawn weights make both deterministic. The fixed sweep pins MEASURED
NUMBERS AND NOT THRESHOLDS: a diff says the integers moved, and the reader judges whether
it is a re-measurement or a bug. The property part draws seed pairs at a fixed generator
and holds floors calibrated under the first measured minima; the printed minima keep the
calibration honest.
"""

from typing import NamedTuple

import numpy as np
import pytest

from diffusion import model, quantized
from mamba import quantized as mamba_twin
from quantized import engine_states
from tests.models import drawn_transformer, plan_of
from transformer import quantized as transformer_twin

# EVERY GATE HERE IS SLOW because it MEASURES: 112 drift runs, each a whole walk of a
# drawn model against its twin. `-m "not slow"` is the inner loop.
pytestmark = pytest.mark.slow

# the structure of the era at a shape a test can afford: the stem, two residual pairs and
# the head, over a quarter of the board's sheet -- two measures
LAYERS = 6
WIDTH = 8
STEPS = 32

WEIGHT_SEEDS = (11, 23, 37, 41)
WALK_SEEDS = (42, 43, 44, 45)


class Shares(NamedTuple):
    """The three numbers a drift report gives as SHARES of what it counted. A property
    gate reads the lowest of each over its trials, and an era's floors stand in the same
    shape -- thus the minimum and the floor it must clear are one type."""

    top1: float
    same_draw: float
    cosine: float


def shares_of(said, counted):
    """the three shares of one drift report; [counted] names the field that says what the
    comparison ran over, which is `cells` on a sheet and `draws` on a chain"""
    over = getattr(said, counted)
    return Shares(said.same_peak / over, said.same_draw / over, said.mean_cosine)


def lowest(here, there):
    """the lower of each share: what the trials so far have read"""
    return Shares(*(min(a, b) for a, b in zip(here, there)))


def drawn_pairs(seed, span, trials):
    """The (weight seed, walk seed) pairs of one property gate, off a FIXED generator: the
    set is the same at every run, thus a floor that breaks is the scheme and never the
    draw. The generator and the span are each era's own, because moving either would move
    the measured minima its floors were calibrated under."""
    rng = np.random.default_rng(seed)
    return tuple(tuple(int(v) for v in rng.integers(1, span, 2)) for _ in range(trials))


def drift(weight_seed, walk_seed, passes):
    """the drift of one drawn model on one walk"""
    coconet = model.Coconet.drawn(weight_seed, LAYERS, WIDTH)
    states, given = model.opening_sheet(engine_states([walk_seed]), STEPS)
    return quantized.drift(coconet, states, given, walk=passes)


# The row `test_a_sweep_states_its_measured_numbers` reads: for each weight seed, summed
# over the four walks, the top-1 count, the same-draw count, the cells counted, and the
# LOWEST mean cosine of the four -- which is the sharpest cosine signal.
SWEPT = {
    11: (3029, 3040, 3299, 0.9984),
    23: (3104, 3033, 3299, 0.9984),
    37: (3095, 3086, 3299, 0.9982),
    41: (3083, 3127, 3299, 0.9960),
}


def sweep_drift(weight_seed, walk_seed):
    """era six's row of the shared sweep at the foot of this file: 16 passes, where the
    cosine signal is sharpest"""
    return drift(weight_seed, walk_seed, 16)


def trial_drift(weight_seed, walk_seed):
    """era six's trial of the shared property gate: 8 passes, the length its floors were
    calibrated at"""
    return drift(weight_seed, walk_seed, 8)


# at 8, 32 and 128 passes of one model: the top-1 count, the cells, the mean cosine, the
# share of activation writes that rode the clamp, and the hottest write in real units
LONG_WALK = {
    8: (411, 445, 0.9982, 0.0, 4.58),
    32: (1436, 1552, 0.9983, 0.0, 4.53),
    128: (5624, 6169, 0.9982, 0.0, 4.98),
}


@pytest.mark.parametrize("passes", sorted(LONG_WALK))
def test_the_long_walk_does_not_compound(passes):
    """THE LONG WALK, AND THE CLAMPS UNDER IT. A redrawn cell enters the context of every
    later pass, thus one model runs at 8, 32 and 128 passes and a cumulative error would
    show as numbers that FALL with the length. A zero clamp count is the finding that the
    format's margin holds."""
    said = drift(11, 42, passes)
    peak, cells, cosine, clamped, hottest = LONG_WALK[passes]
    assert (said.same_peak, said.cells) == (peak, cells)
    assert said.mean_cosine == pytest.approx(cosine, abs=5e-5)
    assert said.activations_clamped == clamped
    assert said.activation_peak == pytest.approx(hottest, abs=5e-3)


# The floors, calibrated on this model's own first measured minima over the CLEAN trials,
# the rule the sibling gates were set by: a fail is a break of the scheme and not a
# re-draw of the set. The first measurement read 0.869, 0.806 and 0.9907.
FLOORS = Shares(top1=0.80, same_draw=0.70, cosine=0.985)

PAIRS = drawn_pairs(0xD21F8, 1_000_001, 60)


# Era four: the step-frame transformer


# THE FEEDBACK AXIS OF THIS ERA IS THE KV RING: a coarsened row is read back by every
# later step, thus an error lives a whole window. The walk runs 40 steps over a window of
# 16, so that every trial wraps it.

TRANSFORMER_SHAPE = {"d": 16, "layers": 2}
TRANSFORMER_HEADS = 4
TRANSFORMER_CONTEXT = 16
TRANSFORMER_STEPS = 40


def transformer_drift(weight_seed, walk_seed):
    """the drift of one drawn model on one walk"""
    return transformer_twin.drift(
        drawn_transformer(weight_seed, heads=TRANSFORMER_HEADS, **TRANSFORMER_SHAPE),
        context=TRANSFORMER_CONTEXT,
        steps=TRANSFORMER_STEPS,
        seed=walk_seed,
    )


# the sweep row of this era, as `SWEPT` above states the columns
TRANSFORMER_SWEPT = {
    11: (358, 379, 384, 0.9975),
    23: (370, 380, 384, 0.9981),
    37: (356, 383, 384, 0.9968),
    41: (352, 382, 384, 0.9972),
}


# THE FLOORS ARE THE ERA'S OWN AND THEY ARE NOT TIGHTENED: the measured minima read
# 0.823, 0.958 and 0.9951, far above them, where the OCaml gate read 0.833, 0.948 and
# 0.9943. A floor tightened onto a measurement turns a re-draw of the set into a failure;
# the printed minima keep the calibration honest.
TRANSFORMER_FLOORS = Shares(top1=0.55, same_draw=0.80, cosine=0.98)

TRANSFORMER_PAIRS = drawn_pairs(7, 1 << 20, 40)


# Era five: the state-space model


# THE FEEDBACK AXIS OF THIS ERA IS THE STATE: a block carries a state no window forgets,
# thus an error accumulates in a register rather than dying with a ring's depth. Both
# models take one step for one step, thus the walk can run past many decay lifetimes.

# The whole plan of the era at a shape a test can afford. The Zamba head brings a SECOND
# source of drift -- a coarse ring, a softmax and a division -- thus the report answers
# for the whole model and not the recurrence alone.
MAMBA_SPELT = "MMZF"
# d 32 over 2 heads and not 16: the attention head width is then a power of FOUR, which
# `ar_quantized.score_shift` needs. The table below was re-measured when `check_shape`
# gained that rule -- the numbers before it were taken at head width 8, where the twin
# scaled by 1/2 and the float model by 1/sqrt(8).
MAMBA_SHAPE = {"d": 32, "heads": 2, "state": 8, "taps": 4}
MAMBA_RING = 16
MAMBA_STEPS = 64


def mamba_drift(weight_seed, walk_seed, steps=MAMBA_STEPS):
    """the drift of one drawn model on one walk"""
    return mamba_twin.drift(
        plan_of(MAMBA_SPELT, seed=weight_seed, **MAMBA_SHAPE),
        steps=steps,
        seed=walk_seed,
        ring=MAMBA_RING,
    )


# the sweep row of this era, as `SWEPT` above states the columns
MAMBA_SWEPT = {
    11: (714, 753, 768, 0.9980),
    23: (712, 760, 768, 0.9980),
    37: (714, 754, 768, 0.9981),
    41: (713, 759, 768, 0.9980),
}


# at 64, 256 and 1024 steps of one model: the top-1 count, the draws and the mean cosine.
# The clamps are NOT a column: the test asserts all three of them at zero outright, which
# is the finding, and a table entry would only restate it three times.
MAMBA_LONG_WALK = {
    64: (183, 192, 0.9980),
    256: (893, 960, 0.9982),
    1024: (3735, 4032, 0.9982),
}


@pytest.mark.parametrize("steps", sorted(MAMBA_LONG_WALK))
def test_the_mamba_long_walk_does_not_compound(steps):
    """THE LONG WALK, AND THE CLAMPS UNDER IT. The state of a block carries forward for
    ever, thus one model runs at 64, 256 and 1024 steps and a cumulative error would
    show as numbers that FALL with the length. They do not -- the top-1 share reads
    0.953, 0.930 and 0.926 -- and a zero clamp count is the finding that the margin
    holds."""
    said = mamba_drift(11, 42, steps=steps)
    peak, draws, cosine = MAMBA_LONG_WALK[steps]
    assert (said.same_peak, said.draws) == (peak, draws)
    assert said.mean_cosine == pytest.approx(cosine, abs=5e-5)
    clamps = said.clamps
    assert (clamps.dt, clamps.beta, clamps.state) == (0, 0, 0)
    assert clamps.dt_seen and clamps.beta_seen and clamps.state_seen


# THE FLOORS ARE THE ERA'S OWN AND THEY ARE NOT TIGHTENED. They are much tighter than era
# four's 0.55, 0.8 and 0.98 for a reason that is a FORMAT and not a virtue: this datapath
# keeps the gate product whole into the norm, where a truncation back to the working class
# cost 0.10 of the cosine on its own. The measured minima read 0.875, 0.969 and 0.9980,
# where the OCaml gate read 0.875, 0.979 and 0.9972.
MAMBA_FLOORS = Shares(top1=0.80, same_draw=0.90, cosine=0.99)

MAMBA_PAIRS = drawn_pairs(7, 1 << 20, 12)


# The two gates of the three eras: one body each


class Sweep(NamedTuple):
    """One era's fixed sweep, as the gate below runs it. [drift] takes a weight seed and a
    walk seed; [counted] names the report field that says what the comparison ran over,
    which is the ONE thing that parts the three bodies; [table] is the pinned row."""

    era: str
    drift: object
    counted: str
    table: dict


SWEEPS = (
    Sweep("six", sweep_drift, "cells", SWEPT),
    # Eras four and five were measured on THIS side and not carried over from the OCaml
    # gates: the drawn weights come from a JAX draw, thus these are a re-measurement of
    # the same scheme on a different draw.
    Sweep("four", transformer_drift, "draws", TRANSFORMER_SWEPT),
    Sweep("five", mamba_drift, "draws", MAMBA_SWEPT),
)


@pytest.mark.parametrize("weight_seed", WEIGHT_SEEDS)
@pytest.mark.parametrize("sweep", SWEEPS, ids=[sweep.era for sweep in SWEEPS])
def test_a_sweep_states_its_measured_numbers(sweep, weight_seed):
    """MEASURED NUMBERS AND NOT THRESHOLDS: a diff here says the integers moved, and the
    reader judges whether it is a re-measurement or a bug. The three eras run ONE body --
    four weight seeds over four walk seeds, summed, against a pinned row."""
    counted = same_peak = same_draw = 0
    low_cosine = 1.0
    for walk_seed in WALK_SEEDS:
        said = sweep.drift(weight_seed, walk_seed)
        counted += getattr(said, sweep.counted)
        same_peak += said.same_peak
        same_draw += said.same_draw
        low_cosine = min(low_cosine, said.mean_cosine)
    peak, draw, over, cosine = sweep.table[weight_seed]
    assert (same_peak, same_draw, counted) == (peak, draw, over)
    assert low_cosine == pytest.approx(cosine, abs=5e-5)


class Trials(NamedTuple):
    """One era's floor gate over drawn seed pairs, as the body below runs it. [drift] and
    [counted] are the sweep's; [pairs] is the era's own fixed set, whose length is its
    trial count; [floors] are the shares every clean trial must clear.

    [releases] IS ERA SIX'S ALONE, and it is the second of the two things that part the
    three bodies: a drawn trunk can outgrow any fixed format, thus a trial whose clamps
    fired is the format's answer and not the scheme's fault. The two frozen eras report
    no such share, and their gates hold every trial to the floors."""

    era: str
    drift: object
    counted: str
    pairs: tuple
    floors: Shares
    releases: bool = False


PROPERTIES = (
    Trials("six", trial_drift, "cells", PAIRS, FLOORS, releases=True),
    Trials("four", transformer_drift, "draws", TRANSFORMER_PAIRS, TRANSFORMER_FLOORS),
    Trials("five", mamba_drift, "draws", MAMBA_PAIRS, MAMBA_FLOORS),
)


@pytest.mark.parametrize("trials", PROPERTIES, ids=[era.era for era in PROPERTIES])
def test_the_floors_hold_on_drawn_seed_pairs(trials, capsys):
    """THE SCHEME AGAINST A SET OF DRAWN MODELS, not the four weight seeds the sweep pins:
    a fail here is a break of the scheme and not a re-draw of the set, thus no floor is
    ever tightened onto a measurement. At Q6 no trial of era six's set clamps."""
    low = Shares(1.0, 1.0, 1.0)
    released = 0
    for weight_seed, walk_seed in trials.pairs:
        said = trials.drift(weight_seed, walk_seed)
        if trials.releases and said.activations_clamped > 0.001:
            released += 1
            continue
        low = lowest(low, shares_of(said, trials.counted))
    # the minima print BEFORE the floors: they are what the reader needs at the moment a
    # floor breaks, and an assert inside the loop loses them
    with capsys.disabled():
        clamped = f", {released} released by their clamps" if trials.releases else ""
        print(
            f"\nera {trials.era}: {len(trials.pairs)} drawn seed pairs{clamped}: "
            f"low top-1 {low.top1:.3f}  low same draw {low.same_draw:.3f}  "
            f"low cosine {low.cosine:.4f}"
        )
    # STRICTLY ABOVE, and for all three: the two frozen eras compared with >= before the
    # three bodies became one. Nothing observable rides on it -- the measured minima stand
    # 0.02 to 0.27 clear of their floors, and no float lands on one.
    assert low.top1 > trials.floors.top1
    assert low.same_draw > trials.floors.same_draw
    assert low.cosine > trials.floors.cosine
