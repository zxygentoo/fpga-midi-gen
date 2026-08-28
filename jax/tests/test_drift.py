"""What the quantization costs: each integer twin against the float model it quantizes.

Era six holds the file and the frozen eras stand at its foot. The shape of the gate is one
shape -- a fixed sweep that pins MEASURED NUMBERS AND NOT THRESHOLDS, and a property part
that draws seed pairs at a fixed generator and holds floors calibrated under the first
measured minima -- and what differs is the feedback axis each era carries.

`Coconet.drawn` and the quantization inside `quantized.drift` read the same draw, thus the
comparison isolates the fixed-point scheme and the sweep reads no file that git ignores.
The randomness is pseudo-randomness with the seed an input, per the project rule, thus
both parts are deterministic.

THE FEEDBACK AXIS OF THIS ERA IS THE WALK, and it is what parts this gate from era five's.
That model held a state that carried an error forward in time; this one holds a sheet --
every cell a pass redraws stands in the context of every later pass, thus an arithmetic
error compounds through the music rather than through a register. The fixed sweep therefore
runs the walk out to 128 passes beside the short ones, which is a quarter of the board's
full budget at a quarter of its sheet.

THE DRAWN WEIGHTS TAKE THE TRAINED NORM'S SCALE, which is `Coconet.drawn`'s own default and
its docstring's argument: at the trainer's opening tenth an untrained trunk decays tenfold
at every layer, and by the third the report reads the resolution floor of the format and
not the arithmetic.

Two parts, the rule of the sibling gates. The fixed sweep pins MEASURED NUMBERS AND NOT
THRESHOLDS: a diff says the integers moved -- judge whether it is a re-measurement or a
bug. The property part draws seed pairs at a fixed generator and holds the floors; the
printed minima keep the calibration honest.
"""

import numpy as np
import pytest

import nn
from diffusion import model, quantized
from mamba import quantized as mamba_twin
from tests.test_mamba import plan_of
from tests.test_transformer import drawn as transformer_model
from transformer import quantized as transformer_twin

# the structure of the era at a shape a test can afford: the stem, two residual pairs and
# the head, over a quarter of the board's sheet -- two measures
LAYERS = 6
WIDTH = 8
STEPS = 32

WEIGHT_SEEDS = (11, 23, 37, 41)
WALK_SEEDS = (42, 43, 44, 45)


def drift(weight_seed, walk_seed, passes):
    """the drift of one drawn model on one walk"""
    coconet = model.Coconet.drawn(weight_seed, LAYERS, WIDTH)
    states, given = model.opening_sheet(nn.engine_states([walk_seed]), STEPS)
    return quantized.drift(coconet, states, given, walk=passes)


# for each weight seed, summed over the four walks: the top-1 count, the same-draw count,
# the cells they were counted over, and the lowest mean cosine of the four. The sharpest
# cosine signal is the lowest walk.
SWEPT = {
    11: (3029, 3040, 3299, 0.9984),
    23: (3104, 3033, 3299, 0.9984),
    37: (3095, 3086, 3299, 0.9982),
    41: (3083, 3127, 3299, 0.9960),
}


@pytest.mark.parametrize("weight_seed", WEIGHT_SEEDS)
def test_the_sweep_states_its_measured_numbers(weight_seed):
    """MEASURED NUMBERS AND NOT THRESHOLDS, the rule of the drift gates: a diff here says
    the integers moved."""
    cells = same_peak = same_draw = 0
    low_cosine = 1.0
    for walk_seed in WALK_SEEDS:
        said = drift(weight_seed, walk_seed, 16)
        cells += said.cells
        same_peak += said.same_peak
        same_draw += said.same_draw
        low_cosine = min(low_cosine, said.mean_cosine)
    wanted_peak, wanted_draw, wanted_cells, wanted_cosine = SWEPT[weight_seed]
    assert (same_peak, same_draw, cells) == (wanted_peak, wanted_draw, wanted_cells)
    assert low_cosine == pytest.approx(wanted_cosine, abs=5e-5)


# at 8, 32 and 128 passes of one model: the top-1 count, the cells, the mean cosine, the
# share of activation writes that rode the clamp, and the hottest write in real units
LONG_WALK = {
    8: (411, 445, 0.9982, 0.0, 4.58),
    32: (1436, 1552, 0.9983, 0.0, 4.53),
    128: (5624, 6169, 0.9982, 0.0, 4.98),
}


@pytest.mark.parametrize("passes", sorted(LONG_WALK))
def test_the_long_walk_does_not_compound(passes):
    """THE LONG WALK, AND THE CLAMPS UNDER IT.

    A redrawn cell enters the context of every later pass, thus a quantization error can
    compound over the walk in a way one pass never shows. The same model runs at 8, 32 and
    128 passes; a cumulative error would show as numbers that FALL with the length. The
    clamps stand beside them because the formats were chosen with margin and not metered on
    a trained checkpoint: a zero here is the finding that the margin holds."""
    said = drift(11, 42, passes)
    peak, cells, cosine, clamped, hottest = LONG_WALK[passes]
    assert (said.same_peak, said.cells) == (peak, cells)
    assert said.mean_cosine == pytest.approx(cosine, abs=5e-5)
    assert said.activations_clamped == clamped
    assert said.activation_peak == pytest.approx(hottest, abs=5e-3)


# The floors, calibrated on this model's own first measured minima over the CLEAN trials,
# the rule the sibling gates were set by: a fail is a break of the scheme and not a re-draw
# of the set. The first measurement read 0.869, 0.806 and 0.9907.
TOP1_FLOOR = 0.80
SAME_DRAW_FLOOR = 0.70
COSINE_FLOOR = 0.985

TRIALS = 60


def test_the_floors_hold_on_drawn_seed_pairs(capsys):
    """A TRIAL THAT CLAMPS IS THE FORMAT'S ANSWER AND NOT THE SCHEME'S FAULT.

    A drawn trunk can outgrow any fixed format, thus a trial whose clamps fired is counted
    and released from the floors, and a trial that does not clamp has no excuse -- the
    floors still hold the arithmetic. At Q6 no drawn trial of this sweep clamps; the release
    guarded three at the retired Q12, where one pair clamped 5.5 percent of its writes and
    read a cosine of 0.87."""
    rng = np.random.default_rng(0xD21F8)
    low_top1 = low_draw = low_cosine = 1.0
    released = 0
    for _ in range(TRIALS):
        weight_seed, walk_seed = (int(v) for v in rng.integers(1, 1_000_001, 2))
        said = drift(weight_seed, walk_seed, 8)
        if said.activations_clamped > 0.001:
            released += 1
            continue
        top1, draw = said.same_peak / said.cells, said.same_draw / said.cells
        low_top1, low_draw = min(low_top1, top1), min(low_draw, draw)
        low_cosine = min(low_cosine, said.mean_cosine)
        assert top1 > TOP1_FLOOR and draw > SAME_DRAW_FLOOR, (
            f"weights {weight_seed}, walk {walk_seed}: top-1 {top1:.3f}, "
            f"same draw {draw:.3f}"
        )
        assert said.mean_cosine > COSINE_FLOOR, (
            f"weights {weight_seed}, walk {walk_seed}: cosine {said.mean_cosine:.4f}"
        )
    # the minima print so that the calibration stays honest: a floor that no trial comes
    # near is a floor that gates nothing
    with capsys.disabled():
        print(
            f"\n{TRIALS} drawn seed pairs, {released} released by their clamps: "
            f"low top-1 {low_top1:.3f}  low same draw {low_draw:.3f}  "
            f"low cosine {low_cosine:.4f}"
        )


# ==================================================================== #
# Era four: the step-frame transformer                                 #
# ==================================================================== #

# THE FEEDBACK AXIS OF THIS ERA IS THE KV RING. A drawn key or value row is coarsened to
# its top byte and stored, and every later step reads it back, thus an arithmetic error
# lives in the ring for a whole window rather than dying with its step. The walk therefore
# runs past the ring -- 40 steps over a window of 16 -- so that every trial wraps it.

TRANSFORMER_SHAPE = {"d": 16, "layers": 2}
TRANSFORMER_HEADS = 4
TRANSFORMER_CONTEXT = 16
TRANSFORMER_STEPS = 40


def transformer_drift(weight_seed, walk_seed):
    """the drift of one drawn model on one walk"""
    return transformer_twin.drift(
        transformer_model(weight_seed, heads=TRANSFORMER_HEADS, **TRANSFORMER_SHAPE),
        context=TRANSFORMER_CONTEXT,
        steps=TRANSFORMER_STEPS,
        seed=walk_seed,
    )


# for each weight seed, summed over the four walks: the top-1 count, the same-draw count,
# the draws they were counted over, and the lowest mean cosine of the four
TRANSFORMER_SWEPT = {
    11: (358, 379, 384, 0.9975),
    23: (370, 380, 384, 0.9981),
    37: (356, 383, 384, 0.9968),
    41: (352, 382, 384, 0.9972),
}


@pytest.mark.parametrize("weight_seed", WEIGHT_SEEDS)
def test_the_transformer_sweep_states_its_measured_numbers(weight_seed):
    """MEASURED NUMBERS AND NOT THRESHOLDS: a diff here says the integers moved.

    They were measured on this side and NOT carried over from the OCaml gate that stood
    before it: the drawn weights come from a JAX draw now, thus the numbers are a
    re-measurement of the same scheme on a different draw. The old table read 363, 355,
    361 and 358 top-1 out of the same 384 draws."""
    draws = same_peak = same_draw = 0
    low_cosine = 1.0
    for walk_seed in WALK_SEEDS:
        said = transformer_drift(weight_seed, walk_seed)
        draws += said.draws
        same_peak += said.same_peak
        same_draw += said.same_draw
        low_cosine = min(low_cosine, said.mean_cosine)
    wanted_peak, wanted_draw, wanted_draws, wanted_cosine = TRANSFORMER_SWEPT[weight_seed]
    assert (same_peak, same_draw, draws) == (wanted_peak, wanted_draw, wanted_draws)
    assert low_cosine == pytest.approx(wanted_cosine, abs=5e-5)


# THE FLOORS ARE THE ERA'S OWN AND THEY ARE NOT TIGHTENED. They were calibrated on
# 2026-08-13 against the model of the token, held for the frame, and they stand where they
# stood: the measured minima of this sweep read 0.823, 0.958 and 0.9951, far above them. A
# floor tightened onto a measurement turns a re-draw of the set into a failure, and the
# printed minima are what keeps the calibration honest instead.
TRANSFORMER_TOP1_FLOOR = 0.55
TRANSFORMER_SAME_DRAW_FLOOR = 0.80
TRANSFORMER_COSINE_FLOOR = 0.98

TRANSFORMER_TRIALS = 40


def test_the_transformer_floors_hold_on_drawn_seed_pairs(capsys):
    """The scheme against a set of drawn models, not the four the sweep pins. A fail is a
    break of the scheme and not a re-draw of the set; the printed minima keep the
    calibration honest.

    The old OCaml gate read 0.833, 0.948 and 0.9943 over 100 pairs of its own draw; this
    side reads 0.823, 0.958 and 0.9951 over 40 of a different draw. The two agree about
    what the scheme costs, which is what the re-measurement had to show."""
    generator = np.random.default_rng(7)
    low_peak = low_draw = 1.0
    low_cosine = 1.0
    for _ in range(TRANSFORMER_TRIALS):
        weight_seed, walk_seed = (int(v) for v in generator.integers(1, 1 << 20, 2))
        said = transformer_drift(weight_seed, walk_seed)
        low_peak = min(low_peak, said.same_peak / said.draws)
        low_draw = min(low_draw, said.same_draw / said.draws)
        low_cosine = min(low_cosine, said.mean_cosine)
    with capsys.disabled():
        print(
            f"\n{TRANSFORMER_TRIALS} drawn seed pairs: low top-1 {low_peak:.3f}  "
            f"low same draw {low_draw:.3f}  low cosine {low_cosine:.4f}"
        )
    assert low_peak >= TRANSFORMER_TOP1_FLOOR
    assert low_draw >= TRANSFORMER_SAME_DRAW_FLOOR
    assert low_cosine >= TRANSFORMER_COSINE_FLOOR


# ==================================================================== #
# Era five: the state-space model                                      #
# ==================================================================== #

# THE FEEDBACK AXIS OF THIS ERA IS THE STATE. A block carries a state that no window
# forgets, thus a quantization error accumulates in a register rather than dying with a
# ring's depth -- and the long walk below is what says whether it does. BOTH MODELS TAKE
# ONE STEP FOR ONE STEP here, thus the comparison is linear in the walk and can run past
# many decay lifetimes.

# The whole plan of the era at a shape a test can afford: two blocks, the Zamba head and
# the feed-forward. The head brings a SECOND source of drift that the trunk does not have
# -- a coarse ring, a softmax and a division -- thus the report answers for the whole model
# and not for the recurrence alone.
MAMBA_SPELT = "MMZF"
MAMBA_SHAPE = {"d": 16, "heads": 2, "state": 8, "taps": 4}
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


# for each weight seed, summed over the four walks: the top-1 count, the same-draw count,
# the draws they were counted over, and the lowest mean cosine of the four
MAMBA_SWEPT = {
    11: (723, 765, 768, 0.9976),
    23: (704, 761, 768, 0.9962),
    37: (709, 751, 768, 0.9962),
    41: (706, 762, 768, 0.9967),
}


@pytest.mark.parametrize("weight_seed", WEIGHT_SEEDS)
def test_the_mamba_sweep_states_its_measured_numbers(weight_seed):
    """MEASURED NUMBERS AND NOT THRESHOLDS: a diff here says the integers moved.

    They were measured on this side and NOT carried over from the OCaml gate that stood
    before it: the drawn weights come from a JAX draw now. The old table read 721, 718, 711
    and 739 top-1 out of the same 768 draws."""
    draws = same_peak = same_draw = 0
    low_cosine = 1.0
    for walk_seed in WALK_SEEDS:
        said = mamba_drift(weight_seed, walk_seed)
        draws += said.draws
        same_peak += said.same_peak
        same_draw += said.same_draw
        low_cosine = min(low_cosine, said.mean_cosine)
    wanted_peak, wanted_draw, wanted_draws, wanted_cosine = MAMBA_SWEPT[weight_seed]
    assert (same_peak, same_draw, draws) == (wanted_peak, wanted_draw, wanted_draws)
    assert low_cosine == pytest.approx(wanted_cosine, abs=5e-5)


# at 64, 256 and 1024 steps of one model: the top-1 count, the draws, the mean cosine, and
# the share of each clamp that fired
MAMBA_LONG_WALK = {
    64: (178, 192, 0.9976),
    256: (886, 960, 0.9979),
    1024: (3736, 4032, 0.9977),
}


@pytest.mark.parametrize("steps", sorted(MAMBA_LONG_WALK))
def test_the_mamba_long_walk_does_not_compound(steps):
    """THE LONG WALK, AND THE CLAMPS UNDER IT.

    The state of a block carries forward for ever, thus a quantization error can compound
    over a walk in a way one step never shows. The same model runs at 64, 256 and 1024
    steps; a cumulative error would show as numbers that FALL with the length. They do not:
    the top-1 share reads 0.927, 0.923 and 0.927 and the cosine stands flat.

    The clamps stand beside them because the formats of this era are chosen with margin and
    not metered on a trained checkpoint: a zero here is the finding that the margin holds,
    and it is the finding the OCaml gate made before it."""
    said = mamba_drift(11, 42, steps=steps)
    peak, draws, cosine = MAMBA_LONG_WALK[steps]
    assert (said.same_peak, said.draws) == (peak, draws)
    assert said.mean_cosine == pytest.approx(cosine, abs=5e-5)
    clamps = said.clamps
    assert (clamps.dt, clamps.beta, clamps.state) == (0, 0, 0)
    assert clamps.dt_seen and clamps.beta_seen and clamps.state_seen


# THE FLOORS ARE THE ERA'S OWN AND THEY ARE NOT TIGHTENED. They were calibrated on
# 2026-08-20 against this model's own first measured minima, and they are much tighter than
# era four's 0.55, 0.8 and 0.98 for a reason that is a format and not a virtue: this
# datapath keeps the gate product whole into the norm that reads it, where a truncation
# back to the working class cost 0.10 of the cosine on its own. A scheme that measures this
# well must be held to it.
MAMBA_TOP1_FLOOR = 0.80
MAMBA_SAME_DRAW_FLOOR = 0.90
MAMBA_COSINE_FLOOR = 0.99

MAMBA_TRIALS = 12


def test_the_mamba_floors_hold_on_drawn_seed_pairs(capsys):
    """The scheme against a set of drawn models, not the four the sweep pins. A fail is a
    break of the scheme and not a re-draw of the set; the printed minima keep the
    calibration honest.

    The old OCaml gate read 0.875, 0.979 and 0.9972 over 60 pairs of its own draw."""
    generator = np.random.default_rng(7)
    low_peak = low_draw = low_cosine = 1.0
    for _ in range(MAMBA_TRIALS):
        weight_seed, walk_seed = (int(v) for v in generator.integers(1, 1 << 20, 2))
        said = mamba_drift(weight_seed, walk_seed)
        low_peak = min(low_peak, said.same_peak / said.draws)
        low_draw = min(low_draw, said.same_draw / said.draws)
        low_cosine = min(low_cosine, said.mean_cosine)
    with capsys.disabled():
        print(
            f"\n{MAMBA_TRIALS} drawn seed pairs: low top-1 {low_peak:.3f}  "
            f"low same draw {low_draw:.3f}  low cosine {low_cosine:.4f}"
        )
    assert low_peak >= MAMBA_TOP1_FLOOR
    assert low_draw >= MAMBA_SAME_DRAW_FLOOR
    assert low_cosine >= MAMBA_COSINE_FLOOR
