"""What the quantization costs: the integer twin against the float model it quantizes.

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

from diffusion import infer, model, quantized

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
    states, given = infer.opening_sheet(quantized.engine_states([walk_seed]), STEPS)
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
