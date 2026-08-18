"""The arithmetic of the draw.

[temper] and [pick] are the two places a rewrite can be plausibly wrong and still make
music: a peak taken over the wrong axis, a min-p floor applied before the temperature, an
inclusive compare in the cumulative walk. Each shifts the distribution a little and
nothing raises.

No mask stands here any more. The era of the token measured its peak over the legal set
alone, because an illegal code could hold the largest logit; no frame is illegal, thus the
peak is the peak.
"""

import numpy as np
import pytest

import data
from transformer import infer


def test_the_peak_of_each_row_weighs_one():
    raw = np.array([[10.0, 9.0, 2.0]])
    weights = infer.temper(raw, temperature=1.0, min_p=0.0)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(np.exp(-1.0))


def test_min_p_is_a_share_of_the_peak_after_the_temperature():
    raw = np.array([[0.0, -3.0, -8.0]])
    weights = infer.temper(raw, temperature=1.0, min_p=0.01)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(np.exp(-3.0))  # 0.0498, above the floor
    assert weights[0, 2] == 0.0  # 0.000335, below it and cut
    # the floor is a share of the peak, thus raising it cuts more
    assert infer.temper(raw, temperature=1.0, min_p=0.1)[0, 1] == 0.0


def test_temperature_flattens_and_sharpens():
    raw = np.array([[0.0, -1.0]])
    warm = infer.temper(raw, temperature=2.0, min_p=0.0)
    cold = infer.temper(raw, temperature=0.5, min_p=0.0)
    assert warm[0, 1] > cold[0, 1]


def test_pick_takes_the_first_class_whose_total_passes_the_draw():
    weights = np.zeros((1, data.CLASSES))
    weights[0, 3] = 0.5
    weights[0, 9] = 0.5
    assert infer.pick(weights, np.array([0.25]))[0] == 3
    assert infer.pick(weights, np.array([0.75]))[0] == 9


def test_pick_falls_to_the_heaviest_class_when_no_total_passes():
    """a draw past every total is rounding, not a choice; it lands on the class that holds
    the mass and never on one the floor cut away"""
    weights = np.zeros((1, data.CLASSES))
    weights[0, 3] = 1.0
    assert infer.pick(weights, np.array([2.0]))[0] == 3


def test_pick_runs_each_row_of_the_batch_on_its_own():
    weights = np.zeros((2, data.CLASSES))
    weights[0, 5] = 1.0
    weights[1, 40] = 1.0
    assert list(infer.pick(weights, np.array([0.5, 0.5]))) == [5, 40]
