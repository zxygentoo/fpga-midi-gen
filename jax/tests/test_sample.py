"""The arithmetic of the draw, which is `sample.py`.

[tempered_weight] and [pick_share] are the two places a rewrite can be plausibly wrong
and still make music: a peak over the wrong axis, a min-p floor applied before the
temperature, an inclusive compare in the cumulative walk. Each shifts the distribution a
little and nothing raises. The INTEGER pick beside them is `quantized.pick`, which
`test_quantized.py` gates."""

import numpy as np
import pytest

import corpus
import sample


def test_the_peak_of_each_row_weighs_one():
    raw = np.array([[10.0, 9.0, 2.0]])
    weights = sample.tempered_weight(raw, temperature=1.0, min_p=0.0)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(np.exp(-1.0))


def test_min_p_is_a_share_of_the_peak_after_the_temperature():
    raw = np.array([[0.0, -3.0, -8.0]])
    weights = sample.tempered_weight(raw, temperature=1.0, min_p=0.01)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(np.exp(-3.0))  # 0.0498, above the floor
    assert weights[0, 2] == 0.0  # 0.000335, below it and cut
    # the floor is a share of the peak, thus raising it cuts more
    assert sample.tempered_weight(raw, temperature=1.0, min_p=0.1)[0, 1] == 0.0


def test_temperature_flattens_and_sharpens():
    raw = np.array([[0.0, -1.0]])
    warm = sample.tempered_weight(raw, temperature=2.0, min_p=0.0)
    cold = sample.tempered_weight(raw, temperature=0.5, min_p=0.0)
    assert warm[0, 1] > cold[0, 1]


def test_pick_takes_the_first_class_whose_total_passes_the_draw():
    weights = np.zeros((1, corpus.CLASSES))
    weights[0, 3] = 0.5
    weights[0, 9] = 0.5
    assert sample.pick_share(weights, np.array([0.25]))[0] == 3
    assert sample.pick_share(weights, np.array([0.75]))[0] == 9


def test_pick_holds_the_top_of_the_uniform_range():
    """The draw is the uniform times the LAST RUNNING TOTAL, thus it is strictly under
    that total and a class always passes. A draw made against a second sum -- numpy
    adds pairwise in sum() and left to right in cumsum() -- could land above every
    running total, and the pick would fall off the end onto a class the floor cut away."""
    weights = np.zeros((1, corpus.CLASSES))
    weights[0, 3] = 1.0
    assert sample.pick_share(weights, np.array([1.0 - 2.0**-24]))[0] == 3


def test_pick_runs_each_row_of_the_batch_on_its_own():
    weights = np.zeros((2, corpus.CLASSES))
    weights[0, 5] = 1.0
    weights[1, 40] = 1.0
    assert list(sample.pick_share(weights, np.array([0.5, 0.5]))) == [5, 40]
