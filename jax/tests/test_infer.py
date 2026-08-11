"""The draw of infer.py must be the draw of the OCaml sampler.

[temper] and [pick] are the two places a rewrite can be plausibly wrong and still make
music: a peak taken over all codes instead of the legal ones, a min-p floor applied
before the temperature, an inclusive compare in the cumulative walk. Each shifts the
distribution a little and nothing raises.

The whole-stream proof is `infer.py --gate`, which needs the OCaml binary and a
checkpoint; the last test runs it when both are present and skips when they are not.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

import data
from transformer import infer

JAX_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = JAX_ROOT.parent / "_train" / "d64-mk-do01-48k-s4-prog.ckpt"
PLAYER = JAX_ROOT.parent / "_build" / "default" / "bin" / "play_transformer.exe"


def test_temper_measures_the_peak_over_the_legal_set_only():
    """an illegal code may hold the largest logit; the peak must ignore it, or every
    legal weight shrinks and min-p cuts music the model wanted"""
    raw = np.array([[10.0, 1.0, 2.0]])
    legal = np.array([[False, True, True]])
    weights = infer.temper(raw, legal, temperature=1.0, min_p=0.0)
    assert weights[0, 0] == 0.0
    assert weights[0, 2] == pytest.approx(1.0)  # the legal peak weighs one
    assert weights[0, 1] == pytest.approx(np.exp(-1.0))


def test_min_p_is_a_share_of_the_peak_after_the_temperature():
    raw = np.array([[0.0, -3.0, -8.0]])
    legal = np.ones((1, 3), dtype=bool)
    weights = infer.temper(raw, legal, temperature=1.0, min_p=0.01)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(np.exp(-3.0))  # 0.0498, above the floor
    assert weights[0, 2] == 0.0  # 0.000335, below it and cut
    # the floor is a share of the peak, thus raising it cuts more
    raised = infer.temper(raw, legal, temperature=1.0, min_p=0.1)
    assert raised[0, 1] == 0.0


def test_temperature_flattens_and_sharpens():
    raw = np.array([[0.0, -1.0]])
    legal = np.ones((1, 2), dtype=bool)
    warm = infer.temper(raw, legal, temperature=2.0, min_p=0.0)
    cold = infer.temper(raw, legal, temperature=0.5, min_p=0.0)
    assert warm[0, 1] > cold[0, 1]


def test_pick_takes_the_first_code_whose_total_passes_the_draw():
    weights = np.zeros((1, data.VOCAB))
    weights[0, 3] = 0.5
    weights[0, 9] = 0.5
    assert infer.pick(weights, np.array([0.25]))[0] == 3
    assert infer.pick(weights, np.array([0.75]))[0] == 9


def test_pick_falls_to_zero_when_the_chosen_weight_is_not_positive():
    """the OCaml guard: a draw past every total lands on a code with no mass, and END is
    the only token that is always legal"""
    weights = np.zeros((1, data.VOCAB))
    weights[0, 3] = 1.0
    assert infer.pick(weights, np.array([2.0]))[0] == data.END


def test_pick_runs_each_row_of_the_batch_on_its_own():
    weights = np.zeros((2, data.VOCAB))
    weights[0, 5] = 1.0
    weights[1, 200] = 1.0
    assert list(infer.pick(weights, np.array([0.5, 0.5]))) == [5, 200]


@pytest.mark.skipif(
    not CHECKPOINT.exists() or not PLAYER.exists(),
    reason="Gate C needs _train/d64-mk-do01-48k-s4-prog.ckpt and dune build",
)
def test_gate_c_the_batched_draw_equals_the_ocaml_sampler():
    """batched on purpose: a solo run passes even when a finished element wrongly
    consumes a draw, because there is nothing queued behind it"""
    done = subprocess.run(
        [
            "python",
            "-m",
            "transformer.infer",
            "--ckpt",
            str(CHECKPOINT),
            "--seeds",
            "1-3",
            "--steps",
            "24",
            "--gate",
        ],
        capture_output=True,
        text=True,
        cwd=JAX_ROOT,
        check=False,
    )
    assert "GATE C PASSED" in done.stdout, done.stdout + done.stderr
