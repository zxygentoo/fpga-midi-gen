"""The trainer must train.

This file exists because of a bug that every other check passed. Converting the CLI to
click renamed `args.lr` to `lr`, and the loop already wrote its rate back into a name of
its own -- so `lr = schedule(step, lr, ...)` fed the schedule its own output and the rate
collapsed geometrically to zero. The model initialised correctly, the shapes were right,
the parity gate passed, and a one-step run reported the exact baseline loss, because a
step-1 loss is measured before the first update. Nothing was wrong except that no
learning happened.

Therefore: run the loop and watch the loss fall. A smoke test at d 8 costs seconds.
"""

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from transformer import train

JAX_ROOT = Path(__file__).resolve().parent.parent
CORPUS = JAX_ROOT / "_data" / "corpus.safetensors"


def test_the_schedule_holds_its_peak():
    """the peak is an input and never moves; a caller that overwrites it is the bug this
    file guards, and this pins the half that lives in the schedule"""
    peak = 1e-3
    rates = [train.schedule(step, peak, 300, 48000) for step in range(1, 1000)]
    assert max(rates) == pytest.approx(peak)
    assert rates[299] == pytest.approx(peak)  # the warmup ends exactly on the peak
    assert rates[-1] > peak * 0.9  # and the cosine has barely begun to fall


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
