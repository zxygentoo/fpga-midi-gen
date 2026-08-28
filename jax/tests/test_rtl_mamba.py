"""The RTL gates of era five: the circuit against the integer twin.

THE ORACLE IS THIS SIDE AND THE CIRCUIT IS THE OTHER. `bin/gate_mamba.exe` drives the
Hardcaml circuit in Cyclesim and prints WHAT IT DID -- the frame the socket face answered
at each step, and every write of the whole residual stream in the order the machine made
them -- and this module states what it must have done, from `mamba/quantized.py` over the
same model, and compares in order. Neither side can pass by agreeing with itself.

The model crosses the seam as a CONTRACT FILE. A tiny model is drawn here, quantized here
and written to a `tmp_path`; the driver reads it and elaborates a circuit from it. EVERY
WIDTH AND THE PLAN travel in that file, thus no flag of this module states a shape.

TWO GATES, AND THE SECOND IS THE ONE THAT FOUND THE FAULTS. Era five's four faults were
all faults of the composition layer -- a weight address whose stride was not the tensor's,
a channel block read at the gate's offset, an operand taken on the address side of a
two-cycle read, and a ring run off its end -- and NONE OF THEM MOVED A FRAME. The stream
gate compares the residual stream after the embed and after every layer, thus a
disagreement names the layer it began in.

The shapes are the ones the expect tests of `source.ml` ran, and each was put there by a
fault: the whole plan at one of each kind, where the region field of every memory is EMPTY
and an address that strided by it would still pass; two blocks and two attention layers,
where both fields appear; three blocks under a plan that interleaves them, where the tap
ring's layer stride runs the top block off the end of its memory if the stride is wrong;
and a wide state and kernel, where a stride written for one K and an address field written
for one N both land.

It SKIPS when the driver is absent -- a clean tree is not a failure. From the repository
root:

    dune build bin/gate_mamba.exe
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from mamba import quantized
from tests.test_mamba import plan_of

ROOT = Path(__file__).resolve().parent.parent.parent
DRIVER = ROOT / "_build" / "default" / "bin" / "gate_mamba.exe"


def drive(subcommand, path, *, seed, steps):
    """the lines the driver states, each as a list of its words"""
    if not DRIVER.exists():
        pytest.skip(f"absent, nothing to gate: {DRIVER.name}")
    done = subprocess.run(
        [
            str(DRIVER),
            subcommand,
            "-int8",
            str(path),
            "-seed",
            str(seed),
            "-steps",
            str(steps),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return [line.split() for line in done.stdout.splitlines()]


def contract(tmp_path, spelt, *, ring=8, **shape):
    """a tiny model of the plan spelt out, drawn here and quantized here, as the file the
    driver reads"""
    twin = quantized.Quantized.of(plan_of(spelt, **shape), ring=ring)
    path = tmp_path / "tiny.int8"
    quantized.save(path, twin)
    return path, quantized.load(path)


@pytest.mark.parametrize(
    "spelt,seed,shape",
    [
        # the whole plan at one layer of each kind, at a live seed and at the standing one
        ("MZF", 42, {}),
        ("MZF", 0, {}),
        # two blocks and two attention layers: both region fields appear, and a memory that
        # ignored one would read another layer's state or another layer's keys
        ("MMZZ", 42, {}),
    ],
)
def test_the_walk_of_the_circuit_is_the_walk_of_the_twin(tmp_path, spelt, seed, shape):
    """CLASS FOR CLASS, STEP FOR STEP. The chain draws the four seats from the soprano
    down; the recurrence carries a state that no window forgets, thus an error in it lives
    for the whole walk and not for one ring's depth."""
    steps = 20
    path, twin = contract(tmp_path, spelt, **shape)
    lines = drive("walk", path, seed=seed, steps=steps)
    circuit = np.array([[int(w) for w in line[3:]] for line in lines if line[0] == "step"])
    assert len(circuit) == steps, f"the driver stated {len(circuit)} steps"
    played, _ = quantized.walk(twin, [seed], steps)
    parted = np.flatnonzero(~(circuit == played[0]).all(axis=-1))
    assert not len(parted), (
        f"the walks part at step {parted[0]}: the circuit drew "
        f"{list(circuit[parted[0]])} and the twin wants {list(played[0][parted[0]])}"
    )


@pytest.mark.parametrize(
    "spelt,seed,shape",
    [
        # the whole plan, at the shape the frame gate runs
        ("MZF", 42, {}),
        # three blocks the plan interleaves: at one block the tap ring's layer field is
        # absent, at two the top block still fits its memory by an accident of rounding,
        # and at three it runs off the end if the stride is wrong
        ("MMZMZF", 7, {"d": 32, "heads": 4, "state": 16}),
        # a wide state and a wide kernel: a stride written for one K and an address field
        # written for one N both land here
        ("MMM", 9, {"state": 32, "taps": 16}),
    ],
)
def test_the_stream_writes_of_the_circuit_are_the_twins(tmp_path, spelt, seed, shape):
    """EVERY WRITE OF THE WHOLE STREAM, IN ORDER: the embed, then the join of each layer.

    A frame gate that fails says only THAT the circuit and the twin parted; this says
    where."""
    steps = 6
    path, twin = contract(tmp_path, spelt, **shape)
    lines = drive("stream", path, seed=seed, steps=steps)
    got = {}
    for word in lines:
        if word[0] == "write":
            got.setdefault(int(word[1]), []).append([int(v) for v in word[3:]])
    want = quantized.streams(twin, [seed], steps)
    assert sorted(got) == list(range(steps)), "the driver skipped a step"
    checked = 0
    for step in sorted(got):
        assert len(got[step]) == len(want[step]), (
            f"step {step}: the circuit wrote {len(got[step])} streams and the twin wants "
            f"{len(want[step])}"
        )
        for at, (made, wanted) in enumerate(zip(got[step], want[step])):
            parted = int((np.array(made) != wanted[0]).sum())
            assert not parted, (
                f"step {step}, stream write {at}: {parted} of {len(made)} elements part"
            )
            checked += 1
    assert checked == steps * (len(twin.plan) + 1)


def test_the_lead_in_draws_nothing_and_moves_no_generator(tmp_path):
    """One bar of silence opens the walk and the generator does not move through it. A twin
    that spent a uniform there would draw a different piece from the same seed, and every
    step of it would be legal music."""
    _, twin = contract(tmp_path, "MZF")
    played, draws = quantized.walk(twin, [1, 7], quantized.LEAD + 2)
    assert (played[:, : quantized.LEAD] == 0).all(), "the lead-in is not silent"
    assert all(not taken for taken in draws[: quantized.LEAD]), "the lead-in drew"
    # the walks of a batch are independent: seed 7 draws what seed 7 draws alone
    alone, _ = quantized.walk(twin, [7], quantized.LEAD + 2)
    assert np.array_equal(alone[0], played[1])
