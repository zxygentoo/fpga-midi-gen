"""The RTL gates of era five: the circuit against the integer twin.

THE ORACLE IS THIS SIDE AND THE CIRCUIT IS THE OTHER. `bin/gate_mamba.exe` drives the
Hardcaml circuit in Cyclesim and prints WHAT IT DID; this module states what it must have
done, from `mamba/quantized.py` over the same model. Neither side can pass by agreeing
with itself. The model crosses the seam as a CONTRACT FILE, and every width and the plan
travel in it, thus no flag of this module states a shape.

TWO GATES, AND THE SECOND IS THE ONE THAT FOUND THE FAULTS. Era five's four faults were
all faults of the composition layer and NONE OF THEM MOVED A FRAME; the stream gate
compares the residual stream after the embed and after every layer, thus a disagreement
names the layer it began in. Each shape below was put here by one of them.

It SKIPS when the driver is absent -- a clean tree is not a failure. From the repository
root:

    dune build bin/gate_mamba.exe
"""


import numpy as np
import pytest

from mamba import quantized
from tests import gate
from tests.models import plan_of

DRIVER = gate.driver("gate_mamba.exe")


built = gate.built_fixture(DRIVER)


def drive(subcommand, path, *, seed, steps):
    """the driver's report, whole; each gate below reads the half it states"""
    return gate.run(DRIVER, subcommand, "-int8", path, "-seed", seed, "-steps", steps)


def contract(tmp_path, spelt, *, ring=8, **shape):
    """a tiny model of the plan spelt out, drawn here and quantized here, as the file the
    driver reads"""
    twin = quantized.Mamba.from_float(plan_of(spelt, **shape), ring=ring)
    path = tmp_path / "tiny.int8"
    quantized.save(path, twin)
    return path, quantized.load(path)


@pytest.mark.parametrize(
    "spelt,seed,shape",
    [
        # the whole plan at one layer of each kind, at a live seed and at the standing one
        ("MZF", 42, {}),
        ("MZF", 0, {}),
        # two blocks and two attention layers: both region fields appear, and a memory
        # that ignored one would read another layer's state or another layer's keys
        ("MMZZ", 42, {}),
    ],
)
def test_the_walk_of_the_circuit_is_the_walk_of_the_twin(tmp_path, spelt, seed, shape):
    """CLASS FOR CLASS, STEP FOR STEP. The chain draws the four seats from the soprano
    down; the recurrence carries a state that no window forgets, thus an error in it lives
    for the whole walk and not for one ring's depth."""
    steps = 20
    path, twin = contract(tmp_path, spelt, **shape)
    circuit = gate.classes_of_walk(drive("walk", path, seed=seed, steps=steps), steps)
    played, _ = quantized.walk(twin, [seed], steps)
    gate.assert_one_walk(circuit, played[0])


@pytest.mark.parametrize(
    "spelt,seed,shape",
    [
        # the whole plan, at the shape the frame gate runs
        ("MZF", 42, {}),
        # three blocks the plan interleaves: at one the tap ring's layer field is absent,
        # at two the top block still fits by an accident of rounding, and at three it runs
        # off the end if the stride is wrong. TWO heads and not four, so the attention
        # head width stays a power of four.
        ("MMZMZF", 7, {"d": 32, "heads": 2, "state": 16}),
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
    stdout = drive("stream", path, seed=seed, steps=steps)
    got = {}
    for word in (line.split() for line in stdout.splitlines() if line):
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
