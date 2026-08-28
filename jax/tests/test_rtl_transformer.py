"""The RTL gate of era four: the circuit against the integer twin.

THE ORACLE IS THIS SIDE AND THE CIRCUIT IS THE OTHER. `bin/gate_transformer.exe` drives the
Hardcaml circuit in Cyclesim and prints WHAT IT DID -- the frame the socket face answered
at each step, and the classes that frame states -- and this module states what it must have
done, from `transformer/quantized.py` over the same model, and compares in order. Neither
side can pass by agreeing with itself.

The model crosses the seam as a CONTRACT FILE. A tiny model is drawn here, quantized here
and written to a `tmp_path`; the driver reads it and elaborates a circuit from it. EVERY
SHAPE NUMBER OF THIS ERA TRAVELS IN THAT FILE -- the width and the layers in the tensors,
the heads, the context and the ALiBi span beside them -- thus no flag of this module states
a shape.

THE COMPARISON IS OVER CLASSES AND NOT OVER FRAME WORDS. The vocabulary and the seat
packing are the corpus library's rule and they stay on the OCaml side; the driver decodes
the frame it answered through `Vocab.classes_of_frame`, thus this side holds no format of
its own. `lib/corpus/vocab.ml` gates that decode against its own inverse.

The shapes are the ones the expect tests of `source.ml` ran, because each was put there by
a fault: one layer, where the ring's layer field is EMPTY and an address that strides by it
would still pass; seed 0, the fixed point of the generator, where every uniform is 0 and
each seat takes the first class the min-p floor left standing; and two layers, where the
layer field appears and a ring that ignored it would read another layer's keys.

It SKIPS when the driver is absent -- a clean tree is not a failure. From the repository
root:

    dune build bin/gate_transformer.exe
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.test_transformer import tiny
from transformer import quantized

ROOT = Path(__file__).resolve().parent.parent.parent
DRIVER = ROOT / "_build" / "default" / "bin" / "gate_transformer.exe"


def drive(path, *, seed, steps):
    """the classes the circuit drew at each step, as the driver states them"""
    if not DRIVER.exists():
        pytest.skip(f"absent, nothing to gate: {DRIVER.name}")
    done = subprocess.run(
        [
            str(DRIVER),
            "walk",
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
    lines = [line.split() for line in done.stdout.splitlines() if line.startswith("step")]
    assert len(lines) == steps, f"the driver stated {len(lines)} steps, wanted {steps}"
    return np.array([[int(word) for word in line[3:]] for line in lines])


def contract(tmp_path, **shape):
    """a tiny model, drawn here and quantized here, as the file the driver reads"""
    twin = tiny(**shape)
    path = tmp_path / "tiny.int8"
    quantized.save(path, twin)
    return path, quantized.load(path)


@pytest.mark.parametrize(
    "seed,layers,heads,d",
    [
        # the shape a test can afford, at a live seed and at the standing one
        (42, 1, 2, 8),
        (0, 1, 2, 8),
        # two layers: the ring's layer field appears, and a ring that ignored it would
        # read another layer's keys
        (42, 2, 2, 8),
        # four heads over a width of 16: the head width is 4, a power of four, and the
        # ALiBi slope of each head differs
        (7, 2, 4, 16),
    ],
)
def test_the_walk_of_the_circuit_is_the_walk_of_the_twin(tmp_path, seed, layers, heads, d):
    """CLASS FOR CLASS, STEP FOR STEP. The chain draws the four seats from the soprano
    down, each reading the stream the seats above it wrote, thus a chain wired the wrong way
    round moves every class after the first -- and the lead-in of one bar must draw nothing
    at all, or the generator stands one draw ahead of the twin's for the whole walk."""
    steps = 20
    path, twin = contract(tmp_path, seed=5, d=d, layers=layers, heads=heads)
    circuit = drive(path, seed=seed, steps=steps)
    played, _ = quantized.walk(twin, [seed], steps)
    parted = np.flatnonzero(~(circuit == played[0]).all(axis=-1))
    assert not len(parted), (
        f"the walks part at step {parted[0]}: the circuit drew "
        f"{list(circuit[parted[0]])} and the twin wants {list(played[0][parted[0]])}"
    )


def test_the_lead_in_draws_nothing_and_moves_no_generator(tmp_path):
    """One bar of silence opens the walk and the generator does not move through it. A twin
    that spent a uniform there would draw a different piece from the same seed, and every
    step of it would be legal music."""
    _, twin = contract(tmp_path, seed=5, d=8, layers=1, heads=2)
    played, draws = quantized.walk(twin, [1, 7], quantized.LEAD + 2)
    assert (played[:, : quantized.LEAD] == 0).all(), "the lead-in is not silent"
    assert all(not taken for taken in draws[: quantized.LEAD]), "the lead-in drew"
    assert all(len(taken) == 4 for taken in draws[quantized.LEAD :])
    # the walks of a batch are independent: seed 7 draws what seed 7 draws alone
    alone, _ = quantized.walk(twin, [7], quantized.LEAD + 2)
    assert np.array_equal(alone[0], played[1])
