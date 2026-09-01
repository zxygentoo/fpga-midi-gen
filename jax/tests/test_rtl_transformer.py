"""The RTL gate of era four: the circuit against the integer twin.

THE ORACLE IS THIS SIDE AND THE CIRCUIT IS THE OTHER. `bin/gate_transformer.exe` drives
the Hardcaml circuit in Cyclesim and prints WHAT IT DID; this module states what it must
have done, from `transformer/quantized.py` over the same model. Neither side can pass by
agreeing with itself. The model crosses the seam as a CONTRACT FILE and every shape number
of the era travels in it, thus no flag of this module states a shape.

THE COMPARISON IS OVER CLASSES AND NOT OVER FRAME WORDS: the vocabulary and the seat
packing stay on the OCaml side, and the driver decodes through `Vocab.classes_of_frame`.
Each shape below was put here by a fault -- one layer, where the ring's layer field is
EMPTY; seed 0, where every uniform is 0; and two layers, where the field appears and a
ring that ignored it would read another layer's keys.

It SKIPS when the driver is absent -- a clean tree is not a failure. From the repository
root:

    dune build bin/gate_transformer.exe
"""


import pytest

from tests import gate
from tests.models import transformer_twin
from transformer import quantized

DRIVER = gate.driver("gate_transformer.exe")


@pytest.fixture(scope="module", autouse=True)
def built():
    """THE SKIP STANDS BEFORE THE WORK AND NOT INSIDE IT: inside `drive`, a tree with no
    `dune build` behind it drew and quantized a model for every case before skipping on
    each."""
    gate.need(DRIVER)


def drive(path, *, seed, steps):
    """the classes the circuit drew at each step, as the driver states them"""
    stdout = gate.run(DRIVER, "walk", "-int8", path, "-seed", seed, "-steps", steps)
    return gate.classes_of_walk(stdout, steps)


def contract(tmp_path, **shape):
    """a tiny model, drawn here and quantized here, as the file the driver reads"""
    twin = transformer_twin(**shape)
    path = tmp_path / "tiny.int8"
    quantized.save(path, twin)
    return path, quantized.load(path)


@pytest.mark.parametrize(
    "seed,layers,heads,d",
    [
        # the shape a test can afford, at a live seed and at the standing one
        (42, 1, 2, 8),
        # seed 0 is the fixed point of xorshift32 and the panel can state it -- all the
        # slide switches down is the rest position of the board. The drawn weights leave
        # class 0 standing at every seat, thus the walk plays nothing after the lead-in,
        # and the BOARD answers the same with the trained model in flash (measured
        # 2026-08-19): all the switches down is a silent board.
        (0, 1, 2, 8),
        # two layers: the ring's layer field appears, and a ring that ignored it would
        # read another layer's keys
        (42, 2, 2, 8),
        # four heads over a width of 16: the head width is 4, a power of four, and the
        # ALiBi slope of each head differs
        (7, 2, 4, 16),
    ],
)
def test_the_walk_of_the_circuit_is_the_walk_of_the_twin(
    tmp_path, seed, layers, heads, d
):
    """CLASS FOR CLASS, STEP FOR STEP: a chain wired the wrong way round moves every class
    after the first, and a lead-in that drew would stand the generator one draw ahead of
    the twin's for the whole walk."""
    steps = 20
    path, twin = contract(tmp_path, seed=5, d=d, layers=layers, heads=heads)
    circuit = drive(path, seed=seed, steps=steps)
    played, _ = quantized.walk(twin, [seed], steps)
    gate.assert_one_walk(circuit, played[0])
