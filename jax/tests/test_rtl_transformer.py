"""The RTL gate of era four: the circuit against the integer twin.

THE ORACLE IS THIS SIDE AND THE CIRCUIT IS THE OTHER. `bin/gate_transformer.exe` drives
the Hardcaml circuit in Cyclesim and prints WHAT IT DID -- the frame the socket face
answered at each step, and the classes that frame states -- and this module states what it
must have done, from `transformer/quantized.py` over the same model, and compares in
order. Neither side can pass by agreeing with itself.

The model crosses the seam as a CONTRACT FILE. A tiny model is drawn here, quantized here
and written to a `tmp_path`; the driver reads it and elaborates a circuit from it. EVERY
SHAPE NUMBER OF THIS ERA TRAVELS IN THAT FILE -- the width and the layers in the tensors,
the heads, the context and the ALiBi span beside them -- thus no flag of this module
states a shape.

THE COMPARISON IS OVER CLASSES AND NOT OVER FRAME WORDS. The vocabulary and the seat
packing are the corpus library's rule and they stay on the OCaml side; the driver decodes
the frame it answered through `Vocab.classes_of_frame`, thus this side holds no format of
its own. `lib/corpus/vocab.ml` gates that decode against its own inverse.

The shapes are the ones the frame benches of `source.ml` ran until the all-era cut moved
the gate here, and each was put there by a fault: one layer, where the ring's layer field
is EMPTY and an address that strides by it would still pass; seed 0, the fixed point of
the generator, where every uniform is 0 and each seat takes the first class the min-p
floor left standing; and two layers, where the layer field appears and a ring that ignored
it would read another layer's keys.

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
    """THE SKIP STANDS BEFORE THE WORK AND NOT INSIDE IT. `gate.need` used to run inside
    `drive`, thus a tree with no `dune build` behind it drew, quantized and wrote a model
    for every case of this file before skipping on each. Module scope asks once."""
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
    """CLASS FOR CLASS, STEP FOR STEP. The chain draws the four seats from the soprano
    down, each reading the stream the seats above it wrote, thus a chain wired the wrong
    way round moves every class after the first -- and the lead-in of one bar must draw
    nothing at all, or the generator stands one draw ahead of the twin's for the whole
    walk."""
    steps = 20
    path, twin = contract(tmp_path, seed=5, d=d, layers=layers, heads=heads)
    circuit = drive(path, seed=seed, steps=steps)
    played, _ = quantized.walk(twin, [seed], steps)
    gate.assert_one_walk(circuit, played[0])
