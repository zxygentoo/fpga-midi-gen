"""The RTL gates of era six: the circuit against the integer twin.

THE ORACLE IS THIS SIDE AND THE CIRCUIT IS THE OTHER. `bin/gate_diffusion.exe` drives the
Hardcaml circuit in Cyclesim and prints WHAT IT DID -- every write of the cell port, every
column the stores took, the logits the head offered, the frames the score face answered --
and this module states what it must have done, from `diffusion/quantized.py` over the same
model, and compares in order. Neither side can pass by agreeing with itself.

The model crosses the seam as a CONTRACT FILE. A tiny model is drawn here, quantized here
and written to a `tmp_path`; the driver reads it and elaborates a circuit from it. The
geometry cannot travel in a file -- T, G and N are the elaboration's -- thus it travels in
the flags this module passes.

Two gates, and each one exists because a whole class of fault does not move a frame:

- THE WALK, PHASE FOR PHASE. The finished canvas alone would pass a walk whose masks are
  one pass out of phase, or one that spends a uniform on a standing cell: both draw a
  canvas, and both draw the WRONG one with no local symptom. The comparison is therefore
  per phase -- the opening, then each pass's mask in the cell order and each pass's draws
  in the cell order -- and the frames close it through the sequencer's own face.
- THE STREAM, COLUMN FOR COLUMN. Era five's four faults were all faults of the composition
  layer -- a weight address whose stride was not the tensor's, a channel block read at the
  gate's offset, an operand taken on the address side of a two-cycle read, and a ring run
  off its end -- and none of them moved a frame.

It SKIPS when the driver is absent -- a clean tree is not a failure. From the repository
root:

    dune build bin/gate_diffusion.exe
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from diffusion import infer, model, quantized

JAX_ROOT = Path(__file__).resolve().parent.parent
ROOT = JAX_ROOT.parent
DRIVER = ROOT / "_build" / "default" / "bin" / "gate_diffusion.exe"

VOICES = model.VOICES


def drive(subcommand, path, *, steps, lanes, walk, seed):
    """the driver's report, one line as a list of its words"""
    if not DRIVER.exists():
        pytest.skip(f"absent, nothing to gate: {DRIVER.name}")
    done = subprocess.run(
        [
            str(DRIVER),
            subcommand,
            "-int8",
            str(path),
            "-steps",
            str(steps),
            "-lanes",
            str(lanes),
            "-walk",
            str(walk),
            "-seed",
            str(seed),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # check=False: the assert carries the stderr into the report, where a
    # CalledProcessError would show the command alone
    assert done.returncode == 0, done.stderr
    return [line.split() for line in done.stdout.splitlines() if line]


def contract_file(tmp_path, *, weight_seed, layers, width):
    """the contract file of one drawn model, and the twin that wrote it"""
    params, stats = model.drawn_params(weight_seed, layers, width)
    twin = quantized.of_params(params, stats)
    path = tmp_path / f"l{layers}-h{width}-s{weight_seed}.int8"
    quantized.save(path, twin)
    return path, twin


def cell_order(steps):
    """Diffusion.cell_order: a step at a time, and the seats of a step inside it. Every
    uniform of the walk is drawn in this order and every write follows it."""
    return [(step, voice) for step in range(steps) for voice in range(VOICES)]


# ---------------------------------------------------------------------
# the walk, phase for phase
# ---------------------------------------------------------------------


def wanted_walk(twin, *, steps, walk, seed):
    """Every write the walk must make, in the order it must make them, and the phase that
    owns each one: the opening, then for each pass its mask and its redraws.

    A disagreement therefore names its phase and not only its index."""
    states, given = infer.opening_canvas(quantized.engine_states([seed]), steps)
    wanted = [
        ("the opening", "CLASS", step, voice, int(given[0, step, voice]))
        for step, voice in cell_order(steps)
    ]
    tally = quantized.counters()
    for at, taken in enumerate(
        quantized.passes(twin, states, given, walk=walk, tally=tally)
    ):
        wanted += [
            (
                f"the mask of pass {at}",
                "MASK",
                step,
                voice,
                int(taken.hidden[0, step, voice]),
            )
            for step, voice in cell_order(steps)
        ]
        wanted += [
            (
                f"a draw of pass {at}",
                "CLASS",
                drawn.step,
                drawn.voice,
                int(drawn.drawn[0]),
            )
            for drawn in taken.draws
            if drawn.hidden[0]
        ]
    return wanted


@pytest.mark.parametrize("seed", [1, 2, 0])
@pytest.mark.parametrize(
    "layers,width,lanes,steps,walk,weight_seed",
    [(6, 8, 2, 6, 3, 1), (4, 7, 3, 5, 4, 2)],
)
def test_the_walk_is_the_twins_walk(
    tmp_path, seed, layers, width, lanes, steps, walk, weight_seed
):
    """SEED 0 IS IN THE GATE. It is the fixed point of xorshift32 -- the panel can state it
    and the engine takes its seed as the SEED cell does -- thus every uniform is 0, every
    cell hides at every pass and every draw takes the top of the grid. The walk that stands
    still is the design, and the gate holds the circuit to that stillness like any other
    walk: the pass counts show it, because every cell is redrawn."""
    path, twin = contract_file(
        tmp_path, weight_seed=weight_seed, layers=layers, width=width
    )
    lines = drive("walk", path, steps=steps, lanes=lanes, walk=walk, seed=seed)
    got = [
        (word[1], int(word[2]), int(word[3]), int(word[4]))
        for word in lines
        if word[0] == "write"
    ]
    want = wanted_walk(twin, steps=steps, walk=walk, seed=seed)
    assert len(got) == len(want), (
        f"the circuit made {len(got)} writes and the twin wants {len(want)}"
    )
    for at, (made, (phase, *wanted)) in enumerate(zip(got, want)):
        assert list(made) == wanted, (
            f"{phase}, write {at}: the circuit wrote {made} and the twin wants "
            f"{tuple(wanted)}"
        )
    # THE FRAMES CLOSE THE WALK THROUGH THE SEQUENCER'S OWN FACE. The driver states the
    # frames of the canvas it drew, thus this side holds the format to nothing of its own:
    # the writes above prove the canvas is the twin's, and this proves the score face
    # states that canvas -- through the Vocab decode, the seat packing, and the silence
    # past T - 1.
    played = [(word[1], word[2]) for word in lines if word[0] == "frame"]
    stated = [(word[1], word[2]) for word in lines if word[0] == "want_frame"]
    assert played and played == stated, (
        f"the score face answered {played} and its canvas states {stated}"
    )


# ---------------------------------------------------------------------
# the stream, column for column
# ---------------------------------------------------------------------


def stem_input(lines, steps):
    """the canvas and the mask the driver drew, read back out of its report: the two facts
    the stem's decode reads, thus this side builds the same input and redraws nothing"""
    classes = np.zeros((1, steps, VOICES), np.int32)
    hidden = np.zeros((1, steps, VOICES), bool)
    for word in lines:
        if word[0] == "canvas":
            classes[0, int(word[1]), int(word[2])] = int(word[3])
        elif word[0] == "hidden":
            hidden[0, int(word[1]), int(word[2])] = bool(int(word[3]))
    return classes, hidden


@pytest.mark.parametrize(
    "name,layers,width,lanes,steps,weight_seed",
    [
        ("H 8, G 2, two pairs, T 6", 6, 8, 2, 6, 1),
        ("H 7, G 3, one pair, T 5", 4, 7, 3, 5, 2),
        # AN IMAGE THAT REALLY BANKS: 1 080 words plan as 1 024 and 512, thus this case
        # reads through the bank mux where the two above read through one bank alone.
        ("H 8, G 4, three pairs, T 6", 8, 8, 4, 6, 3),
        # A STORE THAT REALLY BANKS: 129 steps of 8 channels make a store of 1 032 columns,
        # which plans as 1 024 and 512, thus this case reads and writes THROUGH the store's
        # bank mux and its write select.
        ("H 8, G 2, one pair, T 129", 4, 8, 2, 129, 4),
        # THE RING WRAPS TWICE. Five columns over four ring slots is one wrap; two pairs
        # and the head behind them read every wrapped column, thus a ring one column short
        # -- or a lag of one instead of two -- writes over a column that is still live.
        ("H 8, G 2, two pairs, T 5", 6, 8, 2, 5, 5),
    ],
)
def test_the_store_writes_are_the_twins(
    tmp_path, name, layers, width, lanes, steps, weight_seed
):
    """Every column the engine writes, against the twin's own `layer_writes`: the address
    stands in the elaboration's map, the datum equals the twin's, and each destination
    column is written exactly one time for each layer. The head writes no store, thus its
    gate is the logit face, read through the ports at every step the level offers."""
    path, twin = contract_file(
        tmp_path, weight_seed=weight_seed, layers=layers, width=width
    )
    # the walk of the driver's own stream gate: it names the first mask's threshold alone
    lines = drive("stream", path, steps=steps, lanes=lanes, walk=8, seed=weight_seed)
    classes, hidden = stem_input(lines, steps)
    want = quantized.layer_writes(
        twin,
        quantized.device_kernels(twin),
        classes,
        hidden,
        quantized.counters(),
    )
    checked = 0
    for word in lines:
        if word[0] == "write":
            at, step, channel = int(word[1]), int(word[2]), int(word[3])
            column = want[at][0, step, :, channel]
            where = f"{name}: L{at} step {step} channel {channel}"
        elif word[0] == "logits":
            step, seat = int(word[1]), int(word[2])
            column = want[-1][0, step, :, seat]
            where = f"{name}: the head, step {step} seat {seat}"
        elif word[0] == "misplaced":
            assert word[1] == "0", f"{name}: {word[1]} turns misplaced their columns"
            continue
        else:
            continue
        checked += 1
        got = np.array([int(value) for value in word[4 if word[0] == "write" else 3 :]])
        assert np.array_equal(got, column), (
            f"{where}: {int((got != column).sum())} of {len(column)} rows part\n"
            f"  want {list(column[:12])}\n  got  {list(got[:12])}"
        )
    # every layer's whole tensor and every offered step, thus a driver that printed nothing
    # cannot pass
    columns = sum(steps * layer.outputs for layer in twin.layers[:-1])
    assert checked == columns + (steps * VOICES), (
        f"{name}: {checked} columns checked, and the shape holds "
        f"{columns + steps * VOICES}"
    )
