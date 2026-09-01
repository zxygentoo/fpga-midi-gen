"""The RTL gates of era six: the circuit against the integer twin.

THE ORACLE IS THIS SIDE AND THE CIRCUIT IS THE OTHER. `bin/gate_diffusion.exe` drives the
Hardcaml circuit in Cyclesim and prints WHAT IT DID; this module states what it must have
done, from `diffusion/quantized.py` over the same model. Neither side can pass by agreeing
with itself. The model crosses the seam as a CONTRACT FILE; the GEOMETRY cannot travel in
one -- T, G and N are the elaboration's -- thus it travels in the flags below.

Two gates, and each exists because a whole class of fault does not move a frame:

- THE WALK, PHASE FOR PHASE. The finished sheet alone would pass a walk whose masks are
  one pass out of phase, or one that spends a uniform on a standing cell. The comparison
  is therefore per phase -- the opening, then each pass's mask and each pass's draws in
  the cell order -- and the frames close it through the sequencer's own face.
- THE STREAM, COLUMN FOR COLUMN. Era five's four faults were all faults of the
  composition layer -- a weight stride, a channel block offset, an operand read one cycle
  early, a ring run off its end -- and none of them moved a frame.

P IS A PARAMETER OF THE STREAM GATE AND NOT OF THE WALK, because its input is data: a
narrow P is a legal sheet, and the composition layer's P-parametric paths keep an oracle
at more than one width. THE WALK CASES STAY AT P 48, where the seat registers reach class
46.

It SKIPS when the driver is absent -- a clean tree is not a failure. From the repository
root:

    dune build bin/gate_diffusion.exe
"""


import numpy as np
import pytest

from diffusion import model, quantized
from quantized import Tally, engine_states
from tests import gate

DRIVER = gate.driver("gate_diffusion.exe")

VOICES = model.VOICES


built = gate.built_fixture(DRIVER)


def drive(subcommand, path, *, steps, lanes, walk, seed, rows=model.ROWS):
    """the driver's report, one line as a list of its words"""
    stdout = gate.run(
        DRIVER,
        subcommand,
        "-int8", path,
        "-steps", steps,
        "-lanes", lanes,
        "-walk", walk,
        "-seed", seed,
        "-rows", rows,
    )
    return [line.split() for line in stdout.splitlines() if line]


def contract_file(tmp_path, *, weight_seed, layers, width):
    """the contract file of one drawn model, and the twin that wrote it"""
    twin = quantized.Coconet.from_float(model.Coconet.drawn(weight_seed, layers, width))
    path = tmp_path / f"l{layers}-h{width}-s{weight_seed}.int8"
    quantized.save(path, twin)
    return path, twin


# the walk, phase for phase


def wanted_walk(twin, *, steps, walk, seed):
    """Every write the walk must make, in the order it must make them, and the phase that
    owns each one: the opening, then for each pass its mask and its redraws.

    A disagreement therefore names its phase and not only its index."""
    states, given = model.opening_sheet(engine_states([seed]), steps)
    wanted = [
        ("the opening", "CLASS", step, voice, int(given[0, step, voice]))
        for step, voice in model.cell_order(steps)
    ]
    tally = Tally()
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
            for step, voice in model.cell_order(steps)
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
    """SEED 0 IS IN THE GATE: it is the fixed point of xorshift32 and the panel can state
    it, thus every uniform is 0, every cell hides at every pass and every draw takes the
    top of the grid. The walk that stands still is the design."""
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
    # THE FRAMES CLOSE THE WALK THROUGH THE SEQUENCER'S OWN FACE: the writes above prove
    # the sheet is the twin's, and this proves the score face states that sheet -- through
    # the Vocab decode, the seat packing, and the silence past T - 1.
    played = [(word[1], word[2]) for word in lines if word[0] == "frame"]
    stated = [(word[1], word[2]) for word in lines if word[0] == "want_frame"]
    assert played and played == stated, (
        f"the score face answered {played} and its sheet states {stated}"
    )


# the stream, column for column


def stem_input(lines, steps):
    """the sheet and the mask the driver drew, read back out of its report: the two facts
    the stem's decode reads, thus this side builds the same input and redraws nothing"""
    classes = np.zeros((1, steps, VOICES), np.int32)
    hidden = np.zeros((1, steps, VOICES), bool)
    for word in lines:
        if word[0] == "sheet":
            classes[0, int(word[1]), int(word[2])] = int(word[3])
        elif word[0] == "hidden":
            hidden[0, int(word[1]), int(word[2])] = bool(int(word[3]))
    return classes, hidden


@pytest.mark.parametrize(
    "name,layers,width,lanes,steps,weight_seed,rows",
    [
        # P 8 IS A SIMULATION'S P AND NOT THE BOARD'S. Every address of the composition
        # layer is P-parametric, thus eight rows run the same shapes at a quarter of the
        # column width; the last case holds the board's own P.
        ("H 8, G 2, two pairs, T 6", 6, 8, 2, 6, 1, 8),
        ("H 7, G 3, one pair, T 5", 4, 7, 3, 5, 2, 8),
        # AN IMAGE THAT REALLY BANKS: 1 080 words plan as 1 024 and 512, thus this case
        # reads through the bank mux where the two above read through one bank alone.
        ("H 8, G 4, three pairs, T 6", 8, 8, 4, 6, 3, 8),
        # A STORE THAT REALLY BANKS: 129 steps of 8 channels make a store of 1 032
        # columns, which plans as 1 024 and 512, thus this case reads and writes THROUGH
        # the store's bank mux and its write select.
        ("H 8, G 2, one pair, T 129", 4, 8, 2, 129, 4, 8),
        # THE RING WRAPS TWICE. Five columns over four ring slots is one wrap; two pairs
        # and the head behind them read every wrapped column, thus a ring one column short
        # -- or a lag of one instead of two -- writes over a column that is still live.
        ("H 8, G 2, two pairs, T 5", 6, 8, 2, 5, 5, 8),
        # AND THE SAME BANKING STORE AT THE BOARD'S OWN P. Six class bits and not three, a
        # chain of 48 stages and not 8, and the tag width the build really carries: what
        # the five cases above hold at a narrow P, this one holds at the elected one.
        ("H 8, G 2, one pair, T 129, P 48", 4, 8, 2, 129, 4, model.ROWS),
    ],
)
def test_the_store_writes_are_the_twins(
    tmp_path, name, layers, width, lanes, steps, weight_seed, rows
):
    """Every column the engine writes, against the twin's own `layer_writes`: the address
    stands in the elaboration's map, the datum equals the twin's, and each destination
    column is written exactly one time for each layer. The head writes no store, thus its
    gate is the logit face, read through the ports at every step the level offers."""
    path, twin = contract_file(
        tmp_path, weight_seed=weight_seed, layers=layers, width=width
    )
    # the walk of the driver's own stream gate: it names the first mask's threshold alone
    lines = drive(
        "stream", path, steps=steps, lanes=lanes, walk=8, seed=weight_seed, rows=rows
    )
    classes, hidden = stem_input(lines, steps)
    want = twin.layer_writes(classes, hidden, Tally(), rows=rows)
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
    # every layer's whole tensor and every offered step, thus a driver that printed
    # nothing cannot pass
    columns = sum(steps * layer.outputs for layer in twin.layers()[:-1])
    assert checked == columns + (steps * VOICES), (
        f"{name}: {checked} columns checked, and the shape holds "
        f"{columns + steps * VOICES}"
    )
