"""The parity gates of the JAX seam: the two places the trainer meets the reference.

Era five runs the same two gates over its own tools; the classes at the foot of the file
carry the paths and the shape flags that differ, and the bodies are shared.

The trainer lives here and the reference lives in OCaml. A GPU run only means something
if the two forwards agree, and a seed only names one walk if the two draws agree. Nothing
else in this tree pins them together -- the unit tests hold each side against itself.

Gate A, the loss. `check_transformer loss` states the loss of the OCaml float model over the
canonical valid windows, and this demands the same number of the JAX forward on the same
windows. A pass proves the two forwards agree everywhere the loss can see. The shape
travels in the OCaml output and this side states none of its own: held apart, the heads
and the span would go on matching a default while one side moved, and the gate would pass
two different models.

Gate C, the walk. `play_transformer` and `infer.py` print the same line for a step, thus
the whole event stream compares as text. This is the gate the sampler needs, because the
draw is where a rewrite is plausibly wrong and still makes music: a peak over the wrong
set, a min-p floor applied before the temperature, an inclusive compare in the cumulative
walk. Each shifts the distribution a little and nothing raises.

Both gates need a checkpoint that git ignores and binaries that dune builds. They SKIP
when those are absent -- a clean tree is not a failure -- and they FAIL when the two sides
disagree. From the repository root:

    dune build bin/check_transformer.exe bin/play_transformer.exe
"""

import hashlib
import re
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import data
from transformer import model

JAX_ROOT = Path(__file__).resolve().parent.parent
ROOT = JAX_ROOT.parent
CHECKPOINT = ROOT / "_train" / "transformer" / "d64-frame-do03-96k-s6-l6-nopos-span4.ckpt"
CORPUS = JAX_ROOT / "_data" / "frames.safetensors"
# the tool's default corpus path is relative to the repository root, and pytest runs in [jax]
CORPUS_JSON = ROOT / "corpus" / "JSB-Chorales-dataset" / "Jsb16thSeparated.json"
BUILT = ROOT / "_build" / "default" / "bin"
CHECK_TRANSFORMER = BUILT / "check_transformer.exe"
PLAYER = BUILT / "play_transformer.exe"

# The loss is a mean of 75 windows of 256 steps through six layers of float32, and the two
# sides reduce in different orders. A disagreement that matters -- a different mask, a
# different phase, a transposed table -- moves the third decimal at least.
TOLERANCE = 2e-4


def need(*paths):
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"absent, nothing to gate: {', '.join(missing)}")


def run(*argv):
    # check=False: the assert below carries the stderr into the report, where a
    # CalledProcessError would show the command alone
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_gate_a_the_loss_of_the_two_forwards_agrees():
    need(CHECKPOINT, CORPUS, CORPUS_JSON, CHECK_TRANSFORMER)
    stated = run(
        str(CHECK_TRANSFORMER),
        "loss",
        "-ckpt",
        str(CHECKPOINT),
        "-corpus",
        str(CORPUS_JSON),
    )
    said = dict(re.findall(r"(\w+) (-?[\d.]+)", stated))
    windows, context = int(said["windows"]), int(said["context"])
    heads, span = int(said["heads"]), int(said["span"])

    params = model.load_params(str(CHECKPOINT))
    corpus = data.load_corpus(CORPUS)
    rows = data.eval_rows(corpus["valid"], context, windows)
    assert len(rows) == windows, "the two sides cut a different count of windows"
    classes, phases = data.stack_rows(rows)

    nll = model.seat_nll(
        params, jnp.asarray(classes), jnp.asarray(phases), heads=heads, span=span
    )
    # nats for each step, which is the sum over the seats and the mean over the steps
    here = float(jnp.mean(jnp.sum(nll, axis=-1)))
    there = float(said["loss"])
    assert here == pytest.approx(there, abs=TOLERANCE), (
        f"the JAX forward says {here:.6f} and the OCaml reference says {there:.6f}"
    )


@pytest.mark.parametrize("seed", [1, 7])
def test_gate_c_the_two_walks_are_the_same_stream(seed):
    need(CHECKPOINT, PLAYER)
    steps = 64
    theirs = run(
        str(PLAYER),
        "-ckpt",
        str(CHECKPOINT),
        "-seed",
        str(seed),
        "-steps",
        str(steps),
    )
    ours = run(
        "uv",
        "run",
        "python",
        "-m",
        "transformer.infer",
        "--ckpt",
        str(CHECKPOINT),
        "--seeds",
        str(seed),
        "--steps",
        str(steps),
    )
    lines = lambda text: [l for l in text.splitlines() if l.startswith("step")]
    here, there = lines(ours), lines(theirs)
    assert here, "the JAX walk printed no step lines"
    assert len(here) == len(there) == steps, (
        f"{len(here)} JAX steps against {len(there)} OCaml steps, wanted {steps}"
    )
    first = next((i for i, (a, b) in enumerate(zip(here, there)) if a != b), None)
    assert first is None, (
        f"the walks part at step {first}:\n  jax   {here[first]}\n  ocaml {there[first]}"
    )


# ==================================================================== #
# Era five: the same two gates, over the state-space model             #
# ==================================================================== #

# the elected model of the era: six blocks, the Zamba head, the feed-forward. The plan and
# the span are in the file, thus neither side states one and neither can drift.
MAMBA_CHECKPOINT = (
    ROOT / "_train" / "mamba" / "d64-mamba-k4-n16-zamba-ff-do03-48k-s7.ckpt"
)
CHECK_MAMBA = BUILT / "check_mamba.exe"
MAMBA_PLAYER = BUILT / "play_mamba.exe"


def test_gate_a_the_mamba_forwards_agree():
    """The recurrence states no shape of its own: the tool prints every width out of the
    file, and this side reads the same file, thus the two cannot drift apart in a flag.
    The context travels in the output because it is a choice of the REFEREE here — a window
    of the recurrence opens on a zero state and the model has no context length at all."""
    need(MAMBA_CHECKPOINT, CORPUS, CORPUS_JSON, CHECK_MAMBA)
    from mamba import model as mamba_model

    stated = run(
        str(CHECK_MAMBA),
        "loss",
        "-ckpt",
        str(MAMBA_CHECKPOINT),
        "-corpus",
        str(CORPUS_JSON),
    )
    said = dict(re.findall(r"(\w+) (-?[\d.]+)", stated))
    windows, context = int(said["windows"]), int(said["context"])

    params = mamba_model.load_params(str(MAMBA_CHECKPOINT))
    corpus = data.load_corpus(CORPUS)
    rows = data.eval_rows(corpus["valid"], context, windows)
    assert len(rows) == windows, "the two sides cut a different count of windows"
    classes, phases = data.stack_rows(rows)

    nll = mamba_model.seat_nll(params, jnp.asarray(classes), jnp.asarray(phases))
    here = float(jnp.mean(jnp.sum(nll, axis=-1)))
    there = float(said["loss"])
    assert here == pytest.approx(there, abs=TOLERANCE), (
        f"the JAX forward says {here:.6f} and the OCaml reference says {there:.6f}"
    )


@pytest.mark.parametrize("seed", [1, 7])
def test_gate_c_the_two_mamba_walks_are_the_same_stream(seed):
    need(MAMBA_CHECKPOINT, MAMBA_PLAYER)
    steps = 64
    theirs = run(
        str(MAMBA_PLAYER),
        "-ckpt",
        str(MAMBA_CHECKPOINT),
        "-seed",
        str(seed),
        "-steps",
        str(steps),
    )
    ours = run(
        "uv",
        "run",
        "python",
        "-m",
        "mamba.infer",
        "--ckpt",
        str(MAMBA_CHECKPOINT),
        "--seeds",
        str(seed),
        "--steps",
        str(steps),
    )
    lines = lambda text: [l for l in text.splitlines() if l.startswith("step")]
    here, there = lines(ours), lines(theirs)
    assert here, "the JAX walk printed no step lines"
    assert len(here) == len(there) == steps, (
        f"{len(here)} JAX steps against {len(there)} OCaml steps, wanted {steps}"
    )
    first = next((i for i, (a, b) in enumerate(zip(here, there)) if a != b), None)
    assert first is None, (
        f"the walks part at step {first}:\n  jax   {here[first]}\n  ocaml {there[first]}"
    )


# ==================================================================== #
# Era six: the quantizer, held through the netlist                     #
# ==================================================================== #

# TWO GATES STAND HERE AND NEITHER OF THEM IS OCAML'S ANY MORE. The two temporary gates
# that welded the two integer twins -- the walk and the drift report -- went with the OCaml
# twin; `tests/test_rtl.py` holds the CIRCUIT against the JAX twin and is what stays.
#
# G0 holds the FLOAT MODEL to a number measured before the Flax round rewrote it, and G1
# holds the QUANTIZATION through the netlist the flash carries. Between them a change to
# either model has nowhere to hide: G0 reads every kernel and every fold of the norm, and
# G1 reads every rounding and every exponent, all the way to the bytes of the Verilog.

DIFFUSION_CHECKPOINT = ROOT / "_train" / "diffusion" / "coconet" / "l48-h20-100k.ckpt"
PIECES = JAX_ROOT / "_data" / "pieces.safetensors"
GEN_VERILOG = ROOT / "_build" / "default" / "board" / "nexys-4" / "gen_verilog.exe"

# The masked loss of the golden checkpoint over the sheets below, MEASURED 2026-08-28
# against the functional model that `diffusion/model.py` carried before the Flax round.
# It is a pinned number and not a threshold: a diff here says the float model moved.
GOLDEN_LOSS = 0.193459
DIFFUSION_CROP = 128


def diffusion_gate_masks(sheets, crop):
    """The Bernoulli-half masks of G0: sheet i on the generator at seed i + 1, one uniform
    for each cell in the cell order, hidden exactly when u * 2^24 < 2^23.

    They come from the shared generator and not from either framework's own draw, thus the
    mask is a fact of this repository and no change of a key rule can move it."""
    import prng
    from diffusion import model as sheet_model

    states = prng.states(np.arange(1, sheets + 1))
    hidden = np.zeros((sheets, crop, sheet_model.VOICES), dtype=bool)
    everyone = np.ones(sheets, dtype=bool)
    for step in range(crop):
        for voice in range(sheet_model.VOICES):
            states, u = prng.uniform(states, everyone)
            hidden[:, step, voice] = u * 2.0**24 < float(1 << 23)
    return hidden


def test_g0_the_float_model_reads_its_measured_loss():
    """THE FLOAT MODEL DOES NOT MOVE. The sheets are deterministic -- the first 128 steps of
    every valid piece that holds them, in corpus order -- and the masks come from the shared
    generator, thus no draw of either framework enters and the number reads the FORWARD
    alone: every kernel, every fold of the norm, every plane, and the reader that loaded
    them.

    It was measured on the functional model this era's `model.py` used to be, and it is
    what said that the Flax module tree moved no number. The tolerance is Gate A's: a mean
    of 76 sheets through 48 layers of float32, where two readings reduce in different
    orders. A disagreement that matters moves the fourth decimal at least."""
    need(DIFFUSION_CHECKPOINT, PIECES)
    from diffusion import model as sheet_model

    pieces = data.load_pieces(str(PIECES))["valid"]
    keep = [
        at for at in range(len(pieces.lengths)) if pieces.lengths[at] >= DIFFUSION_CROP
    ]
    classes = np.stack([pieces.classes[at][:DIFFUSION_CROP] for at in keep])
    hidden = diffusion_gate_masks(len(keep), DIFFUSION_CROP)

    coconet = sheet_model.Coconet.load(DIFFUSION_CHECKPOINT)
    values = []
    for at in range(0, len(keep), 16):
        rows = slice(at, at + 16)
        said, _ = coconet(
            sheet_model.planes(jnp.asarray(classes[rows]), jnp.asarray(hidden[rows]))
        )
        nll = np.asarray(sheet_model.nll_of_logits(said, jnp.asarray(classes[rows])))
        values += [float(row[mask].mean()) for row, mask in zip(nll, hidden[rows])]
    here = float(np.mean(values))
    assert here == pytest.approx(GOLDEN_LOSS, abs=TOLERANCE), (
        f"the model reads {here:.6f} and the golden checkpoint measured {GOLDEN_LOSS:.6f}"
    )


# the netlist the flash holds: the golden candidate at T 128, G 5, N 512
GOLDEN_NETLIST_MD5 = "4e367cef6e38b2ae1f06ab3cf42a9c42"


def quantized_checkpoint(tmp_path):
    """the contract file of the golden candidate, written by the JAX quantizer"""
    path = tmp_path / "l48-h20-100k.int8"
    run(
        "uv",
        "run",
        "python",
        "-m",
        "diffusion.infer",
        "quantize",
        "--ckpt",
        str(DIFFUSION_CHECKPOINT),
        "--out",
        str(path),
    )
    return path


def test_g1_the_quantizer_states_the_golden_netlist(tmp_path):
    """THE CIRCUIT DOES NOT MOVE. The elaboration reads the contract file the JAX quantizer
    writes, and the Verilog it states must be the golden's byte for byte: one rounding, one
    exponent or one fold out of place moves a weight, and a moved weight moves the netlist.

    It is the gate of the quantizer and it costs one second. A different md5 says the
    quantization parted; diff the norm ROM first -- the gains and the biases -- and then the
    weight ROM."""
    need(DIFFUSION_CHECKPOINT, GEN_VERILOG)
    run(str(GEN_VERILOG), "-int8", str(quantized_checkpoint(tmp_path)), str(tmp_path))
    said = hashlib.md5((tmp_path / "top.v").read_bytes()).hexdigest()
    assert said == GOLDEN_NETLIST_MD5, (
        f"the JAX quantizer states the netlist {said} and the golden is "
        f"{GOLDEN_NETLIST_MD5}"
    )
