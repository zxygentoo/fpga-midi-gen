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
# Era six: the same two gates, over the masked canvas                  #
# ==================================================================== #

# the golden candidate of the era; the shape is in the file, thus neither side states one
DIFFUSION_CHECKPOINT = ROOT / "_train" / "diffusion" / "coconet" / "l48-h20-100k.ckpt"
PIECES = JAX_ROOT / "_data" / "pieces.safetensors"
CHECK_DIFFUSION = BUILT / "check_diffusion.exe"
DIFFUSION_PLAYER = BUILT / "play_diffusion.exe"


def diffusion_gate_masks(canvases, crop):
    """The Bernoulli-half masks of Gate A: canvas i on the generator at seed i + 1, one
    uniform for each cell in the cell order, hidden exactly when u * 2^24 < 2^23. This is
    Diffusion.gate_mask drawn from the batched twin of the generator, thus the two sides
    hide the same cells by construction and no tolerance covers the mask."""
    import prng
    from diffusion import model as canvas_model

    states = prng.states(np.arange(1, canvases + 1))
    hidden = np.zeros((canvases, crop, canvas_model.VOICES), dtype=bool)
    everyone = np.ones(canvases, dtype=bool)
    for step in range(crop):
        for voice in range(canvas_model.VOICES):
            states, u = prng.uniform(states, everyone)
            hidden[:, step, voice] = u * 2.0**24 < float(1 << 23)
    return hidden


def test_gate_a_the_diffusion_forwards_agree():
    """The canvases are deterministic -- the first [crop] steps of every valid piece that
    holds them, in corpus order -- and the masks come from the shared generator, thus no
    draw of either framework enters and the number reads the forward alone: every kernel,
    every fold of the norm, every plane. The crop travels in the tool's output, as the
    shapes of the sibling gates do."""
    need(DIFFUSION_CHECKPOINT, PIECES, CORPUS_JSON, CHECK_DIFFUSION)
    from diffusion import model as canvas_model

    stated = run(
        str(CHECK_DIFFUSION),
        "loss",
        "-ckpt",
        str(DIFFUSION_CHECKPOINT),
        "-corpus",
        str(CORPUS_JSON),
        "-crop",
        "128",
    )
    said = dict(re.findall(r"(\w+) (-?[\d.]+)", stated))
    canvases, crop = int(said["canvases"]), int(said["crop"])

    pieces = data.load_pieces(str(PIECES))["valid"]
    keep = [at for at in range(len(pieces.lengths)) if pieces.lengths[at] >= crop]
    assert len(keep) == canvases, "the two sides kept a different count of canvases"
    classes = np.stack([pieces.classes[at][:crop] for at in keep])
    hidden = diffusion_gate_masks(canvases, crop)

    params, stats = canvas_model.load_params(str(DIFFUSION_CHECKPOINT))
    values = []
    for at in range(0, canvases, 16):
        chunk_classes = jnp.asarray(classes[at : at + 16])
        chunk_hidden = jnp.asarray(hidden[at : at + 16])
        said_logits, _ = canvas_model.logits(
            params, stats, canvas_model.planes(chunk_classes, chunk_hidden)
        )
        nll = np.asarray(canvas_model.nll_of_logits(said_logits, chunk_classes))
        for row in range(len(nll)):
            mask = hidden[at + row]
            values.append(float(nll[row][mask].mean()))
    here = float(np.mean(values))
    there = float(said["loss"])
    assert here == pytest.approx(there, abs=TOLERANCE), (
        f"the JAX forward says {here:.6f} and the OCaml reference says {there:.6f}"
    )


@pytest.mark.parametrize("seed,crop,walk", [(3, 128, 32), (7, 32, 8)])
def test_gate_c_the_two_canvas_walks_are_the_same_stream(seed, crop, walk):
    """The walk gate of the era: the seeded opening, the anneal thresholds, the tempered
    picks -- every uniform from the shared generator in the pinned cell order, thus the
    whole canvas compares as text. One row runs the full canvas at a board-like budget and
    one runs small, so the gate crosses every rule without owning the suite's clock."""
    need(DIFFUSION_CHECKPOINT, DIFFUSION_PLAYER)
    theirs = run(
        str(DIFFUSION_PLAYER),
        "-ckpt",
        str(DIFFUSION_CHECKPOINT),
        "-seeds",
        str(seed),
        "-steps",
        str(crop),
        "-walk",
        str(walk),
    )
    ours = run(
        "uv",
        "run",
        "python",
        "-m",
        "diffusion.infer",
        "sample",
        "--ckpt",
        str(DIFFUSION_CHECKPOINT),
        "--seeds",
        str(seed),
        "--crop",
        str(crop),
        "--walk",
        str(walk),
    )
    lines = lambda text: [l for l in text.splitlines() if l.startswith("step")]
    here, there = lines(ours), lines(theirs)
    assert here, "the JAX walk printed no step lines"
    assert len(here) == len(there) == crop, (
        f"{len(here)} JAX steps against {len(there)} OCaml steps, wanted {crop}"
    )
    first = next((i for i, (a, b) in enumerate(zip(here, there)) if a != b), None)
    assert first is None, (
        f"the walks part at step {first}:\n  jax   {here[first]}\n  ocaml {there[first]}"
    )
