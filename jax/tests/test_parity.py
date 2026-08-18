"""The parity gates of the JAX seam: the two places the trainer meets the reference.

The trainer lives here and the reference lives in OCaml. A GPU run only means something
if the two forwards agree, and a seed only names one walk if the two draws agree. Nothing
else in this tree pins them together -- the unit tests hold each side against itself.

Gate A, the loss. `checkpoint_tool loss` states the loss of the OCaml float model over the
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

    dune build bin/checkpoint_tool.exe bin/play_transformer.exe
"""

import re
import subprocess
from pathlib import Path

import jax.numpy as jnp
import pytest

import data
from transformer import model

JAX_ROOT = Path(__file__).resolve().parent.parent
ROOT = JAX_ROOT.parent
CHECKPOINT = ROOT / "_train" / "d64-frame-do03-96k-s6-l6-nopos-span4.ckpt"
CORPUS = JAX_ROOT / "_data" / "frames.safetensors"
# the tool's default corpus path is relative to the repository root, and pytest runs in [jax]
CORPUS_JSON = ROOT / "corpus" / "JSB-Chorales-dataset" / "Jsb16thSeparated.json"
BUILT = ROOT / "_build" / "default" / "bin"
CHECKPOINT_TOOL = BUILT / "checkpoint_tool.exe"
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
    done = subprocess.run(argv, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_gate_a_the_loss_of_the_two_forwards_agrees():
    need(CHECKPOINT, CORPUS, CORPUS_JSON, CHECKPOINT_TOOL)
    stated = run(
        str(CHECKPOINT_TOOL),
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
