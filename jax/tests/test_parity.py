"""The parity gates of the JAX seam: what holds the two sides of a model together.

Four kinds of gate stand here, and they are not the same kind of thing.

G0, THE FLOAT MODEL, one for each era. A number MEASURED and pinned, and not a threshold:
the loss of the era's float model over its canonical windows. It reads the forward alone
-- every table, every fold of the norm, every plane, and the reader that loaded them --
thus a rewrite above the seam that moves no number reads it back. Each one is pinned
BEFORE the rewrite that must read it back.

G1, THE QUANTIZER THROUGH THE NETLIST, one for each era. The elaboration reads the
contract file the JAX quantizer writes, and the Verilog it states must be the golden's
byte for byte: one rounding, one exponent or one fold out of place moves a weight, and a
moved weight moves the netlist. It is the gate of the quantizer and it costs one second.
Between G0 and G1 a change to a model has nowhere to hide.

THE WELDS ARE GONE. Gate A and gate C held a JAX trainer to the OCaml float model beside
it, and each frozen era had a pair; both pairs went with the twins that replaced them. G0
holds a forward to a measured number, `test_rtl_<era>.py` holds a draw to the circuit
itself, and `test_drift.py` holds a twin to the float model it quantizes -- thus nothing
is left for a weld to say, and no gate of this file reads an OCaml float model any more.

Every gate needs a checkpoint that git ignores and binaries that dune builds. They SKIP
when those are absent -- a clean tree is not a failure -- and they FAIL when the two sides
disagree. From the repository root:

    dune build bin/gate_transformer.exe bin/gate_mamba.exe
    dune build board/nexys-4/gen_verilog.exe
"""

import hashlib
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

import data
from transformer import model

ROOT = data.JAX_ROOT.parent
CHECKPOINT = ROOT / "_train" / "transformer" / "d64-frame-do03-96k-s6-l6-nopos-span4.ckpt"
CORPUS = data.FRAMES
BUILT = ROOT / "_build" / "default" / "bin"

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


def contract_file(era, checkpoint, tmp_path):
    """the contract file of the era's golden checkpoint, written by its JAX quantizer"""
    path = tmp_path / (checkpoint.stem + ".int8")
    run(
        "uv",
        "run",
        "python",
        "-m",
        f"{era}.infer",
        "quantize",
        "--ckpt",
        str(checkpoint),
        "--out",
        str(path),
    )
    return path


def netlist_md5(argv, tmp_path):
    """the md5 of the Verilog the elaborator states into [tmp_path]"""
    run(*argv, str(tmp_path))
    return hashlib.md5((tmp_path / "top.v").read_bytes()).hexdigest()


def windows_of(corpus, shape):
    """the canonical valid windows of an era, as classes and phases"""
    rows = data.eval_rows(corpus["valid"], shape["context"], shape["windows"])
    assert len(rows) == shape["windows"], "the corpus cut a different count of windows"
    classes, phases = data.stack_rows(rows)
    return jnp.asarray(classes), jnp.asarray(phases)


def seat_loss(nll):
    """nats for each step, which is the sum over the seats and the mean over the steps"""
    return float(jnp.mean(jnp.sum(nll, axis=-1)))


# ==================================================================== #
# Era four: the transformer                                            #
# ==================================================================== #

# The shape of the canonical reading, as `check_transformer loss` printed it on
# 2026-08-28, the day before the all-era cut deleted that tool. The file states the width
# and the layer count; the heads, the context and the slope span are the draw of the era,
# thus G0 carries them here and nothing else states them any more.
TRANSFORMER_SHAPE = {"windows": 75, "context": 256, "heads": 4, "span": 4}

# The loss of the elected checkpoint over those windows, MEASURED 2026-08-28 against the
# functional model `transformer/model.py` carries before the Flax round. The OCaml
# reference beside it stated the same 1.628177.
TRANSFORMER_LOSS = 1.628177

# The netlist of the elected checkpoint, MEASURED 2026-08-28 on `develop bb3b943`: era
# four's own `gen_verilog` of `7c3d356^` (with `_train/transformer/` before the checkpoint
# name) over HEAD's `top.ml`, with `~e` read as `~model`. The netlist moved at the
# unification round -- the one divider, the nine-bit `Mac` functor -- and the build-log
# records it as MET +0.005 at default directives.
TRANSFORMER_NETLIST_MD5 = "4ab6e292c6c26ca0befa68fb6026d3f5"

GATE_TRANSFORMER = BUILT / "gate_transformer.exe"


def test_g0_the_transformer_reads_its_measured_loss():
    """THE FLOAT MODEL DOES NOT MOVE. The windows are deterministic and no draw enters,
    thus the number reads the forward alone."""
    need(CHECKPOINT, CORPUS)
    classes, phases = windows_of(data.load_corpus(CORPUS), TRANSFORMER_SHAPE)
    held = model.Transformer.load(
        str(CHECKPOINT),
        heads=TRANSFORMER_SHAPE["heads"],
        span=TRANSFORMER_SHAPE["span"],
    )
    here = seat_loss(held.seat_nll(classes, phases))
    assert here == pytest.approx(TRANSFORMER_LOSS, abs=TOLERANCE), (
        f"the model reads {here:.6f} and the elected checkpoint measured "
        f"{TRANSFORMER_LOSS:.6f}"
    )


def test_g1_the_transformer_quantizer_states_its_netlist(tmp_path):
    """THE CIRCUIT DOES NOT MOVE. A different md5 says the quantization parted; diff the
    seat and phase tables first -- they share one exponent -- and then the layer ROM."""
    need(CHECKPOINT, GATE_TRANSFORMER)
    said = netlist_md5(
        [
            str(GATE_TRANSFORMER),
            "verilog",
            "-int8",
            str(contract_file("transformer", CHECKPOINT, tmp_path)),
        ],
        tmp_path,
    )
    assert said == TRANSFORMER_NETLIST_MD5, (
        f"the JAX quantizer states the netlist {said} and the golden is "
        f"{TRANSFORMER_NETLIST_MD5}"
    )


# ==================================================================== #
# Era five: the same gates, over the state-space model                 #
# ==================================================================== #

# the elected model of the era: six blocks, the Zamba head, the feed-forward. The plan and
# the span are in the file, thus neither side states one and neither can drift.
MAMBA_CHECKPOINT = (
    ROOT / "_train" / "mamba" / "d64-mamba-k4-n16-zamba-ff-do03-48k-s7.ckpt"
)

# The shape of the canonical reading, as `check_mamba loss` printed it on 2026-08-28, the
# day before the all-era cut deleted that tool. Only two numbers stand here: every width,
# the plan and the span come out of the file, and the context is a choice of the REFEREE
# -- a window of the recurrence opens on a zero state and the model has no context length
# at all.
MAMBA_SHAPE = {"windows": 75, "context": 256}

# The loss of the elected checkpoint over those windows, MEASURED 2026-08-28 against the
# functional model `mamba/model.py` carries before the Flax round. The OCaml reference
# beside it stated the same 1.640810.
MAMBA_LOSS = 1.640810

# The netlist of the elected checkpoint, MEASURED 2026-08-28 on `develop bb3b943`: era
# five's own `gen_verilog` of `46b1243^` over HEAD's `top.ml`, with `~e` read as `~model`.
# It is the number the unification round proved on 2026-08-23; five days of refactors to
# `lib/nn` and `lib/core` did not move it.
MAMBA_NETLIST_MD5 = "a648db223e3cb91896c23c0881f24634"

GATE_MAMBA = BUILT / "gate_mamba.exe"


def test_g0_the_mamba_reads_its_measured_loss():
    """THE FLOAT MODEL DOES NOT MOVE. The recurrence states no shape of its own: every
    width comes out of the file, thus the number reads the forward alone."""
    need(MAMBA_CHECKPOINT, CORPUS)
    from mamba import model as mamba_model

    classes, phases = windows_of(data.load_corpus(CORPUS), MAMBA_SHAPE)
    held = mamba_model.Mamba.load(str(MAMBA_CHECKPOINT))
    here = seat_loss(held.seat_nll(classes, phases))
    assert here == pytest.approx(MAMBA_LOSS, abs=TOLERANCE), (
        f"the model reads {here:.6f} and the elected checkpoint measured "
        f"{MAMBA_LOSS:.6f}"
    )


def test_g1_the_mamba_quantizer_states_its_netlist(tmp_path):
    """THE CIRCUIT DOES NOT MOVE. A different md5 says the quantization parted; diff the
    decay tensors first -- one ulp of `exp` moves a q_value by one -- and then the ROM."""
    need(MAMBA_CHECKPOINT, GATE_MAMBA)
    said = netlist_md5(
        [
            str(GATE_MAMBA),
            "verilog",
            "-int8",
            str(contract_file("mamba", MAMBA_CHECKPOINT, tmp_path)),
        ],
        tmp_path,
    )
    assert said == MAMBA_NETLIST_MD5, (
        f"the JAX quantizer states the netlist {said} and the golden is "
        f"{MAMBA_NETLIST_MD5}"
    )


# ==================================================================== #
# Era six: the quantizer, held through the netlist                     #
# ==================================================================== #

# TWO GATES STAND HERE AND NEITHER OF THEM IS OCAML'S ANY MORE. The two temporary gates
# that welded the two integer twins -- the walk and the drift report -- went with the
# OCaml twin; `tests/test_rtl.py` holds the CIRCUIT against the JAX twin and is what
# stays.

DIFFUSION_CHECKPOINT = ROOT / "_train" / "diffusion" / "coconet" / "l48-h20-100k.ckpt"
PIECES = data.PIECES
GEN_VERILOG = ROOT / "_build" / "default" / "board" / "nexys-4" / "gen_verilog.exe"

# The masked loss of the golden checkpoint over the sheets below, MEASURED 2026-08-28
# against the functional model that `diffusion/model.py` carried before the Flax round.
DIFFUSION_LOSS = 0.193459
DIFFUSION_CROP = 128

# the netlist the flash holds: the golden candidate at T 128, G 5, N 512, MEASURED
# 2026-08-28 by this gate on `develop bb3b943`
DIFFUSION_NETLIST_MD5 = "4e367cef6e38b2ae1f06ab3cf42a9c42"


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
    for step, voice in sheet_model.cell_order(crop):
        states, u = prng.uniform(states, everyone)
        hidden[:, step, voice] = u * 2.0**24 < float(1 << 23)
    return hidden


def test_g0_the_float_model_reads_its_measured_loss():
    """THE FLOAT MODEL DOES NOT MOVE. The sheets are deterministic -- the first 128 steps
    of every valid piece that holds them, in corpus order -- and the masks come from the
    shared generator, thus no draw of either framework enters and the number reads the
    FORWARD alone: every kernel, every fold of the norm, every plane, and the reader that
    loaded them.

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
    assert here == pytest.approx(DIFFUSION_LOSS, abs=TOLERANCE), (
        f"the model reads {here:.6f} and the golden checkpoint measured "
        f"{DIFFUSION_LOSS:.6f}"
    )


def test_g1_the_quantizer_states_the_golden_netlist(tmp_path):
    """THE CIRCUIT DOES NOT MOVE. The elaboration reads the contract file the JAX
    quantizer writes, and the Verilog it states must be the golden's byte for byte: one
    rounding, one exponent or one fold out of place moves a weight, and a moved weight
    moves the netlist.

    It is the gate of the quantizer and it costs one second. A different md5 says the
    quantization parted; diff the norm ROM first -- the gains and the biases -- and then
    the weight ROM."""
    need(DIFFUSION_CHECKPOINT, GEN_VERILOG)
    said = netlist_md5(
        [
            str(GEN_VERILOG),
            "-int8",
            str(contract_file("diffusion", DIFFUSION_CHECKPOINT, tmp_path)),
        ],
        tmp_path,
    )
    assert said == DIFFUSION_NETLIST_MD5, (
        f"the JAX quantizer states the netlist {said} and the golden is "
        f"{DIFFUSION_NETLIST_MD5}"
    )
