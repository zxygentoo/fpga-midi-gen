"""The parity gates of the JAX seam: what holds the two sides of a model together.

Two kinds of gate stand here, one of each for every era.

G0, THE FLOAT MODEL: a number MEASURED and pinned, and not a threshold -- the loss of the
era's float model over its canonical windows. It reads the forward alone, thus a rewrite
above the seam that moves no number reads it back, and each one is pinned BEFORE the
rewrite that must.

G1, THE QUANTIZER THROUGH THE NETLIST: the elaboration reads the contract file the JAX
quantizer writes, and the Verilog it states must be the golden's byte for byte. One
rounding, one exponent or one fold out of place moves a weight, and a moved weight moves
the netlist. Between G0 and G1 a change to a model has nowhere to hide.

Every gate needs a checkpoint that git ignores and binaries that dune builds. They SKIP
when those are absent -- a clean tree is not a failure -- and they FAIL when the two sides
disagree. From the repository root:

    dune build bin/gate_transformer.exe bin/gate_mamba.exe
    dune build board/nexys-4/gen_verilog.exe
"""

import hashlib

import jax.numpy as jnp
import numpy as np
import pytest

import corpus
import prng
from diffusion import model as sheet_model
from diffusion import train as sheet_train
from mamba import model as mamba_model
from tests import gate
from tests.gate import need, run
from transformer import model

ROOT = gate.ROOT
CHECKPOINT = ROOT / "_train" / "transformer" / "d64-frame-do03-96k-s6-l6-nopos-span4.ckpt"
CORPUS = corpus.FRAMES

# The loss is a mean of 75 windows of 256 steps through six layers of float32, and the two
# sides reduce in different orders. A disagreement that matters -- a different mask, a
# different phase, a transposed table -- moves the third decimal at least.
TOLERANCE = 2e-4


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
        cwd=corpus.JAX_ROOT,
    )
    return path


def netlist_md5(argv, tmp_path):
    """the md5 of the Verilog the elaborator states into [tmp_path]"""
    run(*argv, str(tmp_path))
    return hashlib.md5((tmp_path / "top.v").read_bytes()).hexdigest()


def windows_of(splits, shape):
    """the canonical valid windows of an era, as classes and phases"""
    rows = corpus.eval_rows(splits["valid"], shape["context"], shape["windows"])
    assert len(rows) == shape["windows"], "the corpus cut a different count of windows"
    classes, phases = corpus.stack_rows(rows)
    return jnp.asarray(classes), jnp.asarray(phases)


def seat_loss(nll):
    """nats for each step, which is the sum over the seats and the mean over the steps"""
    return float(jnp.mean(jnp.sum(nll, axis=-1)))


# ==================================================================== #
# Era four: the transformer                                            #
# ==================================================================== #

# The shape of the canonical reading. The file states the width and the layer count; the
# heads, the context and the slope span are the draw of the era, thus G0 carries them here
# and nothing else states them any more.
TRANSFORMER_SHAPE = {"windows": 75, "context": 256, "heads": 4, "span": 4}

# The loss of the elected checkpoint over those windows, MEASURED 2026-08-28 against the
# functional model `transformer/model.py` carries before the Flax round. The OCaml
# reference beside it stated the same 1.628177.
TRANSFORMER_LOSS = 1.628177

# The netlist of the elected checkpoint, re-pinned by this gate through
# `gate_transformer.exe verilog` at the end of the lifts into `lib/nn`.
# IT OWES A VIVADO BUILD before it merges.
TRANSFORMER_NETLIST_MD5 = "a106ff1a991ed756f4c78af99b8d5b35"

GATE_TRANSFORMER = gate.driver("gate_transformer.exe")


def test_g0_the_transformer_reads_its_measured_loss():
    """THE FLOAT MODEL DOES NOT MOVE. The windows are deterministic and no draw enters,
    thus the number reads the forward alone."""
    need(CHECKPOINT, CORPUS)
    classes, phases = windows_of(corpus.load_corpus(CORPUS), TRANSFORMER_SHAPE)
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

# The shape of the canonical reading. Only two numbers stand here: every width, the plan
# and the span come out of the file, and the context is a choice of the REFEREE -- a
# window of the recurrence opens on a zero state and the model has no context length.
MAMBA_SHAPE = {"windows": 75, "context": 256}

# The loss of the elected checkpoint over those windows, MEASURED 2026-08-28 against the
# functional model `mamba/model.py` carries before the Flax round. The OCaml reference
# beside it stated the same 1.640810.
MAMBA_LOSS = 1.640810

# The netlist of the elected checkpoint, re-pinned by this gate through
# `gate_mamba.exe verilog` at the end of the lifts into `lib/nn`.
# IT OWES A VIVADO BUILD before it merges.
MAMBA_NETLIST_MD5 = "e6abe8c20c983a930b99a626c18a9b13"

GATE_MAMBA = gate.driver("gate_mamba.exe")


def test_g0_the_mamba_reads_its_measured_loss():
    """THE FLOAT MODEL DOES NOT MOVE. The recurrence states no shape of its own: every
    width comes out of the file, thus the number reads the forward alone."""
    need(MAMBA_CHECKPOINT, CORPUS)
    classes, phases = windows_of(corpus.load_corpus(CORPUS), MAMBA_SHAPE)
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

# `tests/test_rtl_diffusion.py` holds the CIRCUIT against the JAX twin; what stands here
# is the float model's loss and the netlist the quantizer states.

DIFFUSION_CHECKPOINT = ROOT / "_train" / "diffusion" / "coconet" / "l48-h20-100k.ckpt"
PIECES = corpus.PIECES
GEN_VERILOG = ROOT / "_build" / "default" / "board" / "nexys-4" / "gen_verilog.exe"

# The masked loss of the golden checkpoint over the sheets below, MEASURED 2026-08-28
# against the functional model that `diffusion/model.py` carried before the Flax round.
DIFFUSION_LOSS = 0.193459
DIFFUSION_CROP = 128

# The golden candidate at T 128, G 5, N 512, AND IT IS WHAT THE FLASH HOLDS: the capture
# at seed 47872 read 840 bytes and 280 messages byte for byte against the twin.
DIFFUSION_NETLIST_MD5 = "ca16397aa3c91be2d8fe4c34736d0834"


def diffusion_gate_masks(sheets, crop):
    """The Bernoulli-half masks of G0: sheet i on the generator at seed i + 1, hidden
    exactly when u * 2^24 < 2^23. They come from the SHARED generator and not from either
    framework's draw, thus no change of a key rule can move them."""
    states = prng.states(np.arange(1, sheets + 1))
    hidden = np.zeros((sheets, crop, sheet_model.VOICES), dtype=bool)
    everyone = np.ones(sheets, dtype=bool)
    for step, voice in sheet_model.cell_order(crop):
        states, u = prng.uniform(states, everyone)
        hidden[:, step, voice] = u * 2.0**24 < float(1 << 23)
    return hidden


def test_g0_the_float_model_reads_its_measured_loss():
    """THE FLOAT MODEL DOES NOT MOVE. The sheets and the masks are deterministic, thus no
    draw of either framework enters and the number reads the FORWARD alone: every kernel,
    every fold of the norm, every plane, and the reader that loaded them. The tolerance
    holds two readings of 48 float32 layers that reduce in different orders."""
    need(DIFFUSION_CHECKPOINT, PIECES)
    pieces = corpus.load_pieces(str(PIECES))["valid"]
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
        nll = np.asarray(sheet_train.nll_of_logits(said, jnp.asarray(classes[rows])))
        values += [float(row[mask].mean()) for row, mask in zip(nll, hidden[rows])]
    here = float(np.mean(values))
    assert here == pytest.approx(DIFFUSION_LOSS, abs=TOLERANCE), (
        f"the model reads {here:.6f} and the golden checkpoint measured "
        f"{DIFFUSION_LOSS:.6f}"
    )


def test_g1_the_quantizer_states_the_golden_netlist(tmp_path):
    """THE CIRCUIT DOES NOT MOVE: one rounding, one exponent or one fold out of place
    moves a weight, and a moved weight moves the netlist. A different md5 says the
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
