"""The parity gates of the JAX seam.

Gate A: the JAX forward must reproduce the OCaml referee. checkpoint_tool eval writes
gate.safetensors -- a fixed valid batch, the loss the OCaml Evaluation protocol computed
on it, and the config that made both. The gate loads the same checkpoint and batch in JAX
and demands the same number. A pass proves the two forward functions agree everywhere the
loss can see; only then do GPU training runs mean anything.

The config travels in the file, so this side states no number of its own. Held here, the
heads and the ALiBi span would go on matching a default while the OCaml run moved, and
the gate would pass two different models.

The gate artifacts are generated, never committed. From the repository root:

    dune exec bin/checkpoint_tool.exe -- eval -ckpt _train/ref-d64.ckpt \
        -out jax/_data/gate.safetensors

The tests skip when the artifacts are absent, and fail when the gate file predates the
entries they read: an absent artifact is a clean tree, a stale one is unproven parity.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from transformer import model

JAX_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = JAX_ROOT.parent / "_train" / "ref-d64.ckpt"
GATE = JAX_ROOT / "_data" / "gate.safetensors"
GATE_ENTRIES = ("codes", "phases", "masks", "loss", "heads", "span")
TOLERANCE = 2e-4


def unpack_masks(words):
    """[batch, length, 8] int32 words -> [batch, length, 256] bool, LSB first."""
    view = words.astype("<i4").view(np.uint8)
    return np.unpackbits(
        view.reshape(words.shape[0], words.shape[1], -1), axis=-1, bitorder="little"
    ).astype(bool)


@pytest.fixture(scope="module")
def gate():
    if not (CHECKPOINT.exists() and GATE.exists()):
        pytest.skip("gate artifacts absent: run checkpoint_tool eval first")
    tensors = load_file(str(GATE))
    missing = [name for name in GATE_ENTRIES if name not in tensors]
    if missing:
        raise AssertionError(
            f"{GATE} holds no {missing}: it predates the entries this gate reads. "
            "Regenerate it with the checkpoint_tool line in this file's header."
        )
    return {
        "params": model.load_params(str(CHECKPOINT)),
        "codes": jnp.asarray(tensors["codes"]),
        "phases": jnp.asarray(tensors["phases"]),
        "masks": jnp.asarray(unpack_masks(tensors["masks"])),
        "loss": float(tensors["loss"][0]),
        "heads": int(tensors["heads"][0]),
        "span": int(tensors["span"][0]),
    }


def test_gate_a_loss(gate):
    ours = float(
        model.loss(
            gate["params"],
            gate["codes"],
            gate["phases"],
            gate["masks"],
            heads=gate["heads"],
            span=gate["span"],
        )
    )
    assert ours == pytest.approx(gate["loss"], abs=TOLERANCE)
