"""The parity gates of the JAX seam.

Gate A: the JAX forward must reproduce the OCaml referee. checkpoint_tool eval writes
gate.safetensors -- a fixed valid batch and the two losses the OCaml Evaluation protocol
computed on it. The gate loads the same checkpoint and batch in JAX and demands the same
numbers. A pass proves the two forward functions agree everywhere the loss can see; only
then do GPU training runs mean anything.

The gate artifacts are generated, never committed. From the repository root:

    dune exec bin/checkpoint_tool.exe -- eval -ckpt _train/ref-d64.ckpt \
        -out jax/_data/gate.safetensors

The tests skip when the artifacts are absent.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

import model

JAX_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = JAX_ROOT.parent / "_train" / "ref-d64.ckpt"
GATE = JAX_ROOT / "_data" / "gate.safetensors"
LAYERS, HEADS = 2, 4
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
    params = model.load_params(str(CHECKPOINT), layers=LAYERS)
    return {
        "params": params,
        "codes": jnp.asarray(tensors["codes"]),
        "phases": jnp.asarray(tensors["phases"]),
        "masks": jnp.asarray(unpack_masks(tensors["masks"])),
        "loss_unmasked": float(tensors["loss_unmasked"][0]),
        "loss_masked": float(tensors["loss_masked"][0]),
    }


def test_gate_a_unmasked(gate):
    ours = float(model.loss(gate["params"], gate["codes"], gate["phases"], heads=HEADS))
    assert ours == pytest.approx(gate["loss_unmasked"], abs=TOLERANCE)


def test_gate_a_masked(gate):
    ours = float(
        model.masked_loss(
            gate["params"], gate["codes"], gate["phases"], gate["masks"], heads=HEADS
        )
    )
    assert ours == pytest.approx(gate["loss_masked"], abs=TOLERANCE)
