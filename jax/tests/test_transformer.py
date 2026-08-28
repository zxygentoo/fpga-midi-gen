"""Era four, the step-frame transformer of docs/transformer.md: the chained head.

ONE RULE OF THIS ERA FAILS SILENTLY, and it is the chain. Under a chain wired the wrong
way round the shapes stay right, the loss still falls, and what is lost is the joint
choice that is the whole reason the chain exists: measured on this era, four heads that
drew in parallel cost 0.3157 nats for each step -- 0.456 bits, sixteen times the seed
spread.

The rest of the era is held elsewhere, and deliberately. `test_parity.py` holds the
forward to its measured loss and the QUANTIZER through the netlist the elaboration states;
`test_draw.py` holds the arithmetic of the draw; `test_quantized.py` holds the integer
rules the quantizer stands on; `test_train.py` holds the loop and the drawn tables. What
is left is the one rule no referee outside this repository can see, because a chain drawn
upward is still a model that trains and still a model that plays -- and the CONTRACT FILE,
whose round trip is the seam itself.

The head itself lives in jax/nn.py, where era five reads it too; it is read here through
this era's own module, as the era's trainer and sampler read it.
"""

import jax
import numpy as np
import pytest

import data
import nn
from transformer import model, quantized


def test_the_chain_conditions_downward():
    """Each seat reads what the seats above it drew, and nothing reads a seat below."""
    params = {
        "seats": jax.random.normal(jax.random.PRNGKey(0), (data.SEATS, data.CLASSES, 8))
    }
    h = np.zeros((1, 3, 8), dtype=np.float32) + 0.5
    base = np.ones((1, 3, data.SEATS), dtype=np.int32)

    def logits(drawn):
        return np.asarray(model.seat_logits(params, h, drawn))

    # the soprano is drawn first, thus it conditions on nothing and every seat under it
    # moves when it changes
    soprano = base.copy()
    soprano[..., 3] = 2
    from_base, from_soprano = logits(base), logits(soprano)
    assert np.allclose(from_base[..., 3, :], from_soprano[..., 3, :])
    for seat in (2, 1, 0):
        assert not np.allclose(from_base[..., seat, :], from_soprano[..., seat, :])

    # the bass is drawn last, thus no seat reads it and the whole readout stands still
    bass = base.copy()
    bass[..., 0] = 2
    assert np.allclose(from_base, logits(bass))


# ==================================================================== #
# The contract file: the seam to the elaboration                       #
# ==================================================================== #


def tiny(seed=5, d=8, layers=2, heads=2, context=16):
    """a drawn model of the era's shape, small enough for a test"""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2 + model.PER_LAYER * layers)
    at = iter(keys)
    params = {
        "seats": jax.random.normal(next(at), (data.SEATS, data.CLASSES, d)) * 0.02,
        "phase": jax.random.normal(next(at), (nn.PHASE_BUCKETS, d)) * 0.02,
        "layers": [
            {
                name: jax.random.normal(next(at), shape) * 0.02
                for name, shape in zip(
                    model.LAYER_TENSORS,
                    [(d, d), (d, d), (d, d), (d, d), (d, 4 * d), (4 * d, d)],
                )
            }
            for _ in range(layers)
        ],
    }
    return quantized.Quantized.of(params, heads=heads, context=context)


def test_the_two_tables_share_one_exponent():
    """The seat rows and the phase row ADD, row for row -- the embedding sums them and the
    Embed op of the circuit walks them as one tensor -- thus one exponent covers both, and
    it comes from the larger peak. Two exponents inside one sum would be two formats."""
    twin = tiny()
    assert twin.tensors[0][1] == twin.tensors[1][1]
    # the layer tensors take their own; nothing forces them together
    assert len(twin.tensors) == len(nn.TABLES) + model.PER_LAYER * twin.layers


def test_the_shape_numbers_that_are_in_no_tensor_travel_in_the_file():
    """The heads only split the width at run time, ALiBi holds no position table, and the
    context is a choice of the draw, thus none of the three is in a checkpoint. The
    ELABORATION reads a file and no flag, therefore they travel here."""
    twin = tiny(heads=4, context=32)
    assert (twin.heads, twin.context, twin.slope_span) == (4, 32, 4)
    # what IS in the tensors is read from them and never stated twice
    assert (twin.d, twin.layers) == (8, 2)


def test_the_contract_file_round_trips_exactly(tmp_path):
    """`save` then `load` is the identity: the seam carries the whole model and nothing of
    it is re-derived on either side of the file."""
    twin = tiny()
    path = tmp_path / "tiny.int8"
    quantized.save(path, twin)
    read = quantized.load(path)
    assert (read.heads, read.context, read.slope_span) == (
        twin.heads,
        twin.context,
        twin.slope_span,
    )
    assert (read.temper.q_value, read.temper.q) == (twin.temper.q_value, twin.temper.q)
    assert read.temper.temperature == twin.temper.temperature
    assert read.min_weight == twin.min_weight
    assert len(read.tensors) == len(twin.tensors)
    for (here, e), (there, then) in zip(read.tensors, twin.tensors):
        assert np.array_equal(here, there) and e == then


def test_a_shape_the_circuit_cannot_hold_refuses_at_the_file():
    """The arithmetic of the circuit is shifts, thus a width that is not a power of two or
    a head width that is not a power of four has no circuit at all. It refuses here, where
    a build fails loudly, and never inside a walk."""
    with pytest.raises(ValueError):
        quantized.check_shape(tiny(d=12))
    with pytest.raises(ValueError):
        quantized.check_shape(tiny(context=20))
    # d 8 over 4 heads is a head width of 2, which is not a power of four
    with pytest.raises(ValueError):
        quantized.check_shape(tiny(d=8, heads=4))
