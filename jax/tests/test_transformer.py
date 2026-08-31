"""Era four, the step-frame transformer of docs/transformer.md: the chained head.

ONE RULE OF THIS ERA FAILS SILENTLY, and it is the chain. Under a chain wired the wrong
way round the shapes stay right, the loss still falls, and what is lost is the joint
choice that is the whole reason the chain exists: measured on this era, four heads that
drew in parallel cost 0.3157 nats for each step -- 0.456 bits, sixteen times the seed
spread.

The rest of the era is held elsewhere, and deliberately. `test_parity.py` holds the
forward to its measured loss and the QUANTIZER through the netlist the elaboration states;
`test_sample.py` holds the arithmetic of the draw; `test_quantized.py` holds the integer
rules the quantizer stands on; `test_train.py` holds the loop. What is left is the one
rule no referee outside this repository can see, because a chain drawn upward is still a
model that trains and still a model that plays -- and the CONTRACT FILE, whose round trip
is the seam itself.

The head itself lives in jax/ar_model.py, where era five reads it too (its twin is
`ar_quantized.QuantizedHead`); it is read here through this era's own module, as the era's
trainer and sampler read it.
"""

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

import ar_quantized
import corpus
from quantized import EXPONENTS, max_exponent
from tests.models import drawn_transformer, transformer_twin
from transformer import model, quantized


def test_the_chain_conditions_downward():
    """Each seat reads what the seats above it drew, and nothing reads a seat below."""
    head = drawn_transformer().head
    h = np.zeros((1, 3, head.d), dtype=np.float32) + 0.5
    base = np.ones((1, 3, corpus.SEATS), dtype=np.int32)

    def logits(drawn_classes):
        return np.asarray(head.logits(h, drawn_classes))

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


def test_the_twin_carries_the_float_models_skeleton():
    """THE TWO TREES ARE ONE TREE, and that is what makes the twin auditable: a reader
    puts `held.layers[k].wq` beside `twin.layers[k].wq` and reads one tensor against its
    own quantization, with nothing to align by hand."""
    held = drawn_transformer()
    twin = quantized.QuantizedTransformer.of(held, context=16)
    assert len(twin.layers) == len(held.layers)
    assert twin.head.seats.shape == held.head.seats.shape
    assert twin.head.phase.shape == held.head.phase.shape
    for here, there in zip(twin.layers, held.layers):
        for name in model.LAYER_TENSORS:
            assert getattr(here, name).values.shape == getattr(there, name).shape


def test_the_two_tables_take_the_larger_peaks_exponent():
    """The seat rows and the phase row ADD, row for row -- the embedding sums them and the
    Embed op of the circuit walks them as one tensor -- thus ONE exponent covers both. The
    module holds one field, so no caller can break the rule; what is still a choice is
    that the exponent comes from the LARGER peak, and a table quantized at another
    tensor's exponent clamps."""
    held = drawn_transformer()
    # the phase table is lifted far past the seats, thus the shared exponent is its own
    held.head.take([held.head.seats[...] * 0.01, held.head.phase[...] * 4.0])
    twin = ar_quantized.QuantizedHead.of(held.head)
    peak = float(np.abs(np.asarray(held.head.phase[...])).max())
    assert twin.e == max_exponent(peak)
    assert np.abs(twin.seats).max() < 127, "the smaller table does not reach the rail"


def test_each_layer_tensor_takes_its_own_exponent():
    """Only the two tables are forced together, and the reason is that their rows add. A
    layer tensor stands alone in its own product, thus it reads its own peak and a
    neighbour's scale does not reach it."""
    held = drawn_transformer()
    layer = held.layers[0]
    # wq alone is lifted; every other tensor of the model stands where it stood
    layer.take(
        [
            tensor * 8.0 if name == "wq" else tensor
            for name, tensor in zip(model.LAYER_TENSORS, layer.tensors())
        ]
    )
    lifted = quantized.QuantizedTransformer.of(held, context=16).layers[0]
    plain = quantized.QuantizedTransformer.of(drawn_transformer(), context=16).layers[0]
    assert lifted.wq.e == plain.wq.e - 3, "eight times the peak is three exponents down"
    for name in model.LAYER_TENSORS[1:]:
        assert getattr(lifted, name).e == getattr(plain, name).e


def test_the_shape_numbers_that_are_in_no_tensor_travel_in_the_file():
    """The heads only split the width at run time, ALiBi holds no position table, and the
    context is a choice of the draw, thus none of the three is in a checkpoint. The
    ELABORATION reads a file and no flag, therefore they travel here."""
    twin = transformer_twin(heads=4, context=32)
    assert (twin.heads, twin.context, twin.slope_span) == (4, 32, 4)
    # what IS in the tensors is read from them and never stated twice
    assert (twin.d, len(twin.layers)) == (8, 2)


def test_the_contract_file_round_trips_exactly(tmp_path):
    """`save` then `load` is the identity: the seam carries the whole model and nothing of
    it is re-derived on either side of the file."""
    twin = transformer_twin()
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
    assert len(read.every_tensor()) == len(twin.every_tensor())
    for here, there in zip(read.every_tensor(), twin.every_tensor()):
        assert np.array_equal(here.values, there.values) and here.e == there.e


def test_a_file_whose_two_tables_disagree_is_refused(tmp_path):
    """`QuantizedHead` holds ONE exponent, thus this side cannot state two; a FILE can,
    and a reader that took the first and dropped the second would sum two formats in
    silence."""
    path = tmp_path / "tiny.int8"
    quantized.save(path, transformer_twin())
    tensors = load_file(str(path))
    tensors[EXPONENTS][1] += 1
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="share one exponent"):
        quantized.load(path)


def test_a_shape_the_circuit_cannot_hold_refuses_at_the_file():
    """The arithmetic of the circuit is shifts, thus a width that is not a power of two or
    a head width that is not a power of four has no circuit at all. It refuses here, where
    a build fails loudly, and never inside a walk."""
    with pytest.raises(ValueError):
        quantized.check_shape(transformer_twin(d=12))
    with pytest.raises(ValueError):
        quantized.check_shape(transformer_twin(context=20))
    # d 8 over 4 heads is a head width of 2, which is not a power of four
    with pytest.raises(ValueError):
        quantized.check_shape(transformer_twin(d=8, heads=4))


def test_the_lead_in_draws_nothing_and_moves_no_generator():
    """One bar of silence opens the walk and the generator does not move through it. A
    twin that spent a uniform there would draw a different piece from the same seed, and
    every step of it would be legal music.

    IT IS THE TWIN AGAINST ITSELF and no circuit is in it, thus it stands here and not in
    `test_rtl_transformer.py`: a gate that mounts a driver it never runs skips on a tree
    with no `dune build` and gates nothing there."""
    twin = transformer_twin(layers=1)
    played, draws = quantized.walk(twin, [1, 7], ar_quantized.LEAD + 2)
    assert (played[:, : ar_quantized.LEAD] == 0).all(), "the lead-in is not silent"
    assert all(not taken for taken in draws[: ar_quantized.LEAD]), "the lead-in drew"
    assert all(len(taken) == 4 for taken in draws[ar_quantized.LEAD :])
    # the walks of a batch are independent: seed 7 draws what seed 7 draws alone
    alone, _ = quantized.walk(twin, [7], ar_quantized.LEAD + 2)
    assert np.array_equal(alone[0], played[1])
