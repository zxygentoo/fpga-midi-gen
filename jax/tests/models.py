"""The drawn models the gates share, at shapes a test can afford.

IT DOES NOT BEGIN WITH `test_`, AND THAT IS THE WHOLE POINT. `tests/` carries no
`__init__.py`, thus `from tests.test_transformer import tiny` imports a SECOND module
object of a file pytest has already collected under its own rootdir name: two copies of
every constant, two draws of every model, and a `__pycache__` that holds both. Anything
one gate module needs from another moves here instead.

Each name says its era, because this module holds both and a bare `drawn` in it would
say neither.
"""

from mamba import model as recurrence
from mamba import train as recurrence_train
from transformer import model as step
from transformer import quantized as step_twin


def drawn_transformer(seed=5, d=8, layers=2, heads=2):
    """a drawn float model of era four's shape, small enough for a test"""
    return step.Transformer.drawn(seed, d, layers, heads=heads)


def transformer_twin(seed=5, d=8, layers=2, heads=2, context=16):
    """the twin of a drawn model of era four's shape"""
    return step_twin.QuantizedTransformer.of(
        drawn_transformer(seed, d, layers, heads), context=context
    )


# Era five's small shape. d 32 over 2 heads gives an attention head width of 16, a power
# of FOUR, which is what `ar_quantized.score_shift` needs to divide by its square root in
# one shift; the OCaml side draws at the same numbers, `Model.For_test.shape`.
MAMBA_SHAPE = {
    "d": 32,
    "layers": 3,
    "heads": 2,
    "state": 8,
    "expand": 2,
    "conv_scale": 0.5,
    "half_lives": None,
    "attention_at": (),
}


def drawn_mamba(seed=3, taps=recurrence.CONV_TAPS, **over):
    """a drawn float model at era five's small shape, or a widening of it"""
    return recurrence.Mamba.drawn(seed, taps=taps, **(MAMBA_SHAPE | over))


def plan_of(spelt, **over):
    """a drawn model of the plan spelt out, at era five's small shape or a widening of
    it"""
    letters = [recurrence_train.PLAN_LETTERS[c] for c in spelt.lower()]
    return drawn_mamba(spelt=letters, layers=len(letters), **over)
