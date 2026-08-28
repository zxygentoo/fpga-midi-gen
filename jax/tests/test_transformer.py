"""Era four, the step-frame transformer of docs/transformer.md: the chained head.

ONE RULE OF THIS ERA FAILS SILENTLY, and it is the chain. Under a chain wired the wrong
way round the shapes stay right, the loss still falls, and what is lost is the joint
choice that is the whole reason the chain exists: measured on this era, four heads that
drew in parallel cost 0.3157 nats for each step -- 0.456 bits, sixteen times the seed
spread.

The rest of the era is held elsewhere, and deliberately. `test_parity.py` holds the
forward against the OCaml reference over the canonical valid windows, and the walk against
its player step line for step line; `test_draw.py` holds the arithmetic of the draw;
`test_train.py` holds the loop and the drawn tables. What is left is the one rule no
referee outside this repository can see, because a chain drawn upward is still a model
that trains and still a model that plays.

The head itself lives in jax/nn.py, where era five reads it too; it is read here through
this era's own module, as the era's trainer and sampler read it.
"""

import jax
import numpy as np

import data
from transformer import model


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
