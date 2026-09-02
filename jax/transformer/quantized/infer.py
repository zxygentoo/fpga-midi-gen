"""The walk of the step-frame twin: what the board runs, step by step.

`quantized/model.py` holds the weights, the formats and the contract file; this half runs
them. THE CIRCUIT MUST EQUAL IT OPERATION FOR OPERATION, not approximately: a rewrite that
is algebraically equal and differently ordered is a different machine, and
`tests/test_rtl_transformer.py` holds the two together.

The style is functional and the engine is frozen: a step gives the engine after it, thus a
walk is a fold and no state hides in a mutable field. The loop, the chain and the draw
stand once in `ar_quantized.py` for both step-frame eras; what is here is era four's own
trunk and the one format it names.
"""

from typing import NamedTuple

import numpy as np

import ar_quantized
import quantized as q
from transformer.quantized import model as qmodel

# THE ONE FORMAT THIS ERA NAMES OF ITS OWN; every other stands in `ar_quantized.py`.
# `KV_Q` is the query, the keys, the values and the context, Q12 in int16. Era five's
# `V_Q` is a 12 of its own and names a block's value rows as well, thus the two are one
# number and not one format.
KV_Q = 12


def projection(y, weight):
    """one of the three projections of a step: one matvec column, Q12 in int16"""
    return q.clamp16(
        ar_quantized.rescale(y @ weight.values, at=ar_quantized.Y_Q + weight.e, to=KV_Q)
    )


class Engine(NamedTuple):
    """One running inference over a batch of walks. Everything is frozen: a step gives
    the engine after it, thus a walk is a fold and no state hides in a mutable field.
    THE RINGS ARE THE CONTEXT -- one slot for each step of the window -- and a walk
    never reads an unwritten slot."""

    twin: qmodel.Transformer
    h: np.ndarray  # [walks, d], Q16 in int32
    kc: np.ndarray  # [walks, layers, slots, d], Q12 int16
    vc: np.ndarray
    position: int  # one forward for each step, thus this counts the steps as well
    states: np.ndarray  # [walks], the generator of each walk


def create_engine(twin, seeds):
    """The origin of a batch of walks: an empty ring, no residual, and the generator at
    the SEED AS IT STANDS. The lead-in is not here -- it is the first steps of the walk
    itself, thus a caller counts the steps the float sampler counts."""
    twin.check_shape()
    walks, d = len(seeds), twin.d
    rings = (walks, len(twin.layers), twin.context, d)
    return Engine(
        twin=twin,
        h=np.zeros((walks, d), np.int64),
        kc=np.zeros(rings, np.int64),
        vc=np.zeros(rings, np.int64),
        position=0,
        states=q.engine_states(seeds),
    )


def forward(engine, classes, phase):
    """one step through the engine: the engine after it"""
    twin = engine.twin
    d, slots = twin.d, twin.context
    newest = engine.position & (slots - 1)
    filled = min(engine.position + 1, slots)
    kc, vc = engine.kc.copy(), engine.vc.copy()
    h = twin.head.embed(classes, phase)
    for at, layer in enumerate(twin.layers):
        y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)

        query, key, value = (
            projection(y, getattr(layer, name)) for name in ("wq", "wk", "wv")
        )
        kc[:, at, newest, :] = ar_quantized.coarse_to_ring(key)
        vc[:, at, newest, :] = ar_quantized.coarse_to_ring(value)
        # the rings of ONE layer: slicing the layer axis here lets `attend` name none
        context = ar_quantized.attend(
            kc[:, at],
            vc[:, at],
            query=query,
            newest=newest,
            filled=filled,
            heads=twin.heads,
            span=twin.slope_span,
            row_q=KV_Q,
        )
        h = ar_quantized.join(h, layer.wo, values=context, at=KV_Q)
        y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)
        hidden = q.clamp16(
            np.maximum(
                ar_quantized.rescale(
                    y @ layer.w1.values,
                    at=ar_quantized.Y_Q + layer.w1.e,
                    to=ar_quantized.HID_Q,
                ),
                0,
            )
        )
        h = ar_quantized.join(h, layer.w2, values=hidden, at=ar_quantized.HID_Q)
    return engine._replace(h=h, kc=kc, vc=vc, position=engine.position + 1)


def next_step(engine):
    """one step of the walk -- `ar_quantized.next_step` over era four's own trunk"""
    return ar_quantized.next_step(engine, forward)


def walk(twin, seeds, steps):
    """the classes of each step of the walk, and the draws behind them"""
    return ar_quantized.walk(create_engine(twin, seeds), steps, forward)
