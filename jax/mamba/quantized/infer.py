"""The walk of the state-space twin: what the board runs, step by step.

`quantized/model.py` holds the weights, the formats and the contract file; this half runs
them. THE CIRCUIT MUST EQUAL IT OPERATION FOR OPERATION, not approximately: a rewrite that
is algebraically equal and differently ordered is a different machine, and
`tests/test_rtl_mamba.py` holds the two together.

The style is functional and the engine is frozen: a step gives the engine after it, thus a
walk is a fold and no state hides in a mutable field. The loop, the chain and the draw
stand once in `ar_quantized.py` for both step-frame eras; what is here is era five's own
recurrence, its attention and the formats they name.

BOTH MODELS TAKE ONE STEP FOR ONE STEP. A block carries a state no window forgets, thus
the clamps of a walk are a finding and `Clamps` counts them: the formats of this era are
chosen with margin and not metered on a trained checkpoint.
"""

from typing import NamedTuple

import numpy as np

import ar_quantized
import quantized as q
from mamba import model as recurrence
from mamba.quantized import model as qmodel

# THE FORMATS THIS ERA NAMES OF ITS OWN; every other stands in `ar_quantized.py`.
# `V_Q` is the value rows of a block and of the attention rings, Q12 in int16. Era four's
# `KV_Q` names an attention ring's four tensors and nothing else, thus the two are one
# number and not one format.
V_Q = 12
S_Q = 12  # the state of the recurrence
ALPHA_Q = 15  # the decay of one step
BETA_Q = 15  # the input coefficient
# the gate product, in an int32: two Q12 values multiply and nothing truncates them before
# the norm that reads them
GATE_Q = 2 * V_Q


def matvec(y, weight, *, transposed, at, to):
    """One matvec column: the terms of a Q[at] vector against a row of the weight.
    [transposed] states that the image holds the tensor with its OUTER axis first, as it
    does for w_in, and the circuit reads that same order."""
    matrix = weight.values.T if transposed else weight.values
    return q.clamp16(ar_quantized.rescale(y @ matrix, at=at + weight.e, to=to))


class Clamps(NamedTuple):
    """The clamps of the walk, and the chances each one had. The formats of this era are
    chosen with margin and not metered on a trained checkpoint, thus a clamp that fires is
    the finding that says which is wrong -- and an error in the STATE carries forward,
    where era four could let a hot signal die with its window."""

    dt: int = 0
    dt_seen: int = 0
    beta: int = 0
    beta_seen: int = 0
    state: int = 0
    state_seen: int = 0

    def counting(self, name, hit, size):
        """[name] and its `_seen` twin, moved together: a counter that rose without its
        chances would read as a rate this walk never had"""
        seen = f"{name}_seen"
        return self._replace(
            **{
                name: getattr(self, name) + int(hit.sum()),
                seen: getattr(self, seen) + size,
            }
        )


class Engine(NamedTuple):
    """One running inference over a batch of walks. Everything is frozen: a step gives
    the engine after it, thus a walk is a fold and no state hides in a mutable field.
    THE STATE AND THE TAPS ARE THE MEMORY and the only things that survive a step."""

    twin: qmodel.Mamba
    state: np.ndarray  # [walks, blocks * d_in * n], Q12
    taps: np.ndarray  # [walks, blocks * channels * taps], Q12
    kc: np.ndarray  # [walks, attentions, ring, d], Q12 in a coarse byte
    vc: np.ndarray
    h: np.ndarray  # [walks, d], Q16
    position: int
    states: np.ndarray  # [walks], the generator of each walk
    clamps: Clamps


def create_engine(twin, seeds):
    """the origin of a batch of walks: a zero state, an empty tap ring, an empty key and
    value ring, and no residual"""
    twin.check_shape()
    shape = twin.widths
    walks, blocks = len(seeds), len(twin.blocks)
    rings = sum(1 for layer in twin.layers if layer.kind == recurrence.ZATTN)
    return Engine(
        twin=twin,
        state=np.zeros((walks, blocks * shape.d_in * shape.state), np.int64),
        taps=np.zeros((walks, blocks * shape.channels * shape.taps), np.int64),
        kc=np.zeros((walks, max(1, rings), twin.ring, shape.d), np.int64),
        vc=np.zeros((walks, max(1, rings), twin.ring, shape.d), np.int64),
        h=np.zeros((walks, shape.d), np.int64),
        position=0,
        states=q.engine_states(seeds),
        clamps=Clamps(),
    )


def block(engine, layer, ordinal, h, state, taps, tally):
    """One block of the trunk: the stream after the residual join, and the clamps it met.
    It writes the state and the taps of its own region IN PLACE, on copies the caller made
    for this step, as the state RAM of the circuit is written in place."""
    shape = engine.twin.widths
    d, d_in, heads, n = shape.d, shape.d_in, shape.heads, shape.state
    width, channels, head = shape.taps, shape.channels, shape.head
    position = engine.position
    y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)
    zxbcdt = matvec(y, layer.w_in, transposed=True, at=ar_quantized.Y_Q, to=V_Q)
    # the convolution: the step's input enters the ring, then a row of [width] terms for
    # each channel, then the SiLU chain over the sums
    tap_base = ordinal * channels * width
    slot = tap_base + (np.arange(channels) * width) + (position & (width - 1))
    taps[:, slot] = zxbcdt[:, d_in : d_in + channels]
    kernel, kernel_e = layer.conv.values, layer.conv.e
    accumulated = np.zeros((len(h), channels), np.int64)
    for k in range(width):
        # TAP k READS THE STEP k BACK, and it reads ZERO while the walk has not run k
        # steps -- thus the origin needs no clearing walk and the rule is a mux
        if position < k:
            continue
        back = tap_base + (np.arange(channels) * width) + ((position - k) & (width - 1))
        accumulated = accumulated + (taps[:, back] * kernel[:, k])
    scaled = ar_quantized.rescale(accumulated, at=V_Q + kernel_e, to=V_Q)
    xbc = ar_quantized.silu(q.clamp16(scaled))
    x = xbc[:, :d_in]
    b = xbc[:, d_in : d_in + n]
    c = xbc[:, d_in + n :]
    # the decay of each head: softplus of the biased draw, then one exp2
    raw = zxbcdt[:, d_in + channels :]
    dt = ar_quantized.softplus(raw + layer.dt_bias)
    rails = (dt == q.INT16_HIGH) | (dt == q.INT16_LOW)
    tally = tally.counting("dt", rails, dt.size)
    alpha = q.exp2_of_magnitude(q.apply_scale(layer.decay, qmodel.DECAY_Q_BITS, dt))
    # the state update and the readout, head by head. [beta] is the inject operand of the
    # head: [state] products of [dt] against B, written before the walk.
    base = ordinal * d_in * n
    read = np.zeros((len(h), d_in), np.int64)
    for hd in range(heads):
        wide = (dt[:, hd, None] * b) >> (V_Q + V_Q - BETA_Q)
        tally = tally.counting("beta", q.clamps16(wide), wide.size)
        beta = q.clamp16(wide)
        lanes = np.arange(hd * head, (hd + 1) * head)
        rows = base + (lanes[:, None] * n) + np.arange(n)
        updated = (
            (alpha[:, hd, None, None] * state[:, rows])
            + (x[:, lanes, None] * beta[:, None, :])
        ) >> ALPHA_Q
        tally = tally.counting("state", q.clamps16(updated), updated.size)
        state[:, rows] = q.clamp16(updated)
        # the readout reads the state the update just wrote, and the skip folds into the
        # row as its last term
        read[:, lanes] = q.clamp16(
            (
                (state[:, rows] * c[:, None, :]).sum(axis=-1)
                + (x[:, lanes] * layer.d_skip[hd])
            )
            >> S_Q
        )
    # THE GATE PRODUCT STAYS WIDE. Both operands are Q12 values well under one, thus a
    # truncation back to Q12 would keep five bits of a product that holds seventeen --
    # right before the norm, which does not care what scale it arrives in.
    gated = read * ar_quantized.silu(zxbcdt[:, :d_in])
    g = ar_quantized.rms_norm_q(gated, at=GATE_Q, width=d_in)
    return ar_quantized.join(h, layer.w_out, values=g, at=V_Q), tally


def attention(engine, layer, ordinal, h, embedding, kc, vc):
    """One attention layer, era four's with one addition: the query and the key read the
    JOINED vector -- the normed stream beside the normed embedding -- thus their walk is
    2d terms where the value's is d."""
    twin = engine.twin
    d, slots = twin.d, twin.ring
    newest = engine.position & (slots - 1)
    filled = min(engine.position + 1, slots)
    y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)
    joined = np.concatenate([y, embedding], axis=-1)

    def project(name, source):
        return matvec(
            source, getattr(layer, name), transposed=False, at=ar_quantized.Y_Q, to=V_Q
        )

    kc[:, ordinal, newest, :] = ar_quantized.coarse_to_ring(project("wk", joined))
    vc[:, ordinal, newest, :] = ar_quantized.coarse_to_ring(project("wv", y))
    # the rings of THIS attention site: slicing the ordinal here lets `attend` name none
    context = ar_quantized.attend(
        kc[:, ordinal],
        vc[:, ordinal],
        query=project("wq", joined),
        newest=newest,
        filled=filled,
        heads=twin.heads,
        span=twin.span,
        row_q=V_Q,
    )
    return ar_quantized.join(h, layer.wo, values=context, at=V_Q)


def feed_forward(twin, layer, h):
    """Era four's feed-forward as a layer of its own: one matvec and a ReLU, Q10. The ReLU
    stands AFTER the clamp of the matvec where the circuit takes it before, which is the
    same integer either way."""
    y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=twin.d)
    hidden = np.maximum(
        matvec(
            y, layer.w1, transposed=False, at=ar_quantized.Y_Q, to=ar_quantized.HID_Q
        ),
        0,
    )
    return ar_quantized.join(h, layer.w2, values=hidden, at=ar_quantized.HID_Q)


def layer_streams(engine, classes, phase):
    """The residual stream after the embed and after each layer of the step the engine
    would take next, in the order the circuit writes them. A frame gate says only THAT the
    circuit and the twin parted; this says where."""
    twin = engine.twin
    state, taps = engine.state.copy(), engine.taps.copy()
    kc, vc = engine.kc.copy(), engine.vc.copy()
    h = twin.head.embed(classes, phase)
    embedding = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=twin.d)
    written = [h]
    tally = engine.clamps
    for layer, ordinal in zip(twin.layers, twin.ordinals()):
        if layer.kind == recurrence.MAMBA:
            h, tally = block(engine, layer, ordinal, h, state, taps, tally)
        elif layer.kind == recurrence.ZATTN:
            h = attention(engine, layer, ordinal, h, embedding, kc, vc)
        else:
            h = feed_forward(twin, layer, h)
        written.append(h)
    return written, engine._replace(
        h=written[-1],
        state=state,
        taps=taps,
        kc=kc,
        vc=vc,
        position=engine.position + 1,
        clamps=tally,
    )


def forward(engine, classes, phase):
    """one step of the recurrence: the engine after it"""
    return layer_streams(engine, classes, phase)[1]


def next_step(engine):
    """one step of the walk -- `ar_quantized.next_step` over era five's own recurrence"""
    return ar_quantized.next_step(engine, forward)


def walk(twin, seeds, steps):
    """the classes of each step of the walk, and the draws behind them"""
    return ar_quantized.walk(create_engine(twin, seeds), steps, forward)
