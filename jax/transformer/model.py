"""The step-frame transformer of docs/transformer_model.md.

One step of music is one position. Four voice classes enter through four tables that sum,
and they leave through the same four tables in a chain from the soprano down.

The network under the head is a decoder with no bias terms, RMSNorm before each sublayer,
ALiBi for the position, and d_ff = 4 d. Matmul precision is pinned to true float32, no
TF32.

Checkpoints are safetensors: tensors "0".."N" in construction order -- seats [4, 48, d],
phase [16, d], then per layer wq wk wv wo [d, d], w1 [d, 4d], w2 [4d, d].
"""

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

from data import SEATS

jax.config.update("jax_default_matmul_precision", "float32")

# The slope of head k is 2^-(SLOPE_SPAN (k+1) / heads). Elected 2026-08-18 over spans 4,
# 8, 16, 24 and 64: the means of 4 and 8 are a dead heat, and the VARIANCE is the finding
# -- 5 to 7 times tighter over six seeds, replicated at two step budgets. Every head is
# then local, and seeds stop latching onto whatever distant structure their init favours.
SLOPE_SPAN = 4
PHASE_BUCKETS = 16
TABLES = ("seats", "phase")
LAYER_TENSORS = ("wq", "wk", "wv", "wo", "w1", "w2")


def load_params(path):
    """The two tables, then six tensors for each layer."""
    tensors = load_file(path)
    count = len(tensors)
    if count < len(TABLES) + 6 or (count - len(TABLES)) % 6:
        raise ValueError(
            f"{path}: {count} tensors is not {TABLES} and six for each layer"
        )
    layers = (count - len(TABLES)) // 6
    params = {name: jnp.asarray(tensors[str(at)]) for at, name in enumerate(TABLES)}
    params["layers"] = [
        dict(
            zip(
                LAYER_TENSORS,
                (
                    jnp.asarray(tensors[str(len(TABLES) + 6 * layer + i)])
                    for i in range(6)
                ),
            )
        )
        for layer in range(layers)
    ]
    return params


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def attention_bias(heads, length, span=SLOPE_SPAN):
    """ALiBi plus the causal wall, [1, heads, length, length]."""
    pos = jnp.arange(length, dtype=jnp.float32)
    distance = pos[:, None] - pos[None, :]
    slopes = -(2.0 ** (-span * (jnp.arange(heads, dtype=jnp.float32) + 1.0) / heads))
    alibi = slopes[None, :, None, None] * distance[None, None, :, :]
    wall = jnp.triu(jnp.ones((length, length), dtype=jnp.float32), k=1) * -1e9
    return alibi + wall[None, None, :, :]


def _dropout(x, rate, key):
    keep = 1.0 - rate
    return jnp.where(jax.random.bernoulli(key, keep, x.shape), x / keep, 0.0)


def embed(params, classes):
    """The input of one position: the four seat rows sum, then the bar phase.

    A shared table with a voice tag cannot work here, and the reason is arithmetic and not
    capacity. Every step carries all four seats, thus the sum of the four tags is the same
    vector at every position -- a bias, which carries nothing -- and what remains is
    symmetric in the four codes. A soprano on 72 over a bass on 48 would give the vector of
    a soprano on 48 under a bass on 72, and the voices would be thrown away on the way in.
    Four tables break the symmetry, and no voice tag is then necessary anywhere."""
    return sum(params["seats"][seat][classes[..., seat]] for seat in range(SEATS))


def hidden(params, classes, phases, *, heads, dropout=0.0, key=None, span=SLOPE_SPAN):
    """classes [batch, length, SEATS] -> [batch, length, d], the residual stream after the
    last layer and before any readout.

    dropout > 0 needs a PRNG [key]; it drops the embedding sum and each residual branch."""
    batch, length = classes.shape[0], classes.shape[1]
    d = params["seats"].shape[-1]
    head_d = d // heads
    bias = attention_bias(heads, length, span)

    def drop(x):
        nonlocal key
        if dropout <= 0.0:
            return x
        key, sub = jax.random.split(key)
        return _dropout(x, dropout, sub)

    h = drop(embed(params, classes) + params["phase"][phases])
    for layer in params["layers"]:
        normed = rms_norm(h)

        def split(x):
            return x.reshape(batch, length, heads, head_d).transpose(0, 2, 1, 3)

        q = split(normed @ layer["wq"])
        k = split(normed @ layer["wk"])
        v = split(normed @ layer["wv"])
        scores = q @ k.transpose(0, 1, 3, 2) * (1.0 / jnp.sqrt(float(head_d))) + bias
        context = jax.nn.softmax(scores, axis=-1) @ v
        merged = context.transpose(0, 2, 1, 3).reshape(batch, length, d)
        h = h + drop(merged @ layer["wo"])
        h = h + drop(jnp.maximum(rms_norm(h) @ layer["w1"], 0.0) @ layer["w2"])
    return h


def seat_logits(params, h, drawn):
    """The chained head: [batch, length, d] -> [batch, length, SEATS, CLASSES].

    Each seat reads the stream that the seats above it have already written:

        h3 = h                   logits(seat 3) = E[3] . rms(h3)
        h2 = h3 + E[3][c3]       logits(seat 2) = E[2] . rms(h2)
        h1 = h2 + E[2][c2]       logits(seat 1) = E[1] . rms(h1)
        h0 = h1 + E[1][c1]       logits(seat 0) = E[0] . rms(h0)

    [drawn] holds the classes the chain conditions on -- the true frame in training, where
    all four heads then run in one pass with no sampling, and the drawn seats at the draw.
    Only seats 3, 2 and 1 are read.

    The chain runs from the soprano down, which keeps the one decision the ear accepted:
    the top voice is chosen first and conditions on no voice under it, as the music is
    written. Four heads that drew in parallel would make the voices conditionally
    independent, and a chord is a joint choice: measured on this model, that costs 0.3157
    nats for each step -- 0.456 bits, sixteen times the seed spread. The chain removes the
    cost for no parameters at all -- parallel heads need the same four tables -- and three
    adds of a vector.

    The table that reads a voice is the table that writes it, which is the tied embedding
    of the era, one time for each seat. Untying the readout was measured and is null: it
    buys nothing and costs three block RAM tiles. What the chain adds is also what the next
    position reads: the input embedding of step t+1 is a3 + a2 + a1 + a0, and the chain
    assembles it one voice at a time."""
    seats = params["seats"]
    stream = h
    logits = [None] * SEATS
    for seat in reversed(range(SEATS)):
        logits[seat] = rms_norm(stream) @ seats[seat].T
        if seat:
            stream = stream + seats[seat][drawn[..., seat]]
    return jnp.stack(logits, axis=-2)


def seat_nll(params, classes, phases, *, heads, dropout=0.0, key=None, span=SLOPE_SPAN):
    """The negative log likelihood of every voice of every step: classes
    [batch, length + 1, SEATS] -> [batch, length, SEATS].

    The caller reduces. The loss does not carry across the encoding and neither does a
    per-prediction mean: report nats for each step, which is the sum over the seats."""
    labels = classes[:, 1:]
    h = hidden(
        params,
        classes[:, :-1],
        phases,
        heads=heads,
        dropout=dropout,
        key=key,
        span=span,
    )
    logp = jax.nn.log_softmax(seat_logits(params, h, labels), axis=-1)
    return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
