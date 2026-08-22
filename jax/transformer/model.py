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

from nn import (
    PHASE_BUCKETS,
    SLOPE_SPAN,
    TABLES,
    attention_bias,
    dropout_masks,
    embed,
    rms_norm,
    seat_logits,
    seat_nll_of_hidden,
)

LAYER_TENSORS = ("wq", "wk", "wv", "wo", "w1", "w2")
PER_LAYER = len(LAYER_TENSORS)


def load_params(path):
    """The two tables, then the tensors of each layer, in construction order."""
    tensors = load_file(path)
    count = len(tensors)
    if count < len(TABLES) + PER_LAYER or (count - len(TABLES)) % PER_LAYER:
        raise ValueError(
            f"{path}: {count} tensors is not {TABLES} and {PER_LAYER} for each layer"
        )

    def layer_at(index):
        base = len(TABLES) + PER_LAYER * index
        return {
            name: jnp.asarray(tensors[str(base + at)])
            for at, name in enumerate(LAYER_TENSORS)
        }

    params = {name: jnp.asarray(tensors[str(at)]) for at, name in enumerate(TABLES)}
    params["layers"] = [
        layer_at(index) for index in range((count - len(TABLES)) // PER_LAYER)
    ]
    return params


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
        return x * dropout_masks(sub, dropout, x.shape)

    def split_heads(x):
        return x.reshape(batch, length, heads, head_d).transpose(0, 2, 1, 3)

    h = drop(embed(params, classes) + params["phase"][phases])
    for layer in params["layers"]:
        normed = rms_norm(h)
        q = split_heads(normed @ layer["wq"])
        k = split_heads(normed @ layer["wk"])
        v = split_heads(normed @ layer["wv"])
        scores = q @ k.transpose(0, 1, 3, 2) * (1.0 / jnp.sqrt(float(head_d))) + bias
        context = jax.nn.softmax(scores, axis=-1) @ v
        merged = context.transpose(0, 2, 1, 3).reshape(batch, length, d)
        h = h + drop(merged @ layer["wo"])
        h = h + drop(jnp.maximum(rms_norm(h) @ layer["w1"], 0.0) @ layer["w2"])
    return h


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
    return seat_nll_of_hidden(params, h, labels)
