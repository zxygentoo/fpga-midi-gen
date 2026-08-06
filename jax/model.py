"""The JAX port of the forward pass of lib/transformer.ml.

The OCaml network is the spec -- decoder-only, ALiBi, a bar-phase table, scale-free
RMSNorm (eps 1e-6 on the mean square), no biases, tied embedding -- and this file must
compute the same function: gate_a.py proves it against the OCaml referee's numbers.
Matmul precision is pinned to true float32, no TF32, so the two sides agree tightly.

Checkpoints are Kaun safetensors: tensors named "0".."N" in construction order --
embed [vocab, d], phase [16, d], then per layer wq wk wv wo [d, d], w1 [d, 4d],
w2 [4d, d].
"""

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

jax.config.update("jax_default_matmul_precision", "float32")

VOCAB = 256
PHASE_BUCKETS = 16
LAYER_TENSORS = ("wq", "wk", "wv", "wo", "w1", "w2")


def load_params(path, layers):
    tensors = load_file(path)
    return {
        "embed": jnp.asarray(tensors["0"]),
        "phase": jnp.asarray(tensors["1"]),
        "layers": [
            dict(
                zip(
                    LAYER_TENSORS,
                    (jnp.asarray(tensors[str(2 + 6 * l + i)]) for i in range(6)),
                )
            )
            for l in range(layers)
        ],
    }


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def attention_bias(heads, length):
    """ALiBi plus the causal wall, [1, heads, length, length]."""
    pos = jnp.arange(length, dtype=jnp.float32)
    distance = pos[:, None] - pos[None, :]
    slopes = -(2.0 ** (-8.0 * (jnp.arange(heads, dtype=jnp.float32) + 1.0) / heads))
    alibi = slopes[None, :, None, None] * distance[None, None, :, :]
    wall = jnp.triu(jnp.ones((length, length), dtype=jnp.float32), k=1) * -1e9
    return alibi + wall[None, None, :, :]


def _dropout(x, rate, key):
    keep = 1.0 - rate
    return jnp.where(jax.random.bernoulli(key, keep, x.shape), x / keep, 0.0)


def logits(params, codes, phases, *, heads, dropout=0.0, key=None):
    """codes, phases: [batch, length] int32 -> [batch, length, vocab] float32.

    dropout > 0 needs a PRNG [key]; it drops the embedding sum and each residual
    branch. The default is the exact forward of the OCaml model."""
    batch, length = codes.shape
    d = params["embed"].shape[1]
    head_d = d // heads
    bias = attention_bias(heads, length)

    def drop(x):
        nonlocal key
        if dropout <= 0.0:
            return x
        key, sub = jax.random.split(key)
        return _dropout(x, dropout, sub)

    h = drop(params["embed"][codes] + params["phase"][phases])
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
    return rms_norm(h) @ params["embed"].T


def _cross_entropy(raw, labels):
    logp = jax.nn.log_softmax(raw, axis=-1)
    picked = jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked)


def loss(params, codes, phases, *, heads, dropout=0.0, key=None):
    """Plain cross entropy over the whole vocabulary: codes [batch, length + 1]."""
    raw = logits(params, codes[:, :-1], phases, heads=heads, dropout=dropout, key=key)
    return _cross_entropy(raw, codes[:, 1:])


def masked_loss(params, codes, phases, masks, *, heads, dropout=0.0, key=None):
    """The control loss of the mask era: the grammar inside the softmax.

    masks: [batch, length, vocab] bool -- the legal set of each label draw."""
    raw = logits(params, codes[:, :-1], phases, heads=heads, dropout=dropout, key=key)
    return _cross_entropy(raw + jnp.where(masks, 0.0, -1e9), codes[:, 1:])
