"""The JAX port of the forward pass of lib/transformer/transformer.ml.

The OCaml network is the spec -- decoder-only, ALiBi, a bar-phase table, scale-free
RMSNorm (eps 1e-6 on the mean square), no biases, tied embedding -- and this file must
compute the same function: tests/test_parity.py proves it against the OCaml referee's
numbers. Matmul precision is pinned to true float32, no TF32, so the two sides agree
tightly.

Checkpoints are Kaun safetensors: tensors named "0".."N" in construction order --
embed [vocab, d], phase [16, d], then per layer wq wk wv wo [d, d], w1 [d, 4d],
w2 [4d, d].
"""

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

jax.config.update("jax_default_matmul_precision", "float32")

VOCAB = 256
SLOPE_SPAN = 8  # the ALiBi paper's exponent span; wider slopes see further
PHASE_BUCKETS = 16
PROGRESS_BUCKETS = 16
# the steps of one bucket at the draw: a chorale runs 228 steps at the median, so a
# bucket of the corpus is 14.2 steps and 16 is the nearest power of two -- a bit-slice in
# the circuit, and a period of 16 x 16 = 256 steps, about one chorale
PROGRESS_STRIDE = 16
LAYER_TENSORS = ("wq", "wk", "wv", "wo", "w1", "w2")


def load_params(path):
    """The tables and the layer count come from the tensor count, as
    Config.of_checkpoint reads them: two or three tables, then six tensors for each
    layer. No caller then states a number the file already answers.

    Two layouts exist and they never collide: without the progress table the count is
    2 + 6L, with it 3 + 6L, and no L makes one look like the other."""
    tensors = load_file(path)
    count = len(tensors)
    if count >= 8 and (count - 2) % 6 == 0:
        tables, layers = 2, (count - 2) // 6
    elif count >= 9 and (count - 3) % 6 == 0:
        tables, layers = 3, (count - 3) // 6
    else:
        raise ValueError(
            f"{path} holds {count} tensors: not two or three tables and six for each layer"
        )
    params = {
        "embed": jnp.asarray(tensors["0"]),
        "phase": jnp.asarray(tensors["1"]),
        "layers": [
            dict(
                zip(
                    LAYER_TENSORS,
                    (jnp.asarray(tensors[str(tables + 6 * layer + i)]) for i in range(6)),
                )
            )
            for layer in range(layers)
        ],
    }
    if tables == 3:
        params["progress"] = jnp.asarray(tensors["2"])
    return params


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def attention_bias(heads, length, span=SLOPE_SPAN):
    """ALiBi plus the causal wall, [1, heads, length, length].

    A head subtracts slope x distance from its logits, so the slope is a recency prior
    and [span] sets how far the gentlest head can see: the slope of head k is
    2^-(span (k+1) / heads), and the penalty at distance D is slope x D. At the paper's
    span of 8 the gentlest slope is 1/256 whatever the head count -- -4 logits at 1024
    tokens, -8 at 2048, which is blind for a chorale phrase. A wider span reaches
    further and stays a power of two, thus a shift in the circuit."""
    pos = jnp.arange(length, dtype=jnp.float32)
    distance = pos[:, None] - pos[None, :]
    slopes = -(2.0 ** (-span * (jnp.arange(heads, dtype=jnp.float32) + 1.0) / heads))
    alibi = slopes[None, :, None, None] * distance[None, None, :, :]
    wall = jnp.triu(jnp.ones((length, length), dtype=jnp.float32), k=1) * -1e9
    return alibi + wall[None, None, :, :]


def _dropout(x, rate, key):
    keep = 1.0 - rate
    return jnp.where(jax.random.bernoulli(key, keep, x.shape), x / keep, 0.0)


def logits(
    params,
    codes,
    phases,
    *,
    heads,
    dropout=0.0,
    key=None,
    span=SLOPE_SPAN,
    progress=None,
):
    """codes, phases: [batch, length] int32 -> [batch, length, vocab] float32.

    dropout > 0 needs a PRNG [key]; it drops the embedding sum and each residual
    branch. The default is the exact forward of the OCaml model.

    [progress] holds the progress bucket of each token and needs the progress table in
    [params]. Absent, the model reads the bar but not the piece."""
    batch, length = codes.shape
    d = params["embed"].shape[1]
    head_d = d // heads
    bias = attention_bias(heads, length, span)

    def drop(x):
        nonlocal key
        if dropout <= 0.0:
            return x
        key, sub = jax.random.split(key)
        return _dropout(x, dropout, sub)

    embedded = params["embed"][codes] + params["phase"][phases]
    if progress is not None:
        embedded = embedded + params["progress"][progress]
    h = drop(embedded)
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


def _cross_entropy(raw, labels, weights):
    """weights [batch, length] excludes the padding of a short piece from the mean; a
    padded position would teach the walk to hold the last chord and emit END for ever"""
    logp = jax.nn.log_softmax(raw, axis=-1)
    picked = jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
    if weights is None:
        return -jnp.mean(picked)
    return -jnp.sum(picked * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def loss(
    params,
    codes,
    phases,
    masks,
    *,
    heads,
    dropout=0.0,
    key=None,
    weights=None,
    span=SLOPE_SPAN,
    progress=None,
):
    """Cross entropy with the grammar inside the softmax: codes [batch, length + 1].

    The model spends no mass on a code the sampler would refuse, thus its raw mass
    outside the legal set stays untrained and the same mask must guard every draw.

    masks: [batch, length, vocab] bool -- the legal set of each label draw."""
    raw = logits(
        params,
        codes[:, :-1],
        phases,
        heads=heads,
        dropout=dropout,
        key=key,
        span=span,
        progress=progress,
    )
    return _cross_entropy(raw + jnp.where(masks, 0.0, -1e9), codes[:, 1:], weights)
