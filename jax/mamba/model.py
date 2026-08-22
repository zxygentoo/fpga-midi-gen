"""The selective state-space model of docs/mamba.md.

One step of music is one step of the recurrence. Four voice classes enter through four
tables that sum, and they leave through the same four tables in a chain from the soprano
down -- the head of era four, unchanged. What changed is the trunk: a Mamba-2 block in its
recurrent form, with a fixed state where era four held a window of keys and values.

THE RECURRENCE HAS TWO FORMS HERE, and the design document sanctions the second one only
after a measurement:

- [block] is one step. The sampler runs it, the OCaml reference runs it, and the circuit
  computes its integers. It is the definition.
- [block_window] is the same recurrence over a whole window from a zero state, written as
  the quadratic form of Mamba-2 -- one decay matrix for each head instead of a walk. The
  trainer runs it. Measured on the baseline shape: 203 ms for each step under a scan of
  [block], against era four's 61, which would put the training round of the prototype past
  thirteen hours. The window form answers in a small fraction of that.

  Two forms of one recurrence is a second thing to keep true, and the price is paid in the
  gate: jax/tests/test_mamba.py holds them to each other, step for step, and a break there
  is a break of the model and not of a test.

The carry of the step form is the whole memory of the model: for each layer a state
[batch, H, P, N] and the K-1 convolution taps behind the step. A training window starts it
at zero and the boot of the walk starts it at zero, thus the seam condition of the corpus
is the condition the model trains on.

A LAYER IS ONE OF TWO KINDS, and the plan of the model is the sequence of them. Six
blocks is the trunk of docs/mamba.md. A plan with one attention layer in it is the hybrid
probe of 2026-08-20: era four's attention SUBLAYER -- 4 heads, ALiBi at the elected span,
the causal wall, no feed-forward under it -- swapped in where a block stood. The swap
takes 16,384 parameters where the block took 27,532, thus the probe cannot win by
capacity, and the compute hypothesis stays open. The attention is imported from
jax/transformer/model.py and not written again.

The attention layer is the one part of this model with a context. A block carries a state
of fixed size and knows nothing of how long the walk has run; the attention layer carries
a ring of the last ATTN_CONTEXT keys and values. A hybrid walk therefore has a context
length where the trunk had none.

Checkpoints are safetensors: tensors "0".."N" in construction order -- seats [4, 48, d],
phase [16, d], then for each layer either the six of a block -- w_in [d, 2 d_in + 2N + H],
conv [d_in + 2N, K], dt_bias [H], a_log [H], d_skip [H], w_out [d_in, d] -- or the four of
an attention layer, wq wk wv wo, each [d, d]. The file states its own plan: a square first
tensor in a group is wq and nothing else can be.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

from data import BAR_STEPS, SEATS
from transformer.model import SLOPE_SPAN, attention_bias

jax.config.update("jax_default_matmul_precision", "float32")

# the phase table IS the bar -- one row for each step of it, as era four states it
PHASE_BUCKETS = BAR_STEPS
TABLES = ("seats", "phase")

# The two kinds of layer. A trunk of blocks alone is the model of docs/mamba.md; a plan
# with an attention layer in it is the hybrid probe, and the attention is ERA FOUR'S,
# imported and not rewritten -- 4 heads, ALiBi at the elected span, the causal wall, and
# nothing else. It is a SUBLAYER and not era four's whole layer: no feed-forward follows
# it, thus the swap takes 16,384 parameters where the block it replaces took 27,532 and a
# win cannot be a win of capacity.
MAMBA, ATTN, ZATTN, MLP = "mamba", "attn", "zattn", "mlp"
LAYER_TENSORS = {
    MAMBA: ("w_in", "conv", "dt_bias", "a_log", "d_skip", "w_out"),
    ATTN: ("wq", "wk", "wv", "wo"),
    ZATTN: ("wq", "wk", "wv", "wo"),
    MLP: ("w1", "w2"),
}

# ZATTN is the shared-memory attention of Zamba, halved: the query and the key read the
# ORIGINAL EMBEDDING beside the residual stream -- wq and wk are [2d, d] over
# concat(rms(h), rms(e)) -- while the value reads the stream alone. Six blocks of
# recurrence smear which note was actually played; the embedding still says it, and
# attention needs it to match on. MLP is era four's feed-forward as a layer of its own, so
# that it can be ablated without touching the attention beside it.
#
# Each kind is its own group in the file and the FIRST TENSOR names it: w_in is
# [d, projection] with projection > 4d, wq is [d, d] for ATTN and [2d, d] for ZATTN, and
# w1 is [d, 4d]. No flag carries the plan.

# The ALiBi span of the attention layers, written into the checkpoint after the last
# layer and read back by [load_params] as a PLAIN FLOAT and not a tensor. It is a constant
# of the trained model, not a weight: it must not reach the optimizer, and it must be
# static so that the bias it builds is a constant of the trace. Era four carried this as a
# flag that "must match the training run"; this era's rule is that the file states every
# width, and a span played back wrong is silently wrong music.
SPAN_KEY = "span"

# The window the attention layer of a hybrid reads at inference. A block has no context
# and this one does, thus the walk of a hybrid carries a ring where the walk of a trunk
# carried nothing. It is era four's training window and era four's ring.
ATTN_CONTEXT = 256

# The Mamba defaults. K is the DRAW of the trainer and no longer a constant of the model:
# the convolution width is a lever of the sweep, thus a checkpoint states its own K and
# every reader takes it from the tensor. The expansion of two sets the inner width.
CONV_TAPS = 4
EXPAND = 2

# The chunk of the semiseparable window form. It is a measurement and not a taste: the
# traffic falls as T/chunk and the inter-chunk scan deepens as T/chunk, thus the two meet
# somewhere and the harness finds where.
CHUNK = 64


class Shape(NamedTuple):
    """The widths a checkpoint states. Everything else derives from them, thus a player
    names no shape and a file cannot disagree with a flag."""

    d: int  # the residual width
    d_in: int  # the inner width, EXPAND * d
    heads: int
    state: int  # N, the state width of one head
    taps: int  # K, the convolution width
    plan: tuple  # the kind of each layer, in order

    @property
    def layers(self):
        return len(self.plan)

    @property
    def head(self):
        """P, the head width: the state is H blocks of P x N"""
        return self.d_in // self.heads

    @property
    def head_d(self):
        """the head width of the attention layer, which splits d and not d_in"""
        return self.d // self.heads

    @property
    def channels(self):
        """the channels the convolution walks: x, then B and C"""
        return self.d_in + 2 * self.state


def kind_of(layer):
    """which kind a layer group is; its own tensors say so, and no flag is consulted"""
    if "w_in" in layer:
        return MAMBA
    if "w1" in layer:
        return MLP
    # the query of the Zamba block reads twice the width, thus its wq is not square
    return ATTN if layer["wq"].shape[0] == layer["wq"].shape[1] else ZATTN


def shape_of(params):
    """The widths and the plan, both out of the file.

    The blocks state the widths; the attention layer states none of its own, because era
    four's layer splits the residual width it is given. The plan comes from the tensor
    names, thus a player names no shape and no flag carries the position of the attention
    layer either."""
    plan = tuple(kind_of(layer) for layer in params["layers"])
    block = next((layer for layer in params["layers"] if kind_of(layer) == MAMBA), None)
    if block is None:
        raise ValueError("a plan of attention alone is not this model")
    d_in, d = block["w_out"].shape
    heads = block["dt_bias"].shape[0]
    state = (block["w_in"].shape[1] - 2 * d_in - heads) // 2
    return Shape(
        d=d,
        d_in=d_in,
        heads=heads,
        state=state,
        taps=block["conv"].shape[1],
        plan=plan,
    )


def load_params(path):
    """The two tables, then the tensors of each layer, in construction order.

    A layer group is six tensors where it is a block and four where it is attention, thus
    the walk is sequential and it reads the kind before it reads the count. The first
    tensor of a group is `w_in` at [d, projection] or `wq` at [d, d], and the projection
    is never d, thus a square first tensor names an attention layer and nothing else
    can."""
    tensors = load_file(path)
    params = {name: jnp.asarray(tensors[str(at)]) for at, name in enumerate(TABLES)}
    d = params["seats"].shape[-1]
    layers, at = [], len(TABLES)
    opens = {(d, d): ATTN, (2 * d, d): ZATTN, (d, 4 * d): MLP}
    while str(at) in tensors:
        # the span stands alone after the last layer, and no layer group opens with a
        # single value -- w_in, wq and w1 are all matrices
        if tensors[str(at)].size == 1:
            break
        kind = opens.get(tensors[str(at)].shape, MAMBA)
        names = LAYER_TENSORS[kind]
        layers.append(
            {name: jnp.asarray(tensors[str(at + k)]) for k, name in enumerate(names)}
        )
        at += len(names)
    span = None
    if str(at) in tensors and tensors[str(at)].size == 1:
        span = float(tensors[str(at)].reshape(()))
        at += 1
    if not layers or at != len(tensors):
        raise ValueError(
            f"{path}: {len(tensors)} tensors are not {TABLES} and whole layer groups"
        )
    params["layers"] = layers
    if span is not None:
        params[SPAN_KEY] = span
    return params


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def initial_carry(shape, batch, context=ATTN_CONTEXT):
    """The origin of the walk: a zero state and empty taps for each block, and an empty
    ring of keys and values for each attention layer.

    A block opens on zeros because a training window opens on zeros. The ring opens the
    same way and carries a count of the steps really taken, thus its unwritten slots are
    masked and the first step of a walk attends to itself alone -- which is the first
    position of a training window exactly."""

    def carry_of(kind):
        if kind == MLP:
            return None
        if kind in (ATTN, ZATTN):
            return (
                jnp.zeros((batch, context, shape.d), jnp.float32),
                jnp.zeros((batch, context, shape.d), jnp.float32),
                jnp.int32(0),
            )
        return (
            jnp.zeros((batch, shape.heads, shape.head, shape.state), jnp.float32),
            jnp.zeros((batch, shape.taps - 1, shape.channels), jnp.float32),
        )

    return [carry_of(kind) for kind in shape.plan]


def split_projection(shape, zxbcdt):
    """the gate, the convolution input and the raw dt -- the order docs/mamba.md states"""
    z = zxbcdt[..., : shape.d_in]
    u = zxbcdt[..., shape.d_in : shape.d_in + shape.channels]
    dt_raw = zxbcdt[..., shape.d_in + shape.channels :]
    return z, u, dt_raw


def split_channels(shape, xbc):
    """x, then B and C: the convolution walks them as one row of channels"""
    return (
        xbc[..., : shape.d_in],
        xbc[..., shape.d_in : shape.d_in + shape.state],
        xbc[..., shape.d_in + shape.state :],
    )


def step_size(layer, dt_raw):
    """dt of one step or of a whole window, and the decay rate that goes with it"""
    return jax.nn.softplus(dt_raw + layer["dt_bias"]), jnp.exp(layer["a_log"])


def convolve(conv, taps, u):
    """One step of the depthwise causal convolution: the sum, and the taps of the step
    after it.

    Tap k reads the input k steps back and tap 0 is the step itself, thus a window that
    has not run K steps yet reads zeros for the taps it does not have. The origin needs no
    clearing walk under this rule, which is why the tap ring of the circuit holds it."""
    history = jnp.concatenate([u[:, None, :], taps], axis=1)
    return jnp.einsum("bkc,ck->bc", history, conv), history[:, :-1, :]


def convolve_window(conv, u):
    """The same convolution over a whole window: u [batch, length, channels] -> the same.

    The window opens on zeros, thus the pad at the head is the tap rule of [convolve] and
    not a convenience of the shape. K is the width of [conv] and nothing else states it."""
    length, taps = u.shape[1], conv.shape[1]
    padded = jnp.pad(u, ((0, 0), (taps - 1, 0), (0, 0)))
    return sum(
        padded[:, taps - 1 - k : taps - 1 - k + length] * conv[:, k] for k in range(taps)
    )


def selective_state(shape, layer, state, x, b, c, dt, a):
    """The state update and the readout of one step.

    The decay is one scalar for each head -- the Mamba-2 form -- and that is what makes
    the block affordable here: six exponentials a step for each layer where Mamba-1 would
    want two thousand.

        S[p, n] <- alpha * S[p, n] + x[p] * (dt * B[n])
        y[p]     = sum_n S[p, n] * C[n] + D * x[p]

    The readout reads the state the update just wrote."""
    x = x.reshape(-1, shape.heads, shape.head)
    alpha = jnp.exp(-dt * a)
    beta = dt[..., None] * b[:, None, :]
    state = alpha[..., None, None] * state + x[..., None] * beta[:, :, None, :]
    read = jnp.einsum("bhpn,bn->bhp", state, c) + layer["d_skip"][None, :, None] * x
    return state, read.reshape(-1, shape.d_in)


def selective_window(shape, layer, x, b, c, dt, a):
    """The same recurrence over a whole window from a zero state, in the quadratic form.

    Unrolling the walk gives one weight for each ordered pair of steps:

        S[t] = sum over s <= t of  decay(t, s) * x[s] (dt[s] B[s])
        y[t] = S[t] . C[t] + D x[t]
             = sum over s <= t of  decay(t, s) dt[s] (B[s] . C[t]) x[s]  +  D x[t]

    and decay(t, s) is the product of the decays between them, which a cumulative sum of
    dt * a turns into one subtraction. The state never appears: what the walk would carry
    in [H, P, N] numbers, this reads out of a [length, length] weight for each head.

    The exponent is a difference of two cumulative sums and it is 0 or less over the half
    the mask keeps, thus the exponential cannot overflow. The mask stands on both sides of
    it: once so that no weight survives above the diagonal, and once inside so that the
    gradient of the exponential never reads the values that were going to be thrown
    away."""
    batch, length = dt.shape[0], dt.shape[1]
    x = x.reshape(batch, length, shape.heads, shape.head)
    cum = jnp.cumsum(dt * a, axis=1)
    gap = cum[:, None, :, :] - cum[:, :, None, :]  # [batch, t, s, heads]
    causal = ~jnp.triu(jnp.ones((length, length), bool), k=1)[None, :, :, None]
    decay = jnp.where(causal, jnp.exp(jnp.where(causal, gap, 0.0)), 0.0)
    weight = decay * dt[:, None, :, :] * jnp.einsum("bsn,btn->bts", b, c)[..., None]
    read = jnp.einsum("btsh,bshp->bthp", weight, x) + layer["d_skip"][None, None, :, None] * x
    return read.reshape(batch, length, shape.d_in)


def selective_window_chunked(shape, layer, x, b, c, dt, a, chunk=CHUNK):
    """The same recurrence again, chunked -- the semiseparable form of Mamba-2.

    [selective_window] is the definition this must equal, and it is the oracle in the gate.
    What parts them is only WHERE the work is done. The quadratic form builds a weight for
    every ordered pair of steps in the window: six [batch, T, T, heads] arrays a layer, 96
    MiB of traffic each way at T 256, for arithmetic a 3060 does in under a millisecond. It
    is bandwidth and not multiplies that costs, and this form removes the traffic
    algebraically rather than making the machine faster.

    A window cut into chunks of [chunk] steps splits the sum over s <= t in two:

    - INSIDE a chunk, the quadratic form again, but [chunk] wide instead of T. The decay
      between two steps of one chunk is a difference of the cumulative sums taken from the
      chunk's own head, thus nothing outside the chunk enters it.
    - ACROSS chunks, one state. Everything before the chunk reaches step t only through
      the state standing at the chunk's head, decayed by the cumulative sum up to t. Each
      chunk gives its successor one [heads, head, state] summary, and a scan of T/chunk
      steps carries them -- four steps at T 256, chunk 64.

    THAT is why this is affordable where a scan of the step form was not: that scan was 256
    deep in tiny kernels, and this one is four deep in whole chunks.

    The state between chunks is exactly the state of [selective_state], thus the two forms
    agree by construction and not by luck; the gate checks the arithmetic, not the algebra.
    A window the chunk does not divide is padded at the TAIL with zero dt and zero x, which
    contributes nothing to any sum and decays nothing, and the pad is cut before the
    return."""
    batch, length = dt.shape[0], dt.shape[1]
    pad = -length % chunk
    if pad:
        dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
        b = jnp.pad(b, ((0, 0), (0, pad), (0, 0)))
        c = jnp.pad(c, ((0, 0), (0, pad), (0, 0)))
    chunks = (length + pad) // chunk
    heads, head = shape.heads, shape.head

    def by_chunk(t, *tail):
        return t.reshape(batch, chunks, chunk, *tail)

    x = by_chunk(x, heads, head)  # [batch, chunks, chunk, heads, head]
    b, c = by_chunk(b, shape.state), by_chunk(c, shape.state)
    dt = by_chunk(dt, heads)
    # the cumulative decay from each chunk's own head, inclusive, and the whole of it
    csum = jnp.cumsum(dt * a, axis=2)
    total = csum[:, :, -1, :]

    # inside a chunk: the quadratic form of [selective_window], [chunk] wide
    gap = csum[:, :, None, :, :] - csum[:, :, :, None, :]  # [batch, chunks, t, s, heads]
    causal = ~jnp.triu(jnp.ones((chunk, chunk), bool), k=1)[None, None, :, :, None]
    decay = jnp.where(causal, jnp.exp(jnp.where(causal, gap, 0.0)), 0.0)
    weight = decay * dt[:, :, None, :, :] * jnp.einsum("zwsn,zwtn->zwts", b, c)[..., None]
    inside = jnp.einsum("zwtsh,zwshp->zwthp", weight, x)

    # what each chunk hands its successor: the state at the chunk's end
    # the decay from step s to the chunk's END, thus csum MINUS the total and never
    # the other way round: the exponent is 0 or less, as it is everywhere here
    landed = jnp.exp(csum - total[:, :, None, :]) * dt  # [batch, chunks, chunk, heads]
    given = jnp.einsum("zwsh,zwshp,zwsn->zwhpn", landed, x, b)

    def carry_chunk(state, one):
        return jnp.exp(-one[0])[:, :, None, None] * state + one[1], state

    # the state STANDING AT each chunk's head, thus the scan gives the carry before the add
    origin = jnp.zeros((batch, heads, head, shape.state), jnp.float32)
    moved = (jnp.moveaxis(total, 1, 0), jnp.moveaxis(given, 1, 0))
    (_, standing) = jax.lax.scan(carry_chunk, origin, moved)
    standing = jnp.moveaxis(standing, 0, 1)  # [batch, chunks, heads, head, state]

    # across chunks: that state, decayed to step t, read out through C
    across = jnp.einsum("zwhpn,zwtn->zwthp", standing, c) * jnp.exp(-csum)[..., None]
    read = inside + across + layer["d_skip"][None, None, None, :, None] * x
    return read.reshape(batch, length + pad, shape.d_in)[:, :length]


def block(shape, layer, carry, y):
    """One layer's branch at one step: the projection, the convolution, the recurrence and
    the gated norm of Mamba-2. The residual join is the caller's, because the dropout of
    the trainer sits between the two."""
    state, taps = carry
    z, u, dt_raw = split_projection(shape, y @ layer["w_in"])
    conv_out, taps = convolve(layer["conv"], taps, u)
    x, b, c = split_channels(shape, jax.nn.silu(conv_out))
    dt, a = step_size(layer, dt_raw)
    state, read = selective_state(shape, layer, state, x, b, c, dt, a)
    return (state, taps), rms_norm(read * jax.nn.silu(z))


def block_window(shape, layer, y):
    """The same branch over a whole window from a zero state. It is [block] with the two
    walks replaced by their window forms; every other line is the same arithmetic on one
    more axis."""
    z, u, dt_raw = split_projection(shape, y @ layer["w_in"])
    x, b, c = split_channels(shape, jax.nn.silu(convolve_window(layer["conv"], u)))
    dt, a = step_size(layer, dt_raw)
    read = selective_window(shape, layer, x, b, c, dt, a)
    return rms_norm(read * jax.nn.silu(z))


def span_of(params):
    """the ALiBi span of this model: the file's if it states one, era four's elected 4 if
    it does not. A caller that knows better -- the trainer, which holds the flag -- passes
    its own and never consults this."""
    return params.get(SPAN_KEY, SLOPE_SPAN)


def query_source(kind, y, e):
    """what the query and the key read: the stream alone, or the stream beside the
    ORIGINAL EMBEDDING as Zamba's shared block reads it. The value reads [y] either way."""
    return jnp.concatenate([y, e], axis=-1) if kind == ZATTN else y


def attention_window(shape, layer, y, source, span):
    """Era four's attention over a whole window, and it is era four's arithmetic: the bias
    comes from `transformer.model` and is not written a second time here.

    [source] feeds wq and wk and [y] feeds wv, thus one function serves both kinds and the
    Zamba variant is a wider wq and wk and nothing else.

    It gives the branch after `wo`, where the block form gives the branch before `w_out`,
    thus the caller adds one thing in both cases."""
    batch, length = y.shape[0], y.shape[1]

    def split_heads(x):
        return x.reshape(batch, length, shape.heads, shape.head_d).transpose(0, 2, 1, 3)

    q = split_heads(source @ layer["wq"])
    k = split_heads(source @ layer["wk"])
    v = split_heads(y @ layer["wv"])
    scale = 1.0 / jnp.sqrt(float(shape.head_d))
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale + attention_bias(
        shape.heads, length, span
    )
    read = jax.nn.softmax(scores, axis=-1) @ v
    merged = read.transpose(0, 2, 1, 3).reshape(batch, length, shape.d)
    return merged @ layer["wo"]


def attention_step(shape, layer, ring, y, source, span):
    """The same attention at one step, against the ring of the keys and values behind it.

    The ring holds the last `context` steps with the newest at the end, thus slot j sits
    `context - 1 - j` steps back and that distance IS the ALiBi distance -- no position
    counter enters the arithmetic. `filled` counts the steps the walk has really run and
    masks the slots below it, thus a walk that has not filled the ring reads the same
    scores a training window reads at the same position.

    The mask is the whole of the difference between this and [attention_window]. A ring
    read one slot late, or a distance counted from the wrong end, lands in the gate that
    holds the two forms to each other."""
    keys, values, filled = ring
    context = keys.shape[1]
    keys = jnp.concatenate([keys[:, 1:], (source @ layer["wk"])[:, None, :]], axis=1)
    values = jnp.concatenate([values[:, 1:], (y @ layer["wv"])[:, None, :]], axis=1)
    filled = jnp.minimum(filled + 1, context)

    def split_heads(x):
        return x.reshape(x.shape[0], x.shape[1], shape.heads, shape.head_d).transpose(
            0, 2, 1, 3
        )

    q = split_heads((source @ layer["wq"])[:, None, :])
    distance = jnp.arange(context - 1, -1, -1, dtype=jnp.float32)
    slopes = -(
        2.0 ** (-span * (jnp.arange(shape.heads, dtype=jnp.float32) + 1.0) / shape.heads)
    )
    bias = jnp.where(distance < filled, slopes[:, None] * distance[None, :], -1e9)
    scale = 1.0 / jnp.sqrt(float(shape.head_d))
    scores = (q @ split_heads(keys).transpose(0, 1, 3, 2)) * scale + bias[None, :, None, :]
    read = jax.nn.softmax(scores, axis=-1) @ split_heads(values)
    merged = read.transpose(0, 2, 1, 3).reshape(-1, shape.d)
    return (keys, values, filled), merged @ layer["wo"]


def feed_forward(layer, y):
    """era four's feed-forward, and it is position-wise, thus one form serves the window
    and the step alike"""
    return jnp.maximum(y @ layer["w1"], 0.0) @ layer["w2"]


def branch_step(shape, kind, layer, carry, y, e, span):
    """one layer's whole branch at one step, whichever kind it is: the thing h adds"""
    if kind == MLP:
        return carry, feed_forward(layer, y)
    if kind in (ATTN, ZATTN):
        return attention_step(shape, layer, carry, y, query_source(kind, y, e), span)
    carry, gated = block(shape, layer, carry, y)
    return carry, gated @ layer["w_out"]


def branch_window(shape, kind, layer, y, e, span):
    """the same branch over a whole window"""
    if kind == MLP:
        return feed_forward(layer, y)
    if kind in (ATTN, ZATTN):
        return attention_window(shape, layer, y, query_source(kind, y, e), span)
    return block_window(shape, layer, y) @ layer["w_out"]


def embed(params, classes):
    """The input of one step: the four seat rows sum.

    A shared table with a voice tag cannot work here, and the reason is arithmetic and not
    capacity. Every step carries all four seats, thus the sum of the four tags is the same
    vector at every position -- a bias, which carries nothing -- and what remains is
    symmetric in the four codes. A soprano on 72 over a bass on 48 would give the vector of
    a soprano on 48 under a bass on 72, and the voices would be thrown away on the way in.
    Four tables break the symmetry, and no voice tag is then necessary anywhere."""
    return sum(params["seats"][seat][classes[..., seat]] for seat in range(SEATS))


def forward_step(params, carry, classes, phases, *, span=None):
    """One step of the walk: the frame that just played goes in, and the residual stream
    the head reads comes out. [carry] is the whole memory of the model and the only thing
    that crosses from one step to the next -- the sampler holds nothing else."""
    shape = shape_of(params)
    reach = span_of(params) if span is None else span
    h = embed(params, classes) + params["phase"][phases]
    # the embedding the Zamba block reads, normalised once: it is the input of layer 0 and
    # it does not change as the stream is written
    e = rms_norm(h)
    out = []
    for kind, layer, before in zip(shape.plan, params["layers"], carry):
        after, branch = branch_step(shape, kind, layer, before, rms_norm(h), e, reach)
        h = h + branch
        out.append(after)
    return out, h


def dropout_masks(key, rate, shape_):
    """the multiplier form of the inverted dropout of era four"""
    keep = 1.0 - rate
    return jax.random.bernoulli(key, keep, shape_) / keep


def hidden(params, classes, phases, *, dropout=0.0, key=None, span=None):
    """classes [batch, length, SEATS] -> [batch, length, d], the residual stream after the
    last layer and before any readout.

    The window opens on a zero state. There is no context parameter and no wall: the
    recurrence cannot see forward, thus causality is the shape of the machine and not a
    mask over it.

    dropout > 0 needs a PRNG [key]; it drops the embedding sum and each residual branch."""
    shape = shape_of(params)
    reach = span_of(params) if span is None else span
    h = embed(params, classes) + params["phase"][phases]
    keys = iter(jax.random.split(key, shape.layers + 1) if dropout > 0.0 else ())

    def drop(x):
        return x if dropout <= 0.0 else x * dropout_masks(next(keys), dropout, x.shape)

    h = drop(h)
    e = rms_norm(h)
    for kind, layer in zip(shape.plan, params["layers"]):
        h = h + drop(branch_window(shape, kind, layer, rms_norm(h), e, reach))
    return h


def seat_logits(params, h, drawn):
    """The chained head: [batch, length, d] -> [batch, length, SEATS, CLASSES].

    Era four's head, carried over to the tensor. Each seat reads the stream that the seats
    above it have already written:

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
    independent, and a chord is a joint choice."""
    seats = params["seats"]
    stream = h
    logits = [None] * SEATS
    for seat in reversed(range(SEATS)):
        logits[seat] = rms_norm(stream) @ seats[seat].T
        if seat:
            stream = stream + seats[seat][drawn[..., seat]]
    return jnp.stack(logits, axis=-2)


def seat_nll(params, classes, phases, *, dropout=0.0, key=None, span=None):
    """The negative log likelihood of every voice of every step: classes
    [batch, length + 1, SEATS] -> [batch, length, SEATS].

    The caller reduces. The loss does not carry across the encoding and neither does a
    per-prediction mean: report nats for each step, which is the sum over the seats. Era
    four speaks this same unit on these same windows, thus the two eras compare."""
    labels = classes[:, 1:]
    h = hidden(params, classes[:, :-1], phases, dropout=dropout, key=key, span=span)
    logp = jax.nn.log_softmax(seat_logits(params, h, labels), axis=-1)
    return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
