"""The selective state-space model of docs/mamba.md.

One step of music is one step of the recurrence. Four voice classes enter through four
tables that sum, and they leave through the same four tables in a chain from the soprano
down -- the head is `ar_model.Head`, era four's, unchanged. What changed is the trunk: a
Mamba-2
block in its recurrent form, with a fixed state where era four held a window of keys and
values.

THE RECURRENCE HAS TWO FORMS HERE, and the design document sanctions the second one only
after a measurement:

- [Block.step] is one step. The sampler runs it, the twin's engine answers it, and the
  circuit computes its integers. It is the definition.
- [Block.window] is the same recurrence over a whole window from a zero state, written as
  the quadratic form of Mamba-2 -- one decay matrix for each head instead of a walk. The
  trainer runs it. Measured on the baseline shape: 203 ms for each step under a scan of
  the step form, against era four's 61, which would put the training round of the
  prototype past thirteen hours. The window form answers in a small fraction of that.

  Two forms of one recurrence is a second thing to keep true, and the price is paid in the
  gate: jax/tests/test_mamba.py holds them to each other, step for step, and a break there
  is a break of the model and not of a test.

A THIRD FORM STOOD HERE AND IT IS GONE: the chunked semiseparable form of Mamba-2, which
would replace the quadratic form's [T, T] weight with a scan over chunks. It was built and
measured on 2026-08-21 -- 1.17 times at T 256 and 1.35 at T 512, under the 1.5 bar the
round set -- and the cost was diagnosed as kernel launch overhead and not the traffic it
removes. It was kept dormant because a change of summation order would have put every new
run outside the seed spread of the thirty checkpoints; the era is frozen now, no retrain
is planned, and what was left was an export alive only because a test read it. The
build-log of that day holds the numbers.

THE NET IS A MODULE TREE AND NOT A DICTIONARY OF TENSORS. A layer of one of three kinds
answers [step] and [window] under one signature, thus the trunk dispatches on nothing and
`quantized.QuantizedMamba` carries the same tree in integers under the same attribute
names.

The carry of the step form is the whole memory of the model: for each block a state
[batch, H, P, N] and the K-1 convolution taps behind the step. A training window starts it
at zero and the boot of the walk starts it at zero, thus the seam condition of the corpus
is the condition the model trains on.

A LAYER IS ONE OF THREE KINDS, and the plan of the model is the sequence of them. Six
blocks is the trunk of docs/mamba.md. A plan with one attention layer in it is the hybrid
probe of 2026-08-20: era four's attention SUBLAYER -- 4 heads, ALiBi at the elected span,
the causal wall, no feed-forward under it -- swapped in where a block stood. The swap
takes 16,384 parameters where the block took 27,532, thus the probe cannot win by
capacity, and the compute hypothesis stays open. The attention is era four's arithmetic
over `ar_model`'s own bias and is not written again.

The attention layer is the one part of this model with a context. A block carries a state
of fixed size and knows nothing of how long the walk has run; the attention layer carries
a ring of the last ATTN_CONTEXT keys and values. A hybrid walk therefore has a context
length where the trunk had none.

Checkpoints are safetensors: tensors "0".."N" in construction order -- seats [4, 48, d],
phase [16, d], then for each layer either the six of a block -- w_in [d, 2 d_in + 2N + H],
conv [d_in + 2N, K], dt_bias [H], a_log [H], d_skip [H], w_out [d_in, d] -- or the four of
an attention layer, wq wk wv wo, or the two of a feed-forward, w1 w2. THE FILE STATES ITS
OWN PLAN: the first tensor of a group names its kind, and the span rides last and alone.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from safetensors.numpy import load_file

import ar_model
from ar_model import SLOPE_SPAN, TABLES
from train import save_checkpoint

# The three kinds of layer. A trunk of blocks alone is the model of docs/mamba.md; a plan
# with an attention layer in it is the hybrid probe, and the attention is ERA FOUR'S -- 4
# heads, ALiBi at the elected span, the causal wall, and nothing else. It is a SUBLAYER
# and not era four's whole layer: no feed-forward follows it, thus the swap takes 16,384
# parameters where the block it replaces took 27,532 and a win cannot be a win of
# capacity.
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

# The window the attention layer of a hybrid reads at inference. A block has no context
# and this one does, thus the walk of a hybrid carries a ring where the walk of a trunk
# carried nothing. It IS the training window, stated once in `ar_model`.
ATTN_CONTEXT = ar_model.TRAINING_WINDOW

# The Mamba defaults. K is the DRAW of the trainer and no longer a constant of the model:
# the convolution width is a lever of the sweep, thus a checkpoint states its own K and
# every reader takes it from the tensor. The expansion of two sets the inner width.
CONV_TAPS = 4
EXPAND = 2

class Shape(NamedTuple):
    """The widths a model states. Everything else derives from them, thus a player names
    no shape and a file cannot disagree with a flag."""

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


# ---------------------------------------------------------------------
# the three kinds of layer
# ---------------------------------------------------------------------

# EACH ANSWERS [step] AND [window] UNDER ONE SIGNATURE -- (shape, carry, y, e, span) and
# (shape, y, e, span) -- thus the trunk walks the plan and dispatches on nothing. A layer
# that reads no embedding and no span takes them and ignores them, which costs a name and
# removes three branches.


class NamedTensors:
    """The float tensors of a layer, in the `LAYER_TENSORS` order of its own kind.

    The order is the CHECKPOINT ORDER and the ROM order behind it, thus it is stated one
    time for the three kinds and read through the layer's own `kind`. `ATTN` and `ZATTN`
    name the same four, thus the widened attention reads its own property safely."""

    def tensors(self):
        return [getattr(self, name)[...] for name in LAYER_TENSORS[self.kind]]

    def take(self, tensors):
        for name, value in zip(LAYER_TENSORS[self.kind], tensors):
            getattr(self, name)[...] = jnp.asarray(value)


class Block(NamedTensors, nnx.Module):
    """One Mamba-2 block in its selective form: the projection, the depthwise causal
    convolution, the recurrence, the gated norm, and the output projection."""

    kind = MAMBA

    def __init__(self, *, d, d_in, heads, state, taps, rngs):
        channels = d_in + 2 * state
        self.w_in = nnx.Param(
            ar_model.normal_at(rngs.params(), (d, 2 * d_in + 2 * state + heads))
        )
        self.conv = nnx.Param(ar_model.normal_at(rngs.params(), (channels, taps)))
        self.dt_bias = nnx.Param(jnp.zeros((heads,), jnp.float32))
        self.a_log = nnx.Param(jnp.zeros((heads,), jnp.float32))
        self.d_skip = nnx.Param(jnp.ones((heads,), jnp.float32))
        self.w_out = nnx.Param(ar_model.normal_at(rngs.params(), (d_in, d)))

    def initial_carry(self, shape, batch, context):
        """the origin of this layer's memory: a zero state and empty taps.

        A block opens on zeros because a training window opens on zeros, thus the seam
        condition of the corpus is the condition the model trains on."""
        del context
        return (
            jnp.zeros((batch, shape.heads, shape.head, shape.state), jnp.float32),
            jnp.zeros((batch, shape.taps - 1, shape.channels), jnp.float32),
        )

    def split_projection(self, shape, zxbcdt):
        """the gate, the convolution input and the raw dt -- the order docs/mamba.md
        states"""
        z = zxbcdt[..., : shape.d_in]
        u = zxbcdt[..., shape.d_in : shape.d_in + shape.channels]
        dt_raw = zxbcdt[..., shape.d_in + shape.channels :]
        return z, u, dt_raw

    def split_channels(self, shape, xbc):
        """x, then B and C: the convolution walks them as one row of channels"""
        return (
            xbc[..., : shape.d_in],
            xbc[..., shape.d_in : shape.d_in + shape.state],
            xbc[..., shape.d_in + shape.state :],
        )

    def step_size(self, dt_raw):
        """dt of one step or of a whole window, and the decay rate that goes with it"""
        return jax.nn.softplus(dt_raw + self.dt_bias[...]), jnp.exp(self.a_log[...])

    def convolve(self, taps, u):
        """One step of the depthwise causal convolution: the sum, and the taps of the step
        after it.

        Tap k reads the input k steps back and tap 0 is the step itself, thus a window
        that has not run K steps yet reads zeros for the taps it does not have. The origin
        needs no clearing walk under this rule, which is why the tap ring of the circuit
        holds it."""
        history = jnp.concatenate([u[:, None, :], taps], axis=1)
        return jnp.einsum("bkc,ck->bc", history, self.conv[...]), history[:, :-1, :]

    def convolve_window(self, u):
        """The same convolution over a whole window: u [batch, length, channels] -> the
        same.

        The window opens on zeros, thus the pad at the head is the tap rule of [convolve]
        and not a convenience of the shape. K is the width of the kernel and nothing else
        states it."""
        conv = self.conv[...]
        length, taps = u.shape[1], conv.shape[1]
        padded = jnp.pad(u, ((0, 0), (taps - 1, 0), (0, 0)))
        return sum(
            padded[:, taps - 1 - k : taps - 1 - k + length] * conv[:, k]
            for k in range(taps)
        )

    def selective_state(self, shape, state, x, b, c, dt, a):
        """The state update and the readout of one step.

        The decay is one scalar for each head -- the Mamba-2 form -- and that is what
        makes the block affordable here: six exponentials a step for each layer where
        Mamba-1 would want two thousand.

            S[p, n] <- alpha * S[p, n] + x[p] * (dt * B[n])
            y[p]     = sum_n S[p, n] * C[n] + D * x[p]

        The readout reads the state the update just wrote."""
        x = x.reshape(-1, shape.heads, shape.head)
        alpha = jnp.exp(-dt * a)
        beta = dt[..., None] * b[:, None, :]
        state = alpha[..., None, None] * state + x[..., None] * beta[:, :, None, :]
        read = (
            jnp.einsum("bhpn,bn->bhp", state, c)
            + self.d_skip[...][None, :, None] * x
        )
        return state, read.reshape(-1, shape.d_in)

    def selective_window(self, shape, x, b, c, dt, a):
        """The same recurrence over a whole window from a zero state, in the quadratic
        form.

        Unrolling the walk gives one weight for each ordered pair of steps:

            S[t] = sum over s <= t of  decay(t, s) * x[s] (dt[s] B[s])
            y[t] = S[t] . C[t] + D x[t]
                 = sum over s <= t of  decay(t, s) dt[s] (B[s] . C[t]) x[s]  +  D x[t]

        and decay(t, s) is the product of the decays between them, which a cumulative sum
        of dt * a turns into one subtraction. The state never appears: what the walk would
        carry in [H, P, N] numbers, this reads out of a [length, length] weight for each
        head.

        The exponent is a difference of two cumulative sums and it is 0 or less over the
        half the mask keeps, thus the exponential cannot overflow. The mask stands on both
        sides of it: once so that no weight survives above the diagonal, and once inside
        so that the gradient of the exponential never reads the values that were going to
        be thrown away."""
        batch, length = dt.shape[0], dt.shape[1]
        x = x.reshape(batch, length, shape.heads, shape.head)
        cum = jnp.cumsum(dt * a, axis=1)
        gap = cum[:, None, :, :] - cum[:, :, None, :]  # [batch, t, s, heads]
        causal = ~jnp.triu(jnp.ones((length, length), bool), k=1)[None, :, :, None]
        decay = jnp.where(causal, jnp.exp(jnp.where(causal, gap, 0.0)), 0.0)
        weight = decay * dt[:, None, :, :] * jnp.einsum("bsn,btn->bts", b, c)[..., None]
        read = (
            jnp.einsum("btsh,bshp->bthp", weight, x)
            + self.d_skip[...][None, None, :, None] * x
        )
        return read.reshape(batch, length, shape.d_in)

    def step(self, shape, carry, y, e, span):
        """this layer's branch at one step: the thing h adds, and the carry after it"""
        del e, span
        state, taps = carry
        z, u, dt_raw = self.split_projection(shape, y @ self.w_in[...])
        conv_out, taps = self.convolve(taps, u)
        x, b, c = self.split_channels(shape, jax.nn.silu(conv_out))
        dt, a = self.step_size(dt_raw)
        state, read = self.selective_state(shape, state, x, b, c, dt, a)
        gated = ar_model.rms_norm(read * jax.nn.silu(z))
        return (state, taps), gated @ self.w_out[...]

    def window(self, shape, y, e, span):
        """The same branch over a whole window from a zero state. It is [step] with the
        two walks replaced by their window forms; every other line is the same arithmetic
        on one more axis."""
        del e, span
        z, u, dt_raw = self.split_projection(shape, y @ self.w_in[...])
        x, b, c = self.split_channels(shape, jax.nn.silu(self.convolve_window(u)))
        dt, a = self.step_size(dt_raw)
        read = self.selective_window(shape, x, b, c, dt, a)
        return ar_model.rms_norm(read * jax.nn.silu(z)) @ self.w_out[...]


class Attention(NamedTensors, nnx.Module):
    """Era four's attention sublayer, and the Zamba widening of it.

    [kind] is a property of the tensors and not a flag: the Zamba query and key read the
    stream BESIDE the original embedding, thus their wq is [2d, d] and nothing else can
    be. One class serves both, because the widening is the whole of the difference."""

    def __init__(self, *, d, wide, rngs):
        source = (2 * d, d) if wide else (d, d)
        self.wq = nnx.Param(ar_model.normal_at(rngs.params(), source))
        self.wk = nnx.Param(ar_model.normal_at(rngs.params(), source))
        self.wv = nnx.Param(ar_model.normal_at(rngs.params(), (d, d)))
        self.wo = nnx.Param(ar_model.normal_at(rngs.params(), (d, d)))

    @property
    def kind(self):
        return ZATTN if self.wq.shape[0] != self.wq.shape[1] else ATTN

    def initial_carry(self, shape, batch, context):
        """an empty ring of keys and values, and a count of the steps really taken.

        The unwritten slots are masked by that count, thus the first step of a walk
        attends to itself alone -- which is the first position of a training window
        exactly."""
        return (
            jnp.zeros((batch, context, shape.d), jnp.float32),
            jnp.zeros((batch, context, shape.d), jnp.float32),
            jnp.int32(0),
        )

    def query_source(self, y, e):
        """what the query and the key read: the stream alone, or the stream beside the
        ORIGINAL EMBEDDING as Zamba's shared block reads it. The value reads [y] either
        way."""
        return jnp.concatenate([y, e], axis=-1) if self.kind == ZATTN else y

    def window(self, shape, y, e, span):
        """Era four's attention over a whole window, and it is era four's arithmetic: the
        bias comes from `ar_model.attention_bias` and is not written a second time here.

        It gives the branch after `wo`, where the block form gives the branch after
        `w_out`, thus the caller adds one thing in either case."""
        batch, length = y.shape[0], y.shape[1]
        source = self.query_source(y, e)

        def split_heads(x):
            return x.reshape(batch, length, shape.heads, shape.head_d).transpose(
                0, 2, 1, 3
            )

        q = split_heads(source @ self.wq[...])
        k = split_heads(source @ self.wk[...])
        v = split_heads(y @ self.wv[...])
        scale = 1.0 / jnp.sqrt(float(shape.head_d))
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale + ar_model.attention_bias(
            shape.heads, length, span
        )
        read = jax.nn.softmax(scores, axis=-1) @ v
        merged = read.transpose(0, 2, 1, 3).reshape(batch, length, shape.d)
        return merged @ self.wo[...]

    def step(self, shape, carry, y, e, span):
        """The same attention at one step, against the ring of the keys and values behind
        it.

        The ring holds the last `context` steps with the newest at the end, thus slot j
        sits `context - 1 - j` steps back and that distance IS the ALiBi distance -- no
        position counter enters the arithmetic. `filled` counts the steps the walk has
        really run and masks the slots below it, thus a walk that has not filled the ring
        reads the same scores a training window reads at the same position.

        The mask is the whole of the difference between this and [window]. A ring read one
        slot late, or a distance counted from the wrong end, lands in the gate that holds
        the two forms to each other."""
        keys, values, filled = carry
        context = keys.shape[1]
        source = self.query_source(y, e)
        keys = jnp.concatenate([keys[:, 1:], (source @ self.wk[...])[:, None, :]], axis=1)
        values = jnp.concatenate([values[:, 1:], (y @ self.wv[...])[:, None, :]], axis=1)
        filled = jnp.minimum(filled + 1, context)

        def split_heads(x):
            return x.reshape(
                x.shape[0], x.shape[1], shape.heads, shape.head_d
            ).transpose(0, 2, 1, 3)

        q = split_heads((source @ self.wq[...])[:, None, :])
        distance = jnp.arange(context - 1, -1, -1, dtype=jnp.float32)
        slopes = ar_model.alibi_slopes(shape.heads, span)
        bias = jnp.where(distance < filled, slopes[:, None] * distance[None, :], -1e9)
        scale = 1.0 / jnp.sqrt(float(shape.head_d))
        scores = (
            q @ split_heads(keys).transpose(0, 1, 3, 2)
        ) * scale + bias[None, :, None, :]
        read = jax.nn.softmax(scores, axis=-1) @ split_heads(values)
        merged = read.transpose(0, 2, 1, 3).reshape(-1, shape.d)
        return (keys, values, filled), merged @ self.wo[...]


class FeedForward(NamedTensors, nnx.Module):
    """Era four's feed-forward as a layer of its own, so that it can be ablated without
    touching the attention beside it.

    It is POSITION-WISE, thus one form serves the window and the step alike and it carries
    no memory at all."""

    kind = MLP

    def __init__(self, *, d, rngs):
        self.w1 = nnx.Param(ar_model.normal_at(rngs.params(), (d, 4 * d)))
        self.w2 = nnx.Param(ar_model.normal_at(rngs.params(), (4 * d, d)))

    def initial_carry(self, shape, batch, context):
        """no memory at all: a position-wise layer carries nothing between steps"""
        del shape, batch, context

    def window(self, shape, y, e, span):
        del shape, e, span
        return jnp.maximum(y @ self.w1[...], 0.0) @ self.w2[...]

    def step(self, shape, carry, y, e, span):
        return carry, self.window(shape, y, e, span)


def layer_of(kind, *, d, d_in, heads, state, taps, rngs):
    """one layer of the plan, by its kind"""
    if kind == MAMBA:
        return Block(d=d, d_in=d_in, heads=heads, state=state, taps=taps, rngs=rngs)
    if kind == MLP:
        return FeedForward(d=d, rngs=rngs)
    return Attention(d=d, wide=kind == ZATTN, rngs=rngs)


def kind_of_group(shape, d):
    """THE FIRST TENSOR OF A CHECKPOINT GROUP NAMES ITS KIND, thus a file walk is
    sequential and it reads the kind before it reads the count. The projection of a block
    is never d and never 4d, thus nothing else can be mistaken for one.

    `quantized.kind_of_image` is the same rule over the CONTRACT FILE, and it is not this
    function: that image holds w_in transposed and it refuses a square query outright,
    because the circuit has no narrow path for one."""
    if shape == (d, d):
        return ATTN
    if shape == (2 * d, d):
        return ZATTN
    if shape == (d, 4 * d):
        return MLP
    return MAMBA


# ---------------------------------------------------------------------
# the trunk
# ---------------------------------------------------------------------


class Trunk(ar_model.Trunk):
    """The skeleton both models of the era carry: the tied head, then the layers of the
    plan.

    [Mamba] fills it with the float tensors and `quantized.QuantizedMamba` with their
    integer twins, UNDER THE SAME ATTRIBUTE NAMES AT EVERY LEVEL, thus the two trees are
    one tree and a reader can audit them layer for layer."""

    @property
    def plan(self):
        """the kind of each layer, in order. No flag carries it on either side of the
        seam: a layer's own tensors say what it is."""
        return tuple(layer.kind for layer in self.layers)


class Mamba(Trunk):
    """The state-space model of the era: the tied head, and the layers of the plan."""

    def __init__(self, plan, *, d, heads, state, taps=CONV_TAPS, expand=EXPAND,
                 span=SLOPE_SPAN, rngs):
        d_in = expand * d
        self.head = ar_model.Head(d, rngs=rngs)
        self.layers = nnx.List(
            [
                layer_of(kind, d=d, d_in=d_in, heads=heads, state=state, taps=taps,
                         rngs=rngs)
                for kind in plan
            ]
        )
        # not a weight and not in the optimizer: the bias it builds must be a constant of
        # the trace. It rides the checkpoint as one scalar after the last layer.
        self.span = span

    @property
    def shape(self):
        """The widths and the plan, out of the model itself.

        The blocks state the widths; the attention layer states none of its own, because
        era four's layer splits the residual width it is given."""
        block = next((l for l in self.layers if l.kind == MAMBA), None)
        if block is None:
            raise ValueError("a plan of attention alone is not this model")
        d_in, d = block.w_out.shape
        heads = block.dt_bias.shape[0]
        return Shape(
            d=d,
            d_in=d_in,
            heads=heads,
            state=(block.w_in.shape[1] - 2 * d_in - heads) // 2,
            taps=block.conv.shape[1],
            plan=self.plan,
        )

    @property
    def d(self):
        return self.head.d

    def initial_carry(self, batch, context=ATTN_CONTEXT):
        """the origin of the walk, layer by layer: each states its own memory"""
        shape = self.shape
        return [layer.initial_carry(shape, batch, context) for layer in self.layers]

    def forward_step(self, carry, classes, phases):
        """One step of the walk: the frame that just played goes in, and the residual
        stream the head reads comes out. [carry] is the whole memory of the model and the
        only thing that crosses from one step to the next -- the sampler holds nothing
        else."""
        shape = self.shape
        h = self.head.embed(classes, phases)
        # the embedding the Zamba block reads, normalised once: it is the input of layer 0
        # and it does not change as the stream is written
        e = ar_model.rms_norm(h)
        out = []
        for layer, before in zip(self.layers, carry):
            after, branch = layer.step(shape, before, ar_model.rms_norm(h), e, self.span)
            h = h + branch
            out.append(after)
        return out, h

    def hidden(self, classes, phases, *, dropout=0.0, key=None):
        """classes [batch, length, SEATS] -> [batch, length, d], the residual stream after
        the last layer and before any readout.

        The window opens on a zero state. There is no context parameter and no wall: the
        recurrence cannot see forward, thus causality is the shape of the machine and not
        a mask over it.

        dropout > 0 needs a PRNG [key]; it drops the embedding sum and each residual
        branch."""
        shape = self.shape
        # the embedding sum, then the one residual branch of each layer
        drop = ar_model.dropout(key, dropout, shape.layers + 1)
        h = drop(self.head.embed(classes, phases))
        e = ar_model.rms_norm(h)
        for layer in self.layers:
            h = h + drop(layer.window(shape, ar_model.rms_norm(h), e, self.span))
        return h

    def describe(self):
        """the shape of this model in one phrase, for the head of a training log"""
        return f"{self.shape}, ALiBi span {self.span}"

    def save(self, path):
        """the whole model as one flat list, then the ALiBi span last and alone.

        The span goes LAST, thus an older file that does not carry it still reads: [load]
        takes whole layer groups and then one scalar if one is there. It is written even
        where no layer attends, which costs four bytes and keeps one rule."""
        save_checkpoint(path, self.every_tensor(), span=self.span)

    @classmethod
    def load(cls, path):
        """The model of one checkpoint. EVERY WIDTH AND THE PLAN COME OUT OF THE FILE: a
        layer group is six tensors where it is a block, four where it is attention and two
        where it is a feed-forward, and the first tensor of a group names which."""
        tensors = load_file(str(path))
        d = tensors["0"].shape[-1]
        plan, groups, at = [], [], len(TABLES)
        while str(at) in tensors:
            # the span stands alone after the last layer, and no layer group opens with a
            # single value -- w_in, wq and w1 are all matrices
            if tensors[str(at)].size == 1:
                break
            kind = kind_of_group(tensors[str(at)].shape, d)
            names = LAYER_TENSORS[kind]
            plan.append(kind)
            groups.append([tensors[str(at + k)] for k in range(len(names))])
            at += len(names)
        span = SLOPE_SPAN
        if str(at) in tensors and tensors[str(at)].size == 1:
            span = float(tensors[str(at)].reshape(()))
            at += 1
        if not plan or at != len(tensors):
            raise ValueError(
                f"{path}: {len(tensors)} tensors are not {TABLES} and whole layer groups"
            )
        block = next(
            (group for kind, group in zip(plan, groups) if kind == MAMBA), None
        )
        if block is None:
            raise ValueError(f"{path}: a plan of attention alone is not this model")
        d_in = block[5].shape[0]
        heads = block[2].shape[0]
        # the draw is thrown away one tensor at a time below: it is the cost of one normal
        # and it buys the one constructor
        held = cls(
            plan,
            d=d,
            heads=heads,
            state=(block[0].shape[1] - 2 * d_in - heads) // 2,
            taps=block[1].shape[1],
            expand=d_in // d,
            span=span,
            rngs=nnx.Rngs(0),
        )
        held.head.take([tensors[str(at)] for at in range(len(TABLES))])
        for layer, group in zip(held.layers, groups):
            layer.take(group)
        return held

    @classmethod
    def drawn(cls, seed, *, d, layers=None, heads, state, taps=CONV_TAPS, expand=EXPAND,
              conv_scale=ar_model.DRAW_SCALE, half_lives=None, attention_at=(),
              spelt=None,
              span=SLOPE_SPAN):
        """A model of DRAWN weights: the trainer's opening, and the shape a gate can
        afford.

        [a_log] is the log of a uniform decay rate in [1, 16] and [dt_bias] the inverse
        softplus of a uniform step in [0.001, 0.1], which is the initialization the Mamba
        papers state. [d_skip] opens at one, thus a layer starts as the skip and learns
        its state from there.

        [half_lives] replaces the uniform draw of dt with the ladder of
        [half_life_ladder], and it is the one lever this initialization holds.

        [spelt] names the kind of every layer; [attention_at] names the layers that are
        era four's attention sublayer instead of a block, and [layers] their count.

        THE CONVOLUTION TAKES 0.02 TOO, and it was measured. The argument against was
        fan-in: four taps at 0.02 pass a fiftieth of their input, the SiLU under them sits
        near its own origin, and B and C open so small that the state has little to learn
        from. 1/sqrt(K) is the fan-in scale and the Mamba reference uses it. Over 4 000
        steps of the baseline shape it read 1.7311 valid against 0.02's 1.7113, thus the
        argument is wrong here: the gated norm rescales the branch in any case, and the
        smaller draw is no worse. One rule covers every matrix of this model.

        THE DRAW IS THE MODULE'S AND NOT ITS INITIALIZER'S, one key for each tensor of the
        checkpoint layout, because the measured numbers of `tests/test_drift.py` read this
        draw and a framework that changed its key rule would move them."""
        d_in = expand * d
        channels = d_in + 2 * state
        plan = spelt or [
            ATTN if at in attention_at else MAMBA for at in range(layers)
        ]
        count = len(TABLES) + sum(len(LAYER_TENSORS[kind]) for kind in plan)
        keys = iter(jax.random.split(jax.random.key(seed), count))
        held = cls(plan, d=d, heads=heads, state=state, taps=taps, expand=expand,
                   span=span, rngs=nnx.Rngs(0))
        held.head.take(
            [ar_model.normal_at(next(keys), s) for s in ar_model.Head.shapes(d)]
        )

        def block_tensors():
            w_in = ar_model.normal_at(next(keys), (d, 2 * d_in + 2 * state + heads))
            conv = ar_model.normal_at(next(keys), (channels, taps), conv_scale)
            step = jax.random.uniform(next(keys), (heads,), minval=0.001, maxval=0.1)
            decay = jax.random.uniform(next(keys), (heads,), minval=1.0, maxval=16.0)
            # the draw above still runs and its key is still spent, thus a ladder run and
            # its baseline differ in dt_bias and in no other tensor of the checkpoint
            if half_lives is not None:
                step = half_life_ladder(heads, half_lives) / decay
            return [
                w_in,
                conv,
                # the inverse softplus of the drawn step: softplus of this is that step
                jnp.log(jnp.expm1(step)),
                jnp.log(decay),
                jnp.ones((heads,), jnp.float32),
                ar_model.normal_at(next(keys), (d_in, d)),
            ]

        for kind, layer in zip(plan, held.layers):
            if kind == MAMBA:
                layer.take(block_tensors())
            elif kind == MLP:
                layer.take(
                    [ar_model.normal_at(next(keys), s) for s in ((d, 4 * d), (4 * d, d))]
                )
            else:
                # the Zamba query and key read the stream beside the embedding, thus
                # [2d,d]
                wide = (2 * d, d) if kind == ZATTN else (d, d)
                layer.take(
                    [
                        ar_model.normal_at(next(keys), s)
                        for s in (wide, wide, (d, d), (d, d))
                    ]
                )
        return held


def half_life_ladder(heads, span):
    """The dt of each head that puts its half-life on a log-spaced ladder.

    A trained state decays as `exp(-dt * a)` for each step, thus its half-life is
    `ln 2 / (dt * a)`. The Mamba draw is uniform in dt and says nothing about the
    half-life; this one names the ladder and solves for the dt that lands on it, over the
    decay rate the head already drew. One head sits at each rung, log-spaced.

    It exists because a measurement asked for it: over the elected prototype the trained
    half-lives collapse -- no head above layer 2 holds a median of more than 7 steps --
    and a state that never learns a phrase-scale memory may simply have opened too far
    from one."""
    low, high = span
    rungs = low * (high / low) ** (
        jnp.arange(heads, dtype=jnp.float32) / max(heads - 1, 1)
    )
    return jnp.log(2.0) / rungs
