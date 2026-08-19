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

Checkpoints are safetensors: tensors "0".."N" in construction order -- seats [4, 48, d],
phase [16, d], then per layer w_in [d, 2 d_in + 2 N + H], conv [d_in + 2 N, K],
dt_bias [H], a_log [H], d_skip [H], w_out [d_in, d].
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

from data import BAR_STEPS, SEATS

jax.config.update("jax_default_matmul_precision", "float32")

# the phase table IS the bar -- one row for each step of it, as era four states it
PHASE_BUCKETS = BAR_STEPS
TABLES = ("seats", "phase")
LAYER_TENSORS = ("w_in", "conv", "dt_bias", "a_log", "d_skip", "w_out")
PER_LAYER = len(LAYER_TENSORS)

# The Mamba defaults, and neither is a lever of this prototype: four taps carry the
# short-range half of the block, and the expansion of two sets the inner width.
CONV_TAPS = 4
EXPAND = 2


class Shape(NamedTuple):
    """The widths a checkpoint states. Everything else derives from them, thus a player
    names no shape and a file cannot disagree with a flag."""

    d: int  # the residual width
    d_in: int  # the inner width, EXPAND * d
    heads: int
    state: int  # N, the state width of one head
    layers: int

    @property
    def head(self):
        """P, the head width: the state is H blocks of P x N"""
        return self.d_in // self.heads

    @property
    def channels(self):
        """the channels the convolution walks: x, then B and C"""
        return self.d_in + 2 * self.state


def shape_of(params):
    d_in, d = params["layers"][0]["w_out"].shape
    heads = params["layers"][0]["dt_bias"].shape[0]
    state = (params["layers"][0]["w_in"].shape[1] - 2 * d_in - heads) // 2
    return Shape(d=d, d_in=d_in, heads=heads, state=state, layers=len(params["layers"]))


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


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def initial_carry(shape, batch):
    """The origin of the recurrence: a zero state and empty taps, for each layer."""
    return [
        (
            jnp.zeros((batch, shape.heads, shape.head, shape.state), jnp.float32),
            jnp.zeros((batch, CONV_TAPS - 1, shape.channels), jnp.float32),
        )
        for _ in range(shape.layers)
    ]


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
    not a convenience of the shape."""
    length = u.shape[1]
    padded = jnp.pad(u, ((0, 0), (CONV_TAPS - 1, 0), (0, 0)))
    return sum(
        padded[:, CONV_TAPS - 1 - k : CONV_TAPS - 1 - k + length] * conv[:, k]
        for k in range(CONV_TAPS)
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


def embed(params, classes):
    """The input of one step: the four seat rows sum.

    A shared table with a voice tag cannot work here, and the reason is arithmetic and not
    capacity. Every step carries all four seats, thus the sum of the four tags is the same
    vector at every position -- a bias, which carries nothing -- and what remains is
    symmetric in the four codes. A soprano on 72 over a bass on 48 would give the vector of
    a soprano on 48 under a bass on 72, and the voices would be thrown away on the way in.
    Four tables break the symmetry, and no voice tag is then necessary anywhere."""
    return sum(params["seats"][seat][classes[..., seat]] for seat in range(SEATS))


def forward_step(params, carry, classes, phases):
    """One step of the walk: the frame that just played goes in, and the residual stream
    the head reads comes out. [carry] is the whole memory of the model and the only thing
    that crosses from one step to the next -- the sampler holds nothing else."""
    shape = shape_of(params)
    h = embed(params, classes) + params["phase"][phases]
    out = []
    for layer, before in zip(params["layers"], carry):
        after, g = block(shape, layer, before, rms_norm(h))
        h = h + g @ layer["w_out"]
        out.append(after)
    return out, h


def dropout_masks(key, rate, shape_):
    """the multiplier form of the inverted dropout of era four"""
    keep = 1.0 - rate
    return jax.random.bernoulli(key, keep, shape_) / keep


def hidden(params, classes, phases, *, dropout=0.0, key=None):
    """classes [batch, length, SEATS] -> [batch, length, d], the residual stream after the
    last layer and before any readout.

    The window opens on a zero state. There is no context parameter and no wall: the
    recurrence cannot see forward, thus causality is the shape of the machine and not a
    mask over it.

    dropout > 0 needs a PRNG [key]; it drops the embedding sum and each residual branch."""
    shape = shape_of(params)
    h = embed(params, classes) + params["phase"][phases]
    keys = iter(jax.random.split(key, shape.layers + 1) if dropout > 0.0 else ())

    def drop(x):
        return x if dropout <= 0.0 else x * dropout_masks(next(keys), dropout, x.shape)

    h = drop(h)
    for layer in params["layers"]:
        h = h + drop(block_window(shape, layer, rms_norm(h)) @ layer["w_out"])
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


def seat_nll(params, classes, phases, *, dropout=0.0, key=None):
    """The negative log likelihood of every voice of every step: classes
    [batch, length + 1, SEATS] -> [batch, length, SEATS].

    The caller reduces. The loss does not carry across the encoding and neither does a
    per-prediction mean: report nats for each step, which is the sum over the seats. Era
    four speaks this same unit on these same windows, thus the two eras compare."""
    labels = classes[:, 1:]
    h = hidden(params, classes[:, :-1], phases, dropout=dropout, key=key)
    logp = jax.nn.log_softmax(seat_logits(params, h, labels), axis=-1)
    return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
