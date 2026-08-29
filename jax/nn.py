"""The common parts of the eras, above the seam.

Here stand the float models' shared parts: the step-frame transformer and the state-space
model share `Head` -- the four tied voice tables, the bar-phase table and the chained
readout over them -- one position rule and one sampling chain, and each model module keeps
what is its own: the trunk, the layer layout and the checkpoint walk. Under them stands
the trainer's rule -- the rate curve and the optax chain that every era updates under --
and not its loop, which is each era's own.

BELOW THE SEAM IS `fixed.py`: the integer rules every twin is built on, which no float
model reads. The two files stand as `lib/nn`'s do, where `quantized.ml` is its own module.

What is one thing across the eras stands here one time: a rule changed here changes every
model at once -- which is the point.

Matmul precision is pinned to true float32 here, no TF32; every model imports this
module, thus the pin holds everywhere.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from safetensors.numpy import save_file

import data
import prng

jax.config.update("jax_default_matmul_precision", "float32")

# The slope of head k is 2^-(SLOPE_SPAN (k+1) / heads). Elected 2026-08-18 over spans 4,
# 8, 16, 24 and 64: the means of 4 and 8 are a dead heat, and the VARIANCE is the finding
# -- 5 to 7 times tighter over six seeds, replicated at two step budgets. Every head is
# then local, and seeds stop latching onto whatever distant structure their init favours.
SLOPE_SPAN = 4
# the phase table IS the bar -- one row for each step of it. Two names for one number let
# the corpus phase and the table part, and a phase outside the table gathers a clamped row
# in silence.
PHASE_BUCKETS = data.BAR_STEPS
TABLES = ("seats", "phase")


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


def dropout_masks(key, rate, shape):
    """the multiplier form of inverted dropout: 0 or 1/keep, one for each element"""
    keep = 1.0 - rate
    return jax.random.bernoulli(key, keep, shape) / keep


# The draw of every matrix of both frozen eras: a normal at this deviation. It is not a
# fan-in rule and it was measured against one -- `Mamba.drawn` records the reading on the
# convolution kernel, where 1/sqrt(K) read worse.
DRAW_SCALE = 0.02


def normal_at(key, shape, scale=DRAW_SCALE):
    """one drawn tensor of a frozen era: a normal, scaled"""
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


# ---------------------------------------------------------------------
# the host-side draw: numpy, float64, and the PRNG of the circuit
# ---------------------------------------------------------------------


def _host_rms_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def temper(raw, temperature, min_p):
    """the tempered weight of each class against the peak, then the min-p floor; the peak
    weighs one, thus min_p is a share of the peak"""
    weights = np.exp((raw - raw.max(axis=1, keepdims=True)) / temperature)
    if min_p > 0.0:
        weights = np.where(weights >= min_p, weights, 0.0)
    return weights


def pick_share(weights, share):
    """The class whose running total passes the draw.

    It takes the uniform and not a draw, thus one function owns both sums and the total is
    the last running total -- never a second sum of the same weights. numpy adds pairwise
    in sum() and left to right in cumsum(), thus two sums of one array differ in the last
    bits, and a draw made against the other sum can land above every running total, where
    no class passes at all.

    Against this total the draw is strictly below it, because the uniform falls under 1 by
    2**-24 at the least. Therefore the walk always ends on a class, and that class always
    holds weight the floor left standing: to reach the last index is to know that no
    earlier total passed, thus the weight there is the difference of two totals across the
    draw. No fallback is necessary, and none is written."""
    running = np.cumsum(weights, axis=1)
    return (running > (share * running[:, -1])[:, None]).argmax(axis=1)


# ---------------------------------------------------------------------
# the head: the tied voice tables, and the chain over them
# ---------------------------------------------------------------------


class Head(nnx.Module):
    """The four tied voice tables and the bar-phase table, and the chained head over them.

    IT IS THE INPUT AND THE READOUT AT ONCE, because the tables are tied: the table that
    reads a voice is the table that writes it. That is why one module carries both
    directions and why neither era holds a table of its own.

    A shared table with a voice tag cannot work here, and the reason is arithmetic and not
    capacity. Every step carries all four seats, thus the sum of the four tags is the same
    vector at every position -- a bias, which carries nothing -- and what remains is
    symmetric in the four codes. A soprano on 72 over a bass on 48 would give the vector
    of a soprano on 48 under a bass on 72, and the voices would be thrown away on the way
    in. Four tables break the symmetry, and no voice tag is then necessary anywhere.

    The two tensors stand FIRST in every checkpoint of both eras, in this order, thus
    [tensors] and [take] are the one statement of that layout."""

    def __init__(self, d, *, rngs):
        self.seats = nnx.Param(normal_at(rngs.params(), (data.SEATS, data.CLASSES, d)))
        self.phase = nnx.Param(normal_at(rngs.params(), (PHASE_BUCKETS, d)))

    @staticmethod
    def shapes(d):
        """the shape of each tensor of [tensors], for a draw that states no shape twice"""
        return [(data.SEATS, data.CLASSES, d), (PHASE_BUCKETS, d)]

    @property
    def d(self):
        """the width of the residual stream: the seat table sizes it"""
        return self.seats.shape[-1]

    def embed(self, classes, phases):
        """The input of one step: the four seat rows and the bar-phase row sum.

        A phase outside the table gathers a clamped row, in silence, which is why the
        table IS the bar and holds one row for each step of it."""
        seats = self.seats[...]
        rows = sum(seats[seat][classes[..., seat]] for seat in range(data.SEATS))
        return rows + self.phase[...][phases]

    def logits(self, h, drawn):
        """The chained head: [batch, length, d] -> [batch, length, SEATS, CLASSES].

        Each seat reads the stream that the seats above it have already written:

            h3 = h                   logits(seat 3) = E[3] . rms(h3)
            h2 = h3 + E[3][c3]       logits(seat 2) = E[2] . rms(h2)
            h1 = h2 + E[2][c2]       logits(seat 1) = E[1] . rms(h1)
            h0 = h1 + E[1][c1]       logits(seat 0) = E[0] . rms(h0)

        [drawn] holds the classes the chain conditions on -- the true frame in training,
        where all four heads then run in one pass with no sampling, and the drawn seats at
        the draw. Only seats 3, 2 and 1 are read.

        The chain runs from the soprano down, which keeps the one decision the ear
        accepted: the top voice is chosen first and conditions on no voice under it, as
        the music is written. Four heads that drew in parallel would make the voices
        conditionally independent, and a chord is a joint choice: measured on era four,
        that costs 0.3157 nats for each step -- 0.456 bits, sixteen times the seed spread.
        The chain removes the cost for no parameters at all -- parallel heads need the
        same four tables -- and three adds of a vector.

        What the chain adds is also what the next step reads: the input embedding of step
        t+1 is a3 + a2 + a1 + a0, and the chain assembles it one voice at a time."""
        seats = self.seats[...]
        stream = h
        logits = [None] * data.SEATS
        for seat in reversed(range(data.SEATS)):
            logits[seat] = rms_norm(stream) @ seats[seat].T
            if seat:
                stream = stream + seats[seat][drawn[..., seat]]
        return jnp.stack(logits, axis=-2)

    def nll(self, h, labels):
        """the negative log likelihood of every voice of every step, over a residual
        stream the era's own [hidden] computed; the caller reduces"""
        logp = jax.nn.log_softmax(self.logits(h, labels), axis=-1)
        return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]

    def draw_frame(self, h, state, temperature, min_p):
        """One step of the chain ON THE HOST, in numpy float64 and on the PRNG of the
        circuit: the soprano first, and each seat under it reading the stream the seats
        above have written.

        The chain is the reason a frame is a joint choice and not four independent ones.
        Seat 0 is the bass and seat 3 the soprano, thus the loop runs down.

        Every walk of the batch draws. A step is one frame and never a sentence of its own
        length, thus no walk of the batch finishes before another and none has to sit out
        a draw while the rest go on."""
        seats = np.asarray(self.seats[...])
        stream = h
        frame = np.zeros((len(h), data.SEATS), dtype=np.int32)
        for seat in reversed(range(data.SEATS)):
            raw = (_host_rms_norm(stream) @ seats[seat].T).astype(np.float64)
            weights = temper(raw, temperature, min_p)
            state, uniform = prng.uniform(state, True)
            frame[:, seat] = pick_share(weights, uniform)
            if seat:
                stream = stream + seats[seat][frame[:, seat]]
        return state, frame

    def tensors(self):
        """the two tables in the order every checkpoint of both eras carries them"""
        return [self.seats[...], self.phase[...]]

    def take(self, tensors):
        """the reverse of [tensors]. The two stand together so that the layout cannot
        drift apart."""
        seats, phase = tensors
        self.seats[...] = jnp.asarray(seats)
        self.phase[...] = jnp.asarray(phase)


# ---------------------------------------------------------------------
# the trunk: what the step-frame trainer takes
# ---------------------------------------------------------------------


class Trunk(nnx.Module):
    """WHAT `frames.train` TAKES: a tied `Head` and a `layers` list under it.

    The step-frame eras are one skeleton -- the head states the frame and the layers carry
    the stream -- thus the three rules that read only that skeleton stand here and not
    once in each era. An era states its own `hidden` and its own `describe`; everything
    below reads `hidden` through `seat_nll` and reads nothing else of the era.

    Era six is NOT a subclass: its sheet is not a stream of frames, and its own trunk
    walks `every_layer` where these walk `layers`."""

    def every_tensor(self):
        """Every tensor of the model in THE ONE ORDER -- the head's tables, then the
        tensors of each layer.

        That order is the checkpoint's, the contract file's and the ROM's at once, and
        this is the one place either tree states it."""
        return self.head.tensors() + [
            tensor for layer in self.layers for tensor in layer.tensors()
        ]

    def seat_nll(self, classes, phases, *, dropout=0.0, key=None):
        """The negative log likelihood of every voice of every step: classes
        [batch, length + 1, SEATS] -> [batch, length, SEATS].

        The caller reduces. The loss does not carry across the encoding and neither does a
        per-prediction mean: report nats for each step, which is the sum over the seats.
        The two eras speak this same unit on these same windows, thus they compare."""
        labels = classes[:, 1:]
        h = self.hidden(classes[:, :-1], phases, dropout=dropout, key=key)
        return self.head.nll(h, labels)

    def parameter_count(self):
        return sum(int(tensor.size) for tensor in self.every_tensor())


# ---------------------------------------------------------------------
# the trainer: the rate curve, the update rule, and the seam of a checkpoint
# ---------------------------------------------------------------------

# THE RULE STANDS HERE AND THE LOOP DOES NOT. Every era runs the same optax chain under
# the same curve, thus a rate read one step late would be read one step late by all three
# at once; but the shape of a step is the era's own -- the sheet folds a batch-norm
# population where the frozen eras draw a dropout mask -- and each `train.py` keeps its
# loop.


def learning_rates(peak, warmup, total):
    """The rate at every step of the run: linear from 0 to [peak] over [warmup] steps,
    then cosine from [peak] to 0 over the rest. A warmup of zero is a constant.

    THE SCHEDULE IS READ ONE STEP LATE OR NOT AT ALL. Optax hands a schedule its own
    update count, which is 0 at the first update where the loop's step is 1, thus a curve
    read at the raw count applies a rate of 0 to the first update and every later rate one
    step behind. The `+ 1` is that correction and `tests/test_train.py` holds it.

    THE TWO ENDS ARE THIS PROJECT'S RULES AND NOT OPTAX'S. A warmup of zero is a constant
    peak, where `warmup_cosine_decay_schedule` would be a bare cosine decay; and a run
    SHORTER THAN ITS OWN WARMUP -- which every short probe is -- never leaves the ramp,
    where optax refuses to build a cosine of a negative length at all."""
    if warmup == 0:
        curve = optax.constant_schedule(peak)
    elif total <= warmup:
        curve = optax.linear_schedule(0.0, peak, warmup)
    else:
        curve = optax.warmup_cosine_decay_schedule(0.0, peak, warmup, total, 0.0)
    return lambda count: curve(count + 1)


def update_rule(*, peak, warmup, total, clip, weight_decay):
    """The update of one step: the global-norm clip, then Adam with a decoupled weight
    decay under the schedule.

    A clip of zero or less is NO CLIP. It is not a clip at zero, which would zero every
    gradient of the run. A weight decay of zero makes AdamW Adam, by arithmetic and not by
    a second code path -- which is what era six's paper asks for."""
    adam = optax.adamw(
        learning_rate=learning_rates(peak, warmup, total),
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    if clip <= 0.0:
        return adam
    return optax.chain(optax.clip_by_global_norm(clip), adam)


def save_checkpoint(path, tensors, span=None):
    """The naming rule of the seam: the tensors named "0" upward, in construction order,
    then the ALiBi span last and alone where the model carries one -- an older file that
    does not still reads, because a reader takes whole layer groups and then one scalar
    if one is there. The era's trainer builds the flat list, because the layer layouts
    are its own."""
    if span is not None:
        tensors = list(tensors) + [np.asarray([span], dtype=np.float32)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({str(i): np.asarray(t) for i, t in enumerate(tensors)}, path)
