"""The common parts of the STEP-FRAME eras, above the seam.

It stands to `sample.py` and `train.py` as `ar_train.py` stands to each era's trainer.
Eras
four and five share `Head` -- the four tied voice tables, the bar-phase table and the
chained readout over them -- one position rule, one norm and one draw scale, and each
model module keeps what is its own: the trunk, the layer layout and the checkpoint walk.
What is one thing across ALL THREE eras is smaller and stands elsewhere: `sample.py` is
the host-side draw and `train.py` the rate curve, the update rule and the checkpoint.

THE CUT RUNS ONE WAY. This module imports the shared ones and none of them imports it
back, thus era six can read them without reading a head it has no frames for. Era six is
not a `Trunk` for the same reason: its sheet is not a stream of frames, and its own trunk
walks `every_layer` where these walk `layers`.

BELOW THE SEAM IS `ar_quantized.py`: the integer rules of these two twins, which no float
model reads.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import corpus
import prng
import sample

# The slope of head k is 2^-(SLOPE_SPAN (k+1) / heads). Elected 2026-08-18 over spans 4,
# 8, 16, 24 and 64: the means of 4 and 8 are a dead heat, and the VARIANCE is the finding
# -- 5 to 7 times tighter over six seeds, replicated at two step budgets. Every head is
# then local, and seeds stop latching onto whatever distant structure their init favours.
SLOPE_SPAN = 4
# THE TRAINING WINDOW, in steps: what `ar_train` cuts a batch to, thus what a player must
# state back and what the referee's eval rows are cut at. Era five's attention ring is
# this number as well -- a block carries no context, and the ring of a hybrid is era
# four's window.
TRAINING_WINDOW = 256


# the phase table IS the bar -- one row for each step of it. Two names for one number let
# the corpus phase and the table part, and a phase outside the table gathers a clamped row
# in silence.
PHASE_BUCKETS = corpus.BAR_STEPS
TABLES = ("seats", "phase")


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def alibi_slopes(heads, span=SLOPE_SPAN):
    """the ALiBi slope of each head, [heads]: head k slopes at -2^-(span (k+1) / heads).

    THE RULE HAS ONE HOME. The window form below and the step form of a Zamba layer both
    read it, and the integer twins read `ar_quantized.slope_exponent`, which is this
    exponent floored -- a rule written twice is a rule two forms can part on."""
    return -(2.0 ** (-span * (jnp.arange(heads, dtype=jnp.float32) + 1.0) / heads))


def attention_bias(heads, length, span=SLOPE_SPAN):
    """ALiBi plus the causal wall, [1, heads, length, length]."""
    pos = jnp.arange(length, dtype=jnp.float32)
    distance = pos[:, None] - pos[None, :]
    slopes = alibi_slopes(heads, span)
    alibi = slopes[None, :, None, None] * distance[None, None, :, :]
    wall = jnp.triu(jnp.ones((length, length), dtype=jnp.float32), k=1) * -1e9
    return alibi + wall[None, None, :, :]


def dropout_masks(key, rate, shape):
    """the multiplier form of inverted dropout: 0 or 1/keep, one for each element"""
    keep = 1.0 - rate
    return jax.random.bernoulli(key, keep, shape) / keep


def dropout(key, rate, count):
    """The `drop` a trunk's forward hands down: a fresh mask at each of [count] calls, or
    the identity where the rate is zero.

    ONE SHAPE FOR BOTH TRUNKS. The two wrote this twice and differently -- one mutated a
    key through `nonlocal`, one walked an iterator of splits -- and the iterator is the
    functional one. [count] is stated by the caller because it is a fact of the trunk: a
    `drop` called more often than the caller said raises StopIteration, where a lazy split
    would quietly hand out a key nobody accounted for.

    NO GATE PINS THE MASKS, and the change of shape here moves them: the key tree of one
    split of n is not the tree of n splits of one. Both eras are FROZEN and their
    checkpoints stand -- nothing that is held moves, and no retrain is planned."""
    if rate <= 0.0:

        def keep_all(x):
            return x

        return keep_all

    keys = iter(jax.random.split(key, count))

    def drop(x):
        return x * dropout_masks(next(keys), rate, x.shape)

    return drop


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
        shape = (corpus.SEATS, corpus.CLASSES, d)
        self.seats = nnx.Param(normal_at(rngs.params(), shape))
        self.phase = nnx.Param(normal_at(rngs.params(), (PHASE_BUCKETS, d)))

    @staticmethod
    def shapes(d):
        """the shape of each tensor of [tensors], for a draw that states no shape twice"""
        return [(corpus.SEATS, corpus.CLASSES, d), (PHASE_BUCKETS, d)]

    @property
    def d(self):
        """the width of the residual stream: the seat table sizes it"""
        return self.seats.shape[-1]

    def embed(self, classes, phases):
        """The input of one step: the four seat rows and the bar-phase row sum.

        A phase outside the table gathers a clamped row, in silence, which is why the
        table IS the bar and holds one row for each step of it."""
        seats = self.seats[...]
        rows = sum(seats[seat][classes[..., seat]] for seat in range(corpus.SEATS))
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
        logits = [None] * corpus.SEATS
        for seat in reversed(range(corpus.SEATS)):
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
        frame = np.zeros((len(h), corpus.SEATS), dtype=np.int32)
        for seat in reversed(range(corpus.SEATS)):
            raw = (_host_rms_norm(stream) @ seats[seat].T).astype(np.float64)
            weights = sample.temper(raw, temperature, min_p)
            state, uniform = prng.uniform(state, True)
            frame[:, seat] = sample.pick_share(weights, uniform)
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
    """WHAT `ar_train.train` TAKES: a tied `Head` and a `layers` list under it.

    The step-frame eras are one skeleton -- the head states the frame and the layers carry
    the stream -- thus the three rules that read only that skeleton stand here and not
    once in each era. An era states its own `hidden` and its own `describe`; everything
    below reads `hidden` through `seat_nll` and reads nothing else of the era.

    Era six is NOT a subclass: its sheet is not a stream of frames, and its own trunk
    walks `every_layer` where these walk `layers`.

    THE SKELETON IS THREE ATTRIBUTES AND A METHOD, and a subclass states all four:
    `self.head` (a `Head`), `self.layers` (the list, in the order of the ROM), and
    `hidden(classes, phases, *, dropout, key)`. `self.d` comes off the head. The integer
    twins hold the same three names and are NOT subclasses --
    `ar_quantized.QuantizedImage` states why."""

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
