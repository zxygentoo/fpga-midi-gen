"""The masked sheet of docs/diffusion.md: the roll, the mask, and the paper's net.

This is Coconet (Huang et al., arXiv 1903.07227) at the paper's size, on this corpus. One
sheet is a crop of 128 sixteenth steps -- eight measures, the excerpt length the paper's
raters heard -- as a PIANO ROLL of `data.CLASSES` rows by four voice channels. The model
is handed the roll with some cells hidden and states a categorical distribution over the
pitch rows for every cell, hidden or not. Nothing here is causal and nothing here draws:
the whole sheet is one input, and a piece is written knowing its own ending.

Four things stand here, because the trainer, the sampler, the integer twin and the two
referees all read them: the sheet and its mask planes, the net with its checkpoint, the
two mask distributions -- the orderless-NADE draw of the training loss and the annealed
Bernoulli of the Gibbs walk -- and the rules of the walk itself.

THE RULES OF THE WALK STAND HERE AND NOT IN ONE OF ITS TWO WALKERS. `opening_sheet`,
`anneal_threshold`, `hidden_cells`, `logits` and `tempered_pick` are what the float walk of
`diffusion/infer.py` and the integer walk of `diffusion/quantized.py` must do IDENTICALLY:
the same opening from the same seed, the same threshold at the same pass, the same uniform
for the same cell. `quantized.drift` and `tests/test_drift.py` compare exactly those two
walks, cell for cell on the same uniform. `lib/diffusion/model.ml` holds the first three
under the same names.

THE NET IS A MODULE TREE AND NOT A LIST OF LAYERS, thus `__call__` reads as the paper's own
diagram and an odd layer count is UNREPRESENTABLE. `diffusion/quantized.py` carries the
same skeleton in integers, thus "the same net in the arithmetic the board holds" is
auditable layer for layer.

WHAT IS PINNED FROM THE PAPER, AND WHERE IT CAME FROM. The referee of this round compares
against a published number, thus every constant of the model carries its source:

- The shape (section 3, equations 6 and 7): L = 64 layers of three-by-three convolution
  over time AND pitch, H = 128 channels, batch normalization with the statistics tied
  across time and pitch, a residual connection past every second layer, and a final
  projection to the four voice channels. About nine million parameters.
- The input (equations 4 and 5): 2I planes -- the four voices of the masked roll and the
  four mask planes. A masked cell shows zero in the roll and one in its mask plane. The
  paper's own equation makes the second plane hot on the CONTEXT and its code release
  makes it hot on the MASK; the release is followed, because the release made the number.
- The training mask: the code release draws the MASKED count uniform on 1 to D and then a
  uniform subset of that size. The paper's line reads "|C| ~ U(1, D)" for the context, but
  its own reweighting term D - d + 1 is the masked count, and a context of all D cells
  would divide by zero.
- The Gibbs schedule (section 5.2, citing Yao et al.): the annealed masking probability,
  with the constants from the code release.

The pitch axis is the paper's reason to convolve over it: "the locality of contrapuntal
rules and their near-invariance to translation, both in time and in pitch space". Pitch in
the channels would have to learn each interval separately at every absolute pitch, from 228
chorales.

Checkpoints are safetensors, "0" upward in construction order: for each layer the kernel,
the two norm terms, and the two population statistics. The layer count follows from the
tensor count and the widths from the shapes, thus a checkpoint reads without a flag.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file

import data
import measure
import nn
import prng

# ---------------------------------------------------------------------
# the sheet: the classes of a crop as the paper's input planes
# ---------------------------------------------------------------------

# The paper has no silence row because its data always sings; this corpus rests in 0.35
# percent of its cells, thus silence is one more class and the paper's constraint -- one
# row for each voice at each step -- still holds.
ROWS = data.CLASSES
VOICES = data.SEATS

# The paper's shape. These are the defaults of the trainer, not a limit of the code: the
# board ladder of a later round subtracts from them and measures each cut.
CROP = 128  # T, eight measures of the sixteenth grid
LAYERS = 64  # L
WIDTH = 128  # H
KERNEL = 3
# the code release's batch_norm_variance_epsilon
NORM_EPSILON = 1e-7

# The gamma initializer of the code release. A norm that opens at a tenth keeps the
# residual branch small, and a trunk of 64 layers then trains from a draw.
NORM_SCALE = 0.1

# THE DRAW IS HE NORMAL AND IT IS UNTRUNCATED: standard deviation sqrt(2 / fan_in) over
# the reach and the input channels, the code release's. `jax.nn.initializers.he_normal()`
# is the TRUNCATED normal -- it cuts at two sigma and rescales -- thus it draws a
# different model, and `nnx.Conv`'s own default is LeCun normal, which is a different
# model again.
HE_NORMAL = jax.nn.initializers.variance_scaling(2.0, "fan_in", "normal")

# what a reader needs to find the layers in a flat file; [NormedConv.tensors] states the
# order
LAYER_TENSORS = 5


def cells(steps):
    """D, the variables of one sheet: one voice at one step.

    It is not the cells of the roll. A cell is a single categorical over the pitch rows,
    thus it is masked whole or not at all, and every count of the round -- the mask draw,
    the loss divisor, the paper's rule of thumb N = I times T -- counts these."""
    return steps * VOICES


def planes(classes, hidden):
    """The paper's input: [batch, steps, ROWS, 2 * VOICES], the masked roll beside the mask.

    A masked cell shows zero in every row of its roll column and one in every row of its
    mask plane. The mask is over cells and broadcasts up the pitch axis, which is what makes
    it readable to a convolution over that axis."""
    roll = jnp.moveaxis(jax.nn.one_hot(classes, ROWS, dtype=jnp.float32), -1, -2)
    masked = jnp.asarray(hidden, jnp.float32)[..., None, :]
    return jnp.concatenate(
        [roll * (1.0 - masked), jnp.broadcast_to(masked, roll.shape)], axis=-1
    )


# ---------------------------------------------------------------------
# the two mask distributions
# ---------------------------------------------------------------------


def orderless_masks(key, batch, steps):
    """The training mask of orderless NADE: a uniform masked count, then a uniform subset of
    that size, one draw for each sheet of the batch.

    The rank of a uniform draw states the subset without a shuffle, and states it for the
    whole batch at one time."""
    count_key, order_key = jax.random.split(key)
    width = cells(steps)
    counts = jax.random.randint(count_key, (batch, 1), 1, width + 1)
    order = jax.random.uniform(order_key, (batch, width))
    ranks = jnp.argsort(jnp.argsort(order, axis=-1), axis=-1)
    return (ranks < counts).reshape(batch, steps, VOICES)


# The annealed masking probability of Yao et al., as the code release pins it:
# `YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7)`. The paper states the formula and names no
# values. They are levers of this round and not decisions.
ANNEAL_LOW = 0.1
ANNEAL_HIGH = 0.9
ANNEAL_SPAN = 0.7


def anneal(step, total):
    """The masking probability at step [step] of [total] Gibbs steps:

        alpha_n = max(LOW, HIGH - n (HIGH - LOW) / (SPAN * total))

    High at the opening, where the chain mixes fast and independent resampling is a poor
    approximation, and settled on [ANNEAL_LOW] after an [ANNEAL_SPAN] share of the walk,
    where blocked Gibbs has become nearly the one-variable-at-a-time chain it
    approximates."""
    return max(
        ANNEAL_LOW,
        ANNEAL_HIGH - (ANNEAL_HIGH - ANNEAL_LOW) * step / (ANNEAL_SPAN * total),
    )


# ---------------------------------------------------------------------
# the rules of the walk: the opening, and the mask of one pass
# ---------------------------------------------------------------------


def opening_sheet(states, steps):
    """`Model.opening_sheet`: a sheet of random notes for each walk of the batch, each voice
    inside the register of its own seat -- one uniform for each cell in the cell order, and
    the class [low + floor(u * width)] over [measure.RANGES].

    WHY THE WALK DOES NOT OPEN ON SILENCE, which is the paper's own opening. The paper's roll
    has no silence row, thus an empty cell there states nothing; THIS roll holds silence as a
    class, so an empty cell would state a REST with the authority of context. Measured
    2026-08-25 over 256 sheets, the two openings are the same instrument, and the silent one
    was removed.

    The draw is over the registers and not the whole roll, because a bass at 81 and a soprano
    at 36 are further from this corpus than a rest is. The product [u * width] is exact on
    the 24-bit grid, thus the circuit states the same class from the same seed."""
    sheets = len(states)
    classes = np.zeros((sheets, steps, VOICES), np.int32)
    everyone = np.ones(sheets, bool)
    lows = np.array([low - data.PITCH_LOW + 1 for low, _ in measure.RANGES])
    widths = np.array([high - low + 1 for low, high in measure.RANGES])
    for step in range(steps):
        for voice in range(VOICES):
            states, u = prng.uniform(states, everyone)
            classes[:, step, voice] = lows[voice] + np.floor(u * widths[voice]).astype(
                np.int32
            )
    return states, classes


def anneal_threshold(step, total):
    """`Model.anneal_threshold`: the masking threshold of pass [step] of [total], on the
    24-bit grid of the generator. A cell hides exactly when its word falls under it."""
    return math.floor(anneal(step, total) * 2.0**prng.UNIFORM_BITS)


def hidden_cells(states, steps, threshold):
    """`Model.hidden_cells`: the mask of one pass -- one uniform for each cell in the cell
    order, step-major and seat-minor, hidden exactly when its word falls under the
    threshold.

    The word compare and the float compare `u * 2^24 < threshold` are one test: the product
    is the word, exactly, on the grid, thus the two walks hide the same cells."""
    everyone = np.ones(len(states), bool)
    hidden = np.zeros((len(states), steps, VOICES), dtype=bool)
    for step in range(steps):
        for voice in range(VOICES):
            states, word = prng.uniform_word(states, everyone)
            hidden[:, step, voice] = word < threshold
    return states, hidden


# ---------------------------------------------------------------------
# the net
# ---------------------------------------------------------------------


class Statistics(NamedTuple):
    """What one norm read: the mean and the variance of a channel over the batch, the
    steps and the rows together.

    A training call gives back the BATCH's own, which the trainer folds into the
    population; any other call gives back the population it read."""

    mean: jax.Array
    variance: jax.Array


class PopulationNorm(nnx.Module):
    """The batch norm of equation 6, with the population the era's trainer folds.

    The statistics are tied across time and pitch -- the mean and the variance of a
    channel run over the batch, the steps and the rows together -- thus a channel means
    one thing wherever it stands, which is the same translation argument that put the
    pitch on an axis.

    IT IS NOT `nnx.BatchNorm`, AND THE POPULATION IS THE REASON. That module keeps an
    exponential moving average at a fixed momentum from its first call; the era's rule is a
    WARMED decay -- `train.population_decay` -- and the fold happens OUTSIDE the gradient,
    from the statistics a training pass gives back. [fold] is where a decay writes."""

    def __init__(self, features, *, norm_scale=NORM_SCALE):
        self.scale = nnx.Param(jnp.full(features, norm_scale, jnp.float32))
        self.shift = nnx.Param(jnp.zeros(features, jnp.float32))
        # the population opens at the prior and nothing reads it until the trainer has
        # filled it: a training pass reads the batch's own
        self.mean = nnx.BatchStat(jnp.zeros(features, jnp.float32))
        self.variance = nnx.BatchStat(jnp.ones(features, jnp.float32))

    def __call__(self, a, training=False):
        """the normed activations, and the statistics the pass read.

        In training the norm reads the batch's own statistics, and it gives them back for
        the trainer to fold into the population. Otherwise it reads the population it was
        handed. A walk of the sampler must not depend on what else is in its batch, and
        neither must a referee that has to reproduce a published number."""
        if training:
            mean = jnp.mean(a, axis=(0, 1, 2))
            variance = jnp.mean(jnp.square(a - mean), axis=(0, 1, 2))
        else:
            mean, variance = self.mean[...], self.variance[...]
        normed = (a - mean) * jax.lax.rsqrt(variance + NORM_EPSILON)
        return normed * self.scale[...] + self.shift[...], Statistics(mean, variance)

    def fold(self, seen, decay):
        """the population keeps [decay] of itself and takes the rest from [seen]. The share is
        the TRAINER's rule and the write is this module's; nothing else may move one."""
        self.mean[...] = decay * self.mean[...] + (1.0 - decay) * seen.mean
        self.variance[...] = decay * self.variance[...] + (1.0 - decay) * seen.variance


class NormedConv(nnx.Module):
    """One layer of the paper: the convolution, then the batch norm of equation 6.

    The convolution carries NO BIAS -- the norm behind it carries the shift -- and its padding
    is zero at both edges, because both edges are real: a crop opens and closes on a boundary
    the model must see, and past the ends of the pitch axis there is no music."""

    def __init__(self, inputs, outputs, *, norm_scale=NORM_SCALE, rngs):
        self.conv = nnx.Conv(
            inputs,
            outputs,
            (KERNEL, KERNEL),
            padding="SAME",
            use_bias=False,
            kernel_init=HE_NORMAL,
            rngs=rngs,
        )
        self.norm = PopulationNorm(outputs, norm_scale=norm_scale)

    def __call__(self, x, training=False):
        return self.norm(self.conv(x), training)

    def tensors(self):
        """the five tensors of this layer in the order of the checkpoint: the kernel, the
        two norm terms, the two population statistics"""
        return [
            self.conv.kernel[...],
            self.norm.scale[...],
            self.norm.shift[...],
            self.norm.mean[...],
            self.norm.variance[...],
        ]

    def take(self, tensors):
        """the reverse of [tensors]: the five of one layer, written in. The two stand together
        so that the layout cannot drift apart."""
        kernel, scale, shift, mean, variance = tensors
        self.conv.kernel[...] = jnp.asarray(kernel)
        self.norm.scale[...] = jnp.asarray(scale)
        self.norm.shift[...] = jnp.asarray(shift)
        self.norm.mean[...] = jnp.asarray(mean)
        self.norm.variance[...] = jnp.asarray(variance)


class ResidualPair(nnx.Module):
    """Two layers and the skip past both: the paper's equation 7, which adds the input of
    the pair after the second norm and activates once on the sum."""

    def __init__(self, width, *, norm_scale=NORM_SCALE, rngs):
        self.first = NormedConv(width, width, norm_scale=norm_scale, rngs=rngs)
        self.second = NormedConv(width, width, norm_scale=norm_scale, rngs=rngs)

    def __call__(self, x, training=False):
        first, seen_first = self.first(x, training)
        second, seen_second = self.second(jax.nn.relu(first), training)
        return jax.nn.relu(x + second), [seen_first, seen_second]


class Trunk(nnx.Module):
    """The skeleton both models of the era carry: a stem, the residual pairs, a head.

    [Coconet] fills it with the float layers and `quantized.QuantizedCoconet` with their
    integer twins, UNDER THE SAME ATTRIBUTE NAMES AT EVERY LEVEL, thus a reader can put the
    two side by side and audit them layer for layer."""

    def every_layer(self):
        """every layer in the order the network runs. The checkpoint, the population fold, the
        quantizer and the contract file all walk a trunk this way, and this is the one place
        either tree states that order."""
        walked = [self.stem]
        for pair in self.pairs:
            walked += [pair.first, pair.second]
        return walked + [self.head]


class Coconet(Trunk):
    """The paper's net: a stem, (L - 2) / 2 residual pairs, a head.

    The first layer takes the 2I input planes to H channels and the last takes H to the four
    voices, thus neither can carry a residual. The head keeps its norm and takes no
    activation, as equation 7 states, thus its norm is a learned scale on the logits.

    THE LAYER COUNT IS EVEN AND AT LEAST FOUR, and the tree is why."""

    def __init__(self, layers, width, *, norm_scale=NORM_SCALE, rngs):
        if layers < 4 or layers % 2:
            raise ValueError(f"{layers} layers is no sheet model")
        self.stem = NormedConv(2 * VOICES, width, norm_scale=norm_scale, rngs=rngs)
        self.pairs = nnx.List(
            [
                ResidualPair(width, norm_scale=norm_scale, rngs=rngs)
                for _ in range((layers - 2) // 2)
            ]
        )
        self.head = NormedConv(width, VOICES, norm_scale=norm_scale, rngs=rngs)

    def __call__(self, sheet, *, training=False, remat=False):
        """The logits of every cell of the sheet: [batch, steps, ROWS, VOICES], and the
        statistics of every norm in the layer order.

        The softmax runs over the ROWS axis, thus the model states a pitch for the masked
        cells and the context alike; the loss and the walk read only the masked ones.

        At inference the statistics are the population it was handed and the caller drops
        them; in training they are the batch's own and the trainer folds them."""

        # THE PAIR IS THE UNIT OF REMATERIALISATION, thus a trunk this deep keeps 31
        # tensors for the backward pass instead of a few hundred. [training] closes in
        # because it decides the SHAPE of the pass and not a value inside it, and the
        # closure costs nothing: [__call__] is traced once for each shape under nnx.jit.
        def run_pair(pair, x):
            return pair(x, training)

        run_pair = nnx.remat(run_pair) if remat else run_pair
        h, stem = self.stem(sheet, training)
        h = jax.nn.relu(h)
        seen = [stem]
        for pair in self.pairs:
            h, pair_seen = run_pair(pair, h)
            seen += pair_seen
        said, head = self.head(h, training)
        return said, seen + [head]

    def parameter_count(self):
        """the trainable parameters; the population statistics are not among them"""
        held = jax.tree.leaves(nnx.state(self, nnx.Param))
        return sum(int(np.prod(t.shape)) for t in held)

    def save(self, path):
        """The whole model as one flat list, in the order the network runs.

        THE POPULATION STATISTICS TRAVEL INSIDE THE CHECKPOINT. They are not parameters, but
        the model cannot state a probability without them, and a reader that had to find them
        in a second file would sooner or later pair them with weights they never saw."""
        flat = [tensor for layer in self.every_layer() for tensor in layer.tensors()]
        nn.save_checkpoint(path, flat)

    @classmethod
    def load(cls, path):
        """The model of one checkpoint. A model of L layers holds 5 L tensors, thus the count
        states the layers and the stem's kernel states the width."""
        tensors = load_file(str(path))
        layers, spare = divmod(len(tensors), LAYER_TENSORS)
        if spare or layers < 4 or layers % 2:
            raise ValueError(f"{path}: {len(tensors)} tensors is no sheet model")
        # the draw is thrown away one tensor at a time below: it is the cost of one He draw
        # and it buys the one constructor
        held = cls(layers, int(tensors["0"].shape[3]), rngs=nnx.Rngs(0))
        for at, layer in enumerate(held.every_layer()):
            base = LAYER_TENSORS * at
            layer.take([tensors[str(base + on)] for on in range(LAYER_TENSORS)])
        return held

    @classmethod
    def drawn(cls, seed, layers, width, norm_scale=1.0):
        """A model of DRAWN weights, at a shape a gate can afford, thus no gate reads a
        checkpoint that git ignores.

        The draw follows the SHAPE of the trainer and not its values. It draws with numpy at
        a stated seed and not with the module's own initializer, because a gate's expected
        numbers must not move when a framework changes its key rule.

        THE NORM SCALE OPENS AT 1.0 AND NOT AT THE TRAINER'S TENTH. At the tenth a DRAWN
        trunk decays its activations tenfold at every layer — a trained norm grows out of
        that opening, an untrained one never leaves it — and by the third layer a gate over
        the integer twin reads the resolution floor of the format instead of the
        arithmetic."""
        held = cls(layers, width, norm_scale=norm_scale, rngs=nnx.Rngs(0))
        rng = np.random.default_rng(seed)
        for at, layer in enumerate(held.every_layer()):
            inputs = 2 * VOICES if at == 0 else width
            outputs = VOICES if at == layers - 1 else width
            deviation = math.sqrt(2.0 / (KERNEL * KERNEL * inputs))
            layer.take(
                [
                    rng.normal(0.0, deviation, (KERNEL, KERNEL, inputs, outputs)).astype(
                        np.float32
                    ),
                    np.full(outputs, norm_scale, np.float32),
                    np.zeros(outputs, np.float32),
                    np.zeros(outputs, np.float32),
                    np.ones(outputs, np.float32),
                ]
            )
        return held


def nll_of_logits(said, classes):
    """the negative log likelihood of the true class of every cell: [batch, steps,
    VOICES], in nats, before any mask or any divisor"""
    logp = jax.nn.log_softmax(said, axis=-2)
    return -jnp.take_along_axis(logp, classes[..., None, :], axis=-2)[..., 0, :]


@nnx.jit
def logits(coconet, classes, hidden):
    """The logits of one pass over the batch, from the masked sheet.

    It takes the model as an ARGUMENT and stands at the module level, thus its compiled form
    is keyed on the shapes and every walk of a run reuses the first compile."""
    said, _ = coconet(planes(classes, hidden))
    return said


def tempered_pick(raw, temperature, uniform):
    """The draw of one cell over the batch: the tempered weights, the min-p floor and the
    pick, row for row.
    [raw] is [sheets, ROWS] float64.

    The era draws with no min-p floor, thus the temper is the peak alone. `nn.pick_share`'s own
    docstring holds the argument that no fallback is needed."""
    return nn.pick_share(nn.temper(raw, temperature, 0.0), uniform)
