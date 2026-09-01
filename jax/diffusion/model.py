"""The masked sheet of docs/diffusion.md: the roll, the mask, and the paper's net.

This is Coconet (Huang et al., arXiv 1903.07227) at the paper's size, on this corpus. One
sheet is a crop of 128 sixteenth steps -- eight measures, the excerpt length the paper's
raters heard -- as a PIANO ROLL of `corpus.CLASSES` rows by four voice channels. The model
is handed the roll with some cells hidden and states a categorical distribution over the
pitch rows for every cell, hidden or not. Nothing here is causal and nothing here draws:
the whole sheet is one input, and a piece is written knowing its own ending.

Four things stand here, because the trainer, the sampler, the integer twin and the two
referees all read them: the sheet and its mask planes, the two mask distributions, the
opening and the cell order of a walk, and the net with its checkpoint. The loop that
spends them, and the draw of one cell, are `diffusion/sample.py`.

THE RULES OF THE WALK STAND HERE AND NOT IN ONE OF ITS TWO WALKERS. `cell_order`,
`opening_sheet`, `hidden_cells` and `Coconet.logits` are what the float walk of
`diffusion/infer.py` and the integer walk of `diffusion/quantized.py` must do
IDENTICALLY: the same opening from the same seed, the same cells hidden at the same pass,
the same uniform for the same cell. `lib/diffusion/model.ml` holds the first three under
the same names.

THE NET IS A MODULE TREE AND NOT A LIST OF LAYERS, thus `__call__` reads as the paper's
own diagram and an odd layer count is UNREPRESENTABLE. `diffusion/quantized.py` carries
the same skeleton in integers, auditable layer for layer.

WHAT IS PINNED FROM THE PAPER, AND WHERE IT CAME FROM. The referee compares against a
published number, thus every constant carries its source:

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

The pitch axis is the paper's reason to convolve over it: "the locality of contrapuntal
rules and their near-invariance to translation, both in time and in pitch space".

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

import corpus
import prng
from train import save_checkpoint

ROWS = corpus.CLASSES
VOICES = corpus.SEATS
PLANES = 2 * VOICES # one class plane and one mask plane for each seat

CROP = 128   # T
LAYERS = 64  # L
WIDTH = 128  # H
KERNEL = 3

NORM_EPSILON = 1e-7
NORM_SCALE = 0.1
HE_NORMAL = jax.nn.initializers.variance_scaling(2.0, "fan_in", "normal")

LAYER_TENSORS = 5


def cells(steps):
    """D, the variables of one sheet: one voice at one step, and NOT the cells of the
    roll. A cell is a single categorical over the pitch rows, thus it is masked whole or
    not at all, and every count of the round counts these."""
    return steps * VOICES


def cell_order(steps):
    """`Model.cell_order`: the (step, voice) pairs of one sheet, A STEP AT A TIME AND THE
    SEATS OF A STEP INSIDE IT.

    IT IS A CONTRACT AND NOT A CONVENIENCE: every uniform of a walk is drawn in this
    order, on the host and in the circuit alike, and another order draws a different
    piece from the same seed with nothing to say so."""
    return [(step, voice) for step in range(steps) for voice in range(VOICES)]


def planes(classes, hidden):
    """The paper's input: [batch, steps, ROWS, PLANES], the masked roll beside the mask.
    A masked cell shows zero in every row of its roll column and one in every row of its
    mask plane; the mask broadcasts up the pitch axis, which is what makes it readable to
    a convolution over that axis."""
    roll = jnp.moveaxis(jax.nn.one_hot(classes, ROWS, dtype=jnp.float32), -1, -2)
    masked = jnp.asarray(hidden, jnp.float32)[..., None, :]
    return jnp.concatenate(
        [roll * (1.0 - masked), jnp.broadcast_to(masked, roll.shape)], axis=-1
    )


def orderless_masks(key, batch, steps):
    """The training mask of orderless NADE: a uniform masked count, then a uniform subset
    of that size, one draw for each sheet. The rank of a uniform draw states the subset
    without a shuffle, for the whole batch at one time."""
    count_key, order_key = jax.random.split(key)
    width = cells(steps)
    counts = jax.random.randint(count_key, (batch, 1), 1, width + 1)
    order = jax.random.uniform(order_key, (batch, width))
    ranks = jnp.argsort(jnp.argsort(order, axis=-1), axis=-1)
    return (ranks < counts).reshape(batch, steps, VOICES)


def opening_sheet(states, steps):
    """`Model.opening_sheet`: a sheet of random notes for each walk of the batch, each
    voice inside the register of its own seat -- one uniform for each cell in the cell
    order, and the class [low + floor(u * width)] over [corpus.VOICE_RANGES].

    WHY IT DOES NOT OPEN ON SILENCE, which is the paper's own opening: the paper's roll
    has no silence row, thus an empty cell states nothing there, where THIS roll would
    state a REST with the authority of context. The two openings measure as the same
    instrument.

    The draw is over the registers and not the whole roll, because a bass at 81 is further
    from this corpus than a rest is. The product [u * width] is exact on the 24-bit grid,
    thus the circuit states the same class from the same seed."""
    sheets = len(states)
    classes = np.zeros((sheets, steps, VOICES), np.int32)
    everyone = np.ones(sheets, bool)
    lows = np.array([low - corpus.PITCH_LOW + 1 for low, _ in corpus.VOICE_RANGES])
    widths = np.array([high - low + 1 for low, high in corpus.VOICE_RANGES])
    for step, voice in cell_order(steps):
        states, u = prng.uniform(states, everyone)
        classes[:, step, voice] = lows[voice] + np.floor(u * widths[voice]).astype(
            np.int32
        )
    return states, classes


def hidden_cells(states, steps, threshold):
    """`Model.hidden_cells`: the mask of one pass -- one uniform for each cell in the
    cell order, hidden exactly when its word falls under the threshold. The word
    compare and the float compare `u * 2^24 < threshold` are one test on the grid, thus
    the two walks hide the same cells."""
    everyone = np.ones(len(states), bool)
    hidden = np.zeros((len(states), steps, VOICES), dtype=bool)
    for step, voice in cell_order(steps):
        states, word = prng.uniform_word(states, everyone)
        hidden[:, step, voice] = word < threshold
    return states, hidden


class Statistics(NamedTuple):
    """What one norm read: the mean and the variance of a channel over the batch, the
    steps and the rows together. A training call gives back the BATCH's own; any other
    gives back the population it read."""

    mean: jax.Array
    variance: jax.Array


class PopulationNorm(nnx.Module):
    """The batch norm of equation 6, with the population the era's trainer folds. The
    statistics are TIED ACROSS TIME AND PITCH, thus a channel means one thing wherever it
    stands -- the same translation argument that put the pitch on an axis.

    IT IS NOT `nnx.BatchNorm`, AND THE POPULATION IS THE REASON: that module keeps a
    moving average at a fixed momentum from its first call, where the era's rule is a
    WARMED decay and the fold happens OUTSIDE the gradient. [fold] is where a decay
    writes."""

    def __init__(self, features, *, norm_scale=NORM_SCALE):
        self.scale = nnx.Param(jnp.full(features, norm_scale, jnp.float32))
        self.shift = nnx.Param(jnp.zeros(features, jnp.float32))
        # the population opens at the prior and nothing reads it until the trainer has
        # filled it: a training pass reads the batch's own
        self.mean = nnx.BatchStat(jnp.zeros(features, jnp.float32))
        self.variance = nnx.BatchStat(jnp.ones(features, jnp.float32))

    def __call__(self, a, training=False):
        """The normed activations, and the statistics the pass read. In training the norm
        reads the BATCH's own and gives them back to fold; otherwise it reads the
        population, because no walk may depend on what else is in its batch."""
        if training:
            mean = jnp.mean(a, axis=(0, 1, 2))
            variance = jnp.mean(jnp.square(a - mean), axis=(0, 1, 2))
        else:
            mean, variance = self.mean[...], self.variance[...]
        normed = (a - mean) * jax.lax.rsqrt(variance + NORM_EPSILON)
        return normed * self.scale[...] + self.shift[...], Statistics(mean, variance)

    def fold(self, seen, decay):
        """the population keeps [decay] of itself and takes the rest from [seen]. The
        share is the TRAINER's rule and the write is this module's; nothing else may move
        one."""
        self.mean[...] = decay * self.mean[...] + (1.0 - decay) * seen.mean
        self.variance[...] = decay * self.variance[...] + (1.0 - decay) * seen.variance


class NormedConv(nnx.Module):
    """One layer of the paper: the convolution, then the batch norm of equation 6. The
    convolution carries NO BIAS -- the norm behind it carries the shift -- and pads with
    zero at both edges, because both edges are real."""

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

    def set_tensors(self, tensors):
        """the reverse of [tensors]; the two stand together so the layout cannot drift"""
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
    """The skeleton both models of the era carry: a stem, the residual pairs, a head. The
    float tree and the integer twin fill it under the SAME attribute names at every
    level."""

    def layers(self):
        """every layer in the order the network runs: the checkpoint, the population fold,
        the quantizer and the contract file all walk a trunk this way"""
        walked = [self.stem]
        for pair in self.pairs:
            walked += [pair.first, pair.second]
        return walked + [self.head]


class Coconet(Trunk):
    """The paper's net: a stem, (L - 2) / 2 residual pairs, a head. The first layer takes
    the input planes to H channels and the last takes H to the four voices, thus neither
    carries a residual; the head keeps its norm and takes no activation, as equation 7
    states. THE LAYER COUNT IS EVEN AND AT LEAST FOUR, and the tree is why."""

    def __init__(self, layer_count, width, *, norm_scale=NORM_SCALE, rngs):
        if layer_count < 4 or layer_count % 2:
            raise ValueError(f"{layer_count} layers is no sheet model")
        self.stem = NormedConv(PLANES, width, norm_scale=norm_scale, rngs=rngs)
        self.pairs = nnx.List(
            [
                ResidualPair(width, norm_scale=norm_scale, rngs=rngs)
                for _ in range((layer_count - 2) // 2)
            ]
        )
        self.head = NormedConv(width, VOICES, norm_scale=norm_scale, rngs=rngs)

    def __call__(self, sheet, *, training=False, remat=False):
        """The logits of every cell of the sheet: [batch, steps, ROWS, VOICES], and the
        statistics of every norm in the layer order. The softmax runs over the ROWS axis,
        thus the model states a pitch for the masked cells and the context alike."""

        # THE PAIR IS THE UNIT OF REMATERIALISATION, thus a trunk this deep keeps 31
        # tensors for the backward pass instead of a few hundred. [training] closes in
        # because it decides the SHAPE of the pass and not a value inside it.
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

    @nnx.jit
    def logits(self, classes, hidden):
        """The logits of one pass over the batch, from the masked sheet: what both walks
        read, with the statistics dropped.

        THE JIT IS BUILT ONCE, WITH THE CLASS, thus its compiled form is keyed on the
        shapes and every walk of a run reuses the first compile -- an `nnx.jit` built
        inside a walker would be a new callable at every call."""
        said, _ = self(planes(classes, hidden))
        return said

    def parameter_count(self):
        """the trainable parameters; the population statistics are not among them"""
        held = jax.tree.leaves(nnx.state(self, nnx.Param))
        return sum(int(np.prod(t.shape)) for t in held)

    def save(self, path):
        """The whole model as one flat list, in the order the network runs. THE POPULATION
        STATISTICS TRAVEL INSIDE IT: they are not parameters, but the model cannot state a
        probability without them."""
        flat = [tensor for layer in self.layers() for tensor in layer.tensors()]
        save_checkpoint(path, flat)

    @classmethod
    def load(cls, path):
        """The model of one checkpoint. A model of L layers holds 5 L tensors, thus the
        count states the layers and the stem's kernel states the width."""
        tensors = load_file(str(path))
        layer_count, spare = divmod(len(tensors), LAYER_TENSORS)
        if spare or layer_count < 4 or layer_count % 2:
            raise ValueError(f"{path}: {len(tensors)} tensors is no sheet model")
        # the draw is thrown away tensor by tensor below; it buys the one constructor
        held = cls(layer_count, int(tensors["0"].shape[3]), rngs=nnx.Rngs(0))
        for at, layer in enumerate(held.layers()):
            base = LAYER_TENSORS * at
            layer.set_tensors([tensors[str(base + on)] for on in range(LAYER_TENSORS)])
        return held

    @classmethod
    def drawn(cls, seed, layer_count, width, norm_scale=1.0):
        """A model of DRAWN weights, at a shape a gate can afford. It draws with NUMPY at
        a stated seed and not with the module's own initializer, so a gate's expected
        numbers do not move when a framework changes its key rule.

        THE NORM SCALE OPENS AT 1.0 AND NOT AT THE TRAINER'S TENTH: at the tenth a drawn
        trunk decays its activations tenfold at every layer, and by the third layer a gate
        over the twin reads the resolution floor of the format instead of the
        arithmetic."""
        held = cls(layer_count, width, norm_scale=norm_scale, rngs=nnx.Rngs(0))
        rng = np.random.default_rng(seed)
        for at, layer in enumerate(held.layers()):
            inputs = PLANES if at == 0 else width
            outputs = VOICES if at == layer_count - 1 else width
            deviation = math.sqrt(2.0 / (KERNEL * KERNEL * inputs))
            layer.set_tensors(
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
