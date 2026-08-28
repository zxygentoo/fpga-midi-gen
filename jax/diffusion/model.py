"""The masked canvas of docs/diffusion.md: the roll, the mask, and the paper's net.

This is Coconet (Huang et al., arXiv 1903.07227) at the paper's size, on this corpus. One
canvas is a crop of 128 sixteenth steps -- eight measures, the excerpt length the paper's
raters heard -- as a PIANO ROLL of `data.CLASSES` rows by four voice channels. The model
is handed the roll with some cells hidden and states a categorical distribution over the
pitch rows for every cell, hidden or not. Nothing here is causal and nothing here draws:
the whole canvas is one input, and a piece is written knowing its own ending.

Three things stand here, because the trainer, the sampler and the two referees all read
them: the canvas and its mask planes, the net with its checkpoint, and the two mask
distributions -- the orderless-NADE draw of the training loss and the annealed Bernoulli
of the Gibbs walk.

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
rules and their near-invariance to translation, both in time and in pitch space". The
proto round of feat/diffusion-proto put pitch in the channels and had to learn each
interval separately at every absolute pitch, from 228 chorales.

Checkpoints are safetensors, "0" upward in construction order, as nn.save_checkpoint
states the rule of the seam: for each layer the kernel, the two norm terms, and the two
population statistics. The layer count follows from the tensor count and the widths from
the shapes, thus a checkpoint reads without a flag.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

import data

# ---------------------------------------------------------------------
# the canvas: the classes of a crop as the paper's input planes
# ---------------------------------------------------------------------

# The roll holds one row for each class of the vocabulary of this repository: row 0 is
# silence, the rows 1 to 46 are the pitches 36 to 81, and row 47 is the spare row the
# vocabulary already keeps. The paper has no silence row because its data always sings;
# this corpus rests in 0.35 percent of the cells inside a piece, thus silence is one more
# class and the paper's constraint -- one row for each voice at each step -- still holds.
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


def cells(steps):
    """D, the variables of one canvas: one voice at one step.

    It is not the cells of the roll. A cell is a single categorical over the pitch rows,
    thus it is masked whole or not at all, and every count of the round -- the mask draw,
    the loss divisor, the paper's rule of thumb N = I times T -- counts these."""
    return steps * VOICES


def planes(classes, hidden):
    """The paper's input: [batch, steps, ROWS, 2 * VOICES], the masked roll beside the
    mask.

    [classes] is [batch, steps, VOICES] and [hidden] is the same shape, true where a cell
    is masked. A masked cell shows zero in every row of its roll column and one in every
    row of its mask plane; the mask is over cells and broadcasts up the pitch axis, which
    is what makes it readable to a convolution over that axis."""
    roll = jnp.moveaxis(jax.nn.one_hot(classes, ROWS, dtype=jnp.float32), -1, -2)
    masked = jnp.asarray(hidden, jnp.float32)[..., None, :]
    return jnp.concatenate(
        [roll * (1.0 - masked), jnp.broadcast_to(masked, roll.shape)], axis=-1
    )


# ---------------------------------------------------------------------
# the two mask distributions
# ---------------------------------------------------------------------


def orderless_masks(key, batch, steps):
    """The training mask of orderless NADE: a uniform masked count, then a uniform subset
    of that size, one draw for each canvas of the batch.

    This is the code release's `OrderlessMaskoutMethod`: `k = choice(D) + 1` cells masked,
    `choice(D, size=k, replace=False)` which ones. The rank of a uniform draw states the
    subset without a shuffle, and it states it for the whole batch at one time."""
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
# the net
# ---------------------------------------------------------------------


def conv(x, kernel):
    """One convolution over the step axis and the pitch axis, and no bias term -- the batch
    norm that follows carries the shift.

    The padding is zero at both edges and both edges are real. A crop opens and closes on a
    boundary the model must be able to see, and past the ends of the pitch axis there is no
    music: the corpus sings 36 to 81 and the vocabulary holds nothing else."""
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def normed_conv(x, params, stats, training):
    """One layer of the paper: the convolution, then the batch norm of equation 6.

    The statistics are tied across time and pitch -- the mean and the variance of a channel
    run over the batch, the steps and the rows together -- thus a channel means one thing
    wherever it stands, which is the same translation argument that put the pitch on an
    axis.

    In training the norm reads the batch's own statistics, and it gives them back for the
    trainer to fold into the population. Otherwise it reads the population it was handed. A
    walk of the sampler must not depend on what else is in its batch, and neither must a
    referee that has to reproduce a published number."""
    a = conv(x, params["kernel"])
    if training:
        mean = jnp.mean(a, axis=(0, 1, 2))
        variance = jnp.mean(jnp.square(a - mean), axis=(0, 1, 2))
    else:
        mean, variance = stats["mean"], stats["variance"]
    normed = (a - mean) * jax.lax.rsqrt(variance + NORM_EPSILON)
    return normed * params["scale"] + params["shift"], {
        "mean": mean,
        "variance": variance,
    }


def residual_pair(x, params, stats, training):
    """Two layers and the skip past both: the paper's equation 7, which adds the input of
    the pair after the second norm and activates once on the sum.

    The pair is the unit of rematerialisation, thus a trunk this deep keeps 31 tensors for
    the backward pass instead of a few hundred."""
    first, seen_first = normed_conv(x, params[0], stats[0], training)
    second, seen_second = normed_conv(jax.nn.relu(first), params[1], stats[1], training)
    return jax.nn.relu(x + second), [seen_first, seen_second]


def logits(params, stats, canvas, training=False, remat=False):
    """The logits of every cell of the canvas: [batch, steps, ROWS, VOICES], and the
    batch-norm statistics the pass read.

    [canvas] is the input planes and the softmax runs over the ROWS axis, thus the model
    states a pitch for every voice of every step -- the masked cells and the context alike.
    The loss reads only the masked ones and the Gibbs walk resamples only the masked ones.

    The first layer takes the 2I input planes to H channels and the last takes H to the
    four voices, thus neither can carry a residual: what remains is 31 pairs. The last
    layer keeps its norm and takes no activation, as equation 7 states, and its norm is
    then a learned scale on the logits.

    Returns the pair (logits, statistics). At inference the statistics are the ones it was
    handed and the caller drops them; in training they are the batch's own and the trainer
    folds them into the population."""
    layers = params["layers"]
    pair = partial(residual_pair, training=training)
    if remat:
        pair = jax.checkpoint(pair)
    h, stem = normed_conv(canvas, layers[0], stats[0], training)
    h = jax.nn.relu(h)
    seen = [stem]
    for at in range(1, len(layers) - 1, 2):
        h, pair_stats = pair(h, layers[at : at + 2], stats[at : at + 2])
        seen += pair_stats
    said, head = normed_conv(h, layers[-1], stats[-1], training)
    return said, seen + [head]


def nll_of_logits(said, classes):
    """the negative log likelihood of the true class of every cell: [batch, steps,
    VOICES], in nats, before any mask or any divisor"""
    logp = jax.nn.log_softmax(said, axis=-2)
    return -jnp.take_along_axis(logp, classes[..., None, :], axis=-2)[..., 0, :]


# ---------------------------------------------------------------------
# the checkpoint: one flat list, and its reader
# ---------------------------------------------------------------------

# The order of one layer's tensors on disk, stated one time: [flat_tensors] writes it and
# [load_params] reads it, and neither spells it out again.
PARAM_FIELDS = ("kernel", "scale", "shift")
STAT_FIELDS = ("mean", "variance")
LAYER_FIELDS = PARAM_FIELDS + STAT_FIELDS
LAYER_TENSORS = len(LAYER_FIELDS)


def flat_tensors(params, stats):
    """The whole model as one list, in the order the network runs.

    The population statistics travel INSIDE the checkpoint. No gradient reaches them and
    they are not parameters, but the model cannot state a probability without them, and a
    reader that had to find them in a second file would sooner or later pair statistics
    with weights they never saw.

    This function and [load_params] are the two halves of one layout. They stand together
    so that they cannot drift apart."""
    flat = []
    for layer, stat in zip(params["layers"], stats):
        flat += [layer[field] for field in PARAM_FIELDS]
        flat += [stat[field] for field in STAT_FIELDS]
    return flat


def load_params(path):
    """The (params, statistics) of a checkpoint.

    A model of L layers holds 5 L tensors, thus the count states the layers and the kernel
    shapes state the width. Nothing else is needed to read one."""
    tensors = load_file(path)
    layers, spare = divmod(len(tensors), LAYER_TENSORS)
    if spare or layers < 3:
        raise ValueError(f"{path}: {len(tensors)} tensors is no canvas model")
    params, stats = [], []
    for layer in range(layers):
        held = {
            field: jnp.asarray(tensors[str(LAYER_TENSORS * layer + at)])
            for at, field in enumerate(LAYER_FIELDS)
        }
        params.append({field: held[field] for field in PARAM_FIELDS})
        stats.append({field: held[field] for field in STAT_FIELDS})
    return {"layers": params}, stats


def drawn_params(seed, layers, width, norm_scale=1.0):
    """A model of DRAWN weights, at a shape a gate can afford: the gates of the circuit and
    of the twin need a model and not a training run, thus one is drawn here and no gate has
    to read a checkpoint that git ignores.

    The draw follows the SHAPE of the trainer and not its values: He normal kernels --
    sqrt(2 / fan_in) over the reach and the input channels -- the norm scale, the shift at
    zero, and the population at mean 0 and variance 1.

    THE NORM SCALE OPENS AT 1.0 AND NOT AT THE TRAINER'S TENTH. At the tenth an L-layer
    DRAWN trunk decays its activations tenfold at every layer -- a trained norm grows out of
    that opening, an untrained one never leaves it -- and by the third layer a gate over the
    integer twin reads the resolution floor of the activation format instead of the
    arithmetic. At 1.0 the drawn trunk holds the O(1) activations a trained model holds,
    which is the regime the twin and the circuit must answer for."""
    rng = np.random.default_rng(seed)
    params, stats = [], []
    for at in range(layers):
        inputs = 2 * VOICES if at == 0 else width
        outputs = VOICES if at == layers - 1 else width
        deviation = math.sqrt(2.0 / (KERNEL * KERNEL * inputs))
        drawn = rng.normal(0.0, deviation, (KERNEL, KERNEL, inputs, outputs))
        params.append(
            {
                "kernel": drawn.astype(np.float32),
                "scale": np.full(outputs, norm_scale, np.float32),
                "shift": np.zeros(outputs, np.float32),
            }
        )
        stats.append(
            {
                "mean": np.zeros(outputs, np.float32),
                "variance": np.ones(outputs, np.float32),
            }
        )
    return {"layers": params}, stats


def parameter_count(params):
    """the trainable parameters; the population statistics are not among them"""
    return sum(int(np.prod(t.shape)) for t in jax.tree.leaves(params))
