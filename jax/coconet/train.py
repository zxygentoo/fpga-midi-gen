"""The trainer of the masked canvas of docs/coconet.md.

Run it from the jax directory as a module:

    uv run python -m coconet.train --steps 200 --ckpt ../_train/coconet/probe.ckpt

One row of a batch is one crop of 128 sixteenth steps, taken uniformly inside one uniformly
drawn chorale. A piece shorter than the crop is dropped, thus the round trains on 228 of
the 229 train chorales, and a crop never reads the padded tail of the export: silence
inside a crop is the real rests of the music, 0.35 percent of the cells.

The loss is orderless NADE, the paper's: mask a uniform subset of the cells, take the
negative log-likelihood of the masked cells under the softmax over the pitch rows, and
scale by one over the masked count. THE NUMBER IS NATS FOR EACH MASKED CELL. It is not the
paper's Table 1 figure, which is nats for each FRAME under Algorithm 1 with five orderings;
that referee lives in referee.py and it is the only thing that compares with 0.57. Nor does
this number compare with the loss of any earlier era of this project.

No transposition augmentation: the paper states none, and the pitch axis of the trunk
carries the equivariance that the shifts used to buy. The elected checkpoint is the best
valid loss -- 228 pieces against nine million parameters memorize, and the valid curve is
the only guard the round has.

THE OPTIMIZER IS PLAIN ADAM, and it is plain Adam by arithmetic and not by a second code
path: --wd is 0 by default, and [nn.adamw] with a weight decay of zero IS Adam. That is the
paper's -- it is ISMIR 2017, where AdamW is arXiv 1711.05101 of November 2017 and ICLR 2019,
and the code release calls `tf.train.AdamOptimizer` with no weight decay, no dropout and no
L2 anywhere. Batch norm and the best-by-valid checkpoint are the whole of its regularisation.

THE LEARNING RATE IS NOT THE PAPER'S. The release carries no flag for it: `lib_hparams`
holds 2**-4 marked "for sigmoids", with 2**-6 commented out above it, and halves it on a
plateau of five epochs. Neither number is stated in the paper and both are large for Adam.
The default here is a modern guess with the warmup and cosine decay of nn.schedule, and it
is the first thing a sweep should settle.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np

import data
import nn
from coconet import model

JAX_ROOT = nn.JAX_ROOT

# The batch norm of the code release: `popmean -= 0.01 * (popmean - batchmean)`, thus the
# population keeps 0.99 of itself at every step.
POP_DECAY = 0.99
# its gamma initializer. A norm that opens at a tenth keeps the residual branch small, and
# a trunk of 64 layers then trains from a draw.
NORM_SCALE = 0.1
# The probes are the same rows for every run, thus two seeds and two shapes compare. It is
# not the training seed and it never moves.
PROBE_SEED = 0


def draw_params(key, layers, width):
    """The parameter tree and the population statistics at step zero, in the order the
    network runs.

    A convolution takes the He normal draw of the code release -- standard deviation
    sqrt(2 / fan_in) over the reach and the input channels -- and carries no bias, because
    the norm behind it carries the shift.

    The statistics open at mean 0 and variance 1. Nothing reads them until the trainer has
    filled them: a training pass reads the batch's own."""
    channels = (
        [(2 * model.VOICES, width)]
        + [(width, width)] * (layers - 2)
        + [(width, model.VOICES)]
    )
    keys = jax.random.split(key, len(channels))

    def layer(key, inputs, outputs):
        shape = (model.KERNEL, model.KERNEL, inputs, outputs)
        deviation = np.sqrt(2.0 / (model.KERNEL * model.KERNEL * inputs))
        return {
            "kernel": jax.random.normal(key, shape, dtype=jnp.float32) * deviation,
            "scale": jnp.full(outputs, NORM_SCALE, jnp.float32),
            "shift": jnp.zeros(outputs, jnp.float32),
        }

    def opening(outputs):
        return {"mean": jnp.zeros(outputs, jnp.float32), "variance": jnp.ones(outputs, jnp.float32)}

    params = [layer(key, *shape) for key, shape in zip(keys, channels)]
    return {"layers": params}, [opening(outputs) for _, outputs in channels]


def population_decay(t):
    """The share of itself the batch-norm population keeps at step [t].

    The population opens at mean 0 and variance 1. At the code release's flat 0.99 it needs
    some hundreds of steps to reach the batch statistics, and every valid number before
    that reads the prior and not the model: measured at step 250 of a 1,500-step probe,
    valid stood at log 48 while the training loss had already fallen to 3.33. The warmed
    decay of the standard recipe -- (1 + t) / (10 + t) until it passes the decay -- makes
    the early population the running mean of every batch so far, and it settles onto the
    release's rate at step 890.

    Training never reads it. A training pass normalises by the batch's own statistics, thus
    this moves the evaluation and the checkpoint and never the gradient."""
    return jnp.minimum(POP_DECAY, (1.0 + t) / (10.0 + t))


def save_checkpoint(path, params, stats):
    """the flat list of model.py, which nn.save_checkpoint names "0" upward"""
    nn.save_checkpoint(path, model.flat_tensors(params, stats))


def masked_nll(said, classes, hidden):
    """The orderless NADE loss of one batch: the negative log-likelihood of the masked
    cells, over the masked count, meaned over the canvases.

    The divisor is the paper's one over |not-C| and it is per canvas, not per batch: every
    canvas of the batch drew its own mask size, and a canvas with three cells hidden must
    not weigh a hundredth of one with three hundred.

    The count is never zero -- the draw masks at least one cell -- thus nothing guards the
    division."""
    nll = model.nll_of_logits(said, classes)
    masked = hidden.astype(jnp.float32)
    return jnp.mean(jnp.sum(nll * masked, axis=(1, 2)) / jnp.sum(masked, axis=(1, 2)))


def make_step(clip, weight_decay, remat):
    """The jitted training step: the mask draw, the loss, one AdamW update, and the fold of
    the batch statistics into the population.

    The mask is drawn inside the step. It is fresh at every step, it reads no corpus, and
    the device already holds the key."""

    def loss(params, stats, classes, key):
        hidden = model.orderless_masks(key, *classes.shape[:2])
        canvas = model.planes(classes, hidden)
        said, seen = model.logits(params, stats, canvas, training=True, remat=remat)
        return masked_nll(said, classes, hidden), seen

    def step_fn(params, stats, state, t, classes, lr, key):
        gradient = jax.value_and_grad(loss, has_aux=True)
        (value, seen), grads = gradient(params, stats, classes, key)
        params, state = nn.adamw(
            state, params, grads, t, lr, clip=clip, weight_decay=weight_decay
        )
        decay = population_decay(t)
        stats = jax.tree.map(
            lambda held, drawn: decay * held + (1.0 - decay) * drawn, stats, seen
        )
        return value, params, stats, state

    return jax.jit(step_fn)


def make_eval():
    """The same loss on a fixed probe, under the POPULATION statistics.

    The probe reads the model the sampler and the referees will read, and not the model
    that a batch of sixteen crops happens to normalize."""

    def eval_fn(params, stats, classes, hidden):
        canvas = model.planes(classes, hidden)
        said, _ = model.logits(params, stats, canvas)
        return masked_nll(said, classes, hidden)

    return jax.jit(eval_fn)


def probe_batches(crops, batch):
    """The fixed rows of the valid curve: one crop of every valid piece and one orderless
    mask for each, drawn one time and never drawn again.

    A training batch draws a fresh mask at every step, and the loss of a canvas depends
    heavily on how much of it is hidden -- a canvas with one cell masked is nearly free and
    one with all of them is nearly the prior. Two steps of the training number hardly
    compare. The probes hold the mask still, and what moves in the number is the model.

    The probe mean and the training mean do not compare WITH EACH OTHER. Read each against
    itself over the run."""
    classes = crops.every_piece(PROBE_SEED)
    hidden = model.orderless_masks(
        jax.random.PRNGKey(PROBE_SEED), len(classes), crops.length
    )
    return [
        (jnp.asarray(classes[at : at + batch]), hidden[at : at + batch])
        for at in range(0, len(classes), batch)
    ]


def eval_loss(eval_fn, params, stats, batches):
    """the mean over the probe canvases; the last batch is short, thus the mean is a sum
    over a count and never a mean of means"""
    total = 0.0
    canvases = 0
    for classes, hidden in batches:
        total += float(eval_fn(params, stats, classes, hidden)) * len(classes)
        canvases += len(classes)
    return total / max(canvases, 1)


def train(
    *,
    corpus_path,
    crop,
    layers,
    width,
    batch,
    steps,
    lr,
    seed,
    warmup,
    clip,
    weight_decay,
    remat,
    log_every,
    eval_every,
    ckpt,
):
    """The loop of the round: the crop draw, the schedule, the step, the fixed valid probes
    and the best-by-valid checkpoint."""
    pieces = data.load_pieces(corpus_path)
    crops = data.Crops(pieces["train"], crop)
    probe = probe_batches(data.Crops(pieces["valid"], crop), batch)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, draw_key = jax.random.split(key)
    params, stats = draw_params(draw_key, layers, width)
    state = nn.optimizer_init(params)
    step_fn = make_step(clip, weight_decay, remat)
    eval_fn = make_eval()
    click.echo(
        f"corpus: {len(crops.rows)} train pieces of {len(pieces['train'].lengths)} hold a "
        f"crop of {crop}; probes {sum(len(c) for c, _ in probe)} valid canvases"
    )
    click.echo(
        f"shape: {layers} layers, {width} channels, kernel {model.KERNEL}, "
        f"{model.ROWS} rows, D {model.cells(crop)}; parameters "
        f"{model.parameter_count(params)}; batch {batch}, seed {seed}, "
        f"remat {'on' if remat else 'off'}"
    )

    best = float("inf")
    losses = []
    started = time.perf_counter()

    def evaluate(step, params, stats):
        nonlocal best
        valid = eval_loss(eval_fn, params, stats, probe)
        mark = ""
        if valid < best:
            best = valid
            mark = "  *"
            if ckpt:
                save_checkpoint(ckpt, params, stats)
        click.echo(f"step {step:5d}  eval  valid {valid:.4f}{mark}")

    for step in range(1, steps + 1):
        classes = crops.batch(rng, batch)
        # a name of its own: [lr] is the peak the schedule reads, and a loop that writes
        # its own peak decays the rate geometrically to zero and trains nothing
        rate = nn.schedule(step, lr, warmup, steps)
        key, step_key = jax.random.split(key)
        value, params, stats, state = step_fn(
            params,
            stats,
            state,
            jnp.float32(step),
            jnp.asarray(classes),
            jnp.float32(rate),
            step_key,
        )
        # the device array and NOT float(value): a read blocks until the step finishes, and
        # the loop then cannot overlap the next crop draw with the compute of this one
        losses.append(value)
        if step % log_every == 0 or step == 1:
            click.echo(f"step {step:5d}  loss {float(jnp.mean(jnp.stack(losses))):.4f}")
            losses = []
        if step % eval_every == 0 or step == steps:
            evaluate(step, params, stats)

    seconds = time.perf_counter() - started
    click.echo(
        f"time: {seconds:.0f} s, {seconds / steps * 1000:.0f} ms each step, "
        f"the evaluations inside"
    )
    click.echo(f"best valid {best:.4f}")
    if ckpt:
        click.echo(f"checkpoint of the best: {ckpt}")


@click.command(help=__doc__)
@click.option(
    "--corpus", "corpus_path", default=str(JAX_ROOT / "_data" / "pieces.safetensors")
)
@click.option("--crop", default=model.CROP, help="T, the steps of one canvas")
@click.option("--layers", default=model.LAYERS, help="L, the paper's 64")
@click.option("--width", default=model.WIDTH, help="H, the paper's 128 channels")
@click.option("--batch", default=8)
@click.option("--steps", default=30000)
@click.option("--lr", default=1e-3)
@click.option("--seed", default=6)
@click.option("--warmup", default=1000)
@click.option("--clip", default=1.0)
@click.option("--wd", "weight_decay", default=0.0)
# MEASURED 2026-08-24 on the RTX 3060, at the paper's shape: remat costs 28 percent of the
# step (batch 8, 327 ms against 432) and buys the memory that a batch of 16 needs -- 16
# without it exhausts a 12 GB card. Batch 8 fits either way, thus the default is off.
@click.option(
    "--remat/--no-remat",
    default=False,
    help="rematerialise each residual pair: 28 percent slower, and what a batch of 16 needs",
)
@click.option("--log-every", default=100)
@click.option("--eval-every", default=1000)
@click.option("--ckpt", default=None)
def main(**flags):
    train(**flags)


if __name__ == "__main__":
    main()
