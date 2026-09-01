"""The trainer of the masked sheet of docs/diffusion.md.

Run it from the jax directory as a module:

    uv run python -m diffusion.train --steps 200 --ckpt ../_train/diffusion/probe.ckpt

One row of a batch is one crop of 128 sixteenth steps, taken uniformly inside one
uniformly drawn chorale. A piece shorter than the crop is dropped, thus the round trains
on 228 of the 229 train chorales, and a crop never reads the padded tail of the export:
silence inside a crop is the real rests of the music, 0.35 percent of the cells.

The loss is orderless NADE, the paper's: mask a uniform subset of the cells, take the
negative log-likelihood of the masked cells, and scale by one over the masked count. THE
NUMBER IS NATS FOR EACH MASKED CELL. It is not the paper's Table 1 figure, which is nats
for each FRAME under Algorithm 1; that referee is diffusion/measure.py, and it is the only
thing that compares with 0.57.

No transposition augmentation: the paper states none, and the pitch axis of the trunk
carries the equivariance. The elected checkpoint is the best valid loss -- 228 pieces
against nine million parameters memorize. THE OPTIMIZER IS PLAIN ADAM, by arithmetic and
not by a second code path: --wd is 0, and AdamW with a weight decay of zero IS Adam. THE
RATE MOVES WITH THE RUNG and the release carries no flag for it -- the board rung wants
1.6e-2 where the ceiling wants the default 3e-3, and docs/diffusion.md records each.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import corpus
from diffusion import model
from train import update_rule

# the batch norm of the code release: the population keeps 0.99 of itself at every step
POP_DECAY = 0.99
# the probes are the same rows for every run, thus two seeds and two shapes compare; it is
# not the training seed and it never moves
PROBE_SEED = 0


def population_decay(t):
    """The share of itself the batch-norm population keeps at step [t].

    The population opens at mean 0 and variance 1, and at the release's flat 0.99 it
    needs hundreds of steps to reach the batch statistics -- valid stood at log 48 at
    step 250 of a probe whose training loss had reached 3.33. The WARMED decay makes
    the early population the running mean of every batch so far, and settles onto 0.99
    at step 890. It is why the norm is not `nnx.BatchNorm`, and training never reads
    it."""
    return jnp.minimum(POP_DECAY, (1.0 + t) / (10.0 + t))


def masked_nll(said, classes, hidden):
    """The orderless NADE loss of one batch, meaned over the sheets. The divisor is the
    paper's one over |not-C| and it is PER SHEET: every sheet drew its own mask size, and
    one with three cells hidden must not weigh a hundredth of one with three hundred."""
    nll = model.nll_of_logits(said, classes)
    masked = hidden.astype(jnp.float32)
    return jnp.mean(jnp.sum(nll * masked, axis=(1, 2)) / jnp.sum(masked, axis=(1, 2)))


def fold_population(coconet, seen, decay):
    """the batch statistics of a training pass, folded into the populations in the layer
    order; IT STANDS OUTSIDE THE GRADIENT, because a training pass reads no population"""
    for layer, statistics in zip(coconet.layers(), seen):
        layer.norm.fold(statistics, decay)


def make_step(remat):
    """The jitted training step: the mask draw, the loss, one AdamW update, and the
    fold of the batch statistics. The mask is drawn INSIDE the step -- it reads no
    corpus, and the device already holds the key."""

    @nnx.jit
    def step_fn(coconet, optimizer, t, classes, key):
        def loss(coconet):
            hidden = model.orderless_masks(key, *classes.shape[:2])
            sheet = model.planes(classes, hidden)
            said, seen = coconet(sheet, training=True, remat=remat)
            return masked_nll(said, classes, hidden), seen

        (value, seen), grads = nnx.value_and_grad(loss, has_aux=True)(coconet)
        optimizer.update(coconet, grads)
        fold_population(coconet, seen, population_decay(t))
        return value

    return step_fn


@nnx.jit
def eval_fn(coconet, classes, hidden):
    """the same loss on a fixed probe, under the POPULATION statistics: the probe reads
    the model the sampler will read, and not the model one batch happens to normalize"""
    said, _ = coconet(model.planes(classes, hidden))
    return masked_nll(said, classes, hidden)


def probe_batches(crops, batch):
    """The fixed rows of the valid curve: one crop of every valid piece and one
    orderless mask for each, drawn ONE time. A sheet's loss depends heavily on how much
    of it is hidden, thus a training number that redraws its mask hardly compares with
    itself; these hold the mask still. The probe mean and the training mean do not
    compare."""
    classes = crops.every_piece(PROBE_SEED)
    hidden = model.orderless_masks(
        jax.random.key(PROBE_SEED), len(classes), crops.length
    )
    return [
        (jnp.asarray(classes[at : at + batch]), hidden[at : at + batch])
        for at in range(0, len(classes), batch)
    ]


def eval_loss(coconet, batches):
    """the mean over the probe sheets, a sum over a count and never a mean of means;
    the sums stay on the device until the loop ends, because a read blocks the next
    dispatch"""
    total = 0.0
    sheets = 0
    for classes, hidden in batches:
        total = total + eval_fn(coconet, classes, hidden) * len(classes)
        sheets += len(classes)
    return float(total) / max(sheets, 1)


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
    """The loop of the round: the crop draw, the step, the fixed valid probes and the
    best-by-valid checkpoint. THE SCHEDULE IS INSIDE THE OPTIMIZER and not in this loop.
    """
    pieces = corpus.load_pieces(corpus_path)
    crops = corpus.Crops(pieces["train"], crop)
    probe = probe_batches(corpus.Crops(pieces["valid"], crop), batch)
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    coconet = model.Coconet(layers, width, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(
        coconet,
        update_rule(
            peak=lr, warmup=warmup, total=steps, clip=clip, weight_decay=weight_decay
        ),
        wrt=nnx.Param,
    )
    step_fn = make_step(remat)
    click.echo(
        f"corpus: {len(crops.rows)} train pieces of "
        f"{len(pieces['train'].lengths)} hold a crop of {crop}; "
        f"probes {sum(len(c) for c, _ in probe)} valid sheets"
    )
    click.echo(
        f"shape: {layers} layers, {width} channels, kernel {model.KERNEL}, "
        f"{model.ROWS} rows, D {model.cells(crop)}; parameters "
        f"{coconet.parameter_count()}; batch {batch}, seed {seed}, "
        f"remat {'on' if remat else 'off'}"
    )

    best = float("inf")
    losses = []
    started = time.perf_counter()

    def evaluate(step):
        nonlocal best
        valid = eval_loss(coconet, probe)
        mark = ""
        if valid < best:
            best = valid
            mark = "  *"
            if ckpt:
                coconet.save(ckpt)
        click.echo(f"step {step:5d}  eval  valid {valid:.4f}{mark}")

    for step in range(1, steps + 1):
        classes = crops.batch(rng, batch)
        key, step_key = jax.random.split(key)
        value = step_fn(
            coconet, optimizer, jnp.float32(step), jnp.asarray(classes), step_key
        )
        # the device array and NOT float(value): a read blocks until the step finishes,
        # and the loop then cannot overlap the next crop draw with the compute of this one
        losses.append(value)
        if step % log_every == 0 or step == 1:
            click.echo(f"step {step:5d}  loss {float(jnp.mean(jnp.stack(losses))):.4f}")
            losses = []
        if step % eval_every == 0 or step == steps:
            evaluate(step)

    seconds = time.perf_counter() - started
    click.echo(
        f"time: {seconds:.0f} s, {seconds / steps * 1000:.0f} ms each step, "
        f"the evaluations inside"
    )
    click.echo(f"best valid {best:.4f}")
    if ckpt:
        click.echo(f"checkpoint of the best: {ckpt}")


@click.command(help=__doc__)
@click.option("--corpus", "corpus_path", default=str(corpus.PIECES))
@click.option("--crop", default=model.CROP, help="T, the steps of one sheet")
@click.option("--layers", default=model.LAYERS, help="L, the paper's 64")
@click.option("--width", default=model.WIDTH, help="H, the paper's 128 channels")
@click.option("--batch", default=8)
@click.option("--steps", default=30000)
# the ceiling's measured rate, as the shape defaults are the ceiling's; a rung's own rate
# is in docs/diffusion.md
@click.option("--lr", default=3e-3)
@click.option("--seed", default=6)
@click.option("--warmup", default=1000)
@click.option("--clip", default=1.0)
@click.option("--wd", "weight_decay", default=0.0)
# measured at the paper's shape: remat costs 28 percent of the step and buys the memory a
# batch of 16 needs; batch 8 fits either way, thus the default is off
@click.option(
    "--remat/--no-remat",
    default=False,
    help="rematerialise each residual pair: 28 percent slower, and what a batch of 16 "
    "needs",
)
@click.option("--log-every", default=100)
@click.option("--eval-every", default=1000)
@click.option("--ckpt", default=None)
def main(**flags):
    train(**flags)


if __name__ == "__main__":
    main()
