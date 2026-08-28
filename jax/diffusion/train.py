"""The trainer of the masked sheet of docs/diffusion.md.

Run it from the jax directory as a module:

    uv run python -m diffusion.train --steps 200 --ckpt ../_train/diffusion/probe.ckpt

One row of a batch is one crop of 128 sixteenth steps, taken uniformly inside one uniformly
drawn chorale. A piece shorter than the crop is dropped, thus the round trains on 228 of
the 229 train chorales, and a crop never reads the padded tail of the export: silence
inside a crop is the real rests of the music, 0.35 percent of the cells.

The loss is orderless NADE, the paper's: mask a uniform subset of the cells, take the
negative log-likelihood of the masked cells under the softmax over the pitch rows, and
scale by one over the masked count. THE NUMBER IS NATS FOR EACH MASKED CELL. It is not the
paper's Table 1 figure, which is nats for each FRAME under Algorithm 1 with five orderings;
that referee lives in diffusion/measure.py and it is the only thing that compares with
0.57. Nor does this number compare with the loss of any earlier era of this project.

No transposition augmentation: the paper states none, and the pitch axis of the trunk
carries the equivariance that the shifts used to buy. The elected checkpoint is the best
valid loss -- 228 pieces against nine million parameters memorize, and the valid curve is
the only guard the round has.

THE OPTIMIZER IS PLAIN ADAM, by arithmetic and not by a second code path: --wd is 0 by
default, and AdamW with a weight decay of zero IS Adam. That is the paper's — the code
release calls `tf.train.AdamOptimizer` with no weight decay, no dropout and no L2 anywhere,
thus batch norm and the best-by-valid checkpoint are the whole of its regularisation.

THE UPDATE RULE AND THE RATE CURVE ARE `nn.update_rule` AND `nn.learning_rates`, which
every era reads; `test_train.py` holds the curve against its closed form.

THE RATE MOVES WITH THE RUNG, and the release carries no flag for it. Measured 2026-08-24
under the warmup and cosine decay of `nn.learning_rates`, the board rung wants 1.6e-2 and
the ceiling 3e-3. The default is the ceiling's, because every other default states the paper's
shape; a rung passes its own, as docs/diffusion.md records them.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import data
import nn
from diffusion import model

JAX_ROOT = nn.JAX_ROOT

# The batch norm of the code release: `popmean -= 0.01 * (popmean - batchmean)`, thus the
# population keeps 0.99 of itself at every step.
POP_DECAY = 0.99
# The probes are the same rows for every run, thus two seeds and two shapes compare. It is
# not the training seed and it never moves.
PROBE_SEED = 0


def population_decay(t):
    """The share of itself the batch-norm population keeps at step [t].

    The population opens at mean 0 and variance 1. At the code release's flat 0.99 it needs
    some hundreds of steps to reach the batch statistics, and every valid number before that
    reads the prior and not the model: at step 250 of a 1,500-step probe, valid stood at
    log 48 while the training loss had already fallen to 3.33. The warmed decay makes the
    early population the running mean of every batch so far, and settles onto the release's
    rate at step 890. IT IS WHY THE NORM IS NOT `nnx.BatchNorm`.

    Training never reads it: a training pass normalises by the batch's own statistics, thus
    this moves the evaluation and the checkpoint and never the gradient."""
    return jnp.minimum(POP_DECAY, (1.0 + t) / (10.0 + t))


def masked_nll(said, classes, hidden):
    """The orderless NADE loss of one batch: the negative log-likelihood of the masked cells,
    over the masked count, meaned over the sheets.

    The divisor is the paper's one over |not-C| and it is PER SHEET: every sheet drew its own
    mask size, and one with three cells hidden must not weigh a hundredth of one with three
    hundred. The count is never zero, thus nothing guards the division."""
    nll = model.nll_of_logits(said, classes)
    masked = hidden.astype(jnp.float32)
    return jnp.mean(jnp.sum(nll * masked, axis=(1, 2)) / jnp.sum(masked, axis=(1, 2)))


def fold_population(coconet, seen, decay):
    """The batch statistics of a training pass, folded into the populations of the norms that
    read them, in the layer order. IT STANDS OUTSIDE THE GRADIENT: a training pass reads no
    population at all, and the populations are `nnx.BatchStat` in any case."""
    for layer, statistics in zip(coconet.every_layer(), seen):
        layer.norm.fold(statistics, decay)


def make_step(remat):
    """The jitted training step: the mask draw, the loss, one AdamW update, and the fold of the
    batch statistics into the population. The mask is drawn INSIDE the step — it reads no
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
    """The same loss on a fixed probe, under the POPULATION statistics.

    The probe reads the model the sampler and the referees will read, and not the model
    that a batch of sixteen crops happens to normalize."""
    said, _ = coconet(model.planes(classes, hidden))
    return masked_nll(said, classes, hidden)


def probe_batches(crops, batch):
    """The fixed rows of the valid curve: one crop of every valid piece and one orderless
    mask for each, drawn one time and never drawn again.

    A training batch draws a fresh mask at every step, and a sheet's loss depends heavily on
    how much of it is hidden, thus two steps of the training number hardly compare. The
    probes hold the mask still, and what moves in the number is the model.

    The probe mean and the training mean do not compare WITH EACH OTHER."""
    classes = crops.every_piece(PROBE_SEED)
    hidden = model.orderless_masks(
        jax.random.PRNGKey(PROBE_SEED), len(classes), crops.length
    )
    return [
        (jnp.asarray(classes[at : at + batch]), hidden[at : at + batch])
        for at in range(0, len(classes), batch)
    ]


def eval_loss(coconet, batches):
    """the mean over the probe sheets; the last batch is short, thus the mean is a sum over a
    count and never a mean of means. The sums stay on the device until the loop ends,
    because a read at every batch blocks the dispatch of the next one."""
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
    best-by-valid checkpoint. THE SCHEDULE IS INSIDE THE OPTIMIZER and not in this loop."""
    pieces = data.load_pieces(corpus_path)
    crops = data.Crops(pieces["train"], crop)
    probe = probe_batches(data.Crops(pieces["valid"], crop), batch)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    coconet = model.Coconet(layers, width, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(
        coconet,
        nn.update_rule(
            peak=lr, warmup=warmup, total=steps, clip=clip, weight_decay=weight_decay
        ),
        wrt=nnx.Param,
    )
    step_fn = make_step(remat)
    click.echo(
        f"corpus: {len(crops.rows)} train pieces of {len(pieces['train'].lengths)} hold a "
        f"crop of {crop}; probes {sum(len(c) for c, _ in probe)} valid sheets"
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
        # the device array and NOT float(value): a read blocks until the step finishes, and
        # the loop then cannot overlap the next crop draw with the compute of this one
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
@click.option(
    "--corpus", "corpus_path", default=str(JAX_ROOT / "_data" / "pieces.safetensors")
)
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
