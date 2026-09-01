"""The recipe of the autoregressive eras: the training loop of eras four and five.

Two models under ONE RECIPE. They read the packed stream of `corpus.py`, draw a batch by
one rule -- a uniform stream then a uniform window -- report nats for each step, and keep
the best checkpoint by valid. The trunk is the whole of the difference, and it lives in
each era's `model.py`. Era six is not here on purpose: the sheet folds a population where
these frames draw a dropout mask, thus its own loop in `diffusion/train.py`.

THE LOSS IS REPORTED AS NATS FOR EACH STEP, the sum over the four seats: a per-prediction
mean divides against a different count in each encoding and compares nothing. THE GRADIENT
takes the mean over the predictions instead -- Adam is blind to the scale, but the
global-norm clip is not, and a loss four times larger would make the clip bite four times
harder.

BOTH ERAS ARE FROZEN and this loop is kept, not run. `tests/test_train.py` watches the
loss fall over sixty steps of each era, which is the whole of what holds it.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import ar_model
import cli
import corpus
from train import update_rule


def make_step(dropout):
    """The jitted training step: the loss over the batch, and one update under the
    schedule. [dropout] closes in because it decides the SHAPE of the pass -- whether a
    mask is drawn at all -- and not a value inside it."""

    @nnx.jit
    def step_fn(held, optimizer, classes, phases, key):
        def loss(held):
            return jnp.mean(held.seat_nll(classes, phases, dropout=dropout, key=key))

        value, grads = nnx.value_and_grad(loss)(held)
        optimizer.update(held, grads)
        return value

    return step_fn


@nnx.jit
def eval_fn(held, classes, phases):
    """the evaluation sums of one batch: the per-step loss and the step count. The
    moving-steps instrument lives in measure.py, where elections read it."""
    steps = jnp.sum(held.seat_nll(classes, phases), axis=-1)
    return jnp.sum(steps), jnp.size(steps)


def device_eval_batches(split, context, limit, batch):
    """`corpus.eval_batches` moved to the device, and that is the whole of what it adds:
    the windows are fixed for the run, thus they cross one time and not at every
    evaluation."""
    return [
        (jnp.asarray(classes), jnp.asarray(phases))
        for classes, phases in corpus.eval_batches(split, context, limit, batch)
    ]


def eval_loss(held, batches):
    """nats for each step, the mean over the evaluation windows; the sums stay on the
    device until the loop ends, because a read at every batch blocks the next dispatch"""
    total = jnp.zeros((), jnp.float32)
    steps = 0
    for classes, phases in batches:
        batch_total, batch_steps = eval_fn(held, classes, phases)
        total += batch_total
        steps += batch_steps
    return float(total) / max(int(steps), 1)


def recipe_options(dropout):
    """The fourteen flags of the RECIPE, which every step-frame trainer takes. Each is a
    parameter of `train` and none is a parameter of a model; only the dropout is an era's
    own number, because it scales with the capacity.

    THE FOURTEEN ARE WRITTEN TWICE, here and in the signature of `train`, linked only
    by `**flags`. Every way of welding the two loses one of the two readings, and a
    flag added here and not there raises at the call."""
    options = [
        click.option("--corpus", "corpus_path", default=str(corpus.FRAMES)),
        click.option(
            "--context",
            default=ar_model.TRAINING_WINDOW,
            help="the training window, in steps",
        ),
        click.option("--batch", default=16),
        click.option("--steps", default=96000),
        click.option("--lr", default=1e-3),
        click.option("--seed", default=6),
        click.option("--warmup", default=300),
        click.option("--wd", "weight_decay", default=0.01),
        click.option("--clip", default=1.0),
        click.option("--dropout", default=dropout),
        click.option("--log-every", default=100),
        click.option("--eval-every", default=1600),
        click.option("--eval-limit", default=128),
        click.option("--ckpt", default=None),
    ]
    return cli.add_options(options)


def train(
    held,
    *,
    corpus_path,
    context,
    batch,
    steps,
    lr,
    seed,
    warmup,
    clip,
    weight_decay,
    dropout,
    log_every,
    eval_every,
    eval_limit,
    ckpt,
    note="",
):
    """The loop of the two eras: the batch draw, the step, the two evaluations and the
    best-by-valid checkpoint.

    [held] is the drawn model this run opens on; the caller builds it, because every flag
    of the shape is the model's own. [note] is what an era says about its shape that
    `describe` cannot."""
    splits = corpus.load_corpus(corpus_path)
    pool = corpus.train_pool(splits)
    train_eval = device_eval_batches(splits["train"], context, eval_limit, batch)
    valid_eval = device_eval_batches(splits["valid"], context, eval_limit, batch)
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    optimizer = nnx.Optimizer(
        held,
        update_rule(
            peak=lr, warmup=warmup, total=steps, clip=clip, weight_decay=weight_decay
        ),
        wrt=nnx.Param,
    )
    step_fn = make_step(dropout)
    corpus_steps = sum(int(split.index[row, 1]) for split, row in pool)
    click.echo(
        f"corpus: {len(pool)} pool streams, {corpus_steps} steps; eval rows: "
        f"{sum(len(b[0]) for b in train_eval)} train, "
        f"{sum(len(b[0]) for b in valid_eval)} valid"
    )
    click.echo(
        f"shape: {held.describe()}; context {context}, dropout {dropout}, seed {seed}"
        f"{note}; parameters {held.parameter_count()}"
    )

    best = float("inf")
    losses = []
    started = time.perf_counter()

    def evaluate(step):
        nonlocal best
        train_all = eval_loss(held, train_eval)
        valid_all = eval_loss(held, valid_eval)
        mark = ""
        if valid_all < best:
            best = valid_all
            mark = "  *"
            if ckpt:
                held.save(ckpt)
        click.echo(
            f"step {step:5d}  eval  train {train_all:.4f}  valid {valid_all:.4f}{mark}"
        )

    for step in range(1, steps + 1):
        classes, phases = corpus.train_batch(rng, pool, batch, context)
        key, step_key = jax.random.split(key)
        value = step_fn(
            held, optimizer, jnp.asarray(classes), jnp.asarray(phases), step_key
        )
        # the device array, NOT float(value): a read blocks until the step finishes and
        # the loop cannot then overlap the next batch draw with this compute
        losses.append(value)
        if step % log_every == 0 or step == 1:
            # nats for each step here too: the mean over the predictions times the seats
            mean = float(jnp.mean(jnp.stack(losses)))
            click.echo(f"step {step:5d}  loss {corpus.SEATS * mean:.4f}")
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

