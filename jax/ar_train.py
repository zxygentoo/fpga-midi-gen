"""The recipe of the autoregressive eras: the training loop of eras four and five.

Era four and era five are two models under ONE RECIPE. They read one corpus, the packed
stream of `data.py`; they draw a batch by one rule, a uniform stream then a uniform
window; they report one number, nats for each step; and they keep one checkpoint, the
best by valid. The trunk is the whole of the difference between them, and the trunk lives
in each era's `model.py`.

Era six is not here on purpose. The sheet folds a population where these frames draw a
dropout mask, and it probes a masked crop where they evaluate two fixed splits: a
different recipe, thus its own loop in `diffusion/train.py`.

The update rule is `nn.update_rule` -- optax's AdamW with a decoupled decay and a
global-norm clip, under the warmup and cosine decay of `nn.learning_rates`. THE SCHEDULE
IS INSIDE THE OPTIMIZER and not in this loop.

The loss is reported as NATS FOR EACH STEP -- the sum over the four seats -- because a
per-prediction mean divides against a different count in each encoding and compares
nothing. The two eras share one encoding and one window rule, thus this number compares
across them and the elected model of era four stands at 1.6282. The moving-steps loss --
the drone detector, and the number the elections read -- is measure.py's instrument and
not this log's.

The gradient takes the mean over the predictions and not the sum over the seats. Adam is
blind to the scale, but the global-norm clip is not: a loss four times larger would make
the clip bite four times harder, and the peak rate and the clip of the recipe would stop
meaning what they meant.

BOTH ERAS ARE FROZEN and this loop is kept, not run: the elected checkpoints stand and no
retrain is planned. `tests/test_train.py` runs it over sixty steps of each era and
watches the loss fall, which is the whole of what holds it.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import data
import nn


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


def eval_batches(split, context, limit, batch):
    """The evaluation windows of one split, on the device.

    They are fixed for the whole run, thus they cross to the device one time and not at
    every evaluation."""
    return [
        (jnp.asarray(classes), jnp.asarray(phases))
        for classes, phases in data.eval_batches(split, context, limit, batch)
    ]


def eval_loss(held, batches):
    """nats for each step, the mean over the evaluation windows.

    The sums stay on the device until the loop ends, because a read at every batch blocks
    the dispatch of the next one."""
    total = jnp.zeros((), jnp.float32)
    steps = 0
    for classes, phases in batches:
        batch_total, batch_steps = eval_fn(held, classes, phases)
        total += batch_total
        steps += batch_steps
    return float(total) / max(int(steps), 1)


def recipe_options(dropout):
    """The fourteen flags of the RECIPE, which every step-frame trainer wears.

    They are one set because the loop is one loop: each is a parameter of `train` and none
    of them is a parameter of a model, thus a change to the recipe's defaults has one home
    and the two eras cannot drift apart on it. Only the dropout is an era's own number,
    because it scales with the capacity the era carries. A trainer states its MODEL's
    flags itself, and nothing else."""
    options = [
        click.option("--corpus", "corpus_path", default=str(data.FRAMES)),
        click.option("--context", default=256, help="the training window, in steps"),
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

    def worn(command):
        # click reads a stack of decorators from the bottom up, thus the reverse keeps
        # --help in the order written above
        for option in reversed(options):
            command = option(command)
        return command

    return worn


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

    [held] is the drawn model this run opens on: the caller builds it, because every flag
    of the shape is the model's own and no two eras spell the same shape. [note] is what
    an era says about its own shape that `describe` cannot -- a flag of the RECIPE and not
    of the model -- and era four has nothing to say."""
    corpus = data.load_corpus(corpus_path)
    pool = data.train_pool(corpus)
    train_eval = eval_batches(corpus["train"], context, eval_limit, batch)
    valid_eval = eval_batches(corpus["valid"], context, eval_limit, batch)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    optimizer = nnx.Optimizer(
        held,
        nn.update_rule(
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
        classes, phases = data.train_batch(rng, pool, batch, context)
        key, step_key = jax.random.split(key)
        value = step_fn(
            held, optimizer, jnp.asarray(classes), jnp.asarray(phases), step_key
        )
        # the device array, NOT float(value): a read blocks until the step finishes, and
        # the loop then cannot overlap the next batch draw and its transfer with the
        # compute of this one. The log below reads, thus the run-ahead stays inside one
        # log window.
        losses.append(value)
        if step % log_every == 0 or step == 1:
            # the training number is nats for each step too: the mean over the predictions
            # times the four seats
            mean = float(jnp.mean(jnp.stack(losses)))
            click.echo(f"step {step:5d}  loss {data.SEATS * mean:.4f}")
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

