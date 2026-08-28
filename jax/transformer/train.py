"""The trainer of the step-frame model of docs/transformer.md.

Run it from the jax directory as a module:

    uv run python -m transformer.train --steps 200

The update rule is `nn.update_rule` -- optax's AdamW with a decoupled decay and a
global-norm clip, under the warmup and cosine decay of `nn.learning_rates`. THE SCHEDULE IS
INSIDE THE OPTIMIZER and not in this loop.

The loss is reported as NATS FOR EACH STEP -- the sum over the four seats -- because a
per-prediction mean divides against a different count in each encoding and compares
nothing. The moving-steps loss -- the drone detector, and the number elections read --
is measure.py's instrument and not this log's.

The gradient takes the mean over the predictions and not the sum over the seats. Adam is
blind to the scale, but the global-norm clip is not: a loss four times larger would make
the clip bite four times harder, and the peak rate and the clip of the recipe would stop
meaning what they meant.

THE ERA IS FROZEN and this trainer is kept, not run: the elected checkpoint stands and no
retrain is planned. `tests/test_train.py` runs it over sixty steps and watches the loss
fall, which is the whole of what holds it.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import data
import nn
from transformer import model

JAX_ROOT = nn.JAX_ROOT


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
    """nats for each step, the mean over the evaluation windows"""
    total = 0.0
    steps = 0
    for classes, phases in batches:
        sums = eval_fn(held, classes, phases)
        total += float(sums[0])
        steps += int(sums[1])
    return total / max(steps, 1)


def train(
    held,
    *,
    corpus_path,
    train_on,
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
    average_top,
):
    """The loop of the era: the batch draw, the step, the two evaluations, the
    best-by-valid checkpoint and the top-K average. [held] is the drawn model this run
    opens on: the CLI builds it, because every flag of the shape is the model's own."""
    corpus = data.load_corpus(corpus_path)
    pool = data.train_pool(corpus, train_on)
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
        f"shape: {held.describe()}; context {context}, dropout {dropout}, seed {seed}; "
        f"parameters {held.parameter_count()}"
    )

    best = float("inf")
    top = []  # (valid, step, the flat host tensors) -- the K best snapshots for averaging
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
            if ckpt and train_on != "all":
                held.save(ckpt)
        # the snapshot crosses to the host only when it can stay: the sort would drop it
        # again, and the copy is the whole model
        if average_top > 0 and (len(top) < average_top or valid_all < top[-1][0]):
            top.append(
                (valid_all, step, [np.asarray(t) for t in held.every_tensor()])
            )
            top.sort(key=lambda entry: entry[0])
            del top[average_top:]
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
        if train_on == "all":
            held.save(ckpt)
            click.echo(f"checkpoint of the last step: {ckpt}")
        else:
            click.echo(f"checkpoint of the best: {ckpt}")
        if average_top > 0 and top:
            averaged = [
                np.mean(np.stack(tensors), axis=0)
                for tensors in zip(*[entry[2] for entry in top])
            ]
            path = ckpt.replace(".ckpt", "-avg.ckpt")
            nn.save_checkpoint(path, averaged)
            click.echo(
                f"average of {len(top)} best snapshots "
                f"(steps {[entry[1] for entry in top]}): {path}"
            )


@click.command(help=__doc__)
@click.option(
    "--corpus", "corpus_path", default=str(JAX_ROOT / "_data" / "frames.safetensors")
)
@click.option("--d", default=64)
@click.option("--layers", default=6)
@click.option("--heads", default=4)
@click.option("--context", default=256, help="the window, in steps")
@click.option("--batch", default=16)
@click.option("--steps", default=96000)
@click.option("--lr", default=1e-3)
@click.option("--seed", default=6)
@click.option("--warmup", default=300)
@click.option("--wd", "weight_decay", default=0.01)
@click.option("--clip", default=1.0)
@click.option("--dropout", default=0.3)
@click.option(
    "--alibi-span",
    "span",
    default=nn.SLOPE_SPAN,
    help="the ALiBi exponent span: the slope of head k is 2^-(span (k+1) / heads). The "
    "draw must state the same.",
)
@click.option(
    "--train-on", type=click.Choice(("train", "train+test", "all")), default="train"
)
@click.option("--log-every", default=100)
@click.option("--eval-every", default=1600)
@click.option("--eval-limit", default=128)
@click.option("--ckpt", default=None)
@click.option(
    "--average-top",
    default=0,
    help="also write the mean of the K best-by-valid snapshots as NAME-avg.ckpt",
)
def main(d, layers, heads, span, seed, **flags):
    train(
        model.Transformer.drawn(seed, d, layers, heads=heads, span=span),
        seed=seed,
        **flags,
    )


if __name__ == "__main__":
    main()
