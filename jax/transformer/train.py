"""The trainer of the step-frame model of docs/transformer_model.md.

Run it from the jax directory as a module:

    uv run python -m transformer.train --steps 200

The optimizer is a hand-rolled AdamW with a decoupled decay and a global-norm clip.

Two numbers come out of every evaluation, and the traps of the design document are the
reason for both. The loss is reported as NATS FOR EACH STEP -- the sum over the four seats
-- because a per-prediction mean divides against a different count in each encoding and
compares nothing. And a second number covers the steps where two or more voices move: 77.91
percent of the voice slots repeat the step before, they dominate the mean, and a model that
holds its chord for ever would score well on the mean and play a drone.

The gradient takes the mean over the predictions and not the sum over the seats. Adam is
blind to the scale, but the global-norm clip is not: a loss four times larger would make
the clip bite four times harder, and the peak rate and the clip of the recipe would stop
meaning what they meant.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np

import data
import nn
from transformer import model

JAX_ROOT = nn.JAX_ROOT


def draw_params(key, d, layers):
    def normal(k, shape):
        return jax.random.normal(k, shape, dtype=jnp.float32) * 0.02

    # the two tables, the skipped key below, then the tensors of each layer
    keys = iter(jax.random.split(key, len(model.TABLES) + 1 + model.PER_LAYER * layers))
    params = {
        "seats": normal(next(keys), (data.SEATS, data.CLASSES, d)),
        "phase": normal(next(keys), (model.PHASE_BUCKETS, d)),
    }
    # The window-position table stood here and the ear dropped it. Its key stays skipped,
    # thus a seed draws the same layers it drew for the runs that elected this model.
    next(keys)
    return params | {
        "layers": [
            {
                "wq": normal(next(keys), (d, d)),
                "wk": normal(next(keys), (d, d)),
                "wv": normal(next(keys), (d, d)),
                "wo": normal(next(keys), (d, d)),
                "w1": normal(next(keys), (d, 4 * d)),
                "w2": normal(next(keys), (4 * d, d)),
            }
            for _ in range(layers)
        ],
    }


def save_checkpoint(path, params):
    """the tables in construction order, then the layers; nn.save_checkpoint states the
    naming rule of the seam"""
    nn.save_checkpoint(
        path,
        [params[name] for name in model.TABLES]
        + [layer[name] for layer in params["layers"] for name in model.LAYER_TENSORS],
    )


def make_step(heads, dropout, clip, weight_decay, span):
    def loss(p, classes, phases, key):
        return jnp.mean(
            model.seat_nll(
                p, classes, phases, heads=heads, dropout=dropout, key=key, span=span
            )
        )

    return nn.make_step(loss, clip=clip, weight_decay=weight_decay)


def make_eval(heads, span):
    def nll(params, classes, phases):
        return model.seat_nll(params, classes, phases, heads=heads, span=span)

    return nn.make_eval(nll)


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
@click.option("--wd", default=0.01)
@click.option("--clip", default=1.0)
@click.option("--dropout", default=0.3)
@click.option(
    "--alibi-span",
    default=model.SLOPE_SPAN,
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
def main(
    corpus_path,
    d,
    layers,
    heads,
    context,
    batch,
    steps,
    lr,
    seed,
    warmup,
    wd,
    clip,
    dropout,
    alibi_span,
    train_on,
    log_every,
    eval_every,
    eval_limit,
    ckpt,
    average_top,
):
    corpus = data.load_corpus(corpus_path)
    pool = data.train_pool(corpus, train_on)
    train_eval = nn.on_device(data.eval_batches(corpus["train"], context, eval_limit, batch))
    valid_eval = nn.on_device(data.eval_batches(corpus["valid"], context, eval_limit, batch))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, draw_key = jax.random.split(key)
    params = draw_params(draw_key, d, layers)
    m = v = jax.tree.map(jnp.zeros_like, params)
    step_fn = make_step(heads, dropout, clip, wd, alibi_span)
    eval_fn = make_eval(heads, alibi_span)
    count = sum(int(np.prod(t.shape)) for t in jax.tree.leaves(params))
    corpus_steps = sum(int(split.index[row, 1]) for split, row in pool)
    click.echo(
        f"corpus: {len(pool)} pool streams, {corpus_steps} steps; eval rows: "
        f"{sum(len(b[0]) for b in train_eval)} train, "
        f"{sum(len(b[0]) for b in valid_eval)} valid; parameters: {count}"
    )

    best = float("inf")
    top = []  # (valid, step, host params) -- the K best snapshots for averaging
    losses = []
    started = time.perf_counter()

    def evaluate(step, params):
        nonlocal best
        train_all, train_moving = nn.eval_loss(eval_fn, params, train_eval)
        valid_all, valid_moving = nn.eval_loss(eval_fn, params, valid_eval)
        mark = ""
        if valid_all < best:
            best = valid_all
            mark = "  *"
            if ckpt and train_on != "all":
                save_checkpoint(ckpt, params)
        # the snapshot crosses to the host only when it can stay: the sort would drop it
        # again, and the copy is the whole parameter tree
        if average_top > 0 and (len(top) < average_top or valid_all < top[-1][0]):
            top.append((valid_all, step, jax.tree.map(np.asarray, params)))
            top.sort(key=lambda entry: entry[0])
            del top[average_top:]
        click.echo(
            f"step {step:5d}  eval  train {train_all:.4f} (moving {train_moving:.4f})"
            f"  valid {valid_all:.4f} (moving {valid_moving:.4f}){mark}"
        )

    for step in range(1, steps + 1):
        classes, phases = data.train_batch(rng, pool, batch, context)
        # a name of its own: [lr] is the peak the schedule reads, and a loop that writes
        # its own peak decays the rate geometrically to zero and trains nothing
        rate = nn.schedule(step, lr, warmup, steps)
        key, step_key = jax.random.split(key)
        value, params, m, v = step_fn(
            params,
            m,
            v,
            jnp.float32(step),
            jnp.asarray(classes),
            jnp.asarray(phases),
            jnp.float32(rate),
            step_key,
        )
        # the device array, NOT float(value): a read blocks until the step finishes, and
        # the loop then cannot overlap the next batch draw and its transfer with the
        # compute of this one. Measured at the baseline config: 27.0 ms each step against
        # 19.3. The log below reads, thus the run-ahead stays inside one log window.
        losses.append(value)
        if step % log_every == 0 or step == 1:
            # the training number is nats for each step too: the mean over the predictions
            # times the four seats
            mean = float(jnp.mean(jnp.stack(losses)))
            click.echo(f"step {step:5d}  loss {data.SEATS * mean:.4f}")
            losses = []
        if step % eval_every == 0 or step == steps:
            evaluate(step, params)

    seconds = time.perf_counter() - started
    click.echo(
        f"time: {seconds:.0f} s, {seconds / steps * 1000:.0f} ms each step, "
        f"the evaluations inside"
    )
    click.echo(f"best valid {best:.4f}")
    if ckpt:
        if train_on == "all":
            save_checkpoint(ckpt, params)
            click.echo(f"checkpoint of the last step: {ckpt}")
        else:
            click.echo(f"checkpoint of the best: {ckpt}")
        if average_top > 0 and top:
            averaged = jax.tree.map(
                lambda *tensors: np.mean(np.stack(tensors), axis=0),
                *[entry[2] for entry in top],
            )
            path = ckpt.replace(".ckpt", "-avg.ckpt")
            save_checkpoint(path, averaged)
            click.echo(
                f"average of {len(top)} best snapshots "
                f"(steps {[entry[1] for entry in top]}): {path}"
            )


if __name__ == "__main__":
    main()
