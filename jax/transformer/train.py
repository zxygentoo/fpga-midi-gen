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
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

import data
from transformer import model

JAX_ROOT = Path(__file__).resolve().parent.parent


def draw_params(key, d, layers):
    def normal(k, shape):
        return jax.random.normal(k, shape, dtype=jnp.float32) * 0.02

    keys = iter(jax.random.split(key, 3 + 6 * layers))
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
    """the tables in construction order, then the layers"""
    tensors = [params[name] for name in model.TABLES] + [
        layer[name] for layer in params["layers"] for name in model.LAYER_TENSORS
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({str(i): np.asarray(t) for i, t in enumerate(tensors)}, path)


def schedule(step, peak, warmup, total):
    """linear warmup to the peak, cosine decay to zero; a warmup of zero is a constant"""
    if warmup == 0:
        return peak
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))


def make_step(heads, dropout, clip, weight_decay, span):
    def step_fn(params, m, v, t, classes, phases, lr, key):
        def loss_fn(p):
            nll = model.seat_nll(
                p, classes, phases, heads=heads, dropout=dropout, key=key, span=span
            )
            return jnp.mean(nll)

        value, grads = jax.value_and_grad(loss_fn)(params)
        if clip > 0.0:
            norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
            scale = clip / jnp.maximum(norm, clip)
            grads = jax.tree.map(lambda g: g * scale, grads)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda m_: m_ / (1 - b1**t), m)
        v_hat = jax.tree.map(lambda v_: v_ / (1 - b2**t), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + weight_decay * p),
            params,
            m_hat,
            v_hat,
        )
        return value, params, m, v

    return jax.jit(step_fn)


def make_eval(heads, span):
    def eval_fn(params, classes, phases):
        nll = model.seat_nll(params, classes, phases, heads=heads, span=span)
        steps = jnp.sum(nll, axis=-1)
        moving = data.moving(classes) >= 2
        return (
            jnp.sum(steps),
            jnp.sum(jnp.where(moving, steps, 0.0)),
            jnp.sum(moving),
            jnp.size(steps),
        )

    return jax.jit(eval_fn)


def eval_loss(eval_fn, params, batches):
    """nats for each step, over every step and over the moving steps alone"""
    total = moved = 0.0
    steps = moves = 0
    for classes, phases in batches:
        sums = eval_fn(params, jnp.asarray(classes), jnp.asarray(phases))
        total += float(sums[0])
        moved += float(sums[1])
        moves += int(sums[2])
        steps += int(sums[3])
    return total / max(steps, 1), moved / max(moves, 1)


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
    train_eval = data.eval_batches(corpus["train"], context, eval_limit, batch)
    valid_eval = data.eval_batches(corpus["valid"], context, eval_limit, batch)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, draw_key = jax.random.split(key)
    params = draw_params(draw_key, d, layers)
    m = v = jax.tree.map(jnp.zeros_like, params)
    step_fn = make_step(heads, dropout, clip, wd, alibi_span)
    eval_fn = make_eval(heads, alibi_span)
    count = sum(int(np.prod(t.shape)) for t in jax.tree.leaves(params))
    corpus_steps = sum(int(split.index[row, 1]) for split, row in pool)
    print(
        f"corpus: {len(pool)} pool streams, {corpus_steps} steps; eval rows: "
        f"{sum(len(b[0]) for b in train_eval)} train, "
        f"{sum(len(b[0]) for b in valid_eval)} valid; parameters: {count}",
        flush=True,
    )

    best = float("inf")
    top = []  # (valid, step, host params) -- the K best snapshots for averaging
    losses = []
    started = time.perf_counter()

    def evaluate(step, params):
        nonlocal best
        train_all, train_moving = eval_loss(eval_fn, params, train_eval)
        valid_all, valid_moving = eval_loss(eval_fn, params, valid_eval)
        mark = ""
        if valid_all < best:
            best = valid_all
            mark = "  *"
            if ckpt and train_on != "all":
                save_checkpoint(ckpt, params)
        if average_top > 0:
            top.append((valid_all, step, jax.tree.map(np.asarray, params)))
            top.sort(key=lambda entry: entry[0])
            del top[average_top:]
        print(
            f"step {step:5d}  eval  train {train_all:.4f} (moving {train_moving:.4f})"
            f"  valid {valid_all:.4f} (moving {valid_moving:.4f}){mark}",
            flush=True,
        )

    for step in range(1, steps + 1):
        classes, phases = data.train_batch(rng, pool, batch, context)
        # a name of its own: [lr] is the peak the schedule reads, and a loop that writes
        # its own peak decays the rate geometrically to zero and trains nothing
        rate = schedule(step, lr, warmup, steps)
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
        losses.append(float(value))
        if step % log_every == 0 or step == 1:
            # the training number is nats for each step too: the mean over the predictions
            # times the four seats
            print(
                f"step {step:5d}  loss {data.SEATS * np.mean(losses):.4f}",
                flush=True,
            )
            losses = []
        if step % eval_every == 0 or step == steps:
            evaluate(step, params)

    seconds = time.perf_counter() - started
    print(
        f"time: {seconds:.0f} s, {seconds / steps * 1000:.0f} ms each step, "
        f"the evaluations inside",
        flush=True,
    )
    print(f"best valid {best:.4f}", flush=True)
    if ckpt:
        if train_on == "all":
            save_checkpoint(ckpt, params)
            print(f"checkpoint of the last step: {ckpt}", flush=True)
        else:
            print(f"checkpoint of the best: {ckpt}", flush=True)
        if average_top > 0 and top:
            averaged = jax.tree.map(
                lambda *tensors: np.mean(np.stack(tensors), axis=0),
                *[entry[2] for entry in top],
            )
            path = ckpt.replace(".ckpt", "-avg.ckpt")
            save_checkpoint(path, averaged)
            print(
                f"average of {len(top)} best snapshots "
                f"(steps {[entry[1] for entry in top]}): {path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
