"""The trainer of the state-space model of docs/mamba.md.

Run it from the jax directory as a module:

    uv run python -m mamba.train --steps 200

The recipe opens where era four closed: the same hand-rolled AdamW with a decoupled decay
and a global-norm clip, the same batch draw -- a uniform stream, then a uniform window --
and the same two numbers out of every evaluation.

The loss is reported as NATS FOR EACH STEP -- the sum over the four seats -- because a
per-prediction mean divides against a different count in each encoding and compares
nothing. Era four and era five share one encoding and one window rule, thus this number
compares across the two eras and the elected model of era four stands at 1.6282.

A second number covers the steps where two or more voices move: 77.91 percent of the voice
slots repeat the step before, they dominate the mean, and a recurrence that decays too fast
degenerates to exactly that predictor. Watch the moving-steps number, not the mean.

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
from mamba import model

JAX_ROOT = Path(__file__).resolve().parent.parent


def half_life_ladder(heads, span):
    """The dt of each head that puts its half-life on a log-spaced ladder.

    A trained state decays as `exp(-dt * a)` for each step, thus its half-life is
    `ln 2 / (dt * a)`. The Mamba draw is uniform in dt and says nothing about the
    half-life; this one names the ladder and solves for the dt that lands on it, over the
    decay rate the head already drew. One head sits at each rung, log-spaced.

    It exists because a measurement asked for it: over the elected prototype the trained
    half-lives collapse -- no head above layer 2 holds a median of more than 7 steps -- and
    a state that never learns a phrase-scale memory may simply have opened too far from
    one."""
    low, high = span
    rungs = low * (high / low) ** (jnp.arange(heads, dtype=jnp.float32) / max(heads - 1, 1))
    return jnp.log(2.0) / rungs


def draw_params(
    key, *, d, layers, heads, state, taps, expand, conv_scale, half_lives,
    attention_at, spelt=None,
):
    """The Mamba defaults, over era four's draw of the matrices.

    [a_log] is the log of a uniform decay rate in [1, 16] and [dt_bias] the inverse
    softplus of a uniform step in [0.001, 0.1], which is the initialization the Mamba
    papers state. [d_skip] opens at one, thus a layer starts as the skip and learns its
    state from there.

    [half_lives] replaces the uniform draw of dt with the ladder of [half_life_ladder],
    and it is the one lever this initialization holds.

    [attention_at] names the layers that are era four's attention sublayer instead of a
    block. Its four matrices take the same normal at 0.02 that every other matrix takes --
    era four drew them that way and one rule covers the whole model.

    THE CONVOLUTION TAKES 0.02 TOO, and it was measured. The argument against was fan-in:
    four taps at 0.02 pass a fiftieth of their input, the SiLU under them sits near its own
    origin, and B and C open so small that the state has little to learn from. 1/sqrt(K) is
    the fan-in scale and the Mamba reference uses it. Over 4 000 steps of the baseline
    shape it read 1.7311 valid against 0.02's 1.7113, thus the argument is wrong here: the
    gated norm rescales the branch in any case, and the smaller draw is no worse. One rule
    covers every matrix of this model."""

    def normal(k, shape, scale=0.02):
        return jax.random.normal(k, shape, dtype=jnp.float32) * scale

    d_in = expand * d
    channels = d_in + 2 * state
    plan = spelt or [
        model.ATTN if at in attention_at else model.MAMBA for at in range(layers)
    ]
    count = len(model.TABLES) + sum(len(model.LAYER_TENSORS[kind]) for kind in plan)
    keys = iter(jax.random.split(key, count))
    params = {
        "seats": normal(next(keys), (data.SEATS, data.CLASSES, d)),
        "phase": normal(next(keys), (model.PHASE_BUCKETS, d)),
    }

    def attention(kind):
        # the Zamba query and key read the stream beside the embedding, thus [2d, d]
        wide = (2 * d, d) if kind == model.ZATTN else (d, d)
        return {
            "wq": normal(next(keys), wide),
            "wk": normal(next(keys), wide),
            "wv": normal(next(keys), (d, d)),
            "wo": normal(next(keys), (d, d)),
        }

    def feed_forward():
        return {
            "w1": normal(next(keys), (d, 4 * d)),
            "w2": normal(next(keys), (4 * d, d)),
        }

    def drawn(kind):
        if kind == model.MLP:
            return feed_forward()
        if kind in (model.ATTN, model.ZATTN):
            return attention(kind)
        return layer()

    def layer():
        w_in = normal(next(keys), (d, 2 * d_in + 2 * state + heads))
        conv = normal(next(keys), (channels, taps), conv_scale)
        step = jax.random.uniform(next(keys), (heads,), minval=0.001, maxval=0.1)
        decay = jax.random.uniform(next(keys), (heads,), minval=1.0, maxval=16.0)
        # the draw above still runs and its key is still spent, thus a ladder run and its
        # baseline differ in dt_bias and in no other tensor of the checkpoint
        if half_lives is not None:
            step = half_life_ladder(heads, half_lives) / decay
        return {
            "w_in": w_in,
            "conv": conv,
            # the inverse softplus of the drawn step: softplus(dt_bias) is that step again
            "dt_bias": jnp.log(jnp.expm1(step)),
            "a_log": jnp.log(decay),
            "d_skip": jnp.ones((heads,), jnp.float32),
            "w_out": normal(next(keys), (d_in, d)),
        }

    return params | {"layers": [drawn(kind) for kind in plan]}


def save_checkpoint(path, params, span=None):
    """the tables, then the layers each by its own kind, then the ALiBi span.

    The span goes LAST and alone, thus an older file that does not carry it still reads:
    the walk of [model.load_params] takes whole layer groups and then one scalar if one
    is there. It is written even where no layer attends, which costs four bytes and keeps
    one rule."""
    tensors = [params[name] for name in model.TABLES] + [
        layer[name]
        for layer in params["layers"]
        for name in model.LAYER_TENSORS[model.kind_of(layer)]
    ]
    if span is not None:
        tensors = tensors + [np.asarray([span], dtype=np.float32)]
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


def make_step(dropout, clip, weight_decay, span):
    def step_fn(params, m, v, t, classes, phases, lr, key):
        def loss_fn(p):
            return jnp.mean(
                model.seat_nll(p, classes, phases, dropout=dropout, key=key, span=span)
            )

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


def make_eval(span):
    def eval_fn(params, classes, phases):
        nll = model.seat_nll(params, classes, phases, span=span)
        steps = jnp.sum(nll, axis=-1)
        moving = data.moving(classes) >= 2
        return (
            jnp.sum(steps),
            jnp.sum(jnp.where(moving, steps, 0.0)),
            jnp.sum(moving),
            jnp.size(steps),
        )

    return jax.jit(eval_fn)


def on_device(batches):
    """the evaluation windows are fixed for the whole run, thus they cross to the device
    one time and not at every evaluation"""
    return [(jnp.asarray(classes), jnp.asarray(phases)) for classes, phases in batches]


def eval_loss(eval_fn, params, batches):
    """nats for each step, over every step and over the moving steps alone"""
    total = moved = 0.0
    steps = moves = 0
    for classes, phases in batches:
        sums = eval_fn(params, classes, phases)
        total += float(sums[0])
        moved += float(sums[1])
        moves += int(sums[2])
        steps += int(sums[3])
    return total / max(steps, 1), moved / max(moves, 1)


PLAN_LETTERS = {"m": model.MAMBA, "a": model.ATTN, "z": model.ZATTN, "f": model.MLP}


def parse_plan(ctx, param, value):
    """The plan spelt out, one letter for each layer: M a block, A era four's attention
    sublayer, Z the Zamba one that also reads the embedding, F the feed-forward.

    "MMMMMMZF" is six blocks under a Zamba head. Given, it wins over --layers and
    --attention-at, which spell the plans of blocks and attention alone."""
    del ctx, param
    letters = [c for c in value.lower() if not c.isspace()]
    unknown = {c for c in letters} - set(PLAN_LETTERS)
    if unknown:
        raise click.BadParameter(
            f"{sorted(unknown)} are not plan letters {sorted(PLAN_LETTERS)}"
        )
    return [PLAN_LETTERS[c] for c in letters] or None


def parse_attention_at(ctx, param, value):
    """the layers that are attention and not a block, 0 first; nothing is the trunk"""
    del ctx, param
    return tuple(int(at) for at in value.split(",")) if value else ()


def parse_half_lives(ctx, param, value):
    """LOW-HIGH in steps, or nothing at all for the Mamba draw"""
    del ctx, param
    if not value:
        return None
    low, high = value.split("-")
    return float(low), float(high)


@click.command(help=__doc__)
@click.option(
    "--corpus", "corpus_path", default=str(JAX_ROOT / "_data" / "frames.safetensors")
)
@click.option("--d", default=64)
@click.option("--layers", default=6)
@click.option("--heads", default=4)
@click.option("--state", default=16, help="N, the state width of one head")
@click.option("--taps", default=model.CONV_TAPS, help="K, the convolution width")
@click.option(
    "--alibi-span",
    "alibi_span",
    default=model.SLOPE_SPAN,
    type=float,
    help="the ALiBi exponent span of the attention layers: the slope of head k is "
    "2^-(span (k+1) / heads), thus a LARGER span reaches further. Era four elected 4 on "
    "a pure transformer; the file records whichever this run used.",
)
@click.option(
    "--plan",
    "spelt",
    default="",
    callback=parse_plan,
    help="the plan spelt out, one letter for each layer: M block, A attention, Z the "
    "Zamba attention that reads the embedding, F feed-forward. Wins over --layers.",
)
@click.option(
    "--attention-at",
    "attention_at",
    default="",
    callback=parse_attention_at,
    help="the layers that take era four's attention sublayer instead of a block, 0 "
    "first; "
    "empty is the trunk of six blocks",
)
@click.option("--expand", default=model.EXPAND, help="d_in = expand * d")
@click.option(
    "--dt-half-lives",
    "half_lives",
    default="",
    callback=parse_half_lives,
    help="LOW-HIGH in steps: open dt on a log-spaced half-life ladder instead of the "
    "uniform Mamba draw, one head at each rung",
)
@click.option(
    "--conv-scale",
    type=float,
    default=0.02,
    help="the draw of the convolution kernel; measured against 1/sqrt(K), see draw_params",
)
@click.option("--context", default=256, help="the training window, in steps")
@click.option("--batch", default=16)
@click.option("--steps", default=96000)
@click.option("--lr", default=1e-3)
@click.option("--seed", default=6)
@click.option("--warmup", default=300)
@click.option("--wd", default=0.01)
@click.option("--clip", default=1.0)
@click.option("--dropout", default=0.2)
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
    state,
    taps,
    alibi_span,
    attention_at,
    spelt,
    half_lives,
    expand,
    conv_scale,
    context,
    batch,
    steps,
    lr,
    seed,
    warmup,
    wd,
    clip,
    dropout,
    train_on,
    log_every,
    eval_every,
    eval_limit,
    ckpt,
    average_top,
):
    corpus = data.load_corpus(corpus_path)
    pool = data.train_pool(corpus, train_on)
    train_eval = on_device(data.eval_batches(corpus["train"], context, eval_limit, batch))
    valid_eval = on_device(data.eval_batches(corpus["valid"], context, eval_limit, batch))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, draw_key = jax.random.split(key)
    params = draw_params(
        draw_key,
        d=d,
        layers=layers,
        heads=heads,
        state=state,
        taps=taps,
        expand=expand,
        conv_scale=conv_scale,
        half_lives=half_lives,
        attention_at=attention_at,
        spelt=spelt,
    )
    m = v = jax.tree.map(jnp.zeros_like, params)
    step_fn = make_step(dropout, clip, wd, alibi_span)
    eval_fn = make_eval(alibi_span)
    count = sum(int(np.prod(t.shape)) for t in jax.tree.leaves(params))
    corpus_steps = sum(int(split.index[row, 1]) for split, row in pool)
    click.echo(
        f"corpus: {len(pool)} pool streams, {corpus_steps} steps; eval rows: "
        f"{sum(len(b[0]) for b in train_eval)} train, "
        f"{sum(len(b[0]) for b in valid_eval)} valid; parameters: {count}"
    )
    click.echo(
        f"shape: {model.shape_of(params)}, dropout {dropout}, seed {seed}, "
        f"dt half-lives {half_lives or 'the Mamba draw'}, ALiBi span {alibi_span}"
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
                save_checkpoint(ckpt, params, span=alibi_span)
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
            evaluate(step, params)

    seconds = time.perf_counter() - started
    click.echo(
        f"time: {seconds:.0f} s, {seconds / steps * 1000:.0f} ms each step, "
        f"the evaluations inside"
    )
    click.echo(f"best valid {best:.4f}")
    if ckpt:
        if train_on == "all":
            save_checkpoint(ckpt, params, span=alibi_span)
            click.echo(f"checkpoint of the last step: {ckpt}")
        else:
            click.echo(f"checkpoint of the best: {ckpt}")
        if average_top > 0 and top:
            averaged = jax.tree.map(
                lambda *tensors: np.mean(np.stack(tensors), axis=0),
                *[entry[2] for entry in top],
            )
            path = ckpt.replace(".ckpt", "-avg.ckpt")
            save_checkpoint(path, averaged, span=alibi_span)
            click.echo(
                f"average of {len(top)} best snapshots "
                f"(steps {[entry[1] for entry in top]}): {path}"
            )


if __name__ == "__main__":
    main()
