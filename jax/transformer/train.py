"""The JAX twin of bin/train_transformer.ml -- the sweep vehicle.

Run it from the jax directory as a module, so that data.py stays on the path:

    uv run python -m transformer.train --steps 200

Same walk, same referee rows, same schedule shape; the recipe knobs -- dropout, weight
decay -- are the reason this trainer exists. Checkpoints are Kaun safetensors, tensors
"0".."N" in the OCaml Params order, so checkpoint_tool and play_transformer consume
them directly. The final board model still comes from the OCaml trainer of record; this
side only finds the recipe.

The optimizer is a hand-rolled AdamW with the decoupled decay of Kaun's, and the
gradient clip is the same global-norm rule. Optimizer parity with OCaml is not required
-- Gate B pins the eval, not the trajectory.
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


def draw_params(key, d, layers, progress=False):
    def normal(k, shape):
        return jax.random.normal(k, shape, dtype=jnp.float32) * 0.02

    keys = iter(jax.random.split(key, 2 + 6 * layers))
    drawn = {
        "embed": normal(next(keys), (model.VOCAB, d)),
        "phase": normal(next(keys), (model.PHASE_BUCKETS, d)),
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
    if progress:
        # a folded key, so that the split above keeps the draw of a run without progress
        drawn["progress"] = normal(
            jax.random.fold_in(key, 1), (model.PROGRESS_BUCKETS, d)
        )
    return drawn


def save_checkpoint(path, params):
    """Kaun order: embed, phase, the progress table when the run has one, then
    wq wk wv wo w1 w2 for each layer."""
    tables = [params["embed"], params["phase"]]
    if "progress" in params:
        tables.append(params["progress"])
    tensors = tables + [
        layer[name] for layer in params["layers"] for name in model.LAYER_TENSORS
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({str(i): np.asarray(t) for i, t in enumerate(tensors)}, path)


def schedule(step, peak, warmup, total):
    """the OCaml schedule: linear warmup to the peak, cosine decay to zero; a warmup of
    zero is the constant peak"""
    if warmup == 0:
        return peak
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))


def make_step(heads, dropout, clip, weight_decay, span, progress=False):
    def step_fn(params, m, v, t, codes, phases, buckets, masks, weights, lr, key):
        def loss_fn(p):
            return model.loss(
                p,
                codes,
                phases,
                masks,
                heads=heads,
                dropout=dropout,
                key=key,
                weights=weights,
                span=span,
                progress=buckets if progress else None,
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


def make_eval(heads, span, progress=False):
    def eval_fn(params, codes, phases, buckets, masks):
        return model.loss(
            params,
            codes,
            phases,
            masks,
            heads=heads,
            span=span,
            progress=buckets if progress else None,
        )

    return jax.jit(eval_fn)


def eval_loss(eval_fn, params, batches):
    total, count = 0.0, 0
    for codes, phases, buckets, masks, rows in batches:
        value = float(
            eval_fn(
                params,
                jnp.asarray(codes),
                jnp.asarray(phases),
                jnp.asarray(buckets),
                jnp.asarray(masks),
            )
        )
        total += value * rows
        count += rows
    return total / count


@click.command(help=__doc__)
@click.option("--corpus", default=str(JAX_ROOT / "_data" / "corpus.safetensors"))
@click.option("--d", default=64)
@click.option("--layers", default=2)
@click.option("--heads", default=4)
@click.option("--context", default=256)
@click.option("--batch", default=16)
@click.option("--steps", default=200)
@click.option("--lr", default=3e-4)
@click.option("--seed", default=1)
@click.option("--warmup", default=0)
@click.option("--wd", default=0.01)
@click.option("--clip", default=1.0)
@click.option("--dropout", default=0.0)
@click.option(
    "--alibi-span",
    default=model.SLOPE_SPAN,
    help="the ALiBi exponent span: the slope of head k is 2^-(span (k+1) / heads). The paper's 8 leaves the gentlest head at -4 logits by 1024 tokens; a wider span sees further and stays a power of two. The draw must state the same span.",
)
@click.option(
    "--progress",
    is_flag=True,
    help="add the piece-position table of docs/transformer_model.md: 16 rows, indexed by which sixteenth of its piece the step of a token sits in. It earns its place at context 256, where a window holds START in 0.7% of rows and the model is otherwise blind to its position; at 512 it holds START in 28% of rows, the table is redundant and the loss is 0.004 worse.",
)
@click.option(
    "--train-on", type=click.Choice(("train", "train+test", "all")), default="train"
)
@click.option("--log-every", default=10)
@click.option("--eval-every", default=100)
@click.option("--eval-limit", default=128)
@click.option(
    "--eval-context",
    "eval_context_flag",
    type=int,
    help="evaluate at this context instead of the training one. The windows come from whole pieces, so a long training context leaves almost none: 149 valid rows at 256, 56 at 512, 6 at 1024, and none at 2048. ALiBi has no position table, so a model trained long evaluates short; 256 also makes every run comparable, as checkpoint_tool does.",
)
@click.option("--ckpt", default=None)
@click.option(
    "--average-top",
    default=0,
    help="also write the mean of the K best-by-valid snapshots as NAME-avg.ckpt",
)
def main(
    corpus,
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
    progress,
    train_on,
    log_every,
    eval_every,
    eval_limit,
    eval_context_flag,
    ckpt,
    average_top,
):

    corpus = data.load_corpus(corpus)
    pool = data.train_pool(corpus, train_on)
    eval_context = eval_context_flag or context
    train_eval = data.eval_batches(corpus["train"], eval_context, eval_limit, batch)
    valid_eval = data.eval_batches(corpus["valid"], eval_context, eval_limit, batch)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, draw_key = jax.random.split(key)
    params = draw_params(draw_key, d, layers, progress)
    m = v = jax.tree.map(jnp.zeros_like, params)
    step_fn = make_step(heads, dropout, clip, wd, alibi_span, progress)
    eval_fn = make_eval(heads, alibi_span, progress)
    count = sum(int(np.prod(t.shape)) for t in jax.tree.leaves(params))
    print(
        f"corpus: {len(pool)} pool pieces; eval rows: "
        f"{sum(b[4] for b in train_eval)} train, {sum(b[4] for b in valid_eval)} valid; "
        f"parameters: {count}",
        flush=True,
    )

    best = float("inf")
    top = []  # (valid, step, host params) -- the K best snapshots for averaging
    losses = []
    started = time.perf_counter()

    def evaluate(step, params):
        nonlocal best
        train_loss = eval_loss(eval_fn, params, train_eval)
        valid_loss = eval_loss(eval_fn, params, valid_eval)
        mark = ""
        if valid_loss < best:
            best = valid_loss
            mark = "  *"
            if ckpt and train_on != "all":
                save_checkpoint(ckpt, params)
        if average_top > 0:
            top.append((valid_loss, step, jax.tree.map(np.asarray, params)))
            top.sort(key=lambda entry: entry[0])
            del top[average_top:]
        print(
            f"step {step:4d}  eval  train {train_loss:.4f}  valid {valid_loss:.4f}{mark}",
            flush=True,
        )

    for step in range(1, steps + 1):
        codes, phases, buckets, masks, weights = data.train_batch(
            rng, pool, batch, context
        )
        lr = schedule(step, lr, warmup, steps)
        key, step_key = jax.random.split(key)
        value, params, m, v = step_fn(
            params,
            m,
            v,
            jnp.float32(step),
            jnp.asarray(codes),
            jnp.asarray(phases),
            jnp.asarray(buckets),
            jnp.asarray(masks),
            jnp.asarray(weights),
            jnp.float32(lr),
            step_key,
        )
        losses.append(float(value))
        if step % log_every == 0 or step == 1:
            print(f"step {step:4d}  loss {np.mean(losses):.4f}", flush=True)
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
            steps = [entry[1] for entry in top]
            print(
                f"average of {len(top)} best snapshots (steps {steps}): {path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
