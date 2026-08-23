"""The trainer of the step-frame model of docs/transformer_model.md.

Run it from the jax directory as a module:

    uv run python -m transformer.train --steps 200

The optimizer is nn.py's AdamW, with a decoupled decay and a global-norm clip.

The loss is reported as NATS FOR EACH STEP -- the sum over the four seats -- because a
per-prediction mean divides against a different count in each encoding and compares
nothing. The moving-steps loss -- the drone detector, and the number elections read --
is measure.py's instrument and not this log's.

The gradient takes the mean over the predictions and not the sum over the seats. Adam is
blind to the scale, but the global-norm clip is not: a loss four times larger would make
the clip bite four times harder, and the peak rate and the clip of the recipe would stop
meaning what they meant.
"""


import click
import jax
import jax.numpy as jnp

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
    def draw(key):
        return draw_params(key, d, layers)

    def describe(params):
        return (
            f"shape: d {params['seats'].shape[-1]}, {len(params['layers'])} layers, "
            f"{heads} heads, context {context}, dropout {dropout}, seed {seed}, "
            f"ALiBi span {alibi_span}"
        )

    nn.train(
        corpus_path=corpus_path,
        train_on=train_on,
        context=context,
        batch=batch,
        steps=steps,
        lr=lr,
        seed=seed,
        warmup=warmup,
        log_every=log_every,
        eval_every=eval_every,
        eval_limit=eval_limit,
        ckpt=ckpt,
        average_top=average_top,
        draw_params=draw,
        step_fn=make_step(heads, dropout, clip, wd, alibi_span),
        eval_fn=make_eval(heads, alibi_span),
        save_checkpoint=save_checkpoint,
        describe=describe,
    )


if __name__ == "__main__":
    main()
