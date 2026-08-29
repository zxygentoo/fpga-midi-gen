"""The trainer of the step-frame model of docs/transformer.md.

Run it from the jax directory as a module:

    uv run python -m transformer.train --steps 200

The loop, the evaluation and the checkpoint policy are `frames.train`, the recipe eras
four and five share; this file is the shape of the model and the flags that spell it.

THE ERA IS FROZEN and this trainer is kept, not run: the elected checkpoint stands and no
retrain is planned.
"""

import click

import data
import frames
import nn
from transformer import model


@click.command(help=__doc__)
@click.option("--corpus", "corpus_path", default=str(data.FRAMES))
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
@click.option("--log-every", default=100)
@click.option("--eval-every", default=1600)
@click.option("--eval-limit", default=128)
@click.option("--ckpt", default=None)
def main(d, layers, heads, span, seed, **flags):
    frames.train(
        model.Transformer.drawn(seed, d, layers, heads=heads, span=span),
        seed=seed,
        **flags,
    )


if __name__ == "__main__":
    main()
