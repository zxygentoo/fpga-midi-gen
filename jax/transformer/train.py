"""The trainer of the step-frame model of docs/transformer.md.

Run it as a module, from any directory of the tree:

    uv run python -m transformer.train --steps 200

The loop, the evaluation and the checkpoint policy are `ar_train.train`; this file is the
shape of the model and the flags that spell it. THE ERA IS FROZEN and this trainer is
kept, not run.
"""

import click

import ar_model
import ar_train
from transformer import model


@click.command(help=__doc__)
@click.option("--d", default=64)
@click.option("--layers", default=6)
@click.option("--heads", default=4)
@click.option(
    "--alibi-span",
    "span",
    default=ar_model.SLOPE_SPAN,
    help="the ALiBi exponent span: the slope of head k is 2^-(span (k+1) / heads). The "
    "draw must state the same.",
)
@ar_train.recipe_options(dropout=0.3)
def main(d, layers, heads, span, seed, **flags):
    ar_train.train(
        model.Transformer.drawn(seed, d, layers, heads=heads, span=span),
        seed=seed,
        **flags,
    )


if __name__ == "__main__":
    main()
