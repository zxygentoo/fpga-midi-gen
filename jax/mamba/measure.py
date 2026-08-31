"""What era five's model predicts, and what it plays: the two commands of the era.

    uv run python -m mamba.measure forced --ckpt ../_train/mamba/NAME.ckpt
    uv run python -m mamba.measure free   --ckpt ../_train/mamba/NAME.ckpt --seeds 1-16

THE INSTRUMENTS ARE NOT HERE. Both halves -- the forced loss and the free walk -- are one
thing across the two step-frame eras and stand in `ar_measure.py`, beside `ar_train.py`.
What is era five's is the two commands below: they name a model and a player, and
`ar_measure` names neither.
"""

from pathlib import Path

import click

import cli
import corpus
from ar_measure import (
    corpus_row,
    loss_lines,
    loss_row,
    mean_over_seeds,
    walk_line,
    walk_row,
)
from mamba import model, quantized


@click.group(help=__doc__)
def main():
    pass


@main.command(help=loss_row.__doc__)
@cli.ckpt_option
@click.option("--corpus", "corpus_path", default=str(corpus.FRAMES))
def forced(ckpt, corpus_path):
    row = loss_row(model.Mamba.load(ckpt), corpus_path)
    for line in loss_lines(Path(ckpt).stem[:22], row):
        click.echo(line)


@main.command(help=walk_row.__doc__)
@cli.ckpt_option
@click.option("--seeds", default="1-16", help="a list, or LOW-HIGH")
@click.option("--steps", default=512)
@click.option("--temperature", default=quantized.ELECTED_TEMPERATURE)
@click.option("--min-p", default=quantized.ELECTED_MIN_P)
@click.option("--ring", default=model.ATTN_CONTEXT, help="the attention ring depth")
@click.option("--corpus", "corpus_path", default=str(corpus.FRAMES))
def free(ckpt, seeds, steps, temperature, min_p, ring, corpus_path):
    # THE IMPORT IS DEFERRED AND IT IS LOAD-BEARING. `mamba.infer` sets
    # JAX_PLATFORMS=cpu at import, which is the audition's policy and not the referee's:
    # a top-level import here would put the `forced` command on the CPU as well, and its
    # eval pass is the one thing in this module that wants the card.
    from mamba import infer

    seeds = cli.parse_seeds(None, None, seeds)
    # [infer.draw] and NOT [infer.sample], which is a click Command and cannot be called
    # with a model
    walks = infer.draw(
        model.Mamba.load(ckpt),
        seeds=seeds,
        steps=steps,
        temperature=temperature,
        min_p=min_p,
        ring=ring,
    )
    click.echo(walk_line("the packed corpus", corpus_row(corpus_path)))
    rows = [walk_row(walk) for walk in walks]
    click.echo(
        walk_line(
            f"T {temperature} min-p {min_p}, {len(rows)} walks", mean_over_seeds(rows)
        )
    )


if __name__ == "__main__":
    main()
