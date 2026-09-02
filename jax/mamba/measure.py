"""What era five's model predicts, and what it plays: the two commands of the era.

    uv run python -m mamba.measure forced --ckpt ../_train/mamba/NAME.ckpt
    uv run python -m mamba.measure free   --ckpt ../_train/mamba/NAME.ckpt --seeds 1-16

THE INSTRUMENTS ARE NOT HERE: both halves stand in `ar_measure.py`, one thing across the
two step-frame eras. What is era five's is the two commands below, which name a model and
a player.
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
from mamba import model
from mamba.quantized import model as qmodel


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
@click.option("--temperature", default=qmodel.ELECTED_TEMPERATURE)
@click.option("--min-p", default=qmodel.ELECTED_MIN_P)
@click.option("--ring", default=model.ATTN_CONTEXT, help="the attention ring depth")
@click.option("--corpus", "corpus_path", default=str(corpus.FRAMES))
def free(ckpt, seeds, steps, temperature, min_p, ring, corpus_path):
    # THE DEFERRED IMPORT IS LOAD-BEARING: `mamba.infer` sets JAX_PLATFORMS=cpu at
    # import, and at the top level that would put `forced` on the CPU as well
    from mamba import infer

    seeds = cli.parse_seeds(None, None, seeds)
    # [infer.draw] and NOT [infer.sample], which is a click Command
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
