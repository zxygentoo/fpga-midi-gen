"""What a model does, measured against the corpus.

TWO WAYS TO ASK, AND THEY DISAGREE. That is the reason this module has two halves and
not one bag of numbers:

- FORCED reads what the model PREDICTS, on the corpus's own windows, with the true frame
  behind every step. It is the loss, cut where the loss hides things.
- FREE reads what the model PLAYS, on its own walks, with only what it drew behind it.

Measured 2026-08-21: under forcing, era four and era five were the same instrument -- the
hold each predicted agreed to a quarter point. On their own walks they part by four
standard errors. A walk visits the states it made; a forced pass visits the corpus's. Ask
both, and never read one as the other.

NOTHING HERE RANKS A MODEL. Ten times in this project a metric has ranked a model against
the ear, and this era added two more: the attention block that reads null on every number
is the one the ear elected, and the draw the numbers matched to the corpus is the one the
ear rejected. Read these beside the corpus row to catch a pathology, and let the ear
elect.

What is here has earned its place; what did not is gone. CADENCED -- the share of
silences that arrive somewhere -- carried a standard error of 6.5 points over thirty
walks and never separated two models; the windowed texture never caught a walk that
degraded; the survivor count of the draw read a median of one for every model ever
measured. They were removed 2026-08-22 rather than left to be misread.

    uv run python -m measure forced --ckpt ../_train/mamba/NAME.ckpt
    uv run python -m measure free   --ckpt ../_train/mamba/NAME.ckpt --seeds 1-16
"""

import statistics as st
from pathlib import Path

import click
import jax.numpy as jnp
import numpy as np

import data
from mamba import model

JAX_ROOT = Path(__file__).resolve().parent
CORPUS = str(JAX_ROOT / "_data" / "frames.safetensors")
CONTEXT = 256  # the training window, thus the window the referee's eval rows cut at
# seat 0 is the bass and seat 3 the soprano, as the chained head reads them
VOICES = ("bass", "tenor", "alto", "soprano")


# ==================================================================== #
# FORCED — what the model predicts, on the corpus's own windows        #
# ==================================================================== #


def losses(params, corpus_path=CORPUS, limit=128, batch=16):
    """The loss cut three ways, and the four seats.

    77.91 percent of the voice slots repeat the step before. They are easy, they dominate
    the mean, and a model that holds its chord for ever scores well on them and plays a
    drone -- thus the mean can hide a model's whole deficit in the quarter of the steps
    that carry the music. MOVING is the steps where two voices or more move; STILL is the
    rest, and it is the complement and not a third measurement.

    Era five's whole story is in the pair: it matched era four on the still steps to the
    fourth decimal and lost 0.075 nats on the moving ones.

    Everything is a sum over a count, never a mean of means -- the last batch of the eval
    rows is short and would otherwise weigh its rows above the rest."""
    corpus = data.load_corpus(corpus_path)
    total = moved = 0.0
    steps = moves = 0
    seats = np.zeros(data.SEATS)
    for classes, phases in data.eval_batches(corpus["valid"], CONTEXT, limit, batch):
        classes, phases = jnp.asarray(classes), jnp.asarray(phases)
        nll = model.seat_nll(params, classes, phases)
        by_step = jnp.sum(nll, axis=-1)
        moving = data.moving(classes) >= 2
        total += float(jnp.sum(by_step))
        moved += float(jnp.sum(jnp.where(moving, by_step, 0.0)))
        moves += int(jnp.sum(moving))
        steps += int(jnp.size(by_step))
        seats += np.asarray(jnp.sum(nll, axis=(0, 1)))
    return {
        "loss": total / steps,
        "moving": moved / max(moves, 1),
        "still": (total - moved) / max(steps - moves, 1),
        "seats": seats / steps,
        "moves": moves,
        "steps": steps,
    }


def loss_lines(label, row):
    return [
        f"{label:<22} loss {row['loss']:.4f}   moving {row['moving']:.4f}   "
        f"still {row['still']:.4f}   "
        f"({row['moves']} of {row['steps']} steps move two voices or more)",
        f"{'':<22} "
        + "   ".join(f"{n} {row['seats'][at]:.4f}" for at, n in enumerate(VOICES)),
    ]


# ==================================================================== #
# FREE — what the model plays, on its own walks                        #
# ==================================================================== #


def of_walk(classes, music=None):
    """The two instruments that ever decided anything, over one walk.

    HOLD is the share of voice slots that repeat the step before, against the corpus's
    78.17 percent on the canonical stream. It reads both failures a walk can have in one
    number: far above the corpus is a drone, far below is jitter. It is what showed that
    the elected draw did not transfer between the eras -- era five held 82.7 percent where
    era four held 80.8 at the same temperature, and needed T 1.2 to reach the corpus.

    ONSETS is the note-ons for each step, against the corpus's 0.81. The decode is the
    rule of the frame and lives in data.py, thus an onset means here what it means on the
    wire; a caller that already holds the decode passes it in.

    Both are dense -- every step and every voice -- which is why they separate models
    where the sparse instruments could not."""
    frames = np.asarray(classes)
    music = data.decode(frames) if music is None else music
    ons = sum(1 for step in music for kind, _ in step if kind == "on")
    return {
        "hold": 100.0 * float(np.mean(frames[1:] == frames[:-1])),
        "onsets": ons / len(music),
    }


def over_seeds(rows):
    """the mean of each instrument over several walks, and the standard error beside it

    THE ERROR IS NOT DECORATION. A single-seed reading did not survive a second seed once
    in this era: a texture gap of 3.7 steps at one seed read 0.3 at the next, and a
    conclusion was withdrawn for it. Two walks are the floor and sixteen is comfortable."""
    return {
        name: (
            st.mean([row[name] for row in rows]),
            st.stdev([row[name] for row in rows]) / len(rows) ** 0.5
            if len(rows) > 1
            else float("nan"),
        )
        for name in rows[0]
    }


def walk_line(label, row):
    """one walk, or the mean of several with its error where [over_seeds] made it"""

    def show(name):
        value = row[name]
        return (
            f"{value[0]:6.2f} +- {value[1]:4.2f}"
            if isinstance(value, tuple)
            else f"{value:6.2f}"
        )

    return f"{label:<22} hold {show('hold')}%   onsets {show('onsets')}"


def corpus_row(corpus_path=CORPUS):
    """the same two numbers over stream zero of the train split: the row every other row
    is read against"""
    split = data.load_corpus(corpus_path)["train"]
    return of_walk(split.classes[: int(split.index[0, 1])])


# ==================================================================== #
# The two commands                                                     #
# ==================================================================== #


@click.group(help=__doc__)
def main():
    pass


@main.command(help=losses.__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--corpus", "corpus_path", default=CORPUS)
def forced(ckpt, corpus_path):
    row = losses(model.load_params(ckpt), corpus_path)
    for line in loss_lines(Path(ckpt).stem[:22], row):
        click.echo(line)


@main.command(help=of_walk.__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--seeds", default="1-16", help="a list, or LOW-HIGH")
@click.option("--steps", default=512)
@click.option("--temperature", default=1.0)
@click.option("--min-p", default=0.05)
@click.option("--ring", default=model.ATTN_CONTEXT, help="the attention ring depth")
@click.option("--corpus", "corpus_path", default=CORPUS)
def free(ckpt, seeds, steps, temperature, min_p, ring, corpus_path):
    from mamba import infer

    seeds = infer.parse_seeds(None, None, seeds)
    walks = infer.sample(
        model.load_params(ckpt),
        seeds=seeds,
        steps=steps,
        temperature=temperature,
        min_p=min_p,
        ring=ring,
    )
    click.echo(walk_line("the packed corpus", corpus_row(corpus_path)))
    rows = [of_walk(walk) for walk in walks]
    click.echo(
        walk_line(f"T {temperature} min-p {min_p}, {len(rows)} walks", over_seeds(rows))
    )


if __name__ == "__main__":
    main()
