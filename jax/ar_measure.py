"""What a step-frame model predicts, and what it plays: the measure of eras four and five.

TWO WAYS TO ASK, AND THEY DISAGREE:

- FORCED reads what the model PREDICTS, on the corpus's own windows, with the true frame
  behind every step. It is the loss, cut where the loss hides things.
- FREE reads what the model PLAYS, on its own walks, with only what it drew behind it.

Under forcing eras four and five were the same instrument; on their own walks they part by
four standard errors, because a walk visits the states it made. Ask both, and never read
one as the other. docs/mamba.md holds the round.

It stands beside `ar_train.py` and for the same reason: nothing below reads an era. Each
era's module keeps only its COMMANDS, which name a model and a player.
"""

import statistics as st

import jax.numpy as jnp
import numpy as np

import ar_model
import corpus
import measure

# the referee cuts its eval rows at the window the models trained on
CONTEXT = ar_model.TRAINING_WINDOW


# FORCED — what the model predicts, on the corpus's own windows


def moving(classes):
    """[batch, length + 1, SEATS] -> [batch, length] the count of voices that move into
    each predicted step.

    77.91 percent of the voice slots repeat the step before: they dominate the mean and
    invite a model that holds its chord for ever. The second number of the report divides
    over the steps where two or more voices move, which is where the music is."""
    return (classes[:, 1:] != classes[:, :-1]).sum(axis=-1)


def loss_row(held, corpus_path=str(corpus.FRAMES), limit=128, batch=16):
    """The loss cut three ways, and the four seats.

    77.91 percent of the voice slots repeat the step before, thus the mean can hide a
    model's whole deficit in the quarter of the steps that carry the music. MOVING is the
    steps where two voices or more move and STILL is the complement. Era five's whole
    story is in the pair: it matched era four on the still steps and lost 0.075 nats on
    the moving ones.

    Everything is a sum over a count, never a mean of means -- the last eval batch is
    short and would otherwise weigh its rows above the rest."""
    splits = corpus.load_corpus(corpus_path)
    total = moved = 0.0
    steps = moves = 0
    seats = np.zeros(corpus.SEATS)
    for classes, phases in corpus.eval_batches(splits["valid"], CONTEXT, limit, batch):
        classes, phases = jnp.asarray(classes), jnp.asarray(phases)
        nll = held.seat_nll(classes, phases)
        by_step = jnp.sum(nll, axis=-1)
        moved_in = moving(classes) >= 2
        total += float(jnp.sum(by_step))
        moved += float(jnp.sum(jnp.where(moved_in, by_step, 0.0)))
        moves += int(jnp.sum(moved_in))
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
        (
            f"{label:<22} loss {row['loss']:.4f}   moving {row['moving']:.4f}   "
            f"still {row['still']:.4f}   "
            f"({row['moves']} of {row['steps']} steps move two voices or more)"
        ),
        f"{'':<22} "
        + "   ".join(
            f"{name} {row['seats'][at]:.4f}"
            for at, name in enumerate(measure.VOICE_NAMES)
        ),
    ]


# FREE — what the model plays, on its own walks


def walk_row(classes):
    """The two instruments that ever decided anything, over one walk.

    HOLD is the share of voice slots that repeat the step before, against the corpus's
    78.17 percent; far above is a drone, far below is jitter. ONSETS is the note-ons for
    each step, against the corpus's 0.81. Both are dense -- every step and every voice --
    which is why they separate models where the sparse instruments could not.

    The arithmetic is the common battery's. [measure.battery_row] computes ten more
    numbers and this era reports none of them: they were dropped from the REPORT and not
    from the arithmetic."""
    row = measure.battery_row(np.asarray(classes)[None])
    return {name: row[name] for name in ("hold", "onsets")}


def corpus_row(corpus_path=str(corpus.FRAMES)):
    """the same two numbers over stream zero of the train split: the row every other row
    is read against"""
    split = corpus.load_corpus(corpus_path)["train"]
    return walk_row(split.classes[: int(split.index[0, 1])])


def mean_over_seeds(rows):
    """The mean of each instrument over several walks, and the standard error beside it.

    THE ERROR IS NOT DECORATION: a single-seed reading did not survive a second seed once
    in this era. Two walks are the floor and sixteen is comfortable."""
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
    """one walk, or the mean of several with its error where [mean_over_seeds] made it"""

    def show(name):
        value = row[name]
        return (
            f"{value[0]:6.2f} +- {value[1]:4.2f}"
            if isinstance(value, tuple)
            else f"{value:6.2f}"
        )

    return f"{label:<22} hold {show('hold')}%   onsets {show('onsets')}"