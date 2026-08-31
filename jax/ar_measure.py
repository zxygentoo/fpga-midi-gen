"""What a step-frame model predicts, and what it plays: the measure of eras four and five.

TWO WAYS TO ASK, AND THEY DISAGREE. That is the reason this module has two halves and
not one bag of numbers:

- FORCED reads what the model PREDICTS, on the corpus's own windows, with the true frame
  behind every step. It is the loss, cut where the loss hides things.
- FREE reads what the model PLAYS, on its own walks, with only what it drew behind it.

Measured 2026-08-21: under forcing, era four and era five were the same instrument -- the
hold each predicted agreed to a quarter point. On their own walks they part by four
standard errors. A walk visits the states it made; a forced pass visits the corpus's. Ask
both, and never read one as the other.

IT STANDS BESIDE `ar_train.py` AND FOR THE SAME REASON. Nothing below reads an era: the
forced half asks a trunk for `seat_nll`, which is `ar_model.Trunk`'s, and the free half
reads a walk that some era already drew. The two eras are one recipe and one measure;
each era's module keeps only its COMMANDS, which name a model and a player.

The instruments of the walk are the common battery's, in jax/measure.py, which also holds
the standing warning that NOTHING HERE RANKS A MODEL. This module reports two of them.
What did not earn its place is not reported: CADENCED -- the share of silences that arrive
somewhere -- carried a standard error of 6.5 points over thirty walks and never separated
two models; the windowed texture never caught a walk that degraded; the survivor count of
the draw read a median of one for every model ever measured. They were dropped 2026-08-22
rather than left to be misread.
"""

import statistics as st

import jax.numpy as jnp
import numpy as np

import ar_model
import corpus
import measure

# the referee cuts its eval rows at the window the models trained on
CONTEXT = ar_model.TRAINING_WINDOW


# ==================================================================== #
# FORCED — what the model predicts, on the corpus's own windows        #
# ==================================================================== #


def loss_row(held, corpus_path=str(corpus.FRAMES), limit=128, batch=16):
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
    splits = corpus.load_corpus(corpus_path)
    total = moved = 0.0
    steps = moves = 0
    seats = np.zeros(corpus.SEATS)
    for classes, phases in corpus.eval_batches(splits["valid"], CONTEXT, limit, batch):
        classes, phases = jnp.asarray(classes), jnp.asarray(phases)
        nll = held.seat_nll(classes, phases)
        by_step = jnp.sum(nll, axis=-1)
        moving = corpus.moving(classes) >= 2
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


# ==================================================================== #
# FREE — what the model plays, on its own walks                        #
# ==================================================================== #


def walk_row(classes):
    """The two instruments that ever decided anything, over one walk.

    HOLD is the share of voice slots that repeat the step before, against the corpus's
    78.17 percent on the canonical stream. It reads both failures a walk can have in one
    number: far above the corpus is a drone, far below is jitter. It is what showed that
    the elected draw did not transfer between the eras -- era five held 82.7 percent where
    era four held 80.8 at the same temperature, and needed T 1.2 to reach the corpus.

    ONSETS is the note-ons for each step, against the corpus's 0.81. The decode is the
    rule of the frame and lives in corpus.py, thus an onset means here what it means on
    the wire.

    Both are dense -- every step and every voice -- which is why they separate models
    where the sparse instruments could not.

    THE ARITHMETIC IS THE COMMON BATTERY'S and not a second copy of it. A walk is a stack
    of one sheet, thus [measure.battery_row] reads the same cells and returns the same two
    numbers; it computes ten more and this era reports none of them, because what was
    dropped above was dropped from the REPORT and not from the arithmetic."""
    row = measure.battery_row(np.asarray(classes)[None])
    return {name: row[name] for name in ("hold", "onsets")}


def mean_over_seeds(rows):
    """the mean of each instrument over several walks, and the standard error beside it

    THE ERROR IS NOT DECORATION. A single-seed reading did not survive a second seed once
    in this era: a texture gap of 3.7 steps at one seed read 0.3 at the next, and a
    conclusion was withdrawn for it. Two walks are the floor and sixteen is comfortable.
    """
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


def corpus_row(corpus_path=str(corpus.FRAMES)):
    """the same two numbers over stream zero of the train split: the row every other row
    is read against"""
    split = corpus.load_corpus(corpus_path)["train"]
    return walk_row(split.classes[: int(split.index[0, 1])])
