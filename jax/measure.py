"""The instruments over a walk: what it looks like beside the corpus.

These are DIAGNOSTICS AGAINST A CORPUS REFERENCE AND NEVER RANKERS. Ten times in this
project a metric has ranked a model against the ear, and one of them was a draw
temperature: 0.6 tripled a repetition metric over six pooled seeds and the ear called the
result dull. Read every number beside the same number over the canonical packed stream,
use it to catch a pathology, and let the ear elect. None of this belongs in a loss.

This sits beside the models, as data.py does. A comparison between two models is only a
comparison if one instrument made both numbers, and a later model measures with this one.

Two questions, two groups of numbers:

- THE TEXTURE, over windows. The walk holds when the last window reads like the first and
  like the corpus. A number over a whole draw cannot answer it, because a good opening
  hides a bad end.
- THE ARRIVAL, over the whole walk. CADENCED is the share of silences that follow a
  sonority held CADENCE_HOLD steps or more, and it is the number aimed at the open
  question of docs/transformer_model.md. The corpus reads 99.2 percent: it never stops
  without arriving somewhere first. The model reads 67 to 73, and that is what the ear
  hears as fractured.
"""

import statistics as st

import numpy as np

import data

CADENCE_HOLD = 6  # the steps a sonority rings before its silence counts as an arrival
A_QUARTER = 4  # steps; the corpus median duration is a quarter note already


def windows(music, span):
    """[music] cut into blocks of [span] steps, each block measured.

    One element of [music] is one step and its events, as data.decode gives them. A note
    counts in the window it STARTS in, thus each note counts one time and in one place.
    The last block is short when [span] does not divide the walk, and its shares still
    divide by its own step count."""
    opened, durations = {}, []
    for step, events in enumerate(music):
        for kind, pitch in events:
            if kind == "on":
                opened[pitch] = step
            elif pitch in opened:
                start = opened.pop(pitch)
                durations.append((start, step - start))
    rows = []
    for index in range(0, len(music), span):
        block = music[index : index + span]
        ons = [sum(1 for k, _ in step if k == "on") for step in block]
        sounded = [float(d) for start, d in durations if index <= start < index + span]
        rows.append(
            {
                "onsets": sum(ons) / max(len(block), 1),
                "single_on": sum(1 for n in ons if n == 1) / max(len(block), 1),
                "median": float(np.median(sounded)) if sounded else float("nan"),
                "short": sum(1 for d in sounded if d < A_QUARTER) / max(len(sounded), 1),
            }
        )
    return rows


def silences(sounding):
    """every silence of a walk: the steps the sonority before it was held, and its length.

    [sounding] is one set of pitches for each step. A silence at the head of the walk has
    no sonority before it and gives no pair."""
    out, at = [], 0
    while at < len(sounding):
        if sounding[at]:
            at += 1
            continue
        end = at
        while end < len(sounding) and not sounding[end]:
            end += 1
        if at > 0:
            held, back = sounding[at - 1], at - 1
            while back >= 0 and sounding[back] == held:
                back -= 1
            out.append((at - 1 - back, end - at))
        at = end
    return out


def of_walk(classes):
    """one walk of frames against the corpus: the silence, its placement, and the texture"""
    sounding = [frozenset(int(c) for c in f if c != data.SILENCE) for f in classes]
    music = data.decode(classes)
    whole = windows(music, len(music))[0]
    gaps = silences(sounding)
    return {
        "silence": 100.0 * sum(1 for s in sounding if not s) / len(sounding),
        "cadenced": (
            100.0 * sum(1 for held, _ in gaps if held >= CADENCE_HOLD) / len(gaps)
            if gaps
            else float("nan")
        ),
        "gap": st.mean([length for _, length in gaps]) if gaps else float("nan"),
        "onsets": whole["onsets"],
        "median": whole["median"],
        "short": whole["short"],
        "four": 100.0 * sum(1 for s in sounding if len(s) == data.SEATS) / len(sounding),
    }


def of_canonical_stream(corpus_path):
    """the same numbers over stream zero of the train split: the row every other row is
    read against"""
    split = data.load_corpus(corpus_path)["train"]
    return split, of_walk(split.classes[: int(split.index[0, 1])])


def mean_of(rows, name):
    """the mean of one measure over several walks, skipping a walk that has none"""
    kept = [row[name] for row in rows if row[name] == row[name]]
    return st.mean(kept) if kept else float("nan")


def window_line(label, index, row):
    return (
        f"{label} {index:3d}  onsets/step {row['onsets']:.2f}   "
        f"single-ON {row['single_on']:.2f}   median dur {row['median']:.1f}   "
        f"under a quarter {row['short']:.2f}"
    )


def walk_line(label, row):
    return (
        f"{label:<18}  silence {row['silence']:5.2f}%   "
        f"cadenced {row['cadenced']:5.1f}%   gap {row['gap']:4.1f}   "
        f"onsets {row['onsets']:.2f}   median {row['median']:4.1f}   "
        f"4-voice {row['four']:4.1f}%"
    )
