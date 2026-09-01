"""The draw of the masked sheet: the annealed schedule, the blocked Gibbs loop and the
draw of one cell.

The float walk of `diffusion/infer.py` and the integer walk of `diffusion/quantized.py`
take THIS loop, in this order, off the same seed; what parts them is arithmetic they hand
in. The sheet itself is `diffusion/model.py` -- `cell_order` states the order of every
uniform and `hidden_cells` the mask of a pass -- and `lib/diffusion/model.ml` holds both
under the same names.

The Gibbs schedule is the paper's section 5.2, citing Yao et al.: the annealed masking
probability, with the constants from the code release.

`sample.py` at the root is the era-free half of the draw, the tempered weight and the
cumulative pick that every era shares.
"""

import math
from typing import NamedTuple

import numpy as np

import prng
from diffusion import model
from sample import pick_share, tempered_weight

# THE ERA DRAWS WITH NO MIN-P FLOOR, where the step-frame eras hold one at 0.05. A Gibbs
# redraw picks one cell against a sheet that is still wrong around it, and a floor that
# trimmed its tail would harden the sheet it opened on.
MIN_P = 0.0

# the annealed masking probability of Yao et al., as the code release pins it:
# `YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7)`; the paper states the formula and no values
ANNEAL_PMIN = 0.1
ANNEAL_PMAX = 0.9
ANNEAL_ALPHA = 0.7


def anneal(n, total):
    """The masking probability at pass [n] of [total]: high at the opening, where the
    chain mixes fast, and settled on [ANNEAL_PMIN] after an [ANNEAL_ALPHA] share of the
    walk. [n] IS THE PASS AND [step] IS THE SIXTEENTH, everywhere in this era."""
    return max(
        ANNEAL_PMIN,
        ANNEAL_PMAX - (ANNEAL_PMAX - ANNEAL_PMIN) * n / (ANNEAL_ALPHA * total),
    )


def anneal_threshold(n, total):
    """`Model.anneal_threshold`: the masking threshold of pass [n] of [total], on the
    24-bit grid of the generator. A cell hides exactly when its word falls under it."""
    return math.floor(anneal(n, total) * 2.0**prng.UNIFORM_BITS)


def tempered_pick(raw, temperature, uniform):
    """the draw of one cell over the batch, row for row; [raw] is [sheets, ROWS]"""
    return pick_share(tempered_weight(raw, temperature, MIN_P), uniform)


class Pass(NamedTuple):
    """One pass over a batch of sheets: it reads a sheet, hides cells, states the logits
    of every cell, and redraws the hidden ones."""

    read: np.ndarray
    hidden: np.ndarray
    logits: np.ndarray
    redrawn: np.ndarray
    states: np.ndarray


def redrawn_sheet(states, sheet, hidden, logits, redraw):
    """[sheet] with every cell [hidden] hid drawn again, in the cell order, and the
    generator behind those draws. It writes a COPY: a pass states the sheet it read
    beside the sheet it left, thus neither may be the other."""
    redrawn = sheet.copy()
    for step, voice in model.cell_order(sheet.shape[1]):
        active = hidden[:, step, voice]
        if active.any():
            states, drawn = redraw(states, logits, step, voice, active)
            redrawn[active, step, voice] = drawn[active]
    return states, redrawn


def gibbs_passes(states, opening, *, passes, forward, redraw):
    """The one blocked-Gibbs loop BOTH walks of this era take, a `Pass` at a time.

    [states] is one generator for each sheet and [opening] the sheet the walk starts
    from; [passes] is N, the count the anneal is scaled on. [forward] takes
    (classes, hidden) and gives the logits of every cell; [redraw] takes
    (states, logits, step, voice, active) and gives (states, drawn), one class for each
    sheet the mask hid. The arithmetic is theirs -- what stands here is WHEN each is
    called and in what order.

    THE TWO WALKS MUST AGREE CELL FOR CELL, thus the order of the draws stands here and
    not twice, and A CELL NO SHEET HID TAKES NO UNIFORM -- [redraw] is never called for
    one, and a walk that spent one there would state a different piece with no gate to
    say so."""
    steps = opening.shape[1]
    sheet = opening
    for n in range(passes):
        states, hidden = model.hidden_cells(states, steps, anneal_threshold(n, passes))
        logits = forward(sheet, hidden)
        states, redrawn = redrawn_sheet(states, sheet, hidden, logits, redraw)
        yield Pass(sheet, hidden, logits, redrawn, states)
        sheet = redrawn
