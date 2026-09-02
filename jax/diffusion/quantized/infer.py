"""The walk of the masked sheet: the Gibbs passes the board runs, and their draw.

`quantized/model.py` holds the weights, the formats and the contract file; this half runs
them. THE ORDER OF OPERATIONS IS THE CONTRACT, and `tests/test_rtl_diffusion.py` holds the
circuit to these integers write for write.

EVERY UNIFORM OF A WALK COMES FROM THE SHARED GENERATOR, in the consumption order
`docs/diffusion_rtl.md` states: the opening, then each pass's masks, then each pass's
redraws, one seed for one SHEET. That order is what gives the era its seed handoff, thus a
draw inserted or skipped anywhere here parts the board from this side for ever.

The loop, the schedule and the order of the cells stand in `diffusion/sample.py`, once for
the float walk and this one. What is here is this walk's arithmetic and the record the
drift report reads.
"""

from typing import NamedTuple

import numpy as np

import prng
import quantized as q
from diffusion import model as sheet
from diffusion import sample
from diffusion.quantized import model as qmodel


def class_weights(twin, raw):
    """The Q15 weight of every class of one cell, over the batch. The logits carry the
    activation Q and the exp2 unit reads Q12, thus the difference SHIFTS UP FIRST:
    unshifted, every weight stands within a fraction of a nat of the peak and the draw
    is uniform. It is not `ar_quantized.tempered_weights`, which reads Q12 already and
    holds a min-p floor."""
    raw = np.asarray(raw, np.int64)
    peak = raw.max(axis=-1, keepdims=True)
    shifted = (raw - peak) << (q.EXP2_IN_Q - qmodel.ACTIVATION_Q)
    return q.exp2_q(q.apply_scale(twin.temper.q_value, twin.temper.q, shifted))


# the walk


class Draw(NamedTuple):
    """One redraw of a pass, over the batch. `hidden` holds the walks the mask hid this
    cell for; THE OTHER WALKS STATE NOTHING -- they consumed no uniform, thus `word` and
    `drawn` carry whatever the batched arithmetic happened to compute."""

    step: int
    voice: int
    hidden: np.ndarray  # [sheets]
    word: np.ndarray  # [sheets], the 24-bit uniform
    drawn: np.ndarray  # [sheets], the class


class Pass(NamedTuple):
    """`sample.Pass` with the `Draw` of every cell of the order beside it."""

    read: np.ndarray  # [sheets, steps, VOICES]
    hidden: np.ndarray  # [sheets, steps, VOICES]
    logits: np.ndarray  # [sheets, steps, ROWS, VOICES], the integer logits
    draws: list  # Draw, in the cell order
    redrawn: np.ndarray  # [sheets, steps, VOICES]
    states: np.ndarray  # [sheets], the generator behind the redraws


def passes(twin, states, given, *, walk):
    """The INTEGER walk of the era, one pass at a time: `sample.gibbs_passes` in the
    arithmetic of the board, with the record the drift report reads.

    The loop, the schedule and the order of the draws stand in `gibbs_passes`, once for
    both walks. What is here is this walk's arithmetic and the `Draw` of every cell of the
    ORDER: a cell nothing hid states an idle record. `given` is the opening, handed over
    rather than drawn here so that one walk cannot open on a sheet the other could not.

    THE CELL LOOPS ARE NOT THE COST OF A WALK, and a round that scans them will find
    that out late: profiled at T128 N512 on the golden shape, the forward is 81.6
    percent of a one-sheet walk and the loops are batched numpy that falls to 1.7
    percent at sixteen."""
    sheets, steps, _ = given.shape
    idle = np.zeros(sheets, np.int64)
    spent = {}

    def forward(classes, hidden):
        return twin(classes, hidden)

    def redraw(states, logits, step, voice, active):
        states, word = prng.uniform_word(states, active)
        drawn = q.pick(class_weights(twin, logits[:, step, :, voice]), word)
        spent[step, voice] = (word, drawn)
        return states, drawn

    for taken in sample.gibbs_passes(
        states, given, passes=walk, forward=forward, redraw=redraw
    ):
        draws = [
            Draw(
                step,
                voice,
                taken.hidden[:, step, voice],
                *spent.get((step, voice), (idle, idle)),
            )
            for step, voice in sheet.cell_order(steps)
        ]
        spent.clear()
        yield Pass(
            taken.read, taken.hidden, taken.logits, draws, taken.redrawn, taken.states
        )


def gibbs(twin, states, given, *, walk):
    """The whole walk: `infer.gibbs` in the arithmetic of the board, giving the sheets and
    the generator behind them so a caller can hold the two side by side. A walk of no
    passes is the opening."""
    classes = given
    for taken in passes(twin, states, given, walk=walk):
        classes, states = taken.redrawn, taken.states
    return classes, states
