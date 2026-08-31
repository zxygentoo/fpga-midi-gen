"""The instruments, over a stack of sheets of class indices.

This is the COMMON HOME of the measurement, as jax/sample.py is of the draw. Everything
here is arithmetic over a [sheets, steps, SEATS] array of class indices and nothing here
knows which era drew it -- a Gibbs sheet, a walk of the packed stream and a corpus crop
all read the same way, and a single walk is a stack of one. What a RECIPE measures with a
model lives beside that recipe: jax/ar_measure.py holds the forced pass and the walk of
the two step-frame eras, and jax/diffusion/measure.py holds the paper's Algorithm 1.

ONE INSTRUMENT HERE READS LOGITS AND NOT CLASSES: the drift count at the end scores a
twin's draw against the float model's on the very uniform the twin took -- what the
quantization costs, which both step-frame twins report. It reads two rows and a draw, and
no model of its own.

A ROW IS A DICT OF INSTRUMENTS AND A LINE IS WHAT PRINTS IT, and every measure module of
this tree spells it the same way: `battery_row` and `battery_lines` here, `walk_row`,
`loss_row`, `corpus_row`, `walk_line` and `loss_lines` in `ar_measure.py`, `tail_row` and
`tail_line` in `diffusion/measure.py`. A function that computes one ends in `_row`; a
function that renders one ends in `_line` or `_lines`.

THE CORPUS ROW IS THE REFEREE OF EVERY NUMBER. A triad share of 40 percent says nothing
until the corpus row beside it says 64, and three faults of the parallel instrument were
found by reading it against the corpus and never against another model.

NOTHING HERE RANKS A MODEL. Ten times in this project a metric has ranked a model against
the ear, and the sheet era added more: the instrument the ear elected reads null on the
mean, and the draw whose numbers matched the corpus is the one the ear rejected. Read
these beside the corpus row to catch a pathology, and let the ear elect.

Of the battery, the ear has spoken for two and against three. Read parallel motion, the
voice pairs and the likelihood; report hold, the onsets and register order to catch a
pathology and elect on neither.
"""

import itertools
from typing import NamedTuple

import numpy as np

import corpus
import prng
import sample

# The intervals a chorale treats as a dissonance, in semitones and modulo the octave: the
# two seconds, the tritone and the two sevenths. The perfect fourth is left out -- it is a
# dissonance against the bass and a consonance between the upper voices, and a measurement
# that cannot tell them apart should not name it. This set puts the corpus at 10.3
# percent, which is the number the proto round recorded.
DISSONANT = (1, 2, 6, 10, 11)
# The triad qualities the battery counts: major and minor, which puts the corpus at 63.9
# percent of the steps that carry three voices. Diminished is left out on the proto
# round's method; adding it moves the corpus row to 67.9, thus it is a labelling choice
# and not a finding either way.
TRIADS = ((0, 4, 7), (0, 3, 7))
# THE TAIL OF THE HARMONY. A frame holding three dissonant pairs or more is rare in the
# corpus -- 3.2 percent of its frames -- where TWO is 20.6 percent and merely a seventh
# chord. A mean can therefore hold while these multiply, and it is the mean that the ear
# disagrees with: one strange chord in a phrase is heard, and it moves the average of 128
# frames by nothing.
CLASH = 3
# seat 0 is the bass and seat 3 the soprano, as the packed stream and the chained head of
# the earlier eras both read them
VOICE_NAMES = ("bass", "tenor", "alto", "soprano")
# The pairs of voices, in the order of the seats between them: the three neighbours, then
# the two that skip a voice, then the outer pair. That order is nearly the order of their
# span in the corpus, thus [voice_pairs] reads left to right as the pitch reach runs out.
PAIRS = tuple(
    sorted(
        itertools.combinations(range(corpus.SEATS), 2), key=lambda p: (p[1] - p[0], p[0])
    )
)


def triad_table():
    """A row for each of the 4096 sets of pitch classes: does this set fit inside some
    triad?

    A step holds at most four voices and therefore at most four pitch classes, thus its
    set is one 12-bit word and the whole question is a table lookup. The table is built
    one time and it makes the battery a vector operation instead of a loop over steps."""
    fits = np.zeros(1 << 12, dtype=bool)
    for quality in TRIADS:
        for root in range(12):
            chord = sum(1 << ((step + root) % 12) for step in quality)
            # every subset of a triad fits inside it, and a set of one or two pitch
            # classes is exactly the case the denominator of [battery_row] excludes
            fits[[at for at in range(1 << 12) if at & ~chord == 0]] = True
    return fits


FITS_A_TRIAD = triad_table()


def pitch_class_words(classes):
    """[sheets, steps, VOICES] -> [sheets, steps] the sounding pitch classes of each
    step as one 12-bit word; a rest sets no bit and a unison sets one bit twice"""
    pitches = corpus.pitches_of_classes(classes) % 12
    bits = np.where(classes != corpus.SILENCE, 1 << pitches, 0)
    return np.bitwise_or.reduce(bits, axis=-1)


def apply_or_zero(of, numbers):
    """[of] applied to [numbers], or ZERO where nothing was heard.

    EVERY INSTRUMENT OF THIS BATTERY IS CONDITIONAL and the condition is the same one: a
    mean over the steps that carry three voices, a share over the pairs that sound, a
    spread over the pitches one seat sang. A sheet that carries none of them has no
    answer, and the battery reads 0.0 rather than nan -- a nan poisons a mean of several
    sheets and prints as a hole, where a zero prints as the finding it is: nothing there.

    THE CORPUS ROW IS WHAT SAYS WHICH ZERO IS WHICH. A zero triad share beside a corpus
    row of 64 is a thin sheet, and no number here means anything without it."""
    return float(of(numbers)) if len(numbers) else 0.0


def dissonant_share(intervals):
    """the share of [intervals] that this corpus treats as a dissonance"""
    return np.isin(intervals, DISSONANT).mean()


def voice_pairs(spans, intervals, pairs_sound):
    """Each pair of voices: how far apart it sits in semitones, and how often it sounds a
    dissonance.

    THIS IS THE INSTRUMENT THAT EXPLAINS THE OTHERS, and the trunk is why. Three-by-three
    convolutions with no pooling reach exactly L rows up the pitch axis at L layers, and a
    voice's pitch lives at its own row of the roll -- thus a pair that sits further apart
    than the reach cannot be seen at all, at any width.

    Measured 2026-08-24 over the three board rungs, the excess dissonance of a pair is a
    function of its span and of nothing else. At L 16 every pair inside 16 semitones reads
    the corpus, and the bass against the soprano, which spans 19.6, reads 16 points over
    it; L 24 takes that pair from 24.6 percent to 16.6. Depth is the reach and width is
    the resolution.

    Read it against the corpus row, as everything here is read. A pair can be too
    consonant as well as too dissonant, and the narrow rungs are both."""
    rows = []
    for at, (low, high) in enumerate(PAIRS):
        sounds = pairs_sound[..., at]
        rows.append(
            {
                "name": f"{VOICE_NAMES[low][:2]}-{VOICE_NAMES[high][:2]}",
                "span": apply_or_zero(np.mean, spans[..., at][sounds]),
                "dissonant": 100.0
                * apply_or_zero(dissonant_share, intervals[..., at][sounds]),
            }
        )
    return rows


def parallel_motion(classes, pitches, sounding):
    """Parallel fifths and octaves, for each thousand pairs that live across a step.

    THE FAULT THAT LIVES BETWEEN FRAMES, and the only instrument here that reads across
    time at all. Two voices a fifth or an octave apart, both moving, landing on the same
    interval. It is the most audible error in four-part writing, because an octave
    collapses two voices into one and the texture thins where nothing else changed.

    Measured 2026-08-25, it is what separates this era's models from the corpus where
    every vertical instrument says they have arrived. Bach writes 0.26 fifths and 0.10
    octaves for each thousand; the board rung writes 2.0 and 1.9, and the paper's own size
    0.8 and 1.3. More Gibbs passes take it down two or three times and then stop -- both
    sizes saturate, one by N 256 and one by N 512 -- and sixty-two times the parameters
    halves it and stops. NEITHER REACHES THE CORPUS.

    The reason is structural, and it is the finding of the round. The loss scores the
    MARGINAL of each cell under its context, and the walk draws those marginals
    INDEPENDENTLY; a joint configuration of two voices across two steps is therefore
    evaluated by neither. A term that stands in no objective cannot be sampled away.

    TWO CORRECTIONS OF 2026-08-25, both found by reading the corpus row and not the
    models.

    THE MOTION MUST BE SIMILAR. A parallel is two voices moving THE SAME WAY from a
    perfect interval to the same perfect interval. The first version asked only that the
    interval class stand before and after, thus contrary motion onto a fifth -- which is
    how a fifth is correctly approached -- counted as a fault. It read 53 percent of the
    corpus's own fifths that way, 10 events of 19. The octaves were untouched, 4 of 4
    already similar.

    THE PAIR MUST KEEP ITS ORDER. The gap was an absolute value, thus a pair that crossed
    could hold its interval class while its interval turned upside down. 3.7 percent of
    the corpus's moving pairs swap order across a step.

    THE DIVISOR IS THE PAIRS THAT MOVE, and it was the pairs that merely SOUND until
    2026-08-25. A parallel needs both voices to move, thus a model that holds its notes
    writes fewer of them for no musical reason at all, and the earlier divisor paid it for
    that. The span round caught this: it took the old number from 1.87 to 0.89 while its
    onsets fell from 0.89 for each step to 0.69 and its held cells rose above the corpus.
    [moving] is reported beside the rates for the same reason -- it is the term that was
    hiding, and a rung whose motion has left the corpus is not to be read on the rates
    alone."""
    moved = (classes[:, 1:] != classes[:, :-1]) & sounding[:, 1:] & sounding[:, :-1]
    fifths = octaves = moving = alive = 0
    for low, high in PAIRS:
        held = (
            sounding[:, :-1, low]
            & sounding[:, :-1, high]
            & sounding[:, 1:, low]
            & sounding[:, 1:, high]
        )
        both = moved[..., low] & moved[..., high] & held
        gap_before = pitches[:, :-1, high] - pitches[:, :-1, low]
        gap_after = pitches[:, 1:, high] - pitches[:, 1:, low]
        # SIMILAR MOTION is what makes a parallel a parallel. Contrary motion that lands
        # on a fifth is how a fifth is correctly approached, and counting it read 53
        # percent of the corpus's own fifths as faults.
        together = np.sign(pitches[:, 1:, low] - pitches[:, :-1, low]) == np.sign(
            pitches[:, 1:, high] - pitches[:, :-1, high]
        )
        # and the pair must keep its order. A fifth whose voices cross becomes a fourth,
        # thus the interval did not hold, and the absolute gap cannot see that. A unison
        # has no side, thus a zero on either end keeps its place.
        straight = np.sign(gap_before) * np.sign(gap_after) >= 0
        parallel = both & together & straight
        before = np.abs(gap_before) % 12
        after = np.abs(gap_after) % 12
        fifths += int((parallel & (before == 7) & (after == 7)).sum())
        octaves += int((parallel & (before == 0) & (after == 0)).sum())
        moving += int(both.sum())
        alive += int(held.sum())
    return {
        "fifths": 1000.0 * fifths / max(moving, 1),
        "octaves": 1000.0 * octaves / max(moving, 1),
        "moving": 100.0 * moving / max(alive, 1),
    }


def tessitura(pitches, sounding):
    """Where each voice SITS, and how often it leaves the register of its seat.

    THE THING NOTHING ELSE HERE COVERS. [voice_pairs] reads DIFFERENCES of pitch and the
    order instrument reads the stacking, thus a texture that slid four semitones as a body
    would move neither of them. The trunk invites exactly that: a three-by-three
    convolution over the pitch axis is EQUIVARIANT in pitch, so a cell knows where it
    stands only when its reach touches an edge of the roll, and at L 16 a cell in the
    middle of 48 rows touches neither. Register error should therefore fall with DEPTH and
    settle at L 48, where every cell reaches both edges.

    The seats overlap by 14 to 18 semitones, which is why the order instrument is weak and
    reads 97 to 99 percent everywhere: a voice can stand in good order and still sing in
    the wrong part of its range.

    The mean says a voice has DRIFTED and the spread says it RANGES too widely; the two
    are different faults and a single number would confuse them. [outside] is the tail --
    the share of sounding cells beyond their own seat's [corpus.VOICE_RANGES] -- and the
    corpus reads zero on it by construction, as the spare row does."""
    seats = []
    outside = alive = 0
    for seat, (low, high) in enumerate(corpus.VOICE_RANGES):
        heard = pitches[..., seat][sounding[..., seat]]
        seats.append(
            {
                "name": VOICE_NAMES[seat],
                "mean": apply_or_zero(np.mean, heard),
                "spread": apply_or_zero(np.std, heard),
            }
        )
        outside += int(((heard < low) | (heard > high)).sum())
        alive += len(heard)
    return {"seats": seats, "outside": 100.0 * outside / max(alive, 1)}


def triad_share(words):
    """the share of pitch-class [words] that fit inside a major or a minor triad"""
    return FITS_A_TRIAD[words].mean()


def clash_share(clashes):
    """the share of steps carrying [CLASH] dissonant pairs or more"""
    return (clashes >= CLASH).mean()


def battery_row(classes):
    """The battery over a set of sheets: [sheets, steps, VOICES] of class indices.

    HOLD is the share of voice slots that repeat the step before. It reads both failures a
    sheet can have in one number: far above the corpus is a drone, far below is jitter.

    ONSETS is the note-ons for each step under the decode of corpus.py, thus an onset
    means here what it means on the wire.

    VOICES is the share of steps carrying 0, 1, 2, 3 and 4 sounding voices. The corpus
    sings all four at 99.8 percent of its steps, thus this row alone catches the thin
    sheet that was the open defect of the proto round.

    TRIADS is the share of steps that fit inside a major or minor triad, over the steps
    that carry THREE VOICES OR MORE. DISSONANT is the share of sounding pairs at a
    dissonant interval. ORDER is the share of steps whose sounding voices stand in
    register order, the bass lowest.

    SPARE is the share of cells drawn on the spare row of the vocabulary, which the corpus
    never sings. It is a smoke number: anything above zero says the model puts mass where
    no music is."""
    sounding = classes != corpus.SILENCE
    voices = sounding.sum(axis=-1)
    words = pitch_class_words(classes)
    pitches = corpus.pitches_of_classes(classes)
    thick = voices >= 3
    pairs_sound = np.stack([sounding[..., a] & sounding[..., b] for a, b in PAIRS], -1)
    spans = np.stack([np.abs(pitches[..., a] - pitches[..., b]) for a, b in PAIRS], -1)
    intervals = spans % 12
    ordered = np.stack([pitches[..., a] <= pitches[..., b] for a, b in PAIRS], -1)
    clashes = (np.isin(intervals, DISSONANT) & pairs_sound).sum(axis=-1)
    music = [corpus.decode(sheet) for sheet in classes]
    ons = sum(1 for piece in music for step in piece for kind, _ in step if kind == "on")
    return {
        "hold": 100.0 * float(np.mean(classes[:, 1:] == classes[:, :-1])),
        "onsets": ons / sum(len(piece) for piece in music),
        "voices": [100.0 * float(np.mean(voices == count)) for count in range(5)],
        "triads": 100.0 * apply_or_zero(triad_share, words[thick]),
        "dissonant": 100.0
        * float(
            np.sum(np.isin(intervals, DISSONANT) & pairs_sound)
            / max(pairs_sound.sum(), 1)
        ),
        "order": 100.0
        * apply_or_zero(np.mean, np.all(ordered | ~pairs_sound, axis=-1)[voices >= 2]),
        "clash": 100.0 * apply_or_zero(clash_share, clashes[sounding.any(axis=-1)]),
        "spare": 100.0 * float(np.mean(classes == corpus.CLASSES - 1)),
        "parallels": parallel_motion(classes, pitches, sounding),
        "register": tessitura(pitches, sounding),
        "pairs": voice_pairs(spans, intervals, pairs_sound),
    }


def battery_lines(label, row):
    """one measurement as two lines; print the corpus row above every other row"""
    instruments = (
        f"{label:<22} hold {row['hold']:5.1f}%   onsets {row['onsets']:4.2f}   "
        f"triads {row['triads']:5.1f}%   dissonant {row['dissonant']:5.1f}%   "
        f"clash {row['clash']:4.1f}%   order {row['order']:5.1f}%"
    )
    counts = "  ".join(
        f"{count} {share:5.1f}%" for count, share in enumerate(row["voices"])
    )
    pairs = [
        f"{pair['name']} {pair['span']:4.1f}st {pair['dissonant']:4.1f}%"
        for pair in row["pairs"]
    ]
    parallels = row["parallels"]
    motion = (
        f"{'':<22} parallel 5ths {parallels['fifths']:5.2f}   "
        f"octaves {parallels['octaves']:5.2f}   (each 1000 pairs that MOVE together; "
        f"{parallels['moving']:4.1f}% of the pairs move)"
    )
    register = row["register"]
    seats = "   ".join(
        f"{seat['name'][:2]} {seat['mean']:4.1f}+-{seat['spread']:3.1f}"
        for seat in register["seats"]
    )
    where = f"{'':<22} register {seats}   outside {register['outside']:4.2f}%"
    half = len(pairs) // 2
    return [
        instruments,
        f"{'':<22} voices sounding {counts}   spare {row['spare']:.3f}%",
        where,
        motion,
        f"{'':<22} " + "   ".join(pairs[:half]),
        f"{'':<22} " + "   ".join(pairs[half:]),
    ]


# The drift: the twin's draw against the float model's, on the one uniform the twin took.
# It is what the quantization costs, and both step-frame twins report it through these.


def cosines(here, there):
    """the cosine of each integer row against the float row of the same place, over a
    batch of [rows, classes]"""
    here, there = np.asarray(here, np.float64), np.asarray(there, np.float64)
    return (here * there).sum(axis=-1) / np.sqrt(
        (here * here).sum(axis=-1) * (there * there).sum(axis=-1)
    )


class Counted(NamedTuple):
    """what a drift report has counted over the draws it has seen"""

    draws: int = 0
    same_peak: int = 0
    same_draw: int = 0
    cosine: float = 0.0


def count_draws(counted, here, there, *, drawn, uniform, temperature, min_p):
    """A BATCH of the twin's rows against the float rows of the same places, on the very
    uniforms the twin took.

    [here] and [there] are [rows, classes] -- the twin's integer logits and the float
    model's, one row for each draw counted. [drawn] is the class the twin picked and
    [uniform] the [0, 1) draw it picked on.

    The float draw runs at the policy the twin was quantized under -- the caller states
    it, because the elected numbers are the twin's and not this instrument's.

    IT IS BATCHED because era six redraws a whole sheet at a time and a step-frame chain
    redraws four seats; the arithmetic is one arithmetic and the batch is what parts
    them."""
    here, there = np.asarray(here, np.float64), np.asarray(there, np.float64)
    weights = sample.temper(there, temperature, min_p)
    return Counted(
        draws=counted.draws + len(here),
        same_peak=counted.same_peak
        + int((here.argmax(axis=-1) == there.argmax(axis=-1)).sum()),
        same_draw=counted.same_draw
        + int((sample.pick_share(weights, uniform) == drawn).sum()),
        cosine=counted.cosine + float(cosines(here, there).sum()),
    )


def count_chain_draws(counted, floated, chain_draws, *, temperature, min_p):
    """one step's CHAIN as a batch of four: the step-frame adapter over `count_draws`.

    A `Draw` of the chain holds a walk axis the drift report does not use -- the report
    runs one walk -- thus every row here is that walk's row."""
    return count_draws(
        counted,
        np.stack([taken.logits[0] for taken in chain_draws]),
        np.stack([floated[taken.seat] for taken in chain_draws]),
        drawn=np.array([taken.drawn[0] for taken in chain_draws]),
        uniform=np.array([float(taken.word[0]) for taken in chain_draws])
        * 2.0**-prng.UNIFORM_BITS,
        temperature=temperature,
        min_p=min_p,
    )
