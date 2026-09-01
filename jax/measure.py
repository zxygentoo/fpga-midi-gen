"""The instruments, over a stack of sheets of class indices.

The common home of the measurement. Everything here is arithmetic over a [sheets, steps,
SEATS] array of class indices and nothing here knows which era drew it -- a Gibbs sheet, a
walk of the packed stream and a corpus crop all read the same way, and a single walk is a
stack of one. What a RECIPE measures with a model lives beside that recipe, in
`ar_measure.py` and `diffusion/measure.py`. The drift count at the end is the one
instrument that reads logits: what the quantization costs.

A ROW IS A DICT OF INSTRUMENTS AND A LINE IS WHAT PRINTS IT. A function that computes one
ends in `_row`, a function that renders one in `_line` or `_lines`, and every measure
module of this tree spells it so.

NOTHING HERE RANKS A MODEL, and the corpus row is the referee of every number. A triad
share of 40 percent says nothing until the corpus row beside it says 64. Ten times in this
project a metric has ranked a model against the ear; read these to catch a pathology and
let the ear elect. Of the battery the ear has spoken FOR parallel motion, the voice pairs
and the likelihood, and AGAINST hold, the onsets and register order. The findings are in
docs/diffusion.md.
"""

import itertools
from typing import NamedTuple

import numpy as np

import corpus
import prng
import sample

# what this corpus calls a dissonance, a triad and a clash


# The intervals a chorale treats as a dissonance, modulo the octave: the two seconds, the
# tritone and the two sevenths. The perfect fourth is left out -- it is a dissonance
# against the bass and a consonance between the upper voices, and one number cannot tell
# them apart. This set puts the corpus at 10.3 percent.
DISSONANT = (1, 2, 6, 10, 11)
# Major and minor, which puts the corpus at 63.9 percent of the steps carrying three
# voices. Diminished is left out on the proto round's method; adding it moves the corpus
# row to 67.9, thus it is a labelling choice and not a finding either way.
TRIADS = ((0, 4, 7), (0, 3, 7))
# THE TAIL OF THE HARMONY: a frame of three dissonant pairs or more is 3.2 percent of the
# corpus, where TWO is 20.6 percent and merely a seventh chord. A mean holds while these
# multiply, and one strange chord in a phrase is heard.
CLASH = 3
# seat 0 is the bass and seat 3 the soprano, as the packed stream and the chained head of
# the earlier eras both read them
VOICE_NAMES = ("bass", "tenor", "alto", "soprano")
# the pairs in the order of the seats between them, which is nearly the order of their
# span in the corpus: [voice_pairs] then reads left to right as the pitch reach runs out
PAIRS = tuple(
    sorted(
        itertools.combinations(range(corpus.SEATS), 2), key=lambda p: (p[1] - p[0], p[0])
    )
)

# the shares the instruments are built out of


def triad_table():
    """A row for each of the 4096 sets of pitch classes: does this set fit inside some
    triad? A step holds at most four pitch classes, thus its set is one 12-bit word and
    the battery is a vector operation instead of a loop over steps."""
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
    """[of] applied to [numbers], or ZERO where nothing was heard. Every instrument
    here is conditional, and a nan poisons a mean of several sheets and prints as a
    hole where a zero prints as the finding it is. The corpus row says which zero is
    which."""
    return float(of(numbers)) if len(numbers) else 0.0


def dissonant_share(intervals):
    """the share of [intervals] that this corpus treats as a dissonance"""
    return np.isin(intervals, DISSONANT).mean()


def triad_share(words):
    """the share of pitch-class [words] that fit inside a major or a minor triad"""
    return FITS_A_TRIAD[words].mean()


def clash_share(clashes):
    """the share of steps carrying [CLASH] dissonant pairs or more"""
    return (clashes >= CLASH).mean()


# the three instruments that read more than one step


def voice_pairs(spans, intervals, pairs_sound):
    """Each pair of voices: its mean span in semitones, and how often it sounds a
    dissonance.

    THE INSTRUMENT THAT EXPLAINS THE OTHERS, because the trunk is a stack of
    three-by-three convolutions: it reaches L rows up the pitch axis at L layers, thus
    a pair sitting further apart than the reach cannot be seen at any width. Depth is
    the reach and width is the resolution; docs/diffusion.md measures both."""
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

    Two voices a fifth or an octave apart, both moving the same way, landing on the same
    interval. It is the most audible error in four-part writing and the only instrument
    here that reads across time; docs/diffusion.md reports that no lever of era six
    reaches it, and why.

    THE DIVISOR IS THE PAIRS THAT MOVE, not the pairs that merely sound: a parallel needs
    both voices to move, thus the older divisor paid a model for holding its notes.
    [moving] is reported beside the rates, because a rung whose motion has left the corpus
    is not to be read on the rates alone."""
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
        # SIMILAR MOTION is what makes a parallel a parallel. Contrary motion onto a fifth
        # is how a fifth is correctly approached, and counting it read 53 percent of the
        # corpus's own fifths as faults.
        together = np.sign(pitches[:, 1:, low] - pitches[:, :-1, low]) == np.sign(
            pitches[:, 1:, high] - pitches[:, :-1, high]
        )
        # and the pair must keep its order: a fifth whose voices cross becomes a fourth,
        # which the absolute gap cannot see. A unison has no side, thus a zero keeps
        # place.
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

    The thing nothing else here covers: [voice_pairs] reads DIFFERENCES of pitch and the
    order instrument reads the stacking, thus a texture that slid bodily would move
    neither. A convolution over the pitch axis is equivariant in pitch, so register error
    falls with DEPTH -- a cell knows where it stands only when its reach touches an edge.

    The mean says a voice has DRIFTED and the spread says it RANGES too widely; the two
    are different faults. [outside] is the share of sounding cells beyond their own
    seat's [corpus.VOICE_RANGES], and the corpus reads zero on it by construction."""
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


# the battery, and the lines that print it


def battery_row(classes):
    """The battery over a set of sheets: [sheets, steps, VOICES] of class indices.

    HOLD is the share of voice slots that repeat the step before, and it reads both
    failures in one number: far above the corpus is a drone, far below is jitter. ONSETS
    is the note-ons for each step under the decode of corpus.py, thus an onset means here
    what it means on the wire. VOICES is the share of steps carrying 0 to 4 sounding
    voices; the corpus sings all four at 99.8 percent of its steps.

    TRIADS is over the steps that carry THREE VOICES OR MORE, DISSONANT is over the
    sounding pairs, and ORDER is the share of steps whose voices stand bass lowest. SPARE
    is a smoke number: the corpus never sings that row."""
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
    """A BATCH of the twin's rows [here] against the float rows [there] of the same
    places, on the very uniform the twin drew [drawn] on.

    It is batched because era six redraws a whole sheet where a step-frame chain redraws
    four seats. The caller states the policy, because the elected numbers are the twin's
    and not this instrument's."""
    here, there = np.asarray(here, np.float64), np.asarray(there, np.float64)
    weights = sample.tempered_weight(there, temperature, min_p)
    return Counted(
        draws=counted.draws + len(here),
        same_peak=counted.same_peak
        + int((here.argmax(axis=-1) == there.argmax(axis=-1)).sum()),
        same_draw=counted.same_draw
        + int((sample.pick_share(weights, uniform) == drawn).sum()),
        cosine=counted.cosine + float(cosines(here, there).sum()),
    )


def count_chain_draws(counted, floated, chain_draws, *, temperature, min_p):
    """one step's CHAIN as a batch of four: the step-frame adapter over `count_draws`. A
    `Draw` holds a walk axis the drift report does not use -- it runs one walk -- thus
    every row here is that walk's row."""
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
