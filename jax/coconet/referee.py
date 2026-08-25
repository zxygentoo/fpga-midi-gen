"""The two measuring referees of docs/coconet.md: the likelihood, and the structure.

    uv run python -m coconet.referee nll     --ckpt ../_train/coconet/NAME.ckpt
    uv run python -m coconet.referee corpus

THE LIKELIHOOD is the paper's Algorithm 1, and it is the one number of this round that
compares outside the repository: Table 1 of arXiv 1903.07227 reads **0.57 +- 0.01** nats
for each frame on the sixteenth grid of this corpus lineage. The protocol is pinned to the
paper and to its code release, because a referee that computes a different number reads
nothing. What that means in practice is stated on [framewise_lls]: the ordering is over
FRAMES, the model conditions on the ground truth of the frames before it and on its own
predictions inside the frame it is writing, and five orderings are averaged in probability
space and not in log space.

THE STRUCTURE BATTERY is the proto round's, held to its method. Its rule is the reason it
is here at all: THE CORPUS ROW IS THE REFEREE OF EVERY NUMBER. A triad share of 40 percent
says nothing until the corpus row beside it says 64. And triads count only over the steps
that carry three voices or more -- counted over every step, a dyad sits inside some triad
for free, and a thin canvas scores well for the wrong reason.

NOTHING HERE RANKS A MODEL, which is the standing warning of jax/measure.py and it has been
earned ten times over in this project. Read these to catch a pathology; let the ear elect.

The battery measures a set of canvases and does not draw them, thus this module knows
nothing about sampling: coconet/infer.py draws and calls in here.
"""

import itertools
import time

import click
import jax
import jax.numpy as jnp
import numpy as np

import data
import nn
from coconet import model

JAX_ROOT = nn.JAX_ROOT
CORPUS = str(JAX_ROOT / "_data" / "pieces.safetensors")


# ==================================================================== #
# THE STRUCTURE BATTERY — what a set of canvases is made of            #
# ==================================================================== #

# The intervals a chorale treats as a dissonance, in semitones and modulo the octave: the
# two seconds, the tritone and the two sevenths. The perfect fourth is left out -- it is a
# dissonance against the bass and a consonance between the upper voices, and a measurement
# that cannot tell them apart should not name it. This set puts the corpus at 10.3 percent,
# which is the number the proto round recorded.
DISSONANT = (1, 2, 6, 10, 11)
# The triad qualities the battery counts: major and minor, which puts the corpus at 63.9
# percent of the steps that carry three voices. Diminished is left out on the proto round's
# method; adding it moves the corpus row to 67.9, thus it is a labelling choice and not a
# finding either way.
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
# THE REGISTER OF EACH SEAT: the lowest and the highest pitch it sings anywhere in this
# corpus. Measured 2026-08-25 over every step of every piece, and the three splits agree
# EXACTLY -- thus this is a fact of the genre and not of a draw. The roll holds 36 to 81,
# which is the union of the four, thus no seat can leave the vocabulary and every violation
# this finds is a voice standing in another voice's register.
RANGES = ((36, 66), (46, 69), (52, 74), (60, 81))
# The pairs of voices, in the order of the seats between them: the three neighbours, then
# the two that skip a voice, then the outer pair. That order is nearly the order of their
# span in the corpus, thus [voice_pairs] reads left to right as the pitch reach runs out.
PAIRS = tuple(
    sorted(itertools.combinations(range(model.VOICES), 2), key=lambda p: (p[1] - p[0], p[0]))
)


def triad_table():
    """A row for each of the 4096 sets of pitch classes: does this set fit inside some
    triad?

    A step holds at most four voices and therefore at most four pitch classes, thus its set
    is one 12-bit word and the whole question is a table lookup. The table is built one time
    and it makes the battery a vector operation instead of a loop over steps."""
    fits = np.zeros(1 << 12, dtype=bool)
    for quality in TRIADS:
        for root in range(12):
            chord = sum(1 << ((step + root) % 12) for step in quality)
            # every subset of a triad fits inside it, and a set of one or two pitch classes
            # is exactly the case the denominator of [structure] excludes
            fits[[at for at in range(1 << 12) if at & ~chord == 0]] = True
    return fits


FITS_A_TRIAD = triad_table()


def pitch_class_words(classes):
    """[canvases, steps, VOICES] -> [canvases, steps] the sounding pitch classes of each
    step as one 12-bit word; a rest sets no bit and a unison sets one bit twice"""
    pitches = data.pitches_of_classes(classes) % 12
    bits = np.where(classes != data.SILENCE, 1 << pitches, 0)
    return np.bitwise_or.reduce(bits, axis=-1)


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
    it; L 24 takes that pair from 24.6 percent to 16.6. Depth is the reach and width is the
    resolution.

    Read it against the corpus row, as everything here is read. A pair can be too consonant
    as well as too dissonant, and the narrow rungs are both."""
    rows = []
    for at, (low, high) in enumerate(PAIRS):
        sounds = pairs_sound[..., at]
        rows.append(
            {
                "name": f"{VOICE_NAMES[low][:2]}-{VOICE_NAMES[high][:2]}",
                "span": float(np.mean(spans[..., at][sounds])) if sounds.any() else 0.0,
                "dissonant": 100.0 * float(np.mean(np.isin(intervals[..., at][sounds], DISSONANT)))
                if sounds.any()
                else 0.0,
            }
        )
    return rows


def parallel_motion(classes, pitches, sounding):
    """Parallel fifths and octaves, for each thousand pairs that live across a step.

    THE FAULT THAT LIVES BETWEEN FRAMES, and the only instrument here that reads across
    time at all. Two voices a fifth or an octave apart, both moving, landing on the same
    interval. It is the most audible error in four-part writing, because an octave collapses
    two voices into one and the texture thins where nothing else changed.

    Measured 2026-08-25, it is what separates this era's models from the corpus where every
    vertical instrument says they have arrived. Bach writes 0.26 fifths and 0.10 octaves for
    each thousand; the board rung writes 2.0 and 1.9, and the paper's own size 0.8 and 1.3.
    More Gibbs passes take it down two or three times and then stop -- both sizes saturate,
    one by N 256 and one by N 512 -- and sixty-two times the parameters halves it and stops.
    NEITHER REACHES THE CORPUS.

    The reason is structural, and it is the finding of the round. The loss scores the
    MARGINAL of each cell under its context, and the walk draws those marginals
    INDEPENDENTLY; a joint configuration of two voices across two steps is therefore
    evaluated by neither. A term that stands in no objective cannot be sampled away.

    TWO CORRECTIONS OF 2026-08-25, both found by reading the corpus row and not the models.

    THE MOTION MUST BE SIMILAR. A parallel is two voices moving THE SAME WAY from a perfect
    interval to the same perfect interval. The first version asked only that the interval
    class stand before and after, thus contrary motion onto a fifth -- which is how a fifth
    is correctly approached -- counted as a fault. It read 53 percent of the corpus's own
    fifths that way, 10 events of 19. The octaves were untouched, 4 of 4 already similar.

    THE PAIR MUST KEEP ITS ORDER. The gap was an absolute value, thus a pair that crossed
    could hold its interval class while its interval turned upside down. 3.7 percent of the
    corpus's moving pairs swap order across a step.

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
        # SIMILAR MOTION is what makes a parallel a parallel. Contrary motion that lands on
        # a fifth is how a fifth is correctly approached, and counting it read 53 percent of
        # the corpus's own fifths as faults.
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

    The mean says a voice has DRIFTED and the spread says it RANGES too widely; the two are
    different faults and a single number would confuse them. [outside] is the tail -- the
    share of sounding cells beyond their own seat's [RANGES] -- and the corpus reads zero
    on it by construction, as the spare row does."""
    seats = []
    outside = alive = 0
    for seat, (low, high) in enumerate(RANGES):
        heard = pitches[..., seat][sounding[..., seat]]
        seats.append(
            {
                "name": VOICE_NAMES[seat],
                "mean": float(heard.mean()) if len(heard) else 0.0,
                "spread": float(heard.std()) if len(heard) else 0.0,
            }
        )
        outside += int(((heard < low) | (heard > high)).sum())
        alive += len(heard)
    return {"seats": seats, "outside": 100.0 * outside / max(alive, 1)}


def structure(classes):
    """The battery over a set of canvases: [canvases, steps, VOICES] of class indices.

    HOLD is the share of voice slots that repeat the step before. It reads both failures a
    canvas can have in one number: far above the corpus is a drone, far below is jitter.

    ONSETS is the note-ons for each step under the decode of data.py, thus an onset means
    here what it means on the wire.

    VOICES is the share of steps carrying 0, 1, 2, 3 and 4 sounding voices. The corpus
    sings all four at 99.8 percent of its steps, thus this row alone catches the thin canvas
    that was the open defect of the proto round.

    TRIADS is the share of steps that fit inside a major or minor triad, over the steps that
    carry THREE VOICES OR MORE. DISSONANT is the share of sounding pairs at a dissonant
    interval. ORDER is the share of steps whose sounding voices stand in register order,
    the bass lowest.

    SPARE is the share of cells drawn on the spare row of the vocabulary, which the corpus
    never sings. It is a smoke number: anything above zero says the model puts mass where no
    music is."""
    sounding = classes != data.SILENCE
    voices = sounding.sum(axis=-1)
    words = pitch_class_words(classes)
    pitches = data.pitches_of_classes(classes)
    thick = voices >= 3
    pairs_sound = np.stack([sounding[..., a] & sounding[..., b] for a, b in PAIRS], -1)
    spans = np.stack([np.abs(pitches[..., a] - pitches[..., b]) for a, b in PAIRS], -1)
    intervals = spans % 12
    ordered = np.stack([pitches[..., a] <= pitches[..., b] for a, b in PAIRS], -1)
    clashes = (np.isin(intervals, DISSONANT) & pairs_sound).sum(axis=-1)
    music = [data.decode(canvas) for canvas in classes]
    ons = sum(1 for piece in music for step in piece for kind, _ in step if kind == "on")
    return {
        "hold": 100.0 * float(np.mean(classes[:, 1:] == classes[:, :-1])),
        "onsets": ons / sum(len(piece) for piece in music),
        "voices": [100.0 * float(np.mean(voices == count)) for count in range(5)],
        "triads": 100.0 * float(np.mean(FITS_A_TRIAD[words[thick]])) if thick.any() else 0.0,
        "dissonant": 100.0
        * float(np.sum(np.isin(intervals, DISSONANT) & pairs_sound) / max(pairs_sound.sum(), 1)),
        "order": 100.0
        * float(np.mean(np.all(ordered | ~pairs_sound, axis=-1)[voices >= 2]))
        if (voices >= 2).any()
        else 0.0,
        "clash": 100.0 * float(np.mean(clashes[sounding.any(axis=-1)] >= CLASH))
        if sounding.any()
        else 0.0,
        "spare": 100.0 * float(np.mean(classes == model.ROWS - 1)),
        "parallels": parallel_motion(classes, pitches, sounding),
        "register": tessitura(pitches, sounding),
        "pairs": voice_pairs(spans, intervals, pairs_sound),
    }


def structure_lines(label, row):
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


def corpus_canvases(corpus_path, split, crop, seed):
    """one crop of every piece of a split that holds one, at a fixed seed: the rows the
    corpus row and the likelihood referee both read"""
    return data.Crops(data.load_pieces(corpus_path)[split], crop).every_piece(seed)


# ==================================================================== #
# THE LIKELIHOOD — the paper's Algorithm 1                             #
# ==================================================================== #

# Table 1 of the paper on this corpus lineage, sixteenth grid, random orderings. It is the
# anchor of the round and never a target to optimise against.
PAPER_NLL = 0.57
# the paper's M: "averaging likelihoods across an ensemble of M = 5 orderings"
ORDERINGS = 5


def frame_ordering(rng, steps):
    """One ordering of Algorithm 1: a permutation of the frames, and a permutation of the
    voices inside each frame.

    The paper's random ordering is over FRAMES and not over all D variables -- that is the
    difference between its framewise measurement and a notewise one. Its chronological
    variant keeps the frames in time order and shuffles only the voices; this round reports
    the random one, which is the row of Table 1 that reads 0.57."""
    return rng.permutation(steps), np.stack(
        [rng.permutation(model.VOICES) for _ in range(steps)]
    )


def forward_in_chunks(forward, classes, hidden, chunk):
    """the log probabilities of a stack of canvases, [chunk] at a time: one canvas of the
    stack is one frame of the piece, thus the stack is as tall as the crop and a 12 GB card
    wants it cut"""
    return np.concatenate(
        [
            np.asarray(
                forward(jnp.asarray(classes[at : at + chunk]), hidden[at : at + chunk])
            )
            for at in range(0, len(classes), chunk)
        ]
    )


def framewise_lls(forward, classes, ordering, chunk):
    """The log-likelihood of every frame of one canvas under one ordering: Algorithm 1.

    THE FRAMES ARE INDEPENDENT GIVEN THE ORDERING, and that is the whole reason this referee
    is affordable. Algorithm 1 restores the ground truth of a frame the moment it finishes
    writing it, thus frame l conditions on the TRUE frames that stand before it in the
    ordering and on nothing the model wrote outside itself. Therefore the T frames run as
    one stack and the walk costs I forward passes and not I times T. The code release does
    exactly this, and this function is its shape.

    Inside a frame the model does condition on itself: voice k reads what the model put in
    the k - 1 voices before it. That is what makes this framewise and not notewise -- the
    frame is the unit of prediction and an error inside one accumulates, which the paper
    states is the point.

    The model's own value is written as the ARGMAX and not as a draw. The paper's Algorithm
    1 samples there; its code release takes the argmax, and the code release is what
    produced 0.57."""
    frames, voices = ordering
    steps = len(classes)
    # canvas l of the stack reveals the frames that stand before position l in the ordering
    states = np.tile(classes, (steps, 1, 1))
    hidden = np.zeros((steps, steps, model.VOICES), dtype=bool)
    for at in range(steps):
        hidden[at, frames[at:], :] = True
    row = np.arange(steps)
    lls = np.zeros(steps, dtype=np.float64)
    for turn in range(model.VOICES):
        said = forward_in_chunks(forward, states, hidden, chunk)
        # the frame each canvas of the stack is writing, and the voice of it whose turn
        # this is: one cell for each canvas of the stack, thus one distribution over the
        # pitch rows for each
        voice = voices[frames, turn]
        logp = said[row, frames, :, voice]
        lls[frames] += logp[row, classes[frames, voice]]
        states[row, frames, voice] = np.argmax(logp, axis=-1)
        hidden[row, frames, voice] = False
    return lls


def piece_nll(forward, classes, rng, orderings, chunk):
    """Algorithm 1 for one canvas, frame by frame: the nats of every frame of it.

    The caller means these, which is Algorithm 1's return, AND keeps them. A mean cannot
    see a rare bad moment and the ear can, thus the frames are the tail and the tail is a
    measurement of its own.

    The orderings are combined IN PROBABILITY SPACE, one frame at a time -- logsumexp over
    the ensemble, less the log of its size. A mean of log-likelihoods would be an
    unnormalised geometric mean and would waste probability mass, and the paper's line is
    explicit about it."""
    lls = np.stack(
        [
            framewise_lls(forward, classes, frame_ordering(rng, len(classes)), chunk)
            for _ in range(orderings)
        ]
    )
    return -(np.logaddexp.reduce(lls, axis=0) - np.log(len(lls)))


def framewise_nll(params, stats, canvases, *, orderings, chunk, seed, report=None):
    """The referee over a set of canvases: Algorithm 1's mean nats for each frame, its
    standard error, and the frames themselves.

    The standard error is over the PIECES, which is what the paper's Table 1 reports beside
    its means. The frames are kept for [tail_line]."""
    # the log softmax runs on the device beside the trunk: the referee then indexes a
    # probability and never normalises one, and the argmax of the two is the same cell
    forward = jax.jit(
        lambda classes, hidden: jax.nn.log_softmax(
            model.logits(params, stats, model.planes(classes, hidden))[0], axis=-2
        )
    )
    rng = np.random.default_rng(seed)
    frames = []
    for at, canvas in enumerate(canvases):
        frames.append(piece_nll(forward, canvas, rng, orderings, chunk))
        if report is not None:
            report(at, float(frames[-1].mean()))
    pieces = np.asarray([piece.mean() for piece in frames])
    return {
        "mean": float(pieces.mean()),
        "error": float(pieces.std() / np.sqrt(len(pieces))),
        "pieces": pieces,
        # [pieces, frames] and not one flat run of them: the tail resamples PIECES, thus
        # its error stands beside the mean's and reads against the same population
        "frames": np.stack(frames),
    }


# nats for one frame above which the ear would call it a wrong moment. A frame is four
# voices, thus 2 nats is a joint probability of 0.14 for the whole sonority.
LOUD = 2.0
# resamples of the pieces behind each percentile. A percentile carries no standard error of
# its own, and two models an eighth of a nat apart cannot be told from each other without
# one -- the round has already been caught by that once, on the parallels.
RESAMPLES = 1000
MARKS = (50, 90, 99)


def tail_shape(frames, seed=0):
    """The tail of the framewise nats: the percentiles of [MARKS], the share of frames over
    [LOUD], and a bootstrap error for each of them.

    [frames] is [pieces, frames]. THE RESAMPLE IS OVER PIECES and not over frames, for the
    reason the mean's own error is: the frames of one chorale are one draw of a composer
    and not 128 of them, thus resampling frames would state an error several times too
    small and every model would separate from every other."""
    rng = np.random.default_rng(seed)
    draws = frames[rng.integers(len(frames), size=(RESAMPLES, len(frames)))]
    draws = draws.reshape(RESAMPLES, -1)
    marks = np.percentile(frames, MARKS)
    loud = 100.0 * np.mean(frames >= LOUD)
    return {
        "marks": marks,
        "mark errors": np.percentile(draws, MARKS, axis=-1).std(axis=-1),
        "loud": loud,
        "loud error": float((100.0 * np.mean(draws >= LOUD, axis=-1)).std()),
    }


def tail_line(frames, seed=0):
    """THE RARE BAD MOMENT, which the mean of [framewise_nll] cannot see.

    One strange chord in a phrase is heard, and it moves the average of 128 frames by
    nothing at all. This is the instrument the ear asked for on 2026-08-25, after it heard
    the ceiling iron out a weirdness that cost the mean 0.008 nats.

    A model with a shorter tail at the same mean is a model that is wrong less often and
    not less badly, which is the trade the ear elects.

    READ IT AGAINST WHAT IT MEASURES. These are corpus canvases, thus a frame of high nats
    is a frame where BACH surprised the model, and not one where the model wrote something
    strange. The two are not the same question, and the second one is [structure]'s clash,
    which reads the model's own draws."""
    read = tail_shape(frames, seed)
    marks = "   ".join(
        f"{name} {value:5.3f} +- {error:.3f}"
        for name, value, error in zip(
            ("median", "90th", "99th"), read["marks"], read["mark errors"]
        )
    )
    return (
        f"{'the tail':<22} {marks}   above {LOUD:.0f} nats "
        f"{read['loud']:4.1f} +- {read['loud error']:.1f}%"
    )


# ==================================================================== #
# The commands                                                         #
# ==================================================================== #


@click.group(help=__doc__)
def main():
    pass


@main.command(help=structure.__doc__)
@click.option("--corpus", "corpus_path", default=CORPUS)
@click.option("--split", default="train", type=click.Choice(data.SPLITS))
@click.option("--crop", default=model.CROP)
@click.option("--seed", default=0, help="the crop draw; fixed, thus the row is fixed")
def corpus(corpus_path, split, crop, seed):
    canvases = corpus_canvases(corpus_path, split, crop, seed)
    for line in structure_lines(f"the corpus, {split}", structure(canvases)):
        click.echo(line)


@main.command(help=framewise_nll.__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--corpus", "corpus_path", default=CORPUS)
@click.option("--split", default="test", type=click.Choice(data.SPLITS))
@click.option("--crop", default=model.CROP)
@click.option("--orderings", default=ORDERINGS, help="the paper's M")
@click.option(
    "--pieces", default=0, help="how many pieces of the split; 0 is all of them"
)
@click.option("--chunk", default=16, help="canvases of one forward pass")
@click.option("--seed", default=0)
def nll(ckpt, corpus_path, split, crop, orderings, pieces, chunk, seed):
    canvases = corpus_canvases(corpus_path, split, crop, seed)
    if pieces:
        canvases = canvases[:pieces]
    params, stats = model.load_params(ckpt)
    started = time.perf_counter()

    def report(at, value):
        done = time.perf_counter() - started
        click.echo(
            f"piece {at + 1:3d} of {len(canvases)}  {value:6.4f}  "
            f"{done / (at + 1):5.1f} s each"
        )

    read = framewise_nll(
        params,
        stats,
        canvases,
        orderings=orderings,
        chunk=chunk,
        seed=seed,
        report=report,
    )
    click.echo(
        f"framewise NLL on {split}, {len(canvases)} pieces, {orderings} orderings: "
        f"{read['mean']:.4f} +- {read['error']:.4f} nats for each frame"
    )
    click.echo(tail_line(read["frames"]))
    click.echo(f"the paper's Table 1 on the sixteenth grid: {PAPER_NLL:.2f} +- 0.01")


if __name__ == "__main__":
    main()
