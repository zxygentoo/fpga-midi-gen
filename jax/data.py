"""The corpus, and the ways to draw from it.

This sits beside the models and not inside one: the corpus is the corpus of the project,
and every era reads it here.

There are two exports of it, because the eras ask two different questions of the same
music:

- THE PACKED STREAM, `frames.safetensors`, for the walk of eras two to five: piece, seam,
  piece, seam, with the rolling coordinate of each step and the stream index per split. A
  split is a set of streams and not a set of pieces. One position is one step, thus a
  window is always full and every position weighs one. The two batching policies are the
  training draw (a uniform stream, then a uniform window) and the fixed evaluation windows
  of the referee (the canonical stream, stride context, from its start).
- THE PIECES, `pieces.safetensors`, for the canvas of era six: whole chorales on one grid,
  one row for each, padded to the longest, with the true length and the legal shift range
  beside the cells. The canvas draw crops inside the true length, thus the padding is a
  fact of the file and never a fact of the music.

There are no masks. No frame is illegal, thus nothing guards a draw.
"""

import numpy as np
from safetensors.numpy import load_file

BAR_STEPS = 16
SEATS = 4
SPLITS = ("train", "valid", "test")

# The vocabulary of the model, from docs/transformer_model.md: class 0 is silence and
# class 1 + i is the pitch PITCH_LOW + i. The corpus states 36 to 81 and the shift rule
# holds each voice inside its own observed range, thus 47 classes cover the music and the
# 48th is spare. Four tables of 48 rows put the six-layer model at 93 percent of the block
# RAM of the device, where four of 129 rows put it at 99 and it does not fit.
CLASSES = 48
PITCH_LOW = 36
# the top of the same corpus study: the classes 1 to 46 cover PITCH_LOW to here, and the
# piano roll of the canvas era holds one row for each of them
PITCH_HIGH = 81
SILENCE = 0


def classes_of_codes(codes):
    """The wire code of a voice becomes its class index, which is one subtraction.

    The two are different numbers on purpose: the flag of bit 7 makes the wire code cheap
    for the circuit, and the class index makes the table small for the model. A code
    outside the corpus is a fault of the export and not a case to handle -- the table has
    no row for it."""
    # bit 7 is the flag and bits 6:0 are the pitch, thus a code that sounds is not
    # merely a code that is not zero: the codes 0x01 to 0x7F are silence with a pitch
    # that no writer sets, and `Frame.pitch_of_code` reads the flag on the other side
    # of the seam. A test on zero would call one of those spare codes a pitch.
    sounds = (codes & 0x80) != 0
    pitches = codes & 0x7F
    inside = ~sounds | ((pitches >= PITCH_LOW) & (pitches < PITCH_LOW + CLASSES - 1))
    if not inside.all():
        bad = np.unique(pitches[~inside])
        raise ValueError(f"pitches outside the {CLASSES}-class vocabulary: {bad}")
    return np.where(sounds, pitches - PITCH_LOW + 1, SILENCE).astype(np.int32)


def classes_of_cells(cells):
    """The pitch cells of the piece export become class indices, which is one subtraction.

    The two exports carry the corpus in the two forms it takes on either side of the
    packer: the stream carries wire codes, where a silent voice is a flag, and a piece
    carries the cells of the file, where a rest is -1. Each export meets the one
    vocabulary here."""
    sounds = cells >= 0
    inside = ~sounds | ((cells >= PITCH_LOW) & (cells <= PITCH_HIGH))
    if not inside.all():
        bad = np.unique(cells[~inside])
        raise ValueError(f"pitches outside the {CLASSES}-class vocabulary: {bad}")
    return np.where(sounds, cells - PITCH_LOW + 1, SILENCE).astype(np.int32)


def pitches_of_classes(classes):
    """the MIDI pitch of each sounding class; a silent class has no pitch"""
    return classes + PITCH_LOW - 1


def decode(frames):
    """The decode of docs/transformer_model.md: the sequencer holds the set of pitches that
    sound, and a frame states the set that must sound. The releases are the first set minus
    the second, the strikes are the second minus the first, and every release goes before
    every strike.

    The rule is over sets and not over seats. Two voices that exchange pitches would send
    the Note On of a pitch before its Note Off, and the synth would stop the new note,
    because the four voices share one channel and a Note Off releases by pitch. Two voices
    on one pitch would send two of each.

    It lives here because it is a rule of the frame, and not of one model: the player, the
    instruments and the corpus reference all read it."""
    music = []
    sounding = set()
    for frame in frames:
        wanted = {int(pitches_of_classes(c)) for c in frame if c != SILENCE}
        music.append(
            [("off", p) for p in sorted(sounding - wanted)]
            + [("on", p) for p in sorted(wanted - sounding)]
        )
        sounding = wanted
    return music


class Split:
    """The packed streams of one split. Stream zero is the canonical one -- every piece at
    shift zero, in the order of the corpus -- and the referee reads it alone."""

    def __init__(self, tensors, name):
        self.classes = classes_of_codes(tensors[f"{name}/frames"])  # [steps, SEATS]
        self.positions = tensors[f"{name}/positions"]
        self.index = tensors[f"{name}/index"]  # [streams, 2]: offset, length

    def window(self, row, start, context):
        """one window of stream [row]: context + 1 frames, and the bar phase of the
        context positions that read them.

        The export carries the rolling coordinate of a 256-step window; the bar phase is
        its low four bits. The high four were the window position, which the ear dropped,
        and the corpus needs no second export to say so."""
        at = int(self.index[row, 0]) + start
        positions = self.positions[at : at + context]
        return (
            self.classes[at : at + context + 1],
            (positions % BAR_STEPS).astype(np.int32),
        )


def load_corpus(path):
    tensors = load_file(str(path))
    return {name: Split(tensors, name) for name in SPLITS}


def train_pool(corpus, train_on):
    """the (split, stream) pool of the -train-on flag"""
    names = {
        "train": ("train",),
        "train+test": ("train", "test"),
        "all": ("train", "test", "valid"),
    }[train_on]
    return [
        (corpus[name], row) for name in names for row in range(len(corpus[name].index))
    ]


def train_row(rng, pool, context):
    """One training row: a uniform stream, then a uniform window of it.

    The piece draw is not uniform -- a long piece holds more windows than a short one --
    which is the objective the endless walk asks for."""
    split, row = pool[rng.integers(len(pool))]
    length = int(split.index[row, 1])
    return split.window(row, rng.integers(length - context), context)


def stack_rows(rows):
    """the rows of (classes, phases) stacked into one batch of each"""
    return tuple(np.stack(column) for column in zip(*rows))


def train_batch(rng, pool, batch, context):
    return stack_rows([train_row(rng, pool, context) for _ in range(batch)])


def eval_rows(split, context, limit):
    """the fixed windows of the referee: the canonical stream, cut at stride context from
    its start"""
    need = context + 1
    length = int(split.index[0, 1])
    if length < need:
        return []
    windows = min(limit, (length - need) // context + 1)
    return [split.window(0, window * context, context) for window in range(windows)]


def eval_batches(split, context, limit, batch):
    rows = eval_rows(split, context, limit)
    return [stack_rows(rows[at : at + batch]) for at in range(0, len(rows), batch)]


class Pieces:
    """The whole pieces of one split, as `corpus_tool pieces` wrote them.

    One row is one piece: [classes] is [pieces, steps, SEATS], the piece stands at steps 0
    to its length, and the tail of the row past that is silence. [lengths] holds the true
    step count of each piece and [shifts] the two ends of its legal transposition range,
    both inclusive.

    The padding is a fact of the file and never a fact of the music. It exists only because
    a rectangular tensor is one shape and the chorales are not; [Crops] reads inside the
    true length and no model ever sees it. That is the whole difference from the proto
    round of feat/diffusion-proto, where the tail WAS the training signal and owned 53.2
    percent of the columns of a canvas.

    [shifts] is unread this round -- the paper of docs/diffusion.md states no transposition
    augmentation and the pitch axis of the trunk carries the equivariance instead. It stays
    because it is the third tensor of the export, and a reader that dropped it would make
    the file and its reader disagree."""

    def __init__(self, tensors, name):
        self.classes = classes_of_cells(tensors[f"{name}/cells"])
        self.lengths = tensors[f"{name}/lengths"]
        self.shifts = tensors[f"{name}/shifts"]  # [pieces, 2]: low, high


def load_pieces(path):
    tensors = load_file(str(path))
    return {name: Pieces(tensors, name) for name in SPLITS}


class Crops:
    """The canvas draw of docs/diffusion.md: a uniform piece, then a uniform crop of it.

    A crop is [length] steps taken inside the true length of a piece, thus it never reads
    the padded tail. A piece shorter than the crop is dropped -- one of the 229 train
    chorales of this corpus is 100 steps -- and the round then trains on 228, 76 and 77.
    Silence inside a crop is therefore the real rests of the music, a fraction of one
    percent of the cells.

    The piece draw is UNIFORM and not weighted by length: one piece is one member of the
    corpus. Weighting by length would state the objective of the packed walk, where a long
    piece holds more windows than a short one, and the walk is over."""

    def __init__(self, pieces, length):
        self.classes = pieces.classes
        self.lengths = pieces.lengths
        self.length = length
        self.rows = np.nonzero(pieces.lengths >= length)[0]
        if not len(self.rows):
            raise ValueError(f"no piece of this split holds a crop of {length} steps")

    def crop(self, rng, row):
        """one uniform crop of piece [row], inside its true length"""
        start = rng.integers(int(self.lengths[row]) - self.length + 1)
        return self.classes[row, start : start + self.length]

    def row(self, rng):
        return self.crop(rng, self.rows[rng.integers(len(self.rows))])

    def batch(self, rng, batch):
        return np.stack([self.row(rng) for _ in range(batch)])

    def every_piece(self, seed):
        """One crop of every piece that holds one, at a fixed seed.

        The valid curve and the referees read these. A fresh crop at every evaluation
        would move the number by the draw as much as by the model, and two steps of a run
        would not compare."""
        rng = np.random.default_rng(seed)
        return np.stack([self.crop(rng, row) for row in self.rows])


def moving(classes):
    """[batch, length + 1, SEATS] -> [batch, length] the count of voices that move into
    each predicted step.

    77.91 percent of the voice slots repeat the step before. They are easy, they dominate
    the mean, and they invite a model that holds its chord for ever to score well and play
    a drone. The second number of the report divides over the steps where two or more
    voices move, which is a quarter of the boundaries and where the music is."""
    return (classes[:, 1:] != classes[:, :-1]).sum(axis=-1)
