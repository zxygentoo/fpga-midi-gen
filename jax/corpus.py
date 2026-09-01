"""The corpus and the vocabulary: the music, and what a class means.

`lib/corpus` is the same pair below the seam -- `Jsb` for the chorales and `Vocab` for the
vocabulary -- and `bin/corpus_tool` writes the two exports this module reads. The
vocabulary stands here and not with a model: class 0 is silence and class 1 + i is the
pitch `PITCH_LOW + i` under every era, thus `decode` means one thing on the wire whichever
model drew the frame.

The two exports ask two different questions of the same music. THE PACKED STREAM,
`frames.safetensors`, is the walk of eras two to five: piece, seam, piece, seam, with the
rolling coordinate of each step and the stream index per split. THE PIECES,
`pieces.safetensors`, is the sheet of era six: whole chorales on one grid, padded to the
longest, with the true length and the legal shift range beside the cells.

There are no masks. No frame is illegal, thus nothing guards a draw.
"""

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

# where `corpus_tool` writes the two exports; git ignores the directory
JAX_ROOT = Path(__file__).resolve().parent
FRAMES = JAX_ROOT / "_data" / "frames.safetensors"
PIECES = JAX_ROOT / "_data" / "pieces.safetensors"

BAR_STEPS = 16
SEATS = 4
SPLITS = ("train", "valid", "test")

# The vocabulary of docs/transformer.md. The corpus states 36 to 81 and the shift rule
# holds each voice inside its own range, thus 47 classes cover the music and the 48th is
# spare. Four tables of 48 rows put the six-layer model at 93 percent of the device's
# block RAM, where four of 129 rows put it at 99 and it does not fit.
CLASSES = 48
PITCH_LOW = 36
PITCH_HIGH = 81
SILENCE = 0

# THE REGISTER OF EACH SEAT, measured over every step of every piece: the three splits
# agree exactly, thus this is a fact of the genre and not of a draw. It stands with the
# vocabulary because era six's opening draws from it as well as the referee reads it;
# `Jsb.voice_ranges` is the OCaml side. PITCH_LOW to PITCH_HIGH is the union of the four.
VOICE_RANGES = ((36, 66), (46, 69), (52, 74), (60, 81))


def classes_of_codes(codes):
    """The wire code of a voice becomes its class index, which is one subtraction. The two
    are different numbers on purpose: bit 7 makes the wire code cheap for the circuit and
    the class index makes the table small for the model."""
    # BIT 7 IS THE FLAG, thus a code that sounds is not merely a code that is not zero:
    # 0x01 to 0x7F are silence with a pitch no writer sets. A test on zero would call one
    # of those spare codes a pitch.
    sounds = (codes & 0x80) != 0
    pitches = codes & 0x7F
    inside = ~sounds | ((pitches >= PITCH_LOW) & (pitches < PITCH_LOW + CLASSES - 1))
    if not inside.all():
        bad = np.unique(pitches[~inside])
        # the WINDOW and not the corpus range: a walk that drew the spare class comes
        # back through here as pitch 82, and that is music, not a fault
        raise ValueError(
            f"pitches outside the vocabulary's {PITCH_LOW} to "
            f"{PITCH_LOW + CLASSES - 2}: {bad}"
        )
    return np.where(sounds, pitches - PITCH_LOW + 1, SILENCE).astype(np.int32)


def classes_of_cells(cells):
    """The pitch cells of the piece export become class indices. The stream carries wire
    codes, where a silent voice is a flag; a piece carries the cells of the file, where a
    rest is -1. Each export meets the one vocabulary here."""
    sounds = cells >= 0
    inside = ~sounds | ((cells >= PITCH_LOW) & (cells <= PITCH_HIGH))
    if not inside.all():
        bad = np.unique(cells[~inside])
        raise ValueError(
            f"pitches outside the corpus's {PITCH_LOW} to {PITCH_HIGH}: {bad}"
        )
    return np.where(sounds, cells - PITCH_LOW + 1, SILENCE).astype(np.int32)


def pitches_of_classes(classes):
    """the MIDI pitch of each sounding class; a silent class has no pitch"""
    return classes + PITCH_LOW - 1


def decode(frames):
    """The decode of docs/transformer.md: the releases are the sounding set minus the
    wanted set, the strikes are the reverse, and every release goes before every strike.

    THE RULE IS OVER SETS AND NOT OVER SEATS. The four voices share one channel and a
    Note Off releases by pitch, thus two voices that exchange pitches would send the Note
    On of a pitch before its Note Off and the synth would stop the new note."""
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
        context positions that read them. The export carries the rolling coordinate of a
        256-step window; the bar phase is its low four bits."""
        at = int(self.index[row, 0]) + start
        positions = self.positions[at : at + context]
        return (
            self.classes[at : at + context + 1],
            (positions % BAR_STEPS).astype(np.int32),
        )


def load_corpus(path):
    tensors = load_file(str(path))
    return {name: Split(tensors, name) for name in SPLITS}


def train_pool(corpus):
    """the (split, stream) pool a training run draws from: every stream of the train split
    and no other -- valid is the referee's and test is held out"""
    train = corpus["train"]
    return [(train, row) for row in range(len(train.index))]


def train_row(rng, pool, context):
    """One training row: a uniform stream, then a uniform window of it. The piece draw is
    then not uniform -- a long piece holds more windows -- which is the objective the
    endless walk asks for."""
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
    """The whole pieces of one split, as `corpus_tool pieces` wrote them: one row is one
    piece, the piece stands at steps 0 to its length, and the tail past that is silence.
    [shifts] holds the two ends of the legal transposition range, both inclusive.

    The padding is a fact of the file and never of the music -- a rectangular tensor is
    one shape and the chorales are not -- and [Crops] reads inside the true length.
    [shifts] is unread this round: docs/diffusion.md states no transposition augmentation
    and the pitch axis of the trunk carries the equivariance instead."""

    def __init__(self, tensors, name):
        self.classes = classes_of_cells(tensors[f"{name}/cells"])
        self.lengths = tensors[f"{name}/lengths"]
        self.shifts = tensors[f"{name}/shifts"]  # [pieces, 2]: low, high


def load_pieces(path):
    tensors = load_file(str(path))
    return {name: Pieces(tensors, name) for name in SPLITS}


class Crops:
    """The sheet draw of docs/diffusion.md: a uniform piece, then a uniform crop inside
    its true length. A piece shorter than the crop is dropped, thus the round trains on
    228 of 229 chorales.

    The piece draw is UNIFORM and not weighted by length: one piece is one member of the
    corpus. Weighting would state the objective of the packed walk, and that walk is
    over."""

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
        """One crop of every piece that holds one, at a FIXED seed: the valid curve and
        the referees read these, and a fresh crop each time would move the number by the
        draw as much as by the model."""
        rng = np.random.default_rng(seed)
        return np.stack([self.crop(rng, row) for row in self.rows])


def moving(classes):
    """[batch, length + 1, SEATS] -> [batch, length] the count of voices that move into
    each predicted step.

    77.91 percent of the voice slots repeat the step before: they dominate the mean and
    invite a model that holds its chord for ever. The second number of the report divides
    over the steps where two or more voices move, which is where the music is."""
    return (classes[:, 1:] != classes[:, :-1]).sum(axis=-1)
