"""The corpus of frames, and the two ways to draw from it.

This sits beside the models and not inside one: the packed stream is the corpus of the
project, and a later model reads the same frames.

Reads the export of corpus_tool -- the frames, the rolling coordinates and the stream
index per split -- and holds the two batching policies: the training draw (a uniform
stream, then a uniform window) and the fixed evaluation windows of the referee (the
canonical stream, stride context, from its start).

A split is a set of packed streams -- piece, seam, piece, seam -- and not a set of pieces.
One position is one step, thus a window is always full and every position weighs one.

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
SILENCE = 0


def classes_of_codes(codes):
    """The wire code of a voice becomes its class index, which is one subtraction.

    The two are different numbers on purpose: the flag of bit 7 makes the wire code cheap
    for the circuit, and the class index makes the table small for the model. A code
    outside the corpus is a fault of the export and not a case to handle -- the table has
    no row for it."""
    sounds = codes != 0
    pitches = codes & 0x7F
    inside = ~sounds | ((pitches >= PITCH_LOW) & (pitches < PITCH_LOW + CLASSES - 1))
    if not inside.all():
        bad = np.unique(pitches[~inside])
        raise ValueError(f"pitches outside the {CLASSES}-class vocabulary: {bad}")
    return np.where(sounds, pitches - PITCH_LOW + 1, SILENCE).astype(np.int32)


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
    return tuple(np.stack([row[field] for row in rows]) for field in range(2))


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


def moving(classes):
    """[batch, length + 1, SEATS] -> [batch, length] the count of voices that move into
    each predicted step.

    77.91 percent of the voice slots repeat the step before. They are easy, they dominate
    the mean, and they invite a model that holds its chord for ever to score well and play
    a drone. The second number of the report divides over the steps where two or more
    voices move, which is a quarter of the boundaries and where the music is."""
    return (classes[:, 1:] != classes[:, :-1]).sum(axis=-1)
