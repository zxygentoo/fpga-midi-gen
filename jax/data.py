"""The data side of the JAX trainer.

Reads the export of corpus_tool -- codes, rolling coordinates, bitpacked masks and the
stream index per split -- and mirrors the two batching policies of the OCaml trainer: the
training draw (a uniform stream, then a uniform window) and the fixed evaluation windows
of the referee (the canonical stream, stride context, from its start). Gate B holds this
file honest: the JAX valid loss and the checkpoint_tool eval of the same checkpoint must
agree, and they only can if these rows equal Evaluation.rows.

A split is a set of packed streams -- piece, seam, piece, seam -- and not a set of pieces.
There is no piece position and no short-piece padding: a window of a packed stream is
always full, thus every position weighs one. The two position tables read one array, the
coordinate of docs/improviser.md: the bar phase is its low four bits and the frame is its
high four.
"""

import numpy as np
from safetensors.numpy import load_file

BAR_STEPS = 16
SPLITS = ("train", "valid", "test")

VOCAB = 256
SEATS = 4
END, START = 0x00, 0xFF
NO_LAST_ON = 128  # a pitch above every ON, thus "the sentence holds no ON yet"
NO_LAST_OFF = -1  # a pitch below every OFF


class Sounding:
    """The batched twin of lib/core/sounding_state.ml: which pitches ring, and what the
    sentence has done so far.

    The grammar is the one the training mask carries, so the sampler must draw inside it
    or the model's untrained mass leaks out: START never -- it is input only; an ON when
    its pitch is silent, a seat of the four is open, and it falls below the last ON; an
    OFF when its pitch rings, the sentence holds no ON yet, and it climbs above the last
    OFF; END always.

    A code carries its kind in bit 7 and its pitch in bits 6:0, thus ON is 128..254 over
    pitch 0..126, OFF is 1..127 over pitch 1..127, code 0 is END and code 255 is START."""

    def __init__(self, batch):
        self.ringing = np.zeros((batch, 128), dtype=bool)
        self.last_on = np.full(batch, NO_LAST_ON, dtype=np.int64)
        self.last_off = np.full(batch, NO_LAST_OFF, dtype=np.int64)

    def legal(self):
        """the grammar flag of every code, [batch, 256] bool"""
        mask = np.zeros((len(self.last_on), VOCAB), dtype=bool)
        seats_free = (self.ringing.sum(axis=1) < SEATS)[:, None]
        no_on_yet = (self.last_on == NO_LAST_ON)[:, None]
        mask[:, 128:255] = (
            ~self.ringing[:, :127]
            & seats_free
            & (np.arange(127)[None, :] < self.last_on[:, None])
        )
        mask[:, 1:128] = (
            self.ringing[:, 1:128]
            & no_on_yet
            & (np.arange(1, 128)[None, :] > self.last_off[:, None])
        )
        mask[:, END] = True
        return mask

    def step(self, code, active):
        """walk one drawn code for each active element; it does not test legality"""
        is_on = (code >= 128) & (code != START)
        is_off = (code >= 1) & (code < 128)
        is_end = code == END
        pitch = np.where(is_on, code - 128, code)
        rows = np.arange(len(code))
        for flag, ringing in ((is_on & active, True), (is_off & active, False)):
            self.ringing[rows[flag], pitch[flag]] = ringing
        self.last_on = np.where(is_on & active, pitch, self.last_on)
        self.last_off = np.where(is_off & active, pitch, self.last_off)
        self.last_on = np.where(is_end & active, NO_LAST_ON, self.last_on)
        self.last_off = np.where(is_end & active, NO_LAST_OFF, self.last_off)


class Split:
    """The packed streams of one split. Stream zero is the canonical one -- every piece at
    shift zero, in the order of the corpus -- and the referee reads it alone."""

    def __init__(self, tensors, name):
        self.codes = tensors[f"{name}/codes"]
        self.positions = tensors[f"{name}/positions"]
        self.masks = tensors[f"{name}/masks"]  # packed words [tokens, 8]
        self.index = tensors[f"{name}/index"]  # [streams, 2]: offset, length

    def window(self, row, start, context):
        """one window of stream [row]: the codes, the two position tables and the masks"""
        at = int(self.index[row, 0]) + start
        positions = self.positions[at : at + context]
        return (
            self.codes[at : at + context + 1],
            (positions % BAR_STEPS).astype(np.int32),
            (positions // BAR_STEPS).astype(np.int32),
            self.masks[at : at + context],
        )


def load_corpus(path):
    tensors = load_file(str(path))
    return {name: Split(tensors, name) for name in SPLITS}


def unpack_masks(words):
    """[.., 8] int32 words -> [.., 256] bool, LSB first."""
    view = words.astype("<i4").view(np.uint8)
    return np.unpackbits(
        view.reshape(*words.shape[:-1], 32), axis=-1, bitorder="little"
    ).astype(bool)


def train_pool(corpus, train_on):
    """the (split, stream) pool of the -train-on flag"""
    names = {
        "train": ("train",),
        "train+test": ("train", "test"),
        "all": ("train", "test", "valid"),
    }[train_on]
    return [
        (corpus[name], row)
        for name in names
        for row in range(len(corpus[name].index))
    ]


def train_row(rng, pool, context):
    """One training row: a uniform stream, then a uniform window of it.

    A window of a packed stream is always full -- there is no short piece to pad and no
    position to drop from the loss -- thus every position weighs one. The piece draw is no
    longer uniform: a long piece holds more windows than a short one, which is the
    objective the endless walk asks for."""
    split, row = pool[rng.integers(len(pool))]
    length = int(split.index[row, 1])
    start = rng.integers(length - context)
    codes, phases, progress, masks = split.window(row, start, context)
    return codes, phases, progress, masks, np.ones(context, dtype=np.float32)


def train_batch(rng, pool, batch, context):
    rows = [train_row(rng, pool, context) for _ in range(batch)]
    codes = np.stack([r[0] for r in rows])
    phases = np.stack([r[1] for r in rows])
    progress = np.stack([r[2] for r in rows])
    masks = unpack_masks(np.stack([r[3] for r in rows]))
    weights = np.stack([r[4] for r in rows])
    return codes, phases, progress, masks, weights


def eval_rows(split, context, limit):
    """the fixed windows of the referee: the canonical stream, cut at stride context from
    its start -- Evaluation.rows in lib/transformer/evaluation.ml"""
    need = context + 1
    length = int(split.index[0, 1])
    if length < need:
        return []
    windows = min(limit, (length - need) // context + 1)
    return [split.window(0, window * context, context) for window in range(windows)]


def eval_batches(split, context, limit, batch):
    rows = eval_rows(split, context, limit)
    batches = []
    for at in range(0, len(rows), batch):
        chunk = rows[at : at + batch]
        codes = np.stack([r[0] for r in chunk])
        phases = np.stack([r[1] for r in chunk])
        progress = np.stack([r[2] for r in chunk])
        masks = unpack_masks(np.stack([r[3] for r in chunk]))
        batches.append((codes, phases, progress, masks, len(chunk)))
    return batches
