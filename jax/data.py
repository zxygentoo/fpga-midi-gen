"""The data side of the JAX trainer.

Reads the export of corpus_tool -- codes, phases, bitpacked masks and the variant index
per split -- and mirrors the two batching policies of the OCaml trainer: the training
draw (uniform piece, then uniform legal shift, then a random window) and the fixed
evaluation windows of the referee (shift zero, stride context, pieces in order). Gate B
holds this file honest: the JAX valid loss and the checkpoint_tool eval of the same
checkpoint must agree, and they only can if these rows equal Evaluation.rows.
"""

import numpy as np
from safetensors.numpy import load_file

BAR_STEPS = 16
PROGRESS_BUCKETS = 16
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


def piece_progress(codes):
    """The progress bucket of each token: which sixteenth of its piece the step sits in.

    The bar phase says where a step is in the bar; nothing else in the stream says where
    it is in the piece. The corpus earns the ratio: a repeated bar of the top line comes
    again at a distance that is more constant as a part of the piece (CV 0.48) than as a
    count of steps (CV 0.58). The step of a token is the count of ENDs before it, and the
    length of the piece is its count of ENDs."""
    ends = codes == 0
    before = np.cumsum(ends) - ends
    total = max(int(ends.sum()), 1)
    bucket = before * PROGRESS_BUCKETS // total
    return np.minimum(bucket, PROGRESS_BUCKETS - 1).astype(np.int32)


class Split:
    def __init__(self, tensors, name):
        self.codes = tensors[f"{name}/codes"]
        self.phases = tensors[f"{name}/phases"]
        self.masks = tensors[f"{name}/masks"]  # packed words [tokens, 8]
        self.index = tensors[
            f"{name}/index"
        ]  # [variants, 4]: piece, shift, offset, length
        pieces = int(self.index[:, 0].max()) + 1 if len(self.index) else 0
        self.by_piece = [[] for _ in range(pieces)]
        for row, (piece, _, _, _) in enumerate(self.index):
            self.by_piece[piece].append(row)
        # progress needs the whole piece, so it is built here and sliced like the phases
        self.progress = np.zeros(len(self.codes), dtype=np.int32)
        for _, _, offset, length in self.index:
            sl = slice(offset, offset + length)
            self.progress[sl] = piece_progress(self.codes[sl])

    def variant(self, row):
        _, _, offset, length = self.index[row]
        sl = slice(offset, offset + length)
        return self.codes[sl], self.phases[sl], self.progress[sl], self.masks[sl]

    def zero_shift_row(self, piece):
        for row in self.by_piece[piece]:
            if self.index[row, 1] == 0:
                return row
        raise ValueError(f"piece {piece} has no zero shift")


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
    """the (split, piece) pool of the -train-on flag"""
    names = {
        "train": ("train",),
        "train+test": ("train", "test"),
        "all": ("train", "test", "valid"),
    }[train_on]
    return [
        (corpus[name], piece)
        for name in names
        for piece in range(len(corpus[name].by_piece))
    ]


def train_row(rng, pool, context):
    """one training row: uniform piece, uniform legal shift, then a random window. A
    short piece pads with the zero word, and the weights drop the padded positions from
    the loss: a padded label would teach the walk to hold the last chord of the piece and
    emit END for ever, the drone. The padding grows with the context — one piece of 306
    at 256 tokens, but 81 of 306 at 512."""
    split, piece = pool[rng.integers(len(pool))]
    rows = split.by_piece[piece]
    codes, phases, progress, masks = split.variant(rows[rng.integers(len(rows))])
    need = context + 1
    length = len(codes)
    if length >= need:
        start = rng.integers(length - need + 1)
        return (
            codes[start : start + need],
            phases[start : start + context],
            progress[start : start + context],
            masks[start : start + context],
            np.ones(context, dtype=np.float32),
        )
    steps = int((codes == 0).sum())
    padded_codes = np.zeros(need, dtype=codes.dtype)
    padded_codes[:length] = codes
    pad = np.arange(length, context)
    padded_phases = np.concatenate([phases[:length], (steps + pad - length) % BAR_STEPS])
    # the piece is over in the padding, thus the last bucket holds it
    padded_progress = np.concatenate(
        [progress[:length], np.full(context - length, PROGRESS_BUCKETS - 1)]
    )
    padded_masks = np.concatenate(
        [masks[:length], np.repeat(masks[length - 1 : length], context - length, axis=0)]
    )
    # position i draws label i + 1: it is real while i + 1 is inside the piece
    weights = (np.arange(context) + 1 < length).astype(np.float32)
    return (
        padded_codes,
        padded_phases.astype(phases.dtype),
        padded_progress.astype(progress.dtype),
        padded_masks,
        weights,
    )


def train_batch(rng, pool, batch, context):
    rows = [train_row(rng, pool, context) for _ in range(batch)]
    codes = np.stack([r[0] for r in rows])
    phases = np.stack([r[1] for r in rows])
    progress = np.stack([r[2] for r in rows])
    masks = unpack_masks(np.stack([r[3] for r in rows]))
    weights = np.stack([r[4] for r in rows])
    return codes, phases, progress, masks, weights


def eval_rows(split, context, limit):
    """the fixed windows of the referee: shift zero, pieces in order, stride context,
    short pieces skipped -- Evaluation.rows in lib/evaluation.ml"""
    rows = []
    need = context + 1
    for piece in range(len(split.by_piece)):
        codes, phases, progress, masks = split.variant(split.zero_shift_row(piece))
        length = len(codes)
        if length < need:
            continue
        for window in range((length - need) // context + 1):
            start = min(window * context, length - need)
            rows.append(
                (
                    codes[start : start + need],
                    phases[start : start + context],
                    progress[start : start + context],
                    masks[start : start + context],
                )
            )
            if len(rows) == limit:
                return rows
    return rows


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
