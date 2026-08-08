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
SPLITS = ("train", "valid", "test")


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

    def variant(self, row):
        _, _, offset, length = self.index[row]
        sl = slice(offset, offset + length)
        return self.codes[sl], self.phases[sl], self.masks[sl]

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
    codes, phases, masks = split.variant(rows[rng.integers(len(rows))])
    need = context + 1
    length = len(codes)
    if length >= need:
        start = rng.integers(length - need + 1)
        return (
            codes[start : start + need],
            phases[start : start + context],
            masks[start : start + context],
            np.ones(context, dtype=np.float32),
        )
    steps = int((codes == 0).sum())
    padded_codes = np.zeros(need, dtype=codes.dtype)
    padded_codes[:length] = codes
    pad = np.arange(length, context)
    padded_phases = np.concatenate([phases[:length], (steps + pad - length) % BAR_STEPS])
    padded_masks = np.concatenate(
        [masks[:length], np.repeat(masks[length - 1 : length], context - length, axis=0)]
    )
    # position i draws label i + 1: it is real while i + 1 is inside the piece
    weights = (np.arange(context) + 1 < length).astype(np.float32)
    return padded_codes, padded_phases.astype(phases.dtype), padded_masks, weights


def train_batch(rng, pool, batch, context):
    rows = [train_row(rng, pool, context) for _ in range(batch)]
    codes = np.stack([r[0] for r in rows])
    phases = np.stack([r[1] for r in rows])
    masks = unpack_masks(np.stack([r[2] for r in rows]))
    weights = np.stack([r[3] for r in rows])
    return codes, phases, masks, weights


def eval_rows(split, context, limit):
    """the fixed windows of the referee: shift zero, pieces in order, stride context,
    short pieces skipped -- Evaluation.rows in lib/evaluation.ml"""
    rows = []
    need = context + 1
    for piece in range(len(split.by_piece)):
        codes, phases, masks = split.variant(split.zero_shift_row(piece))
        length = len(codes)
        if length < need:
            continue
        for window in range((length - need) // context + 1):
            start = min(window * context, length - need)
            rows.append(
                (
                    codes[start : start + need],
                    phases[start : start + context],
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
        masks = unpack_masks(np.stack([r[2] for r in chunk]))
        batches.append((codes, phases, masks, len(chunk)))
    return batches
