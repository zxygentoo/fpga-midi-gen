"""What every era's TRAINING shares: the rate curve, the update rule, and the checkpoint
a run writes.

Not the loop. Eras four and five share one in `ar_train.py` and era six keeps its own in
`diffusion/train.py`, because the shape of a step is the era's own -- the sheet folds a
batch-norm population where the step frames draw a dropout mask. What is here is what all
three do identically, and a rate read one step late would be read one step late by all
three at once.

THE CHECKPOINT STANDS HERE AND NOT WITH A MODEL: a checkpoint is what a run writes, and
its naming rule is the seam every era's `load` reads back. The CONTRACT FILE is a
different archive and a different seam -- that one is `quantized.py`, below the
quantization, and it carries a model no trainer ever sees again.

THE MATMUL PRECISION PIN LIVES HERE, and it must run before any model is built. Every
model module reaches this one for `save_checkpoint` and both trainers reach it for the
rule, thus the pin holds everywhere and no era can quietly train in TF32.
"""

from pathlib import Path

import jax
import numpy as np
import optax
from safetensors.numpy import save_file

# TRUE FLOAT32 AND NO TF32. The docstring above states why the pin stands in this module.
jax.config.update("jax_default_matmul_precision", "float32")


def learning_rates(peak, warmup, total):
    """The rate at every step of the run: linear from 0 to [peak] over [warmup] steps,
    then cosine from [peak] to 0 over the rest. A warmup of zero is a constant.

    THE SCHEDULE IS READ ONE STEP LATE OR NOT AT ALL. Optax hands a schedule its own
    update count, which is 0 at the first update where the loop's step is 1, thus a curve
    read at the raw count applies a rate of 0 to the first update and every later rate one
    step behind. The `+ 1` is that correction and `tests/test_train.py` holds it.

    THE TWO ENDS ARE THIS PROJECT'S RULES AND NOT OPTAX'S. A warmup of zero is a constant
    peak, where `warmup_cosine_decay_schedule` would be a bare cosine decay; and a run
    SHORTER THAN ITS OWN WARMUP -- which every short probe is -- never leaves the ramp,
    where optax refuses to build a cosine of a negative length at all."""
    if warmup == 0:
        curve = optax.constant_schedule(peak)
    elif total <= warmup:
        curve = optax.linear_schedule(0.0, peak, warmup)
    else:
        curve = optax.warmup_cosine_decay_schedule(0.0, peak, warmup, total, 0.0)
    return lambda count: curve(count + 1)


def update_rule(*, peak, warmup, total, clip, weight_decay):
    """The update of one step: the global-norm clip, then Adam with a decoupled weight
    decay under the schedule.

    A clip of zero or less is NO CLIP. It is not a clip at zero, which would zero every
    gradient of the run. A weight decay of zero makes AdamW Adam, by arithmetic and not by
    a second code path -- which is what era six's paper asks for."""
    adam = optax.adamw(
        learning_rate=learning_rates(peak, warmup, total),
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    if clip <= 0.0:
        return adam
    return optax.chain(optax.clip_by_global_norm(clip), adam)


def save_checkpoint(path, tensors, span=None):
    """The naming rule of the seam: the tensors named "0" upward, in construction order,
    then the ALiBi span last and alone where the model carries one -- an older file that
    does not still reads, because a reader takes whole layer groups and then one scalar
    if one is there. The era's trainer builds the flat list, because the layer layouts
    are its own."""
    if span is not None:
        tensors = list(tensors) + [np.asarray([span], dtype=np.float32)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({str(i): np.asarray(t) for i, t in enumerate(tensors)}, path)
