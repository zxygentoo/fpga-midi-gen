"""What every era's training shares: the rate curve, the update rule, and the checkpoint
a run writes.

Not the loop. Eras four and five share one in `ar_train.py` and era six keeps its own in
`diffusion/train.py`, because the shape of a step is the era's own. A checkpoint stands
here and not with a model, because it is what a RUN writes; the contract file is a
different archive and a different seam, in `quantized.py`.
"""

from pathlib import Path

import jax
import numpy as np
import optax
from safetensors.numpy import save_file

# TRUE FLOAT32 AND NO TF32. It must run before any model is built; every model module
# reaches this one for `save_checkpoint`, thus the pin holds everywhere.
jax.config.update("jax_default_matmul_precision", "float32")


def learning_rates(peak, warmup, total):
    """The rate at every step: linear from 0 to [peak] over [warmup] steps, then cosine to
    0 over the rest. A warmup of zero is a constant; a run shorter than its warmup never
    leaves the ramp, where optax refuses a cosine of negative length.

    THE `+ 1` IS NOT SPARE. Optax hands a schedule its update count, which is 0 where the
    loop's step is 1; without the correction the first update takes a rate of 0."""
    if warmup == 0:
        curve = optax.constant_schedule(peak)
    elif total <= warmup:
        curve = optax.linear_schedule(0.0, peak, warmup)
    else:
        curve = optax.warmup_cosine_decay_schedule(0.0, peak, warmup, total, 0.0)
    return lambda count: curve(count + 1)


def update_rule(*, peak, warmup, total, clip, weight_decay):
    """The global-norm clip, then Adam with a decoupled weight decay under the schedule.

    A clip of zero or less is NO CLIP -- the node is left out, not set to zero, which
    would zero every gradient of the run."""
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
    """The naming rule of the seam: the tensors named "0" upward in construction order,
    then the ALiBi span last and alone where the model carries one. A reader takes whole
    layer groups and then one scalar if one is there, thus an older file still reads."""
    if span is not None:
        tensors = list(tensors) + [np.asarray([span], dtype=np.float32)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({str(i): np.asarray(t) for i, t in enumerate(tensors)}, path)
