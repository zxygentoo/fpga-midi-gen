"""The sampling: given the logits of one step and a uniform, which class.

The float half of one policy. `quantized.pick` is the integer twin over Q15 weights and a
24-bit word, `quantized.temper_of` and `min_weight_of` fold the temperature and the floor
into the machine's constants, and `lib/nn/sampler.ml` is the circuit that does both at
once. The two must state the same policy; `tests/test_sample.py` and
`tests/test_quantized.py` state the numbers each must give.

Every era draws through here, on raw logits and a uniform from `prng.py`. What one era
alone draws stands with that era: `diffusion/sample.py` is the walk of era six.
"""

import numpy as np


def tempered_weight(raw, temperature, min_p):
    """the tempered weight of each class against the peak, then the min-p floor; the peak
    weighs one, thus min_p is a share of the peak"""
    weights = np.exp((raw - raw.max(axis=1, keepdims=True)) / temperature)
    return weights if min_p <= 0.0 else np.where(weights >= min_p, weights, 0.0)


def pick_share(weights, share):
    """The class whose running total passes the draw.

    ONE FUNCTION OWNS BOTH SUMS: numpy adds pairwise in sum() and left to right in
    cumsum(), thus a draw made against a second sum of the same weights can land above
    every running total and no class passes. Against the last running total the draw is
    strictly below, thus a class always stands and no fallback is necessary."""
    running = np.cumsum(weights, axis=1)
    return (running > (share * running[:, -1])[:, None]).argmax(axis=1)
