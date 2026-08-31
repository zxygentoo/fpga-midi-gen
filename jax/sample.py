"""The sampling: given the logits of one step and a uniform, which class.

THE FLOAT HALF OF ONE POLICY. Its integer twin is `quantized.pick` over Q15 weights and
a 24-bit word, and `quantized.temper_of` and `min_weight_of` are the temperature and the
floor folded into the machine's constants. THE TWO MUST STATE THE SAME POLICY: what
parts them is only the arithmetic, and `tests/test_sample.py` and
`tests/test_quantized.py` state the numbers each must give.

Every era draws through here. The step-frame chain redraws one seat at a time and era six
redraws a cell of a sheet, but the question is the same question and the answer is one
pair of functions. `prng.py` gives the uniform they consume; `lib/nn/sampler.ml` is the
circuit that does both at once.

NOTHING HERE READS A MODEL. The caller hands over the raw logits, thus a walk, a referee
and a drift report all reach this on the same terms.
"""

import numpy as np


def temper(raw, temperature, min_p):
    """the tempered weight of each class against the peak, then the min-p floor; the peak
    weighs one, thus min_p is a share of the peak"""
    weights = np.exp((raw - raw.max(axis=1, keepdims=True)) / temperature)
    if min_p > 0.0:
        weights = np.where(weights >= min_p, weights, 0.0)
    return weights


def pick_share(weights, share):
    """The class whose running total passes the draw.

    It takes the uniform and not a draw, thus one function owns both sums and the total is
    the last running total -- never a second sum of the same weights. numpy adds pairwise
    in sum() and left to right in cumsum(), thus two sums of one array differ in the last
    bits, and a draw made against the other sum can land above every running total, where
    no class passes at all.

    Against this total the draw is strictly below it, because the uniform falls under 1 by
    2**-24 at the least. Therefore the walk always ends on a class, and that class always
    holds weight the floor left standing: to reach the last index is to know that no
    earlier total passed, thus the weight there is the difference of two totals across the
    draw. No fallback is necessary, and none is written."""
    running = np.cumsum(weights, axis=1)
    return (running > (share * running[:, -1])[:, None]).argmax(axis=1)
