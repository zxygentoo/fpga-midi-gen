"""The batched twin of lib/core/prng.ml: Marsaglia xorshift32.

The OCaml module is the reference and the circuit computes the same recurrence, thus a
walk here agrees with the software, the simulation and the board byte for byte. That is
the point: a sweep in JAX can nominate a seed and an audition in OCaml plays it.

One state per batch element, held as uint32. Only an ACTIVE element advances -- a
finished walk must consume nothing, or every walk behind it in the batch shifts.
"""

import numpy as np

MASK32 = 0xFFFFFFFF

# Prng.uniform_bits: the grid a uniform stands on, three bytes of the walk. A threshold,
# an anneal table and an integer draw are all sized on it and each reads the width here.
UNIFORM_BITS = 24


def create(seed):
    """Prng.create: the walk that starts at [seed] itself, which is the rule of the
    board's SEED cell -- thus 0 is a seed like any other and its walk stands still. Take
    [create_folded] where the seed only has to name a walk."""
    if seed & MASK32 != seed:
        raise ValueError("Prng: the seed must fit 32 bits")
    return seed


def create_folded(seed):
    """Prng.create_folded: any integer names a walk. The mask comes after the mix, thus a
    wide seed still reaches the low bits; zero is no state, thus it takes the top one. A
    seed already inside the range names itself."""
    folded = (seed ^ (seed >> 32)) & MASK32
    return MASK32 if folded == 0 else folded


def states(seeds):
    """the starting state of each walk, as one uint32 array"""
    return np.array([create_folded(int(s)) for s in seeds], dtype=np.uint32)


def step(state):
    """one step over a batch, and its draw: the low 8 bits of the new state, which is the
    byte the circuit gives"""
    wide = state.astype(np.uint64)
    wide ^= (wide << np.uint64(13)) & np.uint64(MASK32)
    wide ^= wide >> np.uint64(17)
    wide ^= (wide << np.uint64(5)) & np.uint64(MASK32)
    wide &= np.uint64(MASK32)
    return wide.astype(np.uint32), (wide & np.uint64(0xFF)).astype(np.int64)


def uniform_word(state, active):
    """Prng.uniform_word over a batch: three steps, and the 24-bit word they make with the
    first byte highest.

    It is the uniform AS THE CIRCUITS TAKE IT -- an integer twin hands its draw a word and
    never a float -- thus [uniform] is this word on the grid and the two cannot part."""
    held = state.copy()
    state, high = step(state)
    state, middle = step(state)
    state, low = step(state)
    return np.where(active, state, held), (high * 256 + middle) * 256 + low


def uniform(state, active):
    """Prng.uniform over a batch: the word of [uniform_word] on the grid of 2 ** -24."""
    state, word = uniform_word(state, active)
    return state, word * (2.0**-UNIFORM_BITS)
