"""The batched twin of lib/core/prng.ml: Marsaglia xorshift32.

The OCaml module is the reference and the circuit computes the same recurrence, so a
walk here agrees with the software, the simulation and the board byte for byte. That
agreement is the point: it lets a sweep in JAX nominate a seed and an audition in OCaml
play the same music.

One state per batch element, held as uint32. A draw carries the state forward, and only
an active element advances -- a finished walk must consume nothing, or every walk queued
behind it in the same batch shifts.
"""

import numpy as np

MASK32 = 0xFFFFFFFF


def create_folded(seed):
    """Prng.create_folded: any integer names a walk.

    The mask comes after the mix, not inside it, so a seed wider than the state still
    reaches the low bits. Zero is no state of the walk, thus it takes the top state. A
    seed already inside the range names itself: seed 7 here is the board's seed 7."""
    folded = (seed ^ (seed >> 32)) & MASK32
    return MASK32 if folded == 0 else folded


def states(seeds):
    """the starting state of each walk, as one uint32 array"""
    return np.array([create_folded(int(s)) for s in seeds], dtype=np.uint32)


def step(state):
    """One step over a batch, and the draw of that step: the low 8 bits of the new state,
    which is the byte the circuit gives."""
    wide = state.astype(np.uint64)
    wide ^= (wide << np.uint64(13)) & np.uint64(MASK32)
    wide ^= wide >> np.uint64(17)
    wide ^= (wide << np.uint64(5)) & np.uint64(MASK32)
    wide &= np.uint64(MASK32)
    return wide.astype(np.uint32), (wide & np.uint64(0xFF)).astype(np.int64)


def uniform(state, active):
    """Prng.uniform over a batch: three steps, one draw over [0, 1).

    The three bytes make a grid of 2 ** -24. [active] holds the walks that draw; the rest
    keep the state they came in with, so a finished element never consumes a draw it
    would not have consumed in a run of its own."""
    held = state.copy()
    state, high = step(state)
    state, middle = step(state)
    state, low = step(state)
    value = ((high * 256 + middle) * 256 + low) * (2.0**-24)
    return np.where(active, state, held), value
