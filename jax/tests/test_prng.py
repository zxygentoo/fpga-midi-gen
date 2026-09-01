"""The batched walk must equal the scalar one.

prng.py vectorises lib/core/prng.ml over a batch, and a vectorised recurrence is where a
wrong dtype or a wrong mask hides: it still produces plausible numbers. These pin the walk
against a scalar reference written straight from the OCaml, and pin the one invariant
batching adds -- an inactive element must consume nothing. The end-to-end proof is gate C
of tests/test_parity.py.
"""

import numpy as np

import prng

MASK32 = 0xFFFFFFFF


def scalar_step(state):
    """lib/core/prng.ml, one state at a time"""
    state ^= (state << 13) & MASK32
    state ^= state >> 17
    state ^= (state << 5) & MASK32
    return state & MASK32


def test_create_folded_matches_the_ocaml_rule():
    # a seed inside the range names itself: seed 7 here is the board's seed 7
    assert prng.create_folded(7) == 7
    # zero is no state of the walk, thus it takes the top state
    assert prng.create_folded(0) == MASK32
    # the mask comes after the mix, so a seed wider than the state reaches the low bits
    wide = (5 << 32) | 9
    assert prng.create_folded(wide) == ((wide ^ (wide >> 32)) & MASK32)


def test_batched_step_equals_the_scalar_walk():
    seeds = [1, 7, 1997, MASK32, 123456789]
    state = prng.states(seeds)
    reference = [prng.create_folded(s) for s in seeds]
    for _ in range(200):
        state, byte = prng.step(state)
        reference = [scalar_step(s) for s in reference]
        assert list(state) == reference
        assert list(byte) == [s & 0xFF for s in reference]


def test_uniform_takes_three_steps_on_the_grid():
    state = prng.states([7])
    active = np.array([True])
    walked = prng.create_folded(7)
    high = walked = scalar_step(walked)
    middle = walked = scalar_step(walked)
    low = walked = scalar_step(walked)
    state, value = prng.uniform(state, active)
    expected = (((high & 0xFF) * 256 + (middle & 0xFF)) * 256 + (low & 0xFF)) * 2.0**-24
    assert value[0] == expected
    assert state[0] == walked
    assert 0.0 <= value[0] < 1.0


def test_the_word_is_the_float_draw_on_its_own_grid():
    """`uniform_word` is the uniform AS THE CIRCUITS TAKE IT -- an integer twin hands its
    draw a 24-bit word and never a float -- and `uniform` is that same word on the grid of
    2^-24. THE TWO MUST NOT PART: the drift report reads an integer draw against a float
    one, and a word off a different walk would read there as quantization cost.

    The batch carries a finished walk, thus the mask is welded too: it holds the STATE and
    not the number, and both forms must hold the same one."""
    state = prng.states([7, 1997, 11])
    active = np.array([True, False, True])
    worded, word = prng.uniform_word(state.copy(), active)
    floated, value = prng.uniform(state.copy(), active)
    assert list(word * 2.0**-prng.UNIFORM_BITS) == list(value)
    assert list(worded) == list(floated)
    assert worded[1] == state[1]  # the finished walk consumed nothing under either form
    assert all(0 <= number < (1 << prng.UNIFORM_BITS) for number in word)


def test_an_inactive_walk_consumes_nothing():
    """the invariant batching adds: a finished element must not advance, or every walk
    beside it in the batch draws numbers a solo run would never have drawn"""
    state = prng.states([7, 7])
    active = np.array([True, False])
    for _ in range(5):
        state, _ = prng.uniform(state, active)
    solo = prng.states([7])
    for _ in range(5):
        solo, _ = prng.uniform(solo, np.array([True]))
    assert state[0] == solo[0]
    assert state[1] == prng.create_folded(7)
