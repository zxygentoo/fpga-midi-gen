"""The grammar of data.Sounding must be the grammar of lib/core/sounding_state.ml.

The mask sits inside the training softmax, so the model's mass outside the legal set is
untrained: a sampler that draws under a looser mask emits tokens the weights never
learned, and a tighter one silently removes music the weights did learn. The cases below
are the ones the OCaml expect test pins, walked through the batched class.
"""

import numpy as np

import data


def on(pitch):
    return 128 + pitch


def off(pitch):
    return pitch


def walk(codes, batch=1):
    state = data.Sounding(batch)
    active = np.ones(batch, dtype=bool)
    for code in codes:
        state.step(np.full(batch, code), active)
    return state


def test_the_ons_of_a_sentence_fall():
    # after ON 67 then ON 64, the run falls: 64 is the ceiling of what may follow
    legal = walk([on(67), on(64)]).legal()[0]
    assert legal[on(60)]
    assert not legal[on(65)]
    assert not legal[on(64)]  # a sounding pitch never opens twice: the S-1 cross-kill
    assert legal[data.END]


def test_an_off_waits_for_the_next_sentence():
    # the sentence holds an ON, thus no OFF may follow it
    assert not walk([on(67), on(64)]).legal()[0][off(67)]
    # END closes the sentence and the OFFs open again
    assert walk([on(67), on(64), data.END]).legal()[0][off(67)]


def test_the_offs_climb():
    state = walk([on(67), on(64), data.END, off(64)])
    legal = state.legal()[0]
    assert legal[off(67)]  # above the last OFF
    assert not legal[off(64)]  # already released, and not above itself


def test_four_seats_and_no_fifth_voice():
    legal = walk([on(70), on(67), on(64), on(60)]).legal()[0]
    assert not legal[on(55)]  # the seats are full even though 55 falls below 60
    assert legal[data.END]


def test_start_is_never_legal_and_end_always_is():
    for codes in ([], [on(60)], [on(60), data.END]):
        legal = walk(codes).legal()[0]
        assert not legal[data.START]
        assert legal[data.END]


def test_the_batch_holds_independent_walks():
    """one element's sentence must not reach another's mask"""
    state = data.Sounding(2)
    state.step(np.array([on(67), data.END]), np.array([True, True]))
    first, second = state.legal()
    assert not first[off(67)]  # element 0 holds an ON, so no OFF
    assert second[data.END] and not second[off(67)]  # element 1 rings nothing
    assert first[on(60)] and second[on(60)]
