"""The step frame: the class map, the decode and the chain.

The decode has an OCaml twin in Frame.events_of_frames of lib/core/frame.ml, and the eight
cases here are the eight of its expect test and of docs/transformer_model.md. Both were
measured against the packed corpus and give its texture -- onsets/step 0.81, single-ON
0.10, median 4.0, under a quarter 0.37 -- which is the number the token era recorded.
"""

import jax
import numpy as np
import pytest

import data
from transformer import model


def frame(seats):
    """one frame from its four classes, seat 0 first: seat 0 is the bass, seat 3 the
    soprano, thus this list reads low to high"""
    return np.array(
        [0 if pitch < 0 else pitch - data.PITCH_LOW + 1 for pitch in seats],
        dtype=np.int32,
    )


SILENT = frame([-1, -1, -1, -1])


def test_the_class_map_is_one_subtraction():
    codes = np.array([[0x00, 0x80 | 36, 0x80 | 81, 0x00]], dtype=np.int32)
    classes = data.classes_of_codes(codes)
    assert classes.tolist() == [[0, 1, 46, 0]]
    # class 47 is the spare the design keeps, thus the table has one row over the corpus
    assert classes.max() < data.CLASSES - 1
    assert data.pitches_of_classes(np.array([1, 46])).tolist() == [36, 81]


def test_a_pitch_outside_the_vocabulary_refuses():
    """the wire states any MIDI pitch and the model does not; the two are different
    questions, and the class map is where they meet"""
    with pytest.raises(ValueError, match="outside"):
        data.classes_of_codes(np.array([[0x80 | 20, 0, 0, 0]], dtype=np.int32))


def test_the_decode_makes_the_eight_cases():
    """the table of docs/transformer_model.md, and the expect test of Frame"""
    cases = {
        "hold": ([frame([60, -1, -1, -1])] * 2, []),
        "strike": ([SILENT, frame([60, -1, -1, -1])], [("on", 60)]),
        "release": ([frame([60, -1, -1, -1]), SILENT], [("off", 60)]),
        "move": (
            [frame([60, -1, -1, -1]), frame([62, -1, -1, -1])],
            [("off", 60), ("on", 62)],
        ),
        # the exchange: a seat walk would send On 60 before Off 60 and the synth would
        # stop the new note, because the voices share one channel
        "exchange": ([frame([-1, 60, 64, -1]), frame([-1, 64, 60, -1])], []),
        # the unison: a set holds a pitch one time, thus one Note On and one Note Off
        "unison in": ([frame([-1, 60, 64, -1]), frame([-1, 60, 60, -1])], [("off", 64)]),
        "unison out": ([frame([-1, 60, 60, -1]), frame([-1, 60, 64, -1])], [("on", 64)]),
        "seam": (
            [frame([48, 55, 60, 64]), SILENT],
            [("off", 48), ("off", 55), ("off", 60), ("off", 64)],
        ),
    }
    for name, (frames, events) in cases.items():
        assert data.decode(np.stack(frames))[1] == events, name


def test_the_decode_keeps_its_three_properties():
    """the properties the legality mask used to carry, now held by the rule itself"""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, data.CLASSES - 1, size=(4096, data.SEATS), dtype=np.int64)
    sounding = set()
    for events in data.decode(frames):
        for kind, pitch in events:
            if kind == "on":
                assert pitch not in sounding, "a strike of a pitch that sounds"
                sounding.add(pitch)
            else:
                assert pitch in sounding, "a release of a pitch that does not sound"
                sounding.discard(pitch)
        assert len(sounding) <= data.SEATS, "five notes sound at the same time"


def test_the_chain_conditions_downward():
    """Each seat reads what the seats above it drew, and nothing reads a seat below.

    A chain wired the wrong way round is silent: the shapes stay right, the loss still
    falls, and the model only loses the joint choice that is the whole reason for it."""
    params = {
        "seats": jax.random.normal(jax.random.PRNGKey(0), (data.SEATS, data.CLASSES, 8))
    }
    h = np.zeros((1, 3, 8), dtype=np.float32) + 0.5
    base = np.ones((1, 3, data.SEATS), dtype=np.int32)

    def logits(drawn):
        return np.asarray(model.seat_logits(params, h, drawn))

    # the soprano is drawn first, thus it conditions on nothing and every seat under it
    # moves when it changes
    soprano = base.copy()
    soprano[..., 3] = 2
    from_base, from_soprano = logits(base), logits(soprano)
    assert np.allclose(from_base[..., 3, :], from_soprano[..., 3, :])
    for seat in (2, 1, 0):
        assert not np.allclose(from_base[..., seat, :], from_soprano[..., seat, :])

    # the bass is drawn last, thus no seat reads it and the whole readout stands still
    bass = base.copy()
    bass[..., 0] = 2
    assert np.allclose(from_base, logits(bass))
