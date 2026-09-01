"""The vocabulary and the decode of jax/corpus.py: what a class means, and what a stream
of frames states on the wire.

NEITHER OF THEM BELONGS TO AN ERA, which is why they stand beside the corpus. A break in
either is wrong music under a model that is perfectly correct.

THE CLASS MAP is where the wire's MIDI pitch meets the model's vocabulary: the wire
states any pitch and the model does not, and a map wrong by one semitone or one seat
round trips nothing. THE DECODE has an OCaml twin in `Frame.events_of_frames`, and the
eight cases here are the eight of its expect test and of docs/transformer.md."""

import numpy as np
import pytest

import corpus


def frame(seats):
    """one frame from its four classes, seat 0 first: seat 0 is the bass, seat 3 the
    soprano, thus this list reads low to high"""
    return np.array(
        [0 if pitch < 0 else pitch - corpus.PITCH_LOW + 1 for pitch in seats],
        dtype=np.int32,
    )


SILENT = frame([-1, -1, -1, -1])


def test_the_class_map_is_one_subtraction():
    codes = np.array([[0x00, 0x80 | 36, 0x80 | 81, 0x00]], dtype=np.int32)
    classes = corpus.classes_of_codes(codes)
    assert classes.tolist() == [[0, 1, 46, 0]]
    # class 47 is the spare the design keeps, thus the table has one row over the corpus
    assert classes.max() < corpus.CLASSES - 1
    assert corpus.pitches_of_classes(np.array([1, 46])).tolist() == [36, 81]


def test_a_pitch_outside_the_vocabulary_refuses():
    """the wire states any MIDI pitch and the model does not; the two are different
    questions, and the class map is where they meet"""
    with pytest.raises(ValueError, match="outside"):
        corpus.classes_of_codes(np.array([[0x80 | 20, 0, 0, 0]], dtype=np.int32))


def test_the_decode_makes_the_eight_cases():
    """the table of docs/transformer.md, and the expect test of Frame"""
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
        assert corpus.decode(np.stack(frames))[1] == events, name


def test_the_decode_keeps_its_three_properties():
    """the properties the legality mask used to carry, now held by the rule itself"""
    rng = np.random.default_rng(0)
    shape = (4096, corpus.SEATS)
    frames = rng.integers(0, corpus.CLASSES - 1, size=shape, dtype=np.int64)
    sounding = set()
    for events in corpus.decode(frames):
        for kind, pitch in events:
            if kind == "on":
                assert pitch not in sounding, "a strike of a pitch that sounds"
                sounding.add(pitch)
            else:
                assert pitch in sounding, "a release of a pitch that does not sound"
                sounding.discard(pitch)
        assert len(sounding) <= corpus.SEATS, "five notes sound at the same time"
