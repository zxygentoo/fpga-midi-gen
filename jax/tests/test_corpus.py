"""jax/corpus.py: what a class means, what a stream of frames states on the wire, and
what a reader cuts out of the packed stream.

NONE OF THE THREE BELONGS TO AN ERA, which is why they stand beside the corpus. A break in
any of them is wrong music under a model that is perfectly correct.

THE CLASS MAP is where the wire's MIDI pitch meets the model's vocabulary: the wire
states any pitch and the model does not, and a map wrong by one semitone or one seat
round trips nothing. THE DECODE has an OCaml twin in `Frame.events_of_frames`, and the
eight cases here are the eight of its expect test and of docs/transformer.md. THE PACKED
STREAM is index arithmetic over the `frames.safetensors` export, thus its cases carry the
skip: a window off by one frame trains every step against itself, and a referee whose
stride slipped reads the same music twice and calls it a second measurement.

The other export, the whole pieces of era six, is read by `tests/test_diffusion.py`."""

import numpy as np
import pytest

import corpus
from tests import gate

needs_frames = gate.needs_frames


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


# the packed stream: what a reader cuts out of it


@needs_frames
def test_a_window_reads_its_own_stream_and_folds_the_position_into_the_bar():
    """A WINDOW IS CONTEXT + 1 FRAMES AND CONTEXT PHASES: the last frame is the one the
    last context step predicts, thus a reader that took context frames alone would train
    every step against itself.

    THE PHASE IS THE POSITION FOLDED INTO THE BAR. The export carries the rolling
    coordinate of the step and the model wants its low four bits, thus a window that opens
    inside a bar opens at that bar's phase and never at zero."""
    split = corpus.load_corpus(str(corpus.FRAMES))["train"]
    classes, phases = split.window(0, 20, 8)
    assert classes.shape == (9, corpus.SEATS) and phases.shape == (8,)
    assert phases.tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    # a stream reads from ITS OWN offset, which is what [index] carries: stream 1 at start
    # 0 is the file at index[1, 0] and not the file at 0
    at = int(split.index[1, 0])
    later, _ = split.window(1, 0, 8)
    assert np.array_equal(later, split.classes[at : at + 9])
    assert not np.array_equal(later, split.window(0, 0, 8)[0])


@needs_frames
def test_a_training_window_is_always_full_and_never_leaves_its_stream():
    """The training draw is a uniform stream, then a uniform window of it. THE LAST LEGAL
    START IS length - context - 1, where the window ends on the last frame of the stream:
    one further and it would read across the seam and hear the next stream's opening as
    the phrase this one closed with."""
    pool = corpus.train_pool(corpus.load_corpus(str(corpus.FRAMES)))
    rng = np.random.default_rng(0)
    context = 32
    for _ in range(64):
        classes, phases = corpus.train_row(rng, pool, context)
        assert classes.shape == (context + 1, corpus.SEATS)
        assert phases.shape == (context,)
    split, row = pool[3]
    at, length = (int(v) for v in split.index[row])
    last, _ = split.window(row, length - context - 1, context)
    assert last.shape == (context + 1, corpus.SEATS)
    assert np.array_equal(last[-1], split.classes[at + length - 1])


@needs_frames
def test_the_referee_reads_the_canonical_stream_at_stride_context():
    """The fixed windows of the referee: stream zero -- every piece at shift zero, in the
    order of the corpus -- cut at STRIDE CONTEXT from its start. The stride is the context
    and not one, thus no frame is predicted twice and the loss is a mean over the music.

    The limit caps the count and the stream caps the limit: a stream too short to hold one
    whole window states NO window rather than a short one, which the batcher below would
    otherwise stack against windows of another length."""
    split = corpus.load_corpus(str(corpus.FRAMES))["valid"]
    context = 256
    length = int(split.index[0, 1])
    rows = corpus.eval_rows(split, context, 4)
    assert len(rows) == 4
    for at, (classes, _) in enumerate(rows):
        assert classes.shape == (context + 1, corpus.SEATS)
        assert np.array_equal(classes, split.window(0, at * context, context)[0])
    # under no limit the count is what the stream holds, the last window ending inside it
    whole = (length - context - 1) // context + 1
    assert len(corpus.eval_rows(split, context, 1 << 20)) == whole
    assert corpus.eval_rows(split, length, 4) == []
