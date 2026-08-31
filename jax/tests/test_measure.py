"""The common battery, over a stack of sheets of class indices.

`measure.py` knows which era drew nothing: a Gibbs sheet, a walk of the packed stream and
a corpus crop all read the same way, thus the instruments are gated here and not inside an
era's file. Each test states a chord, a motion or a register whose answer is known by
hand, and the corpus row at the foot is the referee every other number is read against.

WHAT AN ERA MEASURES WITH ITS OWN MODEL IS NOT HERE: era six's likelihood referee is
`tests/test_diffusion.py`, and the forced pass and free walk of the step-frame eras are
`ar_measure.py`'s. The two names of era six this file does touch -- `model.VOICES` and
`model.CROP` -- are the shape of a sheet and nothing more.
"""

import itertools

import numpy as np
import pytest

import corpus
import measure
from diffusion import measure as referee
from diffusion import model
from tests import gate

PIECES = corpus.PIECES
needs_corpus = gate.needs_corpus


def held(pitches):
    """one sheet of two steps holding one sonority; None is a rest, seat 0 first"""
    classes = [
        corpus.SILENCE if pitch is None else pitch - corpus.PITCH_LOW + 1
        for pitch in pitches
    ]
    return np.tile(np.asarray(classes, dtype=np.int32), (1, 2, 1))


def test_the_battery_counts_a_triad_and_names_a_dissonance():
    """the two instruments that carry a chord, on chords whose answer is known by hand"""
    # a C major triad with the root doubled, then a cluster; a sheet is two steps because
    # the hold instrument reads the step before and one step has none
    row = measure.battery_row(held([36, 43, 52, 60]))
    assert row["triads"] == pytest.approx(100.0)
    assert row["dissonant"] == pytest.approx(0.0)
    assert row["order"] == pytest.approx(100.0)
    cluster = measure.battery_row(held([36, 37, 38, 39]))
    assert cluster["triads"] == pytest.approx(0.0)
    assert cluster["dissonant"] > 60.0


def test_the_battery_counts_triads_over_the_thick_steps_alone():
    """The method of the proto round, and the reason it is the method: a dyad sits inside
    some triad for free, thus counting every step flatters a thin sheet. A step of two
    voices must not reach the number at all."""
    row = measure.battery_row(held([36, 43, None, None]))
    assert row["triads"] == pytest.approx(0.0)  # no step carries three voices
    assert row["voices"][2] == pytest.approx(100.0)


def test_the_battery_sees_a_voice_out_of_register():
    """the bass above the tenor is the failure the order instrument is for"""
    assert measure.battery_row(held([60, 48, 52, 55]))["order"] == pytest.approx(0.0)


def test_the_pairs_stand_in_the_order_of_the_seats_between_them():
    """The battery reads left to right as the pitch reach runs out, thus the neighbours
    come first and the outer pair last. Every pair appears one time."""
    assert len(measure.PAIRS) == 6
    assert set(measure.PAIRS) == set(itertools.combinations(range(model.VOICES), 2))
    gaps = [high - low for low, high in measure.PAIRS]
    assert gaps == sorted(gaps)
    assert measure.PAIRS[0] == (0, 1) and measure.PAIRS[-1] == (0, 3)


def test_the_pair_instrument_reads_the_span_of_each_pair():
    """a chord whose spans are known by hand: the bass a twelfth under the soprano, and
    the two inner voices a third apart"""
    row = measure.battery_row(held([48, 64, 67, 79]))["pairs"]
    spans = {pair["name"]: pair["span"] for pair in row}
    assert spans["ba-so"] == pytest.approx(31.0)
    assert spans["te-al"] == pytest.approx(3.0)
    assert spans["ba-te"] == pytest.approx(16.0)


def test_a_rest_leaves_its_pairs_out_of_the_span():
    """a pair sounds only when both of its voices do, thus a rest is not an interval of
    zero and it must not pull a span down"""
    row = measure.battery_row(held([48, None, 67, 79]))["pairs"]
    spans = {pair["name"]: pair["span"] for pair in row}
    assert spans["ba-al"] == pytest.approx(19.0)
    assert spans["ba-te"] == 0.0 and spans["te-so"] == 0.0


def test_the_clash_counts_the_frame_and_not_the_pair():
    """The tail instrument. A seventh chord holds two dissonant pairs and is ordinary; a
    frame is a clash when three of its six pairs or more are dissonant. The mean
    dissonance cannot tell those apart, which is the whole reason this number exists."""
    # G B D F, a dominant seventh: G-F is a tone and B-F a tritone, thus two pairs
    seventh = measure.battery_row(held([55, 59, 62, 65]))
    assert seventh["dissonant"] > 0.0 and seventh["clash"] == pytest.approx(0.0)
    # a cluster: five of the six pairs are seconds or sevenths
    assert measure.battery_row(held([48, 49, 50, 51]))["clash"] == pytest.approx(100.0)


def moving(first, second):
    """one sheet of two steps, each a list of pitches with None for a rest"""
    return np.stack([held(first)[0, 0], held(second)[0, 0]])[None]


def test_the_parallel_instrument_catches_the_fifth_and_the_octave():
    """The fault that lives BETWEEN frames, on motions whose answer is known by hand. Two
    voices a fifth apart, both moving, landing on a fifth."""
    read = measure.battery_row(moving([48, 55, 64, 72], [50, 57, 64, 72]))["parallels"]
    assert read["fifths"] > 0.0 and read["octaves"] == pytest.approx(0.0)
    read = measure.battery_row(moving([48, 60, 64, 67], [50, 62, 64, 67]))["parallels"]
    assert read["octaves"] > 0.0 and read["fifths"] == pytest.approx(0.0)


def test_the_parallel_rate_is_per_moving_pair_and_not_per_sounding_one():
    """A parallel needs BOTH voices to move. A divisor of the pairs that merely sound pays
    a model for holding its notes, which the span round of 2026-08-25 caught it doing: the
    rate halved while the onsets fell a fifth below the corpus.

    One sheet of four steps holds one parallel fifth and one held step. Under the pairs
    that move it reads the whole of the motion; under the pairs that sound it would read
    half of it, for a sheet that wrote exactly the same fault."""
    sheet = np.stack(
        [
            held(row)[0, 0]
            for row in ([48, 55, 64, 72], [50, 57, 64, 72], [50, 57, 64, 72])
        ]
    )[None]
    read = measure.battery_row(sheet)["parallels"]
    # the bass and the tenor move together over step 1 and stand still over step 2, thus
    # one of the two live steps of that pair moves and the fault owns all of it
    assert read["fifths"] == pytest.approx(1000.0)
    assert read["moving"] < 100.0


def test_contrary_motion_onto_a_fifth_is_not_a_parallel():
    """The correction of 2026-08-25. The bass falls a tritone and the tenor rises one,
    thus the pair stands a fifth apart before and a twelfth after -- the same interval
    class, by CONTRARY motion. That is how a fifth is correctly approached, and counting
    it read 53 percent of the corpus's own fifths as faults."""
    read = measure.battery_row(moving([60, 67, 74, 79], [54, 73, 74, 79]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0)


def test_a_pair_that_crosses_holds_no_interval():
    """A fifth whose voices swap places became a fourth, thus its interval did not hold.
    The absolute gap cannot see the crossing -- both ends read seven -- and the order of
    the pair is what says otherwise. Here the bass leaps above the tenor and both rise."""
    read = measure.battery_row(moving([60, 67, 80, 84], [75, 68, 80, 84]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0)


def test_a_voice_that_holds_makes_no_parallel():
    """Two voices that keep a fifth while ONE of them stands still is oblique motion,
    which counterpoint permits and the ear does not object to. Only a pair that moves
    together can be parallel."""
    read = measure.battery_row(moving([48, 55, 64, 72], [48, 55, 65, 72]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0)


def test_contrary_motion_makes_no_parallel():
    """the two voices move, and the interval between them changes; nothing is parallel"""
    read = measure.battery_row(moving([48, 55, 64, 72], [50, 53, 64, 72]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0) and read["octaves"] == pytest.approx(0.0)


def test_a_silent_frame_is_not_a_clash():
    """a frame nobody sings holds no pair at all, thus it must leave the tail alone rather
    than count as clean"""
    assert measure.battery_row(held([None, None, None, None]))["clash"] == pytest.approx(
        0.0
    )


def test_the_register_sees_a_texture_that_slid_where_nothing_else_does():
    """A texture in good order, correctly spaced, and sitting a whole octave too low. The
    order instrument and the voice pairs both read it CLEAN -- the stacking holds and
    every span is unchanged -- thus the register mean is the only thing that can see it.

    And the tail is coarse on purpose. The seats overlap by 14 to 18 semitones, so a drop
    of an octave puts only the SOPRANO under its own floor of 60; the other three are
    still inside ranges their neighbours share. [outside] is a backstop for a gross
    departure and the mean is the sensitive instrument."""
    right = measure.battery_row(held([50, 59, 65, 71]))["register"]
    low = measure.battery_row(held([38, 47, 53, 59]))["register"]
    assert right["outside"] == pytest.approx(0.0)
    for seat, moved in zip(right["seats"], low["seats"]):
        assert moved["mean"] == pytest.approx(seat["mean"] - 12)
    # one seat of the four, thus a quarter of the sounding cells
    assert low["outside"] == pytest.approx(25.0)


def test_the_register_tells_drift_from_over_ranging():
    """The mean says a voice has moved and the spread says it wanders; a sheet that holds
    one chord has no spread at all, and one that alternates two has the half-distance."""
    still = measure.battery_row(held([50, 59, 65, 71]))["register"]
    assert all(seat["spread"] == pytest.approx(0.0) for seat in still["seats"])
    swung = np.concatenate([held([48, 59, 65, 71]), held([52, 59, 65, 71])], axis=1)
    read = measure.battery_row(swung)["register"]
    assert read["seats"][0]["mean"] == pytest.approx(50.0)
    assert read["seats"][0]["spread"] == pytest.approx(2.0)


def test_a_rest_leaves_its_seat_out_of_the_register():
    """a seat that does not sing states no pitch, thus it must not pull the mean toward
    the silence row"""
    read = measure.battery_row(held([50, None, 65, 71]))["register"]
    assert read["seats"][1]["mean"] == pytest.approx(0.0)
    assert read["outside"] == pytest.approx(0.0)


@needs_corpus
def test_the_corpus_row_stands_where_the_proto_round_left_it():
    """The corpus row is the referee of every number of the battery, thus the battery is
    read against the corpus and the corpus is pinned here. These are the sixteenth-grid
    figures; the proto round measured the eighth grid and its dissonance reads the same
    10.2 percent."""
    row = measure.battery_row(referee.corpus_sheets(str(PIECES), "train", model.CROP, 0))
    assert row["voices"][4] == pytest.approx(99.8, abs=0.1)
    assert row["triads"] == pytest.approx(63.9, abs=0.2)
    assert row["dissonant"] == pytest.approx(10.3, abs=0.2)
    assert row["hold"] == pytest.approx(76.9, abs=0.2)
    assert row["clash"] == pytest.approx(2.9, abs=0.3)
    # the horizontal referee: Bach essentially never writes them, thus any rate far above
    # this is a fault of the model and not a taste of the corpus. The divisor is the pairs
    # that MOVE, thus a rung cannot buy the number by holding its notes, and the share
    # that moves stands beside it to catch a rung whose motion has left the corpus.
    parallels = row["parallels"]
    # the fifths halved on 2026-08-25 when similar motion became a condition of the count
    assert parallels["fifths"] < 1.5 and parallels["octaves"] < 1.5
    assert 10.0 < parallels["moving"] < 40.0
    assert row["spare"] == 0.0
    # the register: the corpus cannot leave its own range, and the four means are the
    # reference every rung is read against
    register = row["register"]
    assert register["outside"] == 0.0
    for seat, mean in zip(register["seats"], (50.9, 59.6, 65.1, 70.6)):
        assert seat["mean"] == pytest.approx(mean, abs=0.2)
    # the reference of the pair instrument: the corpus is near 10 percent dissonant at
    # every span, thus a rung that reads 27 at the widest pair has a reach fault and not
    # a taste
    pairs = {pair["name"]: pair for pair in row["pairs"]}
    assert pairs["ba-so"]["span"] == pytest.approx(19.7, abs=0.2)
    assert pairs["al-so"]["span"] == pytest.approx(5.5, abs=0.2)
    assert all(8.0 < pair["dissonant"] < 12.0 for pair in row["pairs"])
