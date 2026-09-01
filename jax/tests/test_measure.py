"""The measurement: the common battery over sheets of class indices, the drift count that
reads two models against each other, and the step-frame referee.

`measure.py` AND `ar_measure.py` UNDER ONE DOCSTRING, as `test_quantized.py` holds
`quantized.py` and `ar_quantized.py`: the cut between the pair runs one way, thus the two
stand together and each test below says which side of it the rule is on. The common
battery knows which era drew nothing -- a Gibbs sheet, a walk of the packed stream and a
corpus crop all read the same way -- and the step-frame half asks eras four and five their
two questions, the FORCED pass over the corpus's own windows and the FREE walk.

Each test states a chord, a motion, a register or a window whose answer is known by hand.
The corpus row at the foot is the referee every battery number is read against, and the
forced pass runs on a stream written here, whose music is arithmetic.

WHAT ERA SIX MEASURES WITH ITS OWN MODEL IS NOT HERE: its likelihood referee is
`tests/test_diffusion.py`. The two names of era six this file does touch -- `model.VOICES`
and `model.CROP` -- are the shape of a sheet and nothing more.
"""

import itertools
import math

import numpy as np
import pytest
from safetensors.numpy import save_file

import ar_measure
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
    """A parallel needs BOTH voices to move, thus a divisor of the pairs that merely sound
    pays a model for holding its notes. One sheet of four steps holds one parallel fifth
    and one held step: per moving pair it reads the whole of the motion, per sounding pair
    half of it, for the same fault."""
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
    """A texture in good order, correctly spaced, and a whole octave too low: the order
    instrument and the voice pairs both read it CLEAN, thus the register mean is the only
    thing that can see it. The tail is coarse on purpose -- the seats overlap by 14 to 18
    semitones, so the drop puts only the soprano under its own floor."""
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
    # this is a fault of the model. The divisor is the pairs that MOVE, thus a rung cannot
    # buy the number by holding its notes.
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


# the drift count: the twin's draw against the float model's


def test_the_cosine_reads_the_shape_of_a_row_and_not_its_scale():
    """The third number of every drift report, at rows whose answer is known by hand: a
    row against itself is 1, a row against a scaling of itself is still 1, and a row
    against one at 45 degrees to it is the root of a half."""
    twin = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    floated = np.array([[1.0, 0.0], [7.0, 0.0], [1.0, 1.0]])
    assert list(measure.cosines(twin, floated)) == pytest.approx([1.0, 1.0, 0.5**0.5])


def test_the_drift_count_adds_a_batch_onto_what_it_has_counted():
    """THE INSTRUMENT ITSELF IS GATED NOWHERE ELSE, and that is the reason this stands
    here: the drift tables are measured numbers and are legitimately re-pinned, thus a
    fault in the instrument would be absorbed into the next re-pin with nothing to say so.

    Two rows over three classes, on numbers chosen by hand. The float rows are one row
    twice; the twin agrees with it on the first and reverses it on the second, thus one of
    the two elects the same class and the cosines are 1 and 9/73. Under a temperature of
    one and a min-p of 0.01 the weights are 1, e^-3 and 0, thus THE PICK LEAVES CLASS 0 AT
    A SHARE OF 1/1.0498 and the two uniforms straddle it; the twin is said to have drawn
    class 0 both times, thus one of the two draws agrees. The count adds onto a report
    that has already seen ten draws, because a walk calls this once for each step."""
    floated = np.array([[0.0, -3.0, -8.0]] * 2)
    twin = np.array([[0.0, -3.0, -8.0], [-8.0, -3.0, 0.0]])
    counted = measure.count_draws(
        measure.Counted(draws=10, same_peak=5, same_draw=4, cosine=3.0),
        twin,
        floated,
        drawn=np.array([0, 0]),
        uniform=np.array([0.95, 0.96]),
        temperature=1.0,
        min_p=0.01,
    )
    assert (counted.draws, counted.same_peak, counted.same_draw) == (12, 6, 5)
    assert counted.cosine == pytest.approx(3.0 + 1.0 + 9.0 / 73.0)


# the step-frame referee: the forced pass, and the error over walks


# The free walk's own arithmetic is the common battery's, gated above; what is this half's
# alone is the CUT the forced pass makes and the error the several walks carry.

# THE STREAM IS WRITTEN HERE AND ITS MUSIC IS ARITHMETIC: the four voices hold a chord for
# four steps and then all move together, and each whole window of the referee's context
# stands one semitone above the one before. Two facts follow that no measurement decides:
# a step moves exactly when its bar phase is 3 modulo 4, and each window opens on a bass
# pitch of its own. The stubs below read those two and never `ar_measure.moving`.

VOICE_BASE = (48, 55, 60, 67)
FIRST_BASS = VOICE_BASE[0] - corpus.PITCH_LOW + 1
# three whole windows of the referee's context, thus a limit of 3 and a batch of 2 leave
# the last batch short
STREAM_STEPS = 3 * ar_measure.CONTEXT + 1


def stream_pitch(step, seat):
    """the pitch of one seat at one step of the written stream"""
    return VOICE_BASE[seat] + ((step // 4) % 2) * 2 + step // ar_measure.CONTEXT


def written_corpus(path):
    """The stream as `corpus_tool` writes one, in the layout `corpus.Split` reads: the
    wire codes, the rolling coordinate of each step, and one stream in the index. All
    three splits carry it, because `load_corpus` reads them all and the referee reads
    valid."""
    steps = np.arange(STREAM_STEPS)
    seats = range(corpus.SEATS)
    frames = np.array(
        [[0x80 | stream_pitch(step, seat) for seat in seats] for step in steps], np.int32
    )
    tensors = {}
    for name in corpus.SPLITS:
        tensors[f"{name}/frames"] = frames
        tensors[f"{name}/positions"] = steps.astype(np.int32)
        tensors[f"{name}/index"] = np.array([[0, STREAM_STEPS]], np.int32)
    save_file(tensors, str(path))
    return str(path)


class Trunk:
    """a stub in the shape of a trained model: `loss_row` reads nothing of one but this"""

    def __init__(self, nll_of):
        self.nll_of = nll_of

    def seat_nll(self, classes, phases):
        return self.nll_of(np.asarray(classes), np.asarray(phases))


def nats_where_the_voices_move(classes, phases):
    """one nat for each seat index on the steps whose bar phase is 3 modulo 4, and nothing
    anywhere else -- which on this stream is exactly the steps where all four voices
    move"""
    del classes
    seats = np.arange(1, corpus.SEATS + 1, dtype=np.float64)
    return np.where((phases % 4 == 3)[..., None], seats, 0.0)


def nats_of_the_window(classes, phases):
    """a cost that is a CONSTANT OF ITS WINDOW: the window's opening bass class, which the
    stream sets one higher at each window, thus the three eval rows cost 0, 1 and 2"""
    rows = (classes[:, 0, 0] - FIRST_BASS).astype(np.float64)
    return np.broadcast_to(rows[:, None, None], (*phases.shape, corpus.SEATS))


def test_the_forced_pass_cuts_the_moving_steps_from_the_still_ones(tmp_path):
    """78 percent of the voice slots repeat the step before, thus a mean can hide a
    model's whole deficit in the quarter of the steps that carry the music. MOVING is the
    steps where two voices or more move, STILL is the complement, and era five's whole
    story was in the pair.

    One step in four moves here and the stub costs its seat index in nats there and
    nothing elsewhere: the moving row reads the whole cost, the still row reads zero, and
    the mean reads a quarter -- which is the hiding this cut exists to stop."""
    path = written_corpus(tmp_path / "frames.safetensors")
    row = ar_measure.loss_row(Trunk(nats_where_the_voices_move), path, limit=3, batch=16)
    assert (row["moves"], row["steps"]) == (192, 768)
    assert row["moving"] == pytest.approx(10.0)  # 1 + 2 + 3 + 4 nats, at every such step
    assert row["still"] == pytest.approx(0.0)
    assert row["loss"] == pytest.approx(2.5)
    assert list(row["seats"]) == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_the_forced_pass_sums_over_a_count_and_never_means_the_means(tmp_path):
    """EVERYTHING IS A SUM OVER A COUNT: the last eval batch is short, and a mean of the
    batch means would weigh its rows above every other row of the run.

    The three windows cost 0, 1 and 2 nats for each seat. At a batch of two the last batch
    holds the costliest of them alone: the sum over the count states 4.0 and a mean of the
    two batch means would state 5.0. The batch size is a fact of the machine and not of
    the music, thus the whole-batch reading must be the same number."""
    path = written_corpus(tmp_path / "frames.safetensors")
    short = ar_measure.loss_row(Trunk(nats_of_the_window), path, limit=3, batch=2)
    whole = ar_measure.loss_row(Trunk(nats_of_the_window), path, limit=3, batch=16)
    assert short["steps"] == whole["steps"] == 768
    assert short["loss"] == pytest.approx(4.0)
    assert short["loss"] == pytest.approx(whole["loss"])


def test_the_error_beside_the_mean_needs_a_second_walk():
    """THE ERROR IS NOT DECORATION: a single-seed reading did not survive a second seed
    once in this era, thus a mean over walks travels with the standard error of that mean.
    One walk has no error and states nan, where a zero would read as certainty."""
    rows = [{"hold": 70.0, "onsets": 0.6}, {"hold": 78.0, "onsets": 1.0}]
    read = ar_measure.mean_over_seeds(rows)
    assert read["hold"] == pytest.approx((74.0, 4.0))
    assert read["onsets"] == pytest.approx((0.8, 0.2))
    alone = ar_measure.mean_over_seeds(rows[:1])
    assert alone["hold"][0] == pytest.approx(70.0)
    assert math.isnan(alone["hold"][1])
