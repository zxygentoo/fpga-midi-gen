"""The masked sheet of the diffusion era: the roll, the mask, the loss and the measure.

Four things here can fail silently, and each one would still train, still sample and still
play -- it would play the wrong piece, or read a number that means nothing:

- the roll. A map wrong by one semitone or one seat round trips nothing, thus every class
  and the corpus itself are pinned.
- the mask. The masked count decides the whole objective; a mask that never hid everything,
  or that hid the pitch rows of a cell unevenly, would train a different model.
- the loss reweighting. One over the masked count is per SHEET, and a batch-wide divisor
  reads the same at every step and is wrong at all of them.
- Algorithm 1. A referee that computes a different number than the paper's reads nothing,
  and the batched form of it is subtle: its frames must be independent given the ordering.
"""

import itertools
import math
import re
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from click.testing import CliRunner
from flax import nnx
from safetensors.numpy import load_file, save_file

import data
import measure
import prng
from diffusion import infer, model, quantized, train
from diffusion import measure as sheet

JAX_ROOT = Path(__file__).resolve().parent.parent
PIECES = JAX_ROOT / "_data" / "pieces.safetensors"
needs_corpus = pytest.mark.skipif(not PIECES.exists(), reason="needs corpus_tool pieces")


def tiny(seed=0, layers=6, width=8):
    """a model small enough for a test; the layer count still holds the paper's shape --
    a stem, two residual pairs and a head"""
    return model.Coconet(layers, width, rngs=nnx.Rngs(seed))


# ---------------------------------------------------------------------
# the roll and the mask planes
# ---------------------------------------------------------------------


def test_a_column_of_the_roll_holds_one_row():
    """one voice at one step sings one pitch, thus its column holds a single one and the
    mask plane beside it is flat"""
    classes = np.arange(model.ROWS, dtype=np.int32)[None, :, None]
    classes = np.tile(classes, (1, 1, model.VOICES))
    hidden = np.zeros(classes.shape, dtype=bool)
    sheet = np.asarray(model.planes(classes, hidden))
    assert sheet.shape == (1, model.ROWS, model.ROWS, 2 * model.VOICES)
    assert (sheet[..., : model.VOICES].sum(axis=-2) == 1).all()
    assert (sheet[..., model.VOICES :] == 0.0).all()


def test_a_neighbour_pitch_is_a_neighbour_row():
    """The reason the pitch is an axis and not a set of channels: a semitone is one row,
    thus a convolution over the rows sees an interval as one shape wherever it stands.

    This is the paper's inductive bias -- the near-invariance of counterpoint to translation
    in pitch -- and the sheets of the proto round had none of it."""
    classes = np.array([[[10, 11, 20, 21]]], dtype=np.int32)
    sheet = np.asarray(model.planes(classes, np.zeros(classes.shape, dtype=bool)))
    rows = [int(np.argmax(sheet[0, 0, :, seat])) for seat in range(model.VOICES)]
    assert rows == [10, 11, 20, 21]


def test_a_masked_cell_shows_zero_in_the_roll_and_one_in_its_plane():
    """equations 4 and 5 of the paper, with the code release's polarity: the mask plane is
    hot where the model must state a pitch, and it is hot up the whole pitch axis"""
    classes = np.array([[[1, 2, 3, 4]]], dtype=np.int32)
    hidden = np.array([[[True, False, False, True]]])
    sheet = np.asarray(model.planes(classes, hidden))
    roll, mask = sheet[..., : model.VOICES], sheet[..., model.VOICES :]
    assert (roll[0, 0, :, 0] == 0.0).all() and (roll[0, 0, :, 3] == 0.0).all()
    assert roll[0, 0, 2, 1] == 1.0 and roll[0, 0, 3, 2] == 1.0
    # the mask is a fact of the CELL, thus it stands in every row of the column
    assert (mask[0, 0, :, 0] == 1.0).all() and (mask[0, 0, :, 1] == 0.0).all()


def test_the_orderless_draw_masks_from_one_cell_to_all_of_them():
    """The paper's training distribution, as its code release states it: the masked count
    is uniform on 1 to D.

    Both ends earn a test. A draw that never masked every cell would never teach the model
    the prior it must state at the opening of a Gibbs walk, and a draw that could mask none
    would divide the loss by zero."""
    steps = 8
    width = model.cells(steps)
    hidden = np.asarray(model.orderless_masks(jax.random.PRNGKey(0), 4096, steps))
    counts = hidden.sum(axis=(1, 2))
    assert counts.min() == 1 and counts.max() == width
    # uniform on 1 to D has the mean (D + 1) / 2; 4096 draws hold that to a percent
    assert abs(counts.mean() - (width + 1) / 2) < 0.02 * width


def test_the_orderless_draw_masks_a_cell_and_never_a_row():
    """a mask that hid some rows of a column and not others would state a pitch the model
    could read around, and the objective would no longer be the paper's"""
    hidden = model.orderless_masks(jax.random.PRNGKey(1), 8, 16)
    classes = np.zeros((8, 16, model.VOICES), dtype=np.int32)
    mask = np.asarray(model.planes(classes, hidden))[..., model.VOICES :]
    assert ((mask == 0.0).all(axis=-2) | (mask == 1.0).all(axis=-2)).all()


def test_the_anneal_falls_from_the_top_and_settles_on_the_floor():
    """the schedule of Yao et al. with the code release's constants: high while the chain
    mixes, and settled on the floor after a [span] share of the walk"""
    walk = 512
    assert model.anneal(0, walk) == pytest.approx(model.ANNEAL_HIGH)
    assert model.anneal(walk - 1, walk) == pytest.approx(model.ANNEAL_LOW)
    at = [model.anneal(step, walk) for step in range(walk)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(at))
    # it reaches the floor after a [span] share of the walk and not at its end
    settles = math.ceil(model.ANNEAL_SPAN * walk)
    assert model.anneal(settles, walk) == pytest.approx(model.ANNEAL_LOW)
    assert model.anneal(settles - 1, walk) > model.ANNEAL_LOW


# ---------------------------------------------------------------------
# the net and its checkpoint
# ---------------------------------------------------------------------


def test_the_net_states_a_distribution_for_every_cell():
    """the softmax runs over the pitch rows, thus every voice of every step carries one --
    the masked cells and the context alike"""
    coconet = tiny()
    classes = np.zeros((2, 16, model.VOICES), dtype=np.int32)
    hidden = model.orderless_masks(jax.random.PRNGKey(2), 2, 16)
    said, seen = coconet(model.planes(classes, hidden))
    assert said.shape == (2, 16, model.ROWS, model.VOICES)
    assert len(seen) == len(coconet.every_layer())
    total = jnp.sum(jax.nn.softmax(said, axis=-2), axis=-2)
    assert np.allclose(np.asarray(total), 1.0, atol=1e-5)


def test_rematerialisation_changes_nothing_but_the_memory():
    """the pair is the unit of remat, thus the trunk keeps 31 tensors instead of hundreds;
    what it computes must not move"""
    coconet = tiny()
    classes = np.zeros((2, 16, model.VOICES), dtype=np.int32)
    sheet = model.planes(classes, model.orderless_masks(jax.random.PRNGKey(3), 2, 16))
    plain, _ = coconet(sheet, training=True)
    kept, _ = coconet(sheet, training=True, remat=True)
    assert np.allclose(np.asarray(plain), np.asarray(kept), atol=1e-5)


def test_the_paper_size_holds_nine_million_parameters():
    """the shape of the round, counted and not assumed: 64 layers of 3 by 3 at 128
    channels, a stem of 2I planes and a head of I"""
    coconet = tiny(layers=model.LAYERS, width=model.WIDTH)
    assert len(coconet.every_layer()) == model.LAYERS
    assert len(coconet.pairs) == (model.LAYERS - 2) // 2
    assert 9.0e6 < coconet.parameter_count() < 9.3e6


def flat_tensors(coconet):
    """the whole model as one list, in the order [Coconet.save] writes it"""
    return [np.asarray(t) for layer in coconet.every_layer() for t in layer.tensors()]


def test_the_checkpoint_states_the_weights_and_the_statistics(tmp_path):
    """The reader takes the layer count from the tensor count and the width from the
    shapes. The population statistics travel inside the file, because a model cannot state
    a probability without them."""
    coconet = tiny(seed=3)
    # a population that is not the opening, thus a reader that dropped it would be caught
    for layer in coconet.every_layer():
        layer.norm.mean[...] = layer.norm.mean[...] + 0.5
        layer.norm.variance[...] = layer.norm.variance[...] + 0.5
    path = str(tmp_path / "sheet.ckpt")
    coconet.save(path)
    read = model.Coconet.load(path)
    pairs = zip(flat_tensors(coconet), flat_tensors(read))
    assert all(np.array_equal(a, b) for a, b in pairs)
    assert len(flat_tensors(read)) == 6 * model.LAYER_TENSORS


def test_the_tree_holds_no_half_pair():
    """THE LAYER COUNT IS EVEN AND AT LEAST FOUR, and the tree is what says so. The
    functional form this module replaced stated the trunk by stride, and an odd count read
    its last layer BOTH as the close of a pair and as the head -- silently, and only in
    the arithmetic."""
    for layers in (3, 5, 7):
        with pytest.raises(ValueError, match="no sheet model"):
            model.Coconet(layers, 8, rngs=nnx.Rngs(0))


def test_the_population_warms_before_it_settles():
    """The population opens at the prior, thus a flat decay makes the first hundreds of
    valid numbers read log(ROWS) whatever the model has learned. The warmed decay must move
    fast at the opening and settle onto the release's rate."""
    assert float(train.population_decay(1.0)) < 0.2
    assert float(train.population_decay(100.0)) < train.POP_DECAY
    assert float(train.population_decay(890.0)) == pytest.approx(train.POP_DECAY)
    assert float(train.population_decay(30000.0)) == pytest.approx(train.POP_DECAY)


def test_the_population_statistics_decide_the_answer():
    """A pass that is not training must read the statistics it was handed and no others.
    If it read the batch's own, one sheet of a Gibbs walk would depend on what else stood
    beside it in the batch, and a referee could not reproduce a number."""
    coconet = tiny()
    classes = np.zeros((2, 16, model.VOICES), dtype=np.int32)
    sheet = model.planes(classes, model.orderless_masks(jax.random.PRNGKey(4), 2, 16))
    said, _ = coconet(sheet)
    # one sheet alone must give what it gave inside the pair
    alone, _ = coconet(sheet[:1])
    assert np.allclose(np.asarray(alone), np.asarray(said[:1]), atol=1e-5)
    for layer in coconet.every_layer():
        layer.norm.mean[...] = layer.norm.mean[...] + 1.0
    other, _ = coconet(sheet)
    assert not np.allclose(np.asarray(said), np.asarray(other))


# ---------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------


def test_the_loss_reads_the_masked_cells_and_no_others():
    """the paper's equation 9 sums over the complement of the context; a loss that read the
    context too would be the code release's default and not the paper's"""
    coconet = tiny()
    classes = np.zeros((1, 8, model.VOICES), dtype=np.int32)
    hidden = np.zeros((1, 8, model.VOICES), dtype=bool)
    hidden[0, 0, 0] = True
    said, _ = coconet(model.planes(classes, hidden))
    logp = jax.nn.log_softmax(said, axis=-2)
    one = -float(logp[0, 0, classes[0, 0, 0], 0])
    assert float(train.masked_nll(said, jnp.asarray(classes), jnp.asarray(hidden))) == (
        pytest.approx(one, abs=1e-5)
    )


def test_the_loss_divides_by_the_count_of_each_sheet():
    """One over the masked count is PER SHEET. A sheet with one cell hidden and one with
    sixteen must weigh the same, thus a batch-wide divisor is wrong -- and it is wrong in a
    way that reads plausible at every step.

    Sheet 0 states nothing and costs log(ROWS) at its single masked cell; sheet 1 is
    certain and costs nothing at all sixteen of its own. The per-sheet divisor reads half
    of log(ROWS); a batch-wide one would read a seventeenth of it."""
    flat = jnp.zeros((4, model.ROWS, model.VOICES))
    sure = jnp.broadcast_to(
        jnp.where(jnp.arange(model.ROWS) == 0, 0.0, -30.0)[None, :, None], flat.shape
    )
    said = jnp.stack([flat, sure])
    classes = jnp.zeros((2, 4, model.VOICES), dtype=jnp.int32)
    hidden = np.zeros((2, 4, model.VOICES), dtype=bool)
    hidden[0, 0, 0] = True
    hidden[1] = True
    assert float(train.masked_nll(said, classes, jnp.asarray(hidden))) == pytest.approx(
        0.5 * np.log(model.ROWS), abs=1e-4
    )


def test_an_untrained_model_reads_the_uniform_prior():
    """a model that has learned nothing must state log(ROWS) nats for each masked cell, and
    a loss that reads far from it at step zero has a scale fault"""
    coconet = tiny(layers=8, width=16)
    classes = np.zeros((4, 16, model.VOICES), dtype=np.int32)
    hidden = model.orderless_masks(jax.random.PRNGKey(5), 4, 16)
    said, _ = coconet(model.planes(classes, hidden), training=True)
    value = float(train.masked_nll(said, jnp.asarray(classes), hidden))
    assert abs(value - np.log(model.ROWS)) < 0.3


# ---------------------------------------------------------------------
# the structure battery
# ---------------------------------------------------------------------


def held(pitches):
    """one sheet of two steps holding one sonority; None is a rest, seat 0 first"""
    classes = [
        data.SILENCE if pitch is None else pitch - data.PITCH_LOW + 1 for pitch in pitches
    ]
    return np.tile(np.asarray(classes, dtype=np.int32), (1, 2, 1))


def test_the_battery_counts_a_triad_and_names_a_dissonance():
    """the two instruments that carry a chord, on chords whose answer is known by hand"""
    # a C major triad with the root doubled, then a cluster; a sheet is two steps because
    # the hold instrument reads the step before and one step has none
    row = measure.structure(held([36, 43, 52, 60]))
    assert row["triads"] == pytest.approx(100.0)
    assert row["dissonant"] == pytest.approx(0.0)
    assert row["order"] == pytest.approx(100.0)
    cluster = measure.structure(held([36, 37, 38, 39]))
    assert cluster["triads"] == pytest.approx(0.0)
    assert cluster["dissonant"] > 60.0


def test_the_battery_counts_triads_over_the_thick_steps_alone():
    """The method of the proto round, and the reason it is the method: a dyad sits inside
    some triad for free, thus counting every step flatters a thin sheet. A step of two
    voices must not reach the number at all."""
    row = measure.structure(held([36, 43, None, None]))
    assert row["triads"] == pytest.approx(0.0)  # no step carries three voices
    assert row["voices"][2] == pytest.approx(100.0)


def test_the_battery_sees_a_voice_out_of_register():
    """the bass above the tenor is the failure the order instrument is for"""
    assert measure.structure(held([60, 48, 52, 55]))["order"] == pytest.approx(0.0)


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
    row = measure.structure(held([48, 64, 67, 79]))["pairs"]
    spans = {pair["name"]: pair["span"] for pair in row}
    assert spans["ba-so"] == pytest.approx(31.0)
    assert spans["te-al"] == pytest.approx(3.0)
    assert spans["ba-te"] == pytest.approx(16.0)


def test_a_rest_leaves_its_pairs_out_of_the_span():
    """a pair sounds only when both of its voices do, thus a rest is not an interval of
    zero and it must not pull a span down"""
    row = measure.structure(held([48, None, 67, 79]))["pairs"]
    spans = {pair["name"]: pair["span"] for pair in row}
    assert spans["ba-al"] == pytest.approx(19.0)
    assert spans["ba-te"] == 0.0 and spans["te-so"] == 0.0


def test_the_clash_counts_the_frame_and_not_the_pair():
    """The tail instrument. A seventh chord holds two dissonant pairs and is ordinary; a
    frame is a clash when three of its six pairs or more are dissonant. The mean dissonance
    cannot tell those apart, which is the whole reason this number exists."""
    # G B D F, a dominant seventh: G-F is a tone and B-F a tritone, thus two pairs
    seventh = measure.structure(held([55, 59, 62, 65]))
    assert seventh["dissonant"] > 0.0 and seventh["clash"] == pytest.approx(0.0)
    # a cluster: five of the six pairs are seconds or sevenths
    assert measure.structure(held([48, 49, 50, 51]))["clash"] == pytest.approx(100.0)


def moving(first, second):
    """one sheet of two steps, each a list of pitches with None for a rest"""
    return np.stack([held(first)[0, 0], held(second)[0, 0]])[None]


def test_the_parallel_instrument_catches_the_fifth_and_the_octave():
    """The fault that lives BETWEEN frames, on motions whose answer is known by hand. Two
    voices a fifth apart, both moving, landing on a fifth."""
    read = measure.structure(moving([48, 55, 64, 72], [50, 57, 64, 72]))["parallels"]
    assert read["fifths"] > 0.0 and read["octaves"] == pytest.approx(0.0)
    read = measure.structure(moving([48, 60, 64, 67], [50, 62, 64, 67]))["parallels"]
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
    read = measure.structure(sheet)["parallels"]
    # the bass and the tenor move together over step 1 and stand still over step 2, thus
    # one of the two live steps of that pair moves and the fault owns all of it
    assert read["fifths"] == pytest.approx(1000.0)
    assert read["moving"] < 100.0


def test_contrary_motion_onto_a_fifth_is_not_a_parallel():
    """The correction of 2026-08-25. The bass falls a tritone and the tenor rises one, thus
    the pair stands a fifth apart before and a twelfth after -- the same interval class, by
    CONTRARY motion. That is how a fifth is correctly approached, and counting it read 53
    percent of the corpus's own fifths as faults."""
    read = measure.structure(moving([60, 67, 74, 79], [54, 73, 74, 79]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0)


def test_a_pair_that_crosses_holds_no_interval():
    """A fifth whose voices swap places became a fourth, thus its interval did not hold.
    The absolute gap cannot see the crossing -- both ends read seven -- and the order of
    the pair is what says otherwise. Here the bass leaps above the tenor and both rise."""
    read = measure.structure(moving([60, 67, 80, 84], [75, 68, 80, 84]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0)


def test_a_voice_that_holds_makes_no_parallel():
    """Two voices that keep a fifth while ONE of them stands still is oblique motion, which
    counterpoint permits and the ear does not object to. Only a pair that moves together
    can be parallel."""
    read = measure.structure(moving([48, 55, 64, 72], [48, 55, 65, 72]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0)


def test_contrary_motion_makes_no_parallel():
    """the two voices move, and the interval between them changes; nothing is parallel"""
    read = measure.structure(moving([48, 55, 64, 72], [50, 53, 64, 72]))["parallels"]
    assert read["fifths"] == pytest.approx(0.0) and read["octaves"] == pytest.approx(0.0)


def test_a_silent_frame_is_not_a_clash():
    """a frame nobody sings holds no pair at all, thus it must leave the tail alone rather
    than count as clean"""
    assert measure.structure(held([None, None, None, None]))["clash"] == pytest.approx(
        0.0
    )


def test_the_likelihood_keeps_its_frames():
    """The mean is Algorithm 1's return and the frames are the tail. A referee that
    averaged them away could not see a model that is wrong rarely and badly."""
    forward = certain_forward(7)
    frames = sheet.piece_nll(
        forward, np.full((16, model.VOICES), 7, np.int32), np.random.default_rng(0), 2, 8
    )
    assert frames.shape == (16,)
    assert float(frames.mean()) < 0.01


def test_the_tail_reads_the_percentiles_and_the_loud_frames():
    """a run of easy frames with one disaster in it: the mean hides it and the tail must
    not"""
    frames = np.full((20, 5), 0.2)
    frames[0, 0] = 40.0
    line = sheet.tail_line(frames)
    assert "median 0.200" in line
    assert re.search(r"above 2 nats\s+1\.0 \+- ", line), line


def test_the_tail_error_resamples_the_pieces_and_not_the_frames():
    """The frames of one chorale are one draw of a composer and not 128 of them. An error
    that resampled frames would read half of the truth, and every model would then separate
    from every other."""
    frames = np.full((20, 5), 0.2)
    frames[0] = 40.0  # one WHOLE piece is the disaster, thus the piece is the unit
    read = sheet.tail_shape(frames)
    assert read["loud"] == pytest.approx(5.0)
    # a binomial over 20 pieces reads 4.9 percent here; over 100 frames it would read 2.2
    assert read["loud error"] > 3.5


def test_the_register_sees_a_texture_that_slid_where_nothing_else_does():
    """A texture in good order, correctly spaced, and sitting a whole octave too low. The
    order instrument and the voice pairs both read it CLEAN -- the stacking holds and every
    span is unchanged -- thus the register mean is the only thing that can see it.

    And the tail is coarse on purpose. The seats overlap by 14 to 18 semitones, so a drop
    of an octave puts only the SOPRANO under its own floor of 60; the other three are still
    inside ranges their neighbours share. [outside] is a backstop for a gross departure and
    the mean is the sensitive instrument."""
    right = measure.structure(held([50, 59, 65, 71]))["register"]
    low = measure.structure(held([38, 47, 53, 59]))["register"]
    assert right["outside"] == pytest.approx(0.0)
    for seat, moved in zip(right["seats"], low["seats"]):
        assert moved["mean"] == pytest.approx(seat["mean"] - 12)
    # one seat of the four, thus a quarter of the sounding cells
    assert low["outside"] == pytest.approx(25.0)


def test_the_register_tells_drift_from_over_ranging():
    """The mean says a voice has moved and the spread says it wanders; a sheet that holds
    one chord has no spread at all, and one that alternates two has the half-distance."""
    still = measure.structure(held([50, 59, 65, 71]))["register"]
    assert all(seat["spread"] == pytest.approx(0.0) for seat in still["seats"])
    swung = np.concatenate([held([48, 59, 65, 71]), held([52, 59, 65, 71])], axis=1)
    read = measure.structure(swung)["register"]
    assert read["seats"][0]["mean"] == pytest.approx(50.0)
    assert read["seats"][0]["spread"] == pytest.approx(2.0)


def test_a_rest_leaves_its_seat_out_of_the_register():
    """a seat that does not sing states no pitch, thus it must not pull the mean toward the
    silence row"""
    read = measure.structure(held([50, None, 65, 71]))["register"]
    assert read["seats"][1]["mean"] == pytest.approx(0.0)
    assert read["outside"] == pytest.approx(0.0)


@needs_corpus
def test_the_corpus_row_stands_where_the_proto_round_left_it():
    """The corpus row is the referee of every number of the battery, thus the battery is
    read against the corpus and the corpus is pinned here. These are the sixteenth-grid
    figures; the proto round measured the eighth grid and its dissonance reads the same
    10.2 percent."""
    row = measure.structure(sheet.corpus_sheets(str(PIECES), "train", model.CROP, 0))
    assert row["voices"][4] == pytest.approx(99.8, abs=0.1)
    assert row["triads"] == pytest.approx(63.9, abs=0.2)
    assert row["dissonant"] == pytest.approx(10.3, abs=0.2)
    assert row["hold"] == pytest.approx(76.9, abs=0.2)
    assert row["clash"] == pytest.approx(2.9, abs=0.3)
    # the horizontal referee: Bach essentially never writes them, thus any rate far above
    # this is a fault of the model and not a taste of the corpus. The divisor is the pairs
    # that MOVE, thus a rung cannot buy the number by holding its notes, and the share that
    # moves stands beside it to catch a rung whose motion has left the corpus.
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


# ---------------------------------------------------------------------
# the likelihood referee
# ---------------------------------------------------------------------


def test_the_ordering_covers_every_frame_and_every_voice():
    """Algorithm 1 walks a permutation of the frames and a permutation of the voices inside
    each. A repeat would score one cell two times and never score another."""
    frames, voices = sheet.frame_ordering(np.random.default_rng(0), 32)
    assert sorted(frames.tolist()) == list(range(32))
    assert all(sorted(row.tolist()) == list(range(model.VOICES)) for row in voices)


def test_algorithm_one_reads_the_true_frames_before_it_and_nothing_after():
    """The mask of the batched form, which is the whole reason the referee is affordable.
    Sheet l of the stack must reveal exactly the frames that stand before position l in
    the ordering; a mask that leaked one frame from after it would read a number far under
    the paper's and look like a triumph."""
    steps = 8
    frames = np.array([3, 0, 5, 1, 7, 2, 6, 4])
    hidden = np.zeros((steps, steps, model.VOICES), dtype=bool)
    for at in range(steps):
        hidden[at, frames[at:], :] = True
    for at in range(steps):
        shown = set(np.nonzero(~hidden[at, :, 0])[0].tolist())
        assert shown == set(frames[:at].tolist())


def test_algorithm_one_scores_every_frame_one_time():
    """the return is nats for each frame, thus every frame must be reached and none twice"""
    coconet = tiny()
    classes = np.zeros((16, model.VOICES), dtype=np.int32)
    forward = partial(sheet.log_probabilities, coconet)
    rng = np.random.default_rng(0)
    lls = sheet.framewise_lls(forward, classes, sheet.frame_ordering(rng, 16), 8)
    assert lls.shape == (16,)
    # every frame holds four voices, thus no frame can be less likely than four uniforms
    assert (lls <= 0.0).all()
    assert (lls >= model.VOICES * np.log(1.0 / model.ROWS) * 3).all()


def certain_forward(row):
    """a stub in the shape of a trained model, sure of one pitch row everywhere: it pins the
    accounting of Algorithm 1 against an answer known by hand"""

    def forward(classes, hidden):
        del hidden
        return jax.nn.log_softmax(
            jnp.broadcast_to(
                jnp.where(jnp.arange(model.ROWS) == row, 0.0, -30.0)[None, None, :, None],
                (*classes.shape[:2], model.ROWS, model.VOICES),
            ),
            axis=-2,
        )

    return forward


def test_algorithm_one_adds_the_four_voices_of_a_frame():
    """The frame is the unit of the measurement, thus its four voices add and the return is
    nats for each frame and not for each cell.

    A sheet of the row the stub is sure of costs nearly nothing. A sheet of any other row
    costs four times the log probability the stub leaves that row -- one for each voice."""
    ordering = sheet.frame_ordering(np.random.default_rng(0), 8)
    forward = certain_forward(7)
    agrees = np.full((8, model.VOICES), 7, dtype=np.int32)
    assert float(-np.mean(sheet.framewise_lls(forward, agrees, ordering, 8))) < 0.01
    misses = np.full((8, model.VOICES), 9, dtype=np.int32)
    assert float(
        -np.mean(sheet.framewise_lls(forward, misses, ordering, 8))
    ) == pytest.approx(model.VOICES * 30.0, rel=0.05)


# ---------------------------------------------------------------------
# the walk, and the trainer end to end
# ---------------------------------------------------------------------


def test_the_walk_rewrites_the_sheet():
    """a walk of a few passes must change the opening it was handed -- a sampler that
    left every cell standing never fired -- and it must state a sheet of the same
    shape, every cell a class of the vocabulary"""
    coconet = tiny()
    states, given = model.opening_sheet(prng.states([1, 2]), 8)
    drawn, _ = infer.gibbs(coconet, given, states, walk=4, temperature=1.0)
    assert drawn.shape == given.shape
    assert (drawn != given).any()
    assert drawn.min() >= 0 and drawn.max() < model.ROWS


def test_a_sheet_walks_the_same_alone_or_in_a_batch():
    """One seed names one SHEET, whatever stands beside it: each walk holds its own
    generator, thus a batch is independent pieces and a sweep's seed crosses to a solo
    audition unchanged. The old sampler could not say this -- its seed named the batch."""
    coconet = tiny()

    def walk(seeds):
        states, given = model.opening_sheet(prng.states(seeds), 8)
        drawn, _ = infer.gibbs(coconet, given, states, walk=3, temperature=1.0)
        return drawn

    alone = walk([7])
    batched = walk([3, 7, 11])
    assert np.array_equal(alone[0], batched[1])


def test_the_tempered_pick_is_the_policy_pick():
    """The rules of Policy.draw_class, held cell for cell: the peak always survives, a
    uniform of 0 lands on the first class that holds weight, and the top of the 24-bit
    grid lands on the LAST class that holds weight -- never past it."""
    raw = np.array([[0.0, -1.0, -10.0], [-10.0, 0.0, -1.0]])
    top = float(0xFFFFFF) * 2.0**-24
    assert list(model.tempered_pick(raw, 1.0, np.array([0.0, 0.0]))) == [0, 0]
    # a class ten nats under the peak weighs 4.5e-5 of it, which the top of the grid
    # reaches -- the same case Policy's own gate walks
    assert list(model.tempered_pick(raw, 1.0, np.array([top, top]))) == [2, 2]
    # the middle of the grid stays on the heavy classes
    assert list(model.tempered_pick(raw, 1.0, np.array([0.5, 0.5]))) == [0, 1]


def test_several_sheets_take_a_file_each():
    """A batch is a set of whole pieces and not one piece in parts. One sheet keeps the
    name the caller gave, thus a single audition writes exactly the file it was asked
    for."""
    assert infer.audition_path("eight.mid", 0, 1) == "eight.mid"
    assert infer.audition_path("a/eight.mid", 0, 4) == "a/eight-0.mid"
    assert infer.audition_path("a/eight.mid", 3, 4) == "a/eight-3.mid"


def test_the_seeded_opening_puts_every_voice_in_its_own_register():
    """The opening that needs no special step. A bass at 81 and a soprano at 36 are further
    from this corpus than a rest is, thus the draw is over each seat's own range and not
    over the whole roll -- and a cell the Bernoulli leaves standing then states a NOTE,
    which is what 99.8 percent of the corpus's cells state."""
    _, drawn = model.opening_sheet(prng.states([7, 8, 9, 10]), 32)
    assert drawn.shape == (4, 32, model.VOICES)
    assert not (drawn == data.SILENCE).any()
    pitches = data.pitches_of_classes(drawn)
    for seat, (low, high) in enumerate(measure.RANGES):
        heard = pitches[..., seat]
        assert heard.min() >= low and heard.max() <= high


def test_the_seeded_opening_is_the_seed_and_nothing_else():
    """the project's rule: the seed is an input, and one seed gives one sheet here, in
    the OCaml reference and on the board"""

    def drawn(seeds):
        return model.opening_sheet(prng.states(seeds), 16)[1]

    assert np.array_equal(drawn([3, 4]), drawn([3, 4]))
    assert not np.array_equal(drawn([3, 4]), drawn([4, 3]))
    # and a row is its seed's, not its batch's
    assert np.array_equal(drawn([3])[0], drawn([9, 3])[1])


@needs_corpus
def test_the_crops_drop_the_piece_that_is_too_short():
    """228 of the 229 train chorales hold eight measures on the sixteenth grid, and the
    round trains on those. The one that does not is 100 steps."""
    pieces = data.load_pieces(str(PIECES))
    counts = {
        name: len(data.Crops(pieces[name], model.CROP).rows) for name in data.SPLITS
    }
    assert counts == {"train": 228, "valid": 76, "test": 77}


@needs_corpus
def test_a_crop_never_reads_the_padded_tail():
    """The padding is a fact of the file and never a fact of the music. A crop that ran into
    it would teach the model the tail prior that owned 53 percent of the proto round's
    sheet."""
    pieces = data.load_pieces(str(PIECES))
    crops = data.Crops(pieces["train"], model.CROP)
    rng = np.random.default_rng(0)
    batch = crops.batch(rng, 256)
    assert batch.shape == (256, model.CROP, model.VOICES)
    # a run of whole silent steps is the tail's signature; inside a piece the corpus falls
    # silent in all four voices at 0.19 percent of its steps and never for long
    silent = (batch == data.SILENCE).all(axis=-1)
    assert float(silent.mean()) < 0.02


@needs_corpus
def test_the_loss_falls_over_a_short_run():
    """the guard test_train.py states for the other eras: a trainer that cannot learn still
    prints a correct step-1 loss, thus only a run of several steps can tell"""
    done = CliRunner().invoke(
        train.main,
        [
            "--layers",
            "8",
            "--width",
            "16",
            "--crop",
            "32",
            "--batch",
            "4",
            "--steps",
            "150",
            "--lr",
            "3e-3",
            "--warmup",
            "10",
            "--seed",
            "4",
            "--log-every",
            "25",
            "--eval-every",
            "1000",
        ],
    )
    assert done.exit_code == 0, done.output
    losses = [float(m) for m in re.findall(r"loss (\d+\.\d+)", done.output)]
    assert len(losses) >= 3, done.output
    # a model that has learned nothing reads log(48) = 3.87 nats for each masked cell
    assert losses[0] > 3.5, done.output
    assert losses[-1] < losses[0] - 0.5, f"the loss did not fall: {losses}"


# ==================================================================== #
# The integer twin: the scalar rules, and the contract file            #
# ==================================================================== #

# THE TWIN IS HELD BY THREE THINGS AND THESE ARE THE SMALLEST OF THEM. `test_rtl.py` holds
# it against the circuit, write for write; `test_drift.py` holds it against the float model
# it quantizes; and `test_parity.py`'s G1 holds the quantizer through the netlist the flash
# carries. What stands here is the arithmetic each of those would break on FIRST -- a
# rounding, an exponent, a table entry -- so that a failure names the rule and not the walk.


@pytest.mark.parametrize(
    "peak,exponent",
    [(0.0, 14), (0.02, 12), (0.08, 10), (127.0, 0), (127.49, 0), (127.5, -1), (1e9, -23)],
)
def test_the_exponent_rule_holds_at_its_boundaries(peak, exponent):
    """The largest e that keeps the peak at 127 or less. 14 caps the all-zero tensor, where
    every exponent fits, and 127.5 is the rounding boundary: it rounds to 128 and the
    exponent has to step down. A tie rounds UP and never away from zero, which is Base's
    `Float.iround_nearest_exn` and not Python's `round`."""
    assert quantized.max_exponent(peak) == exponent


def test_the_byte_is_symmetric_and_ties_round_up():
    """The byte is two's complement and the negative end is not used: the clamp is -127 and
    not -128, thus the image is symmetric and a negated weight is a negated byte."""
    q, e = quantized.quantize(np.array([0.02, -0.01, 0.0]))
    assert (list(q), e) == ([82, -41, 0], 12)
    assert quantized.round_half_up([-5.5, -2.5, 2.5, 5.5]).tolist() == [-5, -2, 3, 6]


def test_the_exp2_table_is_the_shared_table():
    """exp2 of -j/256 in Q15, the one table the samplers of every era read. Entry 0 is the
    peak 2^15, a full fractional step halves, and the last entry sits one table step above
    one half. The whole table was compared entry for entry against
    `Nn_quantized.Constants.exp2_bits` when it was written, and no entry differed: the two
    libm implementations agree here."""
    table = quantized.EXP2_TABLE
    assert (table[0], table[128], table[255]) == (32768, 23170, 16428)
    assert quantized.exp2_of_magnitude(np.int64(4096)) == 16384
    # a magnitude of 16 or more is zero, and the shift may not run past the host word
    assert quantized.exp2_of_magnitude(np.int64(1 << 20)) == 0


def test_the_temper_is_log2e_over_the_temperature():
    """log2(e) / T at a Q one below log2(e)'s own: the extra bit is headroom for the
    temperature, because the circuits carry this constant on an 18-bit signed port."""
    assert quantized.temper_of(1.0) == (23637, 14)
    assert quantized.temper_of(0.5) == (47274, 14)
    with pytest.raises(ValueError):
        quantized.temper_of(0.0)


def test_the_twin_carries_the_float_models_skeleton():
    """THE TWO TREES ARE ONE TREE, and that is what makes the twin auditable: a reader
    puts `coconet.pairs[k].first` beside `twin.pairs[k].first` and reads one layer against
    its own quantization, with nothing to align by hand."""
    coconet = model.Coconet.drawn(5, 6, 8)
    twin = quantized.QuantizedCoconet.of(coconet)
    assert len(twin.pairs) == len(coconet.pairs)
    for here, there in zip(twin.every_layer(), coconet.every_layer()):
        assert here.kernel.shape == there.conv.kernel.shape
        assert here.outputs == len(there.norm.scale[...])
    # the stem reads the planes, the head states the voices, and each pair is two layers
    assert twin.stem.inputs == 2 * model.VOICES
    assert twin.head.outputs == model.VOICES
    assert twin.pairs[0].first.outputs == twin.pairs[0].second.inputs


def test_the_contract_file_round_trips_exactly(tmp_path):
    """`save` then `load` is the identity: the seam carries the whole model and nothing of
    it is re-derived on either side of the file."""
    twin = quantized.QuantizedCoconet.of(model.Coconet.drawn(5, 6, 8), 0.9)
    path = tmp_path / "round-trip.int8"
    quantized.save(path, twin)
    read = quantized.load(path)
    assert read.temper == twin.temper
    assert len(read.pairs) == len(twin.pairs)
    for here, there in zip(read.every_layer(), twin.every_layer()):
        assert here.e == there.e
        assert np.array_equal(np.asarray(here.kernel[...]), np.asarray(there.kernel[...]))
        assert np.array_equal(here.gain_q_value[...], there.gain_q_value[...])
        assert np.array_equal(here.gain_q[...], there.gain_q[...])
        assert np.array_equal(here.bias[...], there.bias[...])


def rebuilt(twin, **parts):
    """the same twin with some of its parts replaced -- the tree's `_replace`"""
    held = {
        "stem": twin.stem,
        "pairs": list(twin.pairs),
        "head": twin.head,
        "temper": twin.temper,
    }
    return quantized.QuantizedCoconet(**{**held, **parts})


def test_a_broken_model_refuses_loudly():
    """`check_shape` states the rules its consumers assume, and the elaboration calls it
    where a bad shape must fail loudly.

    THE SHORT AND THE ODD TRUNK ARE NOT AMONG THEM ANY MORE: a [QuantizedCoconet] is a
    stem, whole pairs and a head by construction, thus neither can be built. What a file
    of the wrong tensor count meets is `load`, and the test below holds that."""
    twin = quantized.QuantizedCoconet.of(model.Coconet.drawn(5, 6, 8))
    head = twin.head
    chopped = quantized.QuantizedNormedConv(
        kernel=head.kernel[...],
        e=head.e,
        gain_q_value=head.gain_q_value[...],
        gain_q=head.gain_q[...],
        bias=head.bias[...][:2],
    )
    with pytest.raises(ValueError, match="do not cover its channels"):
        quantized.check_shape(rebuilt(twin, head=chopped))
    # a head that reads a width the pair before it did not write
    narrow = quantized.QuantizedCoconet.of(model.Coconet.drawn(5, 6, 4))
    with pytest.raises(ValueError, match="the layer before it"):
        quantized.check_shape(rebuilt(twin, head=narrow.head))


def test_a_contract_file_of_the_wrong_shape_refuses_loudly(tmp_path):
    """The file is a FLAT list and the tree is not, thus the count is checked where the
    two meet: a file that holds no whole residual pairs is no model of this era."""
    twin = quantized.QuantizedCoconet.of(model.Coconet.drawn(5, 6, 8))
    path = tmp_path / "short.int8"
    quantized.save(path, twin)
    tensors = load_file(str(path))
    for name in (str(quantized.LAYER_TENSORS * 5 + on) for on in range(5)):
        del tensors[name]
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="no quantized sheet model"):
        quantized.load(path)
