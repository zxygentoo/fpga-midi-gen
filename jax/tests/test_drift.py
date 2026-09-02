"""What the quantization costs: each integer twin against the float model it quantizes.

THE INSTRUMENT LIVES HERE AND SO DOES EVERY WALK THAT READS IT. No twin module carries a
drift: the report gates nothing a build depends on, thus what measures it is a test and
the twins hold the arithmetic alone. `count_draws` scores a batch of the twin's rows
against the float rows of the same places, ON THE VERY UNIFORM THE TWIN TOOK, and each
era's walk below teacher-forces the float model on the twin's own history.

Era six holds the file and the frozen eras stand at its foot. Every era takes the same
shape of gate and what differs is its FEEDBACK AXIS -- era four's KV ring, era five's
state, era six's sheet -- because that is what decides whether an arithmetic error dies
with its step or compounds.

THREE PARTS, and the drawn weights make the two sweeping ones deterministic. The fixed
sweep pins MEASURED NUMBERS AND NOT THRESHOLDS: a diff says the integers moved, and the
reader judges whether it is a re-measurement or a bug. The property part draws seed pairs
at a fixed generator and holds floors calibrated under the first measured minima; the
printed minima keep the calibration honest. The third is one walk of the ELECTED
checkpoint, which is the only place a trained trunk's own drift is measured.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import ar_model
import corpus
import diffusion.quantized.infer
import diffusion.quantized.model
import mamba.infer
import mamba.quantized.infer
import mamba.quantized.model
import prng
import quantized as q
import sample as s
import transformer.quantized.infer
import transformer.quantized.model
from diffusion import model, sample
from tests import gate
from tests.models import drawn_transformer, plan_of

# The instrument: the twin's draw against the float model's, on the one uniform the twin
# took. All three walks below count through it, thus one rule states what a drift is.


def cosines(twin_logits, float_logits):
    """the cosine of each integer row against the float row of the same place, over a
    batch of [rows, classes]"""
    twin = np.asarray(twin_logits, np.float64)
    floated = np.asarray(float_logits, np.float64)
    return (twin * floated).sum(axis=-1) / np.sqrt(
        (twin * twin).sum(axis=-1) * (floated * floated).sum(axis=-1)
    )


class Counted(NamedTuple):
    """what a drift report has counted over the draws it has seen"""

    draws: int = 0
    same_peak: int = 0
    same_draw: int = 0
    cosine: float = 0.0


def count_draws(
    counted, twin_logits, float_logits, *, drawn, uniform, temperature, min_p
):
    """A BATCH of the twin's rows against the float rows of the same places, on the very
    uniform the twin drew [drawn] on.

    It is batched because era six redraws a whole sheet where a step-frame chain redraws
    four seats. The caller states the policy, because the elected numbers are the twin's
    and not this instrument's."""
    twin = np.asarray(twin_logits, np.float64)
    floated = np.asarray(float_logits, np.float64)
    weights = s.tempered_weight(floated, temperature, min_p)
    return Counted(
        draws=counted.draws + len(twin),
        same_peak=counted.same_peak
        + int((twin.argmax(axis=-1) == floated.argmax(axis=-1)).sum()),
        same_draw=counted.same_draw
        + int((s.pick_share(weights, uniform) == drawn).sum()),
        cosine=counted.cosine + float(cosines(twin, floated).sum()),
    )


def count_chain_draws(counted, floated, chain_draws, *, temperature, min_p):
    """one step's CHAIN as a batch of four: the step-frame adapter over `count_draws`. A
    `Draw` holds a walk axis the drift report does not use -- it runs one walk -- thus
    every row here is that walk's row."""
    return count_draws(
        counted,
        np.stack([draw.logits[0] for draw in chain_draws]),
        np.stack([floated[draw.seat] for draw in chain_draws]),
        drawn=np.array([draw.drawn[0] for draw in chain_draws]),
        uniform=np.array([float(draw.word[0]) for draw in chain_draws])
        * 2.0**-prng.UNIFORM_BITS,
        temperature=temperature,
        min_p=min_p,
    )


def test_the_cosine_reads_the_shape_of_a_row_and_not_its_scale():
    """The third number of every drift report, at rows whose answer is known by hand: a
    row against itself is 1, a row against a scaling of itself is still 1, and a row
    against one at 45 degrees to it is the root of a half."""
    twin = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    floated = np.array([[1.0, 0.0], [7.0, 0.0], [1.0, 1.0]])
    assert list(cosines(twin, floated)) == pytest.approx([1.0, 1.0, 0.5**0.5])


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
    counted = count_draws(
        Counted(draws=10, same_peak=5, same_draw=4, cosine=3.0),
        twin,
        floated,
        drawn=np.array([0, 0]),
        uniform=np.array([0.95, 0.96]),
        temperature=1.0,
        min_p=0.01,
    )
    assert (counted.draws, counted.same_peak, counted.same_draw) == (12, 6, 5)
    assert counted.cosine == pytest.approx(3.0 + 1.0 + 9.0 / 73.0)


# EVERY GATE THAT WALKS A MODEL IS SLOW, and each carries the mark: 112 drift runs, each
# a whole walk of a drawn model against its twin, and one over the elected checkpoint.
# `-m "not slow"` is the inner loop, and it still holds the instrument's two hand gates
# above -- they are arithmetic on numbers written out by hand.


# Era six: the masked sheet


# the structure of the era at a shape a test can afford: the stem, two residual pairs and
# the head, over a quarter of the board's sheet -- two measures
LAYERS = 6
WIDTH = 8
STEPS = 32

WEIGHT_SEEDS = (11, 23, 37, 41)
WALK_SEEDS = (42, 43, 44, 45)


class Shares(NamedTuple):
    """The three numbers a drift report gives as SHARES of what it counted. A property
    gate reads the lowest of each over its trials, and an era's floors stand in the same
    shape -- thus the minimum and the floor it must clear are one type."""

    top1: float
    same_draw: float
    cosine: float


def shares_of(report, counted):
    """the three shares of one drift report; [counted] names the field that says what the
    comparison ran over, which is `cells` on a sheet and `draws` on a chain"""
    over = getattr(report, counted)
    return Shares(report.same_peak / over, report.same_draw / over, report.mean_cosine)


def lowest(here, there):
    """the lower of each share: what the trials so far have read"""
    return Shares(*(min(a, b) for a, b in zip(here, there)))


def drawn_pairs(seed, span, trials):
    """The (weight seed, walk seed) pairs of one property gate, off a FIXED generator: the
    set is the same at every run, thus a floor that breaks is the scheme and never the
    draw. The generator and the span are each era's own, because moving either would move
    the measured minima its floors were calibrated under."""
    rng = np.random.default_rng(seed)
    return tuple(tuple(int(v) for v in rng.integers(1, span, 2)) for _ in range(trials))


class SheetDrift(NamedTuple):
    """What era six's quantization costs, measured on the walk the board takes."""

    passes: int
    cells: int  # the redrawn cells: the comparisons of the report
    same_peak: int  # the cells where both models elect the same class
    same_draw: int  # the cells where both models pick the same class
    mean_cosine: float


def sheet_drift(
    coconet,
    states,
    given,
    *,
    walk,
    temperature=diffusion.quantized.model.ELECTED_TEMPERATURE,
):
    """The quantized walk of one sheet, scored against the float model cell for cell.

    At every pass the float model is TEACHER-FORCED on the engine's sheet and mask, and
    the same-draw share reads the float draw ON THE VERY UNIFORM THE ENGINE TOOK, thus
    what stands between them is the arithmetic alone. The quantization happens here,
    from the float model handed in, thus the pair cannot slip."""
    twin = diffusion.quantized.model.Coconet.from_float(coconet, temperature)
    counted = Counted()
    for taken in diffusion.quantized.infer.passes(twin, states, given, walk=walk):
        floated = np.asarray(
            coconet.logits(jnp.asarray(taken.read), jnp.asarray(taken.hidden)),
            dtype=np.float64,
        )
        for drawn in taken.draws:
            active = drawn.hidden
            # a cell no sheet hid took no uniform and is no comparison
            if not active.any():
                continue
            # `count_draws`, the instrument above: this era sends the cell's whole batch
            # of sheets where a chain sends its four seats. `sample.MIN_P` is the
            # one thing that parts the two calls.
            counted = count_draws(
                counted,
                taken.logits[active, drawn.step, :, drawn.voice],
                floated[active, drawn.step, :, drawn.voice],
                drawn=drawn.drawn[active],
                uniform=drawn.word[active] * 2.0**-prng.UNIFORM_BITS,
                temperature=temperature,
                min_p=sample.MIN_P,
            )
    return SheetDrift(
        passes=walk,
        cells=counted.draws,
        same_peak=counted.same_peak,
        same_draw=counted.same_draw,
        mean_cosine=1.0 if counted.draws == 0 else counted.cosine / counted.draws,
    )


def drift(weight_seed, walk_seed, passes):
    """the drift of one drawn model on one walk"""
    coconet = model.Coconet.drawn(weight_seed, LAYERS, WIDTH)
    states, given = model.opening_sheet(q.engine_states([walk_seed]), STEPS)
    return sheet_drift(coconet, states, given, walk=passes)


# The row `test_a_sweep_states_its_measured_numbers` reads: for each weight seed, summed
# over the four walks, the top-1 count, the same-draw count, the cells counted, and the
# LOWEST mean cosine of the four -- which is the sharpest cosine signal.
SWEPT = {
    11: (3029, 3040, 3299, 0.9984),
    23: (3104, 3033, 3299, 0.9984),
    37: (3095, 3086, 3299, 0.9982),
    41: (3083, 3127, 3299, 0.9960),
}


def sweep_drift(weight_seed, walk_seed):
    """era six's row of the shared sweep at the foot of this file: 16 passes, where the
    cosine signal is sharpest"""
    return drift(weight_seed, walk_seed, 16)


def trial_drift(weight_seed, walk_seed):
    """era six's trial of the shared property gate: 8 passes, the length its floors were
    calibrated at"""
    return drift(weight_seed, walk_seed, 8)


# at 8, 32 and 128 passes of one model: the top-1 count, the cells and the mean cosine
LONG_WALK = {
    8: (411, 445, 0.9982),
    32: (1436, 1552, 0.9983),
    128: (5624, 6169, 0.9982),
}


@pytest.mark.slow
@pytest.mark.parametrize("passes", sorted(LONG_WALK))
def test_the_long_walk_does_not_compound(passes):
    """THE LONG WALK. A redrawn cell enters the context of every later pass, thus one
    model runs at 8, 32 and 128 passes and a cumulative error would show as numbers that
    FALL with the length."""
    report = drift(11, 42, passes)
    peak, cells, cosine = LONG_WALK[passes]
    assert (report.same_peak, report.cells) == (peak, cells)
    assert report.mean_cosine == pytest.approx(cosine, abs=5e-5)


# The ELECTED checkpoint, at the numbers `docs/diffusion_rtl.md` states for it: the climb
# table of 2026-08-26 reads 97.2 percent top-1, 0.9998 cosine and 95.1 percent same draw
# at seed 42, T 128, 32 passes, and this pins the same walk to the counts behind them.
# THE DRAWN MODELS ABOVE CANNOT REPLACE IT: they hold the SCHEME, and what a trained
# trunk's own weights cost is measured only on a trained trunk.
ELECTED = gate.ROOT / "weights" / "diffusion.ckpt"
ELECTED_SEED = 42
ELECTED_PASSES = 32
ELECTED_DRIFT = (6148, 6013, 6326, 0.99983)


@pytest.mark.slow
def test_the_elected_checkpoint_drifts_as_the_chapter_states():
    """The one drift measurement the chapter carries that no gate made. It is a MEASURED
    NUMBER and not a threshold: a diff says the twin moved against the float model, and
    the reader judges whether it is a re-measurement or a bug."""
    gate.need(ELECTED)
    coconet = model.Coconet.load(str(ELECTED))
    states, given = model.opening_sheet(q.engine_states([ELECTED_SEED]), model.CROP)
    report = sheet_drift(coconet, states, given, walk=ELECTED_PASSES)
    peak, draw, cells, cosine = ELECTED_DRIFT
    assert (report.same_peak, report.same_draw, report.cells) == (peak, draw, cells)
    assert report.mean_cosine == pytest.approx(cosine, abs=5e-5)



# The floors, calibrated on this model's own first measured minima over the CLEAN trials,
# the rule the sibling gates were set by: a fail is a break of the scheme and not a
# re-draw of the set. The first measurement read 0.869, 0.806 and 0.9907.
FLOORS = Shares(top1=0.80, same_draw=0.70, cosine=0.985)

PAIRS = drawn_pairs(0xD21F8, 1_000_001, 60)


# Era four: the step-frame transformer


# THE FEEDBACK AXIS OF THIS ERA IS THE KV RING: a coarsened row is read back by every
# later step, thus an error lives a whole window. The walk runs 40 steps over a window of
# 16, so that every trial wraps it.

TRANSFORMER_SHAPE = {"d": 16, "layers": 2}
TRANSFORMER_HEADS = 4
TRANSFORMER_CONTEXT = 16
TRANSFORMER_STEPS = 40


@nnx.jit
def window_float_row(held, window, phases, drawn, at):
    """The float logits of the seats of ONE step, teacher-forced on the twin's history.

    It takes the model as an ARGUMENT at the module level, thus its compiled form is keyed
    on the shapes and every step of a drift run reuses the first compile. [window] is
    padded to the context and [at] is the last real position, which the causal wall keeps
    from seeing the padding."""
    h = held.hidden(window, phases)[:, at, None, :]
    return held.head.logits(h, drawn[None])[0, 0]


class RingDrift(NamedTuple):
    """What era four's quantization costs, measured on the walk the board takes."""

    steps: int  # the steps of the walk, the silent lead-in inside
    draws: int  # four for each drawn step: one for each seat of the chain
    same_peak: int  # the draws where both models elect the same class
    same_draw: int  # the draws where both models pick the same class
    mean_cosine: float


def ring_drift(model, *, context, steps, seed):
    """The quantized walk, scored against the float model draw for draw.

    ONE WEIGHTS SOURCE AND ONE POLICY: the walk quantizes `model` itself. The float
    pass is TEACHER-FORCED on the quantized history and chain, and the same-draw share
    reads the float draw on the very uniform the engine took, thus the report measures
    the quantization and never a walk that parted for another reason."""
    twin = transformer.quantized.model.Transformer.from_float(model, context=context)
    engine = transformer.quantized.infer.create_engine(twin, [seed])
    history = []
    counted = Counted()
    for at in range(steps):
        engine, classes, chain_draws = transformer.quantized.infer.next_step(engine)
        # THE HISTORY IS THE TWIN'S: the window the float pass sees before this step is
        # the window the engine's own ring held
        window = list(history)
        history.append(classes[0])
        if not chain_draws or not window:
            continue
        # ONE shape for the whole run: right-padded to [context], read at the last real
        # position
        low = max(0, at - context)
        length = at - low
        rows = np.zeros((1, context, corpus.SEATS), dtype=np.int32)
        rows[0, :length] = np.stack(window[low:])
        phases = np.zeros((1, context), dtype=np.int32)
        phases[0, :length] = np.arange(low, at) % ar_model.PHASE_BUCKETS
        floated = np.asarray(
            window_float_row(model, rows, phases, classes, length - 1)
        ).astype(np.float64)
        counted = count_chain_draws(
            counted,
            floated,
            chain_draws,
            temperature=q.ELECTED_TEMPERATURE,
            min_p=q.ELECTED_MIN_P,
        )
    return RingDrift(
        steps=steps,
        draws=counted.draws,
        same_peak=counted.same_peak,
        same_draw=counted.same_draw,
        mean_cosine=counted.cosine / max(1, counted.draws),
    )


def transformer_drift(weight_seed, walk_seed):
    """the drift of one drawn model on one walk"""
    return ring_drift(
        drawn_transformer(weight_seed, heads=TRANSFORMER_HEADS, **TRANSFORMER_SHAPE),
        context=TRANSFORMER_CONTEXT,
        steps=TRANSFORMER_STEPS,
        seed=walk_seed,
    )


# the sweep row of this era, as `SWEPT` above states the columns
TRANSFORMER_SWEPT = {
    11: (358, 379, 384, 0.9975),
    23: (370, 380, 384, 0.9981),
    37: (356, 383, 384, 0.9968),
    41: (352, 382, 384, 0.9972),
}


# THE FLOORS ARE THE ERA'S OWN AND THEY ARE NOT TIGHTENED: the measured minima read
# 0.823, 0.958 and 0.9951, far above them, where the OCaml gate read 0.833, 0.948 and
# 0.9943. A floor tightened onto a measurement turns a re-draw of the set into a failure;
# the printed minima keep the calibration honest.
TRANSFORMER_FLOORS = Shares(top1=0.55, same_draw=0.80, cosine=0.98)

TRANSFORMER_PAIRS = drawn_pairs(7, 1 << 20, 40)


# Era five: the state-space model


# THE FEEDBACK AXIS OF THIS ERA IS THE STATE: a block carries a state no window forgets,
# thus an error accumulates in a register rather than dying with a ring's depth. Both
# models take one step for one step, thus the walk can run past many decay lifetimes.

# The whole plan of the era at a shape a test can afford. The Zamba head brings a SECOND
# source of drift -- a coarse ring, a softmax and a division -- thus the report answers
# for the whole model and not the recurrence alone.
MAMBA_SPELT = "MMZF"
# d 32 over 2 heads and not 16: the attention head width is then a power of FOUR, which
# `ar_quantized.score_shift` needs. The table below was re-measured when `check_shape`
# gained that rule -- the numbers before it were taken at head width 8, where the twin
# scaled by 1/2 and the float model by 1/sqrt(8).
MAMBA_SHAPE = {"d": 32, "heads": 2, "state": 8, "taps": 4}
MAMBA_RING = 16
MAMBA_STEPS = 64


@nnx.jit
def stream_float_row(held, stream, drawn):
    """the float logits of the seats of one step, on the stream the step before it left"""
    return held.head.logits(stream[:, None, :], drawn[None])[0, 0]


class StateDrift(NamedTuple):
    """What era five's quantization costs, measured on the walk the board takes."""

    steps: int  # the steps of the walk, the silent lead-in inside
    draws: int  # four for each drawn step: one for each seat of the chain
    same_peak: int  # the draws where both models elect the same class
    same_draw: int  # the draws where both models pick the same class
    mean_cosine: float
    clamps: mamba.quantized.infer.Clamps  # the twin's own, no other era reports one


def state_drift(model, *, steps, seed, ring=mamba.quantized.model.ELECTED_RING):
    """The quantized walk, scored against the float model draw for draw.

    ONE WEIGHTS SOURCE AND ONE POLICY: the walk quantizes `model` itself, thus the pair
    cannot slip. The float pass is TEACHER-FORCED on the quantized history and on the
    quantized chain -- it reads the classes the engine drew and conditions each seat on
    the classes the engine chose -- thus what the report measures is the quantization and
    never a walk that parted for another reason.

    BOTH MODELS TAKE ONE STEP FOR ONE STEP. Era four had to re-run a whole window at every
    step, which made a long comparison quadratic; here each carries its own memory, thus
    the walk can run past many decay lifetimes -- which it must, because a state error is
    cumulative in a way era four never had.

    The same-draw share reads the float draw on the very uniform the engine took, thus a
    difference there is the arithmetic and not the generator."""
    twin = mamba.quantized.model.Mamba.from_float(model, ring=ring)
    engine = mamba.quantized.infer.create_engine(twin, [seed])
    carry = model.initial_carry(1, context=ring)
    counted = Counted()
    stream = None
    for at in range(steps):
        engine, classes, chain_draws = mamba.quantized.infer.next_step(engine)
        # THE CHAIN OF A STEP READS THE STREAM OF THE STEP BEFORE IT, on both sides: the
        # float row must be the row that same forward states and never the one this
        # step's classes make
        if chain_draws and stream is not None:
            floated = np.asarray(stream_float_row(model, stream, classes)).astype(
                np.float64
            )
            counted = count_chain_draws(
                counted,
                floated,
                chain_draws,
                temperature=q.ELECTED_TEMPERATURE,
                min_p=q.ELECTED_MIN_P,
            )
        carry, stream = mamba.infer.float_step(
            model,
            carry,
            np.asarray(classes, np.int32),
            np.array([at % ar_model.PHASE_BUCKETS], np.int32),
        )
    return StateDrift(
        steps=steps,
        draws=counted.draws,
        same_peak=counted.same_peak,
        same_draw=counted.same_draw,
        mean_cosine=counted.cosine / max(1, counted.draws),
        clamps=engine.clamps,
    )


def mamba_drift(weight_seed, walk_seed, steps=MAMBA_STEPS):
    """the drift of one drawn model on one walk"""
    return state_drift(
        plan_of(MAMBA_SPELT, seed=weight_seed, **MAMBA_SHAPE),
        steps=steps,
        seed=walk_seed,
        ring=MAMBA_RING,
    )


# the sweep row of this era, as `SWEPT` above states the columns
MAMBA_SWEPT = {
    11: (714, 753, 768, 0.9980),
    23: (712, 760, 768, 0.9980),
    37: (714, 754, 768, 0.9981),
    41: (713, 759, 768, 0.9980),
}


# at 64, 256 and 1024 steps of one model: the top-1 count, the draws and the mean cosine.
# The clamps are NOT a column: the test asserts all three of them at zero outright, which
# is the finding, and a table entry would only restate it three times.
MAMBA_LONG_WALK = {
    64: (183, 192, 0.9980),
    256: (893, 960, 0.9982),
    1024: (3735, 4032, 0.9982),
}


@pytest.mark.slow
@pytest.mark.parametrize("steps", sorted(MAMBA_LONG_WALK))
def test_the_mamba_long_walk_does_not_compound(steps):
    """THE LONG WALK, AND THE CLAMPS UNDER IT. The state of a block carries forward for
    ever, thus one model runs at 64, 256 and 1024 steps and a cumulative error would
    show as numbers that FALL with the length. They do not -- the top-1 share reads
    0.953, 0.930 and 0.926 -- and a zero clamp count is the finding that the margin
    holds."""
    report = mamba_drift(11, 42, steps=steps)
    peak, draws, cosine = MAMBA_LONG_WALK[steps]
    assert (report.same_peak, report.draws) == (peak, draws)
    assert report.mean_cosine == pytest.approx(cosine, abs=5e-5)
    clamps = report.clamps
    assert (clamps.dt, clamps.beta, clamps.state) == (0, 0, 0)
    assert clamps.dt_seen and clamps.beta_seen and clamps.state_seen


# THE FLOORS ARE THE ERA'S OWN AND THEY ARE NOT TIGHTENED. They are much tighter than era
# four's 0.55, 0.8 and 0.98 for a reason that is a FORMAT and not a virtue: this datapath
# keeps the gate product whole into the norm, where a truncation back to the working class
# cost 0.10 of the cosine on its own. The measured minima read 0.875, 0.969 and 0.9980,
# where the OCaml gate read 0.875, 0.979 and 0.9972.
MAMBA_FLOORS = Shares(top1=0.80, same_draw=0.90, cosine=0.99)

MAMBA_PAIRS = drawn_pairs(7, 1 << 20, 12)


# The two gates of the three eras: one body each


class Sweep(NamedTuple):
    """One era's fixed sweep, as the gate below runs it. [drift] takes a weight seed and a
    walk seed; [counted] names the report field that says what the comparison ran over,
    which is the ONE thing that parts the three bodies; [table] is the pinned row."""

    era: str
    drift: object
    counted: str
    table: dict


SWEEPS = (
    Sweep("six", sweep_drift, "cells", SWEPT),
    # Eras four and five were measured on THIS side and not carried over from the OCaml
    # gates: the drawn weights come from a JAX draw, thus these are a re-measurement of
    # the same scheme on a different draw.
    Sweep("four", transformer_drift, "draws", TRANSFORMER_SWEPT),
    Sweep("five", mamba_drift, "draws", MAMBA_SWEPT),
)


@pytest.mark.parametrize("weight_seed", WEIGHT_SEEDS)
@pytest.mark.slow
@pytest.mark.parametrize("sweep", SWEEPS, ids=[sweep.era for sweep in SWEEPS])
def test_a_sweep_states_its_measured_numbers(sweep, weight_seed):
    """MEASURED NUMBERS AND NOT THRESHOLDS: a diff here says the integers moved, and the
    reader judges whether it is a re-measurement or a bug. The three eras run ONE body --
    four weight seeds over four walk seeds, summed, against a pinned row."""
    counted = same_peak = same_draw = 0
    low_cosine = 1.0
    for walk_seed in WALK_SEEDS:
        report = sweep.drift(weight_seed, walk_seed)
        counted += getattr(report, sweep.counted)
        same_peak += report.same_peak
        same_draw += report.same_draw
        low_cosine = min(low_cosine, report.mean_cosine)
    peak, draw, over, cosine = sweep.table[weight_seed]
    assert (same_peak, same_draw, counted) == (peak, draw, over)
    assert low_cosine == pytest.approx(cosine, abs=5e-5)


class Trials(NamedTuple):
    """One era's floor gate over drawn seed pairs, as the body below runs it. [drift] and
    [counted] are the sweep's; [pairs] is the era's own fixed set, whose length is its
    trial count; [floors] are the shares every clean trial must clear."""

    era: str
    drift: object
    counted: str
    pairs: tuple
    floors: Shares


PROPERTIES = (
    Trials("six", trial_drift, "cells", PAIRS, FLOORS),
    Trials("four", transformer_drift, "draws", TRANSFORMER_PAIRS, TRANSFORMER_FLOORS),
    Trials("five", mamba_drift, "draws", MAMBA_PAIRS, MAMBA_FLOORS),
)


@pytest.mark.slow
@pytest.mark.parametrize("trials", PROPERTIES, ids=[era.era for era in PROPERTIES])
def test_the_floors_hold_on_drawn_seed_pairs(trials, capsys):
    """THE SCHEME AGAINST A SET OF DRAWN MODELS, not the four weight seeds the sweep pins:
    a fail here is a break of the scheme and not a re-draw of the set, thus no floor is
    ever tightened onto a measurement. Every trial of every era is held to the floors: at
    Q6 no trial of era six's set rode its clamps, thus the release rule that once stood
    here had nothing left to release."""
    low = Shares(1.0, 1.0, 1.0)
    for weight_seed, walk_seed in trials.pairs:
        report = trials.drift(weight_seed, walk_seed)
        low = lowest(low, shares_of(report, trials.counted))
    # the minima print BEFORE the floors: they are what the reader needs at the moment a
    # floor breaks, and an assert inside the loop loses them
    with capsys.disabled():
        print(
            f"\nera {trials.era}: {len(trials.pairs)} drawn seed pairs: "
            f"low top-1 {low.top1:.3f}  low same draw {low.same_draw:.3f}  "
            f"low cosine {low.cosine:.4f}"
        )
    # STRICTLY ABOVE, and for all three: the two frozen eras compared with >= before the
    # three bodies became one. Nothing observable rides on it -- the measured minima stand
    # 0.02 to 0.27 clear of their floors, and no float lands on one.
    assert low.top1 > trials.floors.top1
    assert low.same_draw > trials.floors.same_draw
    assert low.cosine > trials.floors.cosine
