"""The integer twin of the step-frame model: the arithmetic the board plays.

The float model of `transformer/model.py` is what the ear elected. This module is the same
network in the arithmetic the board can hold -- int8 weights with a power-of-two exponent
for each tensor, int8 KV rings, int16 activations and the draw in integers -- and the
circuit of era four must equal it operation for operation, not approximately.

THE ORDER OF OPERATIONS IS THE CONTRACT. A rewrite that is algebraically equal and
differently ordered is a different machine.

IT CARRIES THE FLOAT MODEL'S SKELETON, `model.Trunk`, under the same attribute names at
every level: `twin.layers[k].wq` is the quantization of `float.layers[k].wq` and nothing
has to be aligned by hand.

The rules that are not this era's come from `nn.py`: the exponent rule, the rounding, the
int16 rails, the temper, the min-p floor, the shared exp2 table and the integer pick stand
there, where every twin reads them.

THE CONTRACT FILE is what crosses the seam to the elaboration. `save` writes it and
`Model.of_int8_checkpoint` reads it; `load` reads it back and a round trip is exact. It
carries the quantized model and the numbers the OCaml `Config` used to carry beside it,
because the elaboration reads a file and no flag:

    tensor          dtype   shape          value
    "0"             int32   [4, 48, d]     the seat tables, int8 in int32
    "1"             int32   [16, d]        the bar-phase table
    "2" .. "6L + 1" int32   [d, d] and so  wq wk wv wo w1 w2 of each layer, in that order
    "exponents"     int32   [6L + 2]       the exponent of each tensor, in the same order
    "heads"         int32   []             they split d at run time
    "context"       int32   []             the attention window, in steps
    "slope_span"    int32   []             the ALiBi exponent span
    "temper"        int32   [2]            the temper: q_value, then q
    "min_weight"    int32   []             the min-p share of the peak weight 2^15

EVERY VALUE TENSOR KEEPS THE SHAPE THE FLOAT TENSOR HAD, and the OCaml reader flattens it
in the row-major order the ROM wants. A shape says what a tensor IS, thus a Python reader
sees the model and not a run of bytes; the flat order is the reader's business.

THE NAMES OF THE VALUE TENSORS ARE THE FLOAT CHECKPOINT'S OWN NAMES, thus one order -- the
construction order -- carries the checkpoint, the file and the ROM, and no reader restates
it. `d` and the layer count are NOT in the file: the seat tensor sizes d and the tensor
count states the layers, as the float reader derives them.

EVERY TENSOR IS INT32 AND THE SCALARS ARE TENSORS. Both are facts of the reader:
`Nx_io.load_safetensors` skips every dtype it does not hold, thus an int8 tensor would
arrive at the elaboration as a hole; and it gives no access to `__metadata__`, thus every
number the elaboration needs travels as a named tensor. The metadata is written as well,
for a reader that has a Python tool in hand: the temperature, the min-p and the
checkpoint are PROVENANCE, because the temper and the floor are already folded.
"""

import math
from typing import NamedTuple

import numpy as np
from flax import nnx
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

import data
import nn
import prng
from nn import TABLES
from transformer import model as step

EXPONENTS = "exponents"
TEMPER = "temper"
MIN_WEIGHT = "min_weight"
HEADS = "heads"
CONTEXT = "context"
SLOPE_SPAN = "slope_span"
# the tensors the file carries beside its numbered weights
BESIDE_THE_WEIGHTS = (EXPONENTS, TEMPER, MIN_WEIGHT, HEADS, CONTEXT, SLOPE_SPAN)

# the policy the ear elected on 2026-08-18: the draw the bitstream commits to. Since
# the all-era cut took the OCaml `Policy` away, these two are the only home of it.
ELECTED_TEMPERATURE = 1.0
ELECTED_MIN_P = 0.05


class Weight(NamedTuple):
    """One tensor of the image: the int8 values in the shape the float tensor had, and the
    exponent that reads them.

    The values are int64 so that a product of two of them cannot wrap, and the file writes
    them back as int32."""

    values: np.ndarray
    e: int

    @classmethod
    def of(cls, tensor, e=None):
        """one float tensor under the exponent rule; [e] overrides the tensor's own peak"""
        q, e = nn.quantize(np.asarray(tensor, np.float64), e=e)
        return cls(np.asarray(q, np.int64), e)


class QuantizedHead(nnx.Module):
    """The two tables of `nn.Head` as the machine holds them -- and ONE exponent over both.

    THE SEAT AND PHASE TABLES SHARE IT and take it from the larger peak: their rows ADD --
    the embedding sums them and the Embed op of the circuit walks them as one tensor --
    thus a difference of exponents would be a difference of formats inside one sum. Here
    that rule is the shape of the module and cannot be broken by a caller.

    The four seat tables are ONE tensor, seat 0 first, and the circuit reaches a row of it
    with a shift and an add from the base: row (seat * CLASSES + class)."""

    def __init__(self, *, seats, phase, e):
        # `nnx.data`: these are the machine's own arrays and never a pytree of device
        # tensors -- the whole engine is host numpy in int64
        self.seats = nnx.data(np.asarray(seats, np.int64))
        self.phase = nnx.data(np.asarray(phase, np.int64))
        self.e = int(e)

    @property
    def d(self):
        return self.seats.shape[-1]

    @classmethod
    def of(cls, head):
        """the float `nn.Head` under the exponent rule, one exponent over both tables"""
        seats, phase = (np.asarray(t, np.float64) for t in head.tensors())
        shared = nn.max_exponent(
            max(float(np.abs(t).max(initial=0.0)) for t in (seats, phase))
        )
        return cls(
            seats=nn.quantize(seats, e=shared)[0],
            phase=nn.quantize(phase, e=shared)[0],
            e=shared,
        )

    def embed(self, classes, phase):
        """the embedding: the four seat rows and the phase row add in the shared exponent,
        then shift to Q16"""
        value = np.broadcast_to(self.phase[phase], (len(classes), self.d)).copy()
        for seat in range(data.SEATS):
            value = value + self.seats[seat, classes[:, seat]]
        return rescale(value, at=self.e, to=H_Q)

    def logits(self, stream, seat):
        """the tied head of one seat: rms_norm of the stream the chain has written so far,
        then that seat's table read backward; Q12 logits over the classes"""
        return (rms_norm(stream, self.d) @ self.seats[seat].T) >> self.e

    def add_row(self, stream, seat, drawn):
        """what the chain adds after a seat draws: the drawn row, in the stream's format"""
        return stream + rescale(self.seats[seat, drawn], at=self.e, to=H_Q)

    def tensors(self):
        """the two tables, in the order of the checkpoint and of the ROM"""
        return [Weight(self.seats, self.e), Weight(self.phase, self.e)]


class QuantizedLayer(nnx.Module):
    """One decoder layer as the machine holds it -- the twin of `model.Layer`, tensor for
    tensor and under the same names. Each of the six takes its OWN exponent; nothing forces
    them together."""

    def __init__(self, weights):
        for name, weight in zip(step.LAYER_TENSORS, weights):
            setattr(self, name, nnx.data(weight))

    @classmethod
    def of(cls, layer):
        """one float [model.Layer] under the exponent rule"""
        return cls([Weight.of(tensor) for tensor in layer.tensors()])

    def tensors(self):
        return [getattr(self, name) for name in step.LAYER_TENSORS]


class QuantizedTransformer(step.Trunk):
    """The model as the bitstream carries it.

    The draw stands beside the layers because the bitstream carries it: one quantization
    serves every seed of a batch, as one bitstream serves every seed of the board. The
    heads, the context and the span are in no tensor, and the elaboration reads a file and
    no flag, thus they stand here too."""

    def __init__(self, *, head, layers, heads, context, slope_span, temper, min_weight):
        self.head = head
        self.layers = nnx.List(list(layers))
        self.heads = int(heads)
        self.context = int(context)
        self.slope_span = int(slope_span)
        self.temper = temper
        self.min_weight = int(min_weight)

    @property
    def d(self):
        return self.head.d

    @classmethod
    def of(
        cls,
        model,
        *,
        context,
        temperature=ELECTED_TEMPERATURE,
        min_p=ELECTED_MIN_P,
    ):
        """The float model in the arithmetic the board holds.

        This is the one quantization of the era -- the drift walk, the audition and the
        elaboration all take their model here, thus the pair under comparison cannot slip.
        The heads and the span come off the model, which is where a player set them."""
        return cls(
            head=QuantizedHead.of(model.head),
            layers=[QuantizedLayer.of(layer) for layer in model.layers],
            heads=model.heads,
            context=context,
            slope_span=model.span,
            temper=nn.Temper.of(temperature),
            min_weight=nn.min_weight_of(min_p),
        )


def check_shape(twin):
    """the rules the consumers assume, refused loudly here rather than inside a walk.

    The arithmetic of the circuit is shifts, thus the shape obeys the shift rules: d and
    the context are powers of two, the head width is a power of four, and the seat table
    holds one row for each seat and class. The two tables share one exponent by the shape
    of `QuantizedHead`, and a FILE that disagrees is refused in `load`."""
    d = twin.d
    if not len(twin.layers):
        raise ValueError("a model of no layers is no step-frame model")
    if d < 1 or d & (d - 1):
        raise ValueError(f"d is {d} and must be a power of two")
    if twin.context < 1 or twin.context & (twin.context - 1):
        raise ValueError(f"the context is {twin.context} and must be a power of two")
    if d % twin.heads:
        raise ValueError(f"{twin.heads} heads do not divide d {d}")
    head_d = d // twin.heads
    if head_d & (head_d - 1) or (head_d.bit_length() - 1) % 2:
        raise ValueError(f"the head width {head_d} must be a power of four")
    if twin.head.seats.size != data.SEATS * data.CLASSES * d:
        raise ValueError("the seat table holds no row for each seat and class")


def save(path, twin):
    """the contract file of `twin`: the module docstring holds the layout and the reasons"""
    check_shape(twin)
    image = twin.every_tensor()
    tensors = {
        str(at): np.asarray(weight.values, np.int32) for at, weight in enumerate(image)
    }
    tensors[EXPONENTS] = np.array([weight.e for weight in image], np.int32)
    tensors[HEADS] = np.array(twin.heads, np.int32)
    tensors[CONTEXT] = np.array(twin.context, np.int32)
    tensors[SLOPE_SPAN] = np.array(twin.slope_span, np.int32)
    tensors[TEMPER] = np.array([twin.temper.q_value, twin.temper.q], np.int32)
    tensors[MIN_WEIGHT] = np.array(twin.min_weight, np.int32)
    save_file(
        tensors,
        str(path),
        metadata={
            "temper_q_value": str(twin.temper.q_value),
            "temper_q": str(twin.temper.q),
            "temperature": repr(twin.temper.temperature),
            "min_weight": str(twin.min_weight),
        },
    )


def load(path):
    """the model of one contract file; a round trip through `save` is exact.

    The temperature is provenance and not arithmetic -- the temper is already folded --
    thus it travels in the metadata alone and only a reader with a Python tool sees it."""
    tensors = load_file(str(path))
    with safe_open(str(path), framework="numpy") as opened:
        metadata = opened.metadata() or {}
    count = len(tensors) - len(BESIDE_THE_WEIGHTS)
    layers, spare = divmod(count - len(TABLES), step.PER_LAYER)
    if count < len(TABLES) + step.PER_LAYER or spare:
        raise ValueError(f"{path}: {len(tensors)} tensors is no quantized step model")
    exponents = tensors[EXPONENTS]
    if exponents[0] != exponents[1]:
        raise ValueError("the seat and phase tables must share one exponent")
    q_value, q = (int(value) for value in tensors[TEMPER])

    def weight_at(at):
        return Weight(np.asarray(tensors[str(at)], np.int64), int(exponents[at]))

    twin = QuantizedTransformer(
        head=QuantizedHead(
            seats=tensors["0"], phase=tensors["1"], e=int(exponents[0])
        ),
        layers=[
            QuantizedLayer(
                [
                    weight_at(len(TABLES) + step.PER_LAYER * at + on)
                    for on in range(step.PER_LAYER)
                ]
            )
            for at in range(layers)
        ],
        heads=int(tensors[HEADS]),
        context=int(tensors[CONTEXT]),
        slope_span=int(tensors[SLOPE_SPAN]),
        temper=nn.Temper(q_value, q, float(metadata.get("temperature", np.nan))),
        min_weight=int(tensors[MIN_WEIGHT]),
    )
    check_shape(twin)
    return twin


# ---------------------------------------------------------------------
# the integer engine: one running inference over a batch of seeds
# ---------------------------------------------------------------------

# The formats of the machine, `Model.Constants`. A Q number holds value * 2^-q.
H_Q = 16  # the residual stream, in int32
Y_Q = 12  # the normed vector, and the score of attention
KV_Q = 12  # the query, the keys, the values and the context: the rings store these
HID_Q = 10  # the feed-forward hidden vector after its ReLU
EPS_Q = int(nn.round_half_up(math.ldexp(1e-6, 2 * Y_Q)))
LOG2E = nn.Temper(int(nn.round_half_up(math.ldexp(1.0 / math.log(2.0), 15))), 15, 1.0)

# the silent lead-in of the boot, in steps: one bar, as the float sampler plays it
LEAD = data.BAR_STEPS


def rescale(value, *, at, to):
    """value * 2^-at as value * 2^-to; the arithmetic shift floors, as the circuit's"""
    if to >= at:
        return value << (to - at)
    return value >> (at - to)


def apply_scale(scale, value):
    """`Constants.apply`: value times a fixed-point multiplier, toward negative infinity"""
    return (value * scale.q_value) >> scale.q


def truncated(numerator, denominator):
    """OCaml's `/` on integers, which goes TOWARD ZERO where numpy's `//` floors.

    Every division of the circuit truncates, thus a floor here would part from it on the
    negative half of the stream and nowhere else -- which is the kind of difference that
    makes music and is still wrong."""
    numerator = np.asarray(numerator, np.int64)
    denominator = np.asarray(denominator, np.int64)
    sign = np.sign(numerator) * np.sign(denominator)
    return sign * (np.abs(numerator) // np.abs(denominator))


def isqrt(values):
    """floor of the square root, over an array: the one answer the [Isqrt] unit gives.

    The float root is correct to a unit at these widths and the two steps settle it; the
    loop is written all the same, because a silently wrong root is a silently wrong norm."""
    values = np.asarray(values, np.int64)
    guess = np.where(values <= 0, 0, np.sqrt(np.maximum(values, 0)).astype(np.int64))
    while True:
        low = np.maximum(guess - ((guess * guess > values) & (guess > 0)), 0)
        high = low + ((low + 1) * (low + 1) <= values)
        if np.array_equal(high, guess):
            return guess
        guess = high


def clamp16(value):
    """the rails of int16: a value that passes them saturates and never wraps"""
    return np.clip(value, nn.INT16_LOW, nn.INT16_HIGH)


def exp2_q(value):
    """`Nn_quantized.exp2_q`: 2^value in Q15 over a Q12 value that is 0 or less.

    Era four exponentiates a nonpositive score, thus the negation stands here and the
    shared table takes the magnitude."""
    return nn.exp2_of_magnitude(-np.asarray(value, np.int64))


def rms_norm(h, d):
    """rms_norm of the residual stream: the sum squares a Q12 copy -- one DSP-sized product
    -- then one isqrt, and one truncating division for each element."""
    copy = h >> 4
    total = (copy * copy).sum(axis=-1, keepdims=True)
    mean = (total >> (d.bit_length() - 1)) + EPS_Q
    return clamp16(truncated(h * 256, isqrt(mean)))


class Engine(NamedTuple):
    """One running inference over a batch of walks. Everything is frozen: a step gives the
    engine after it, as `Quantized.Engine` does.

    THE RINGS ARE THE CONTEXT. `kc` and `vc` hold the coarsened key and value rows of every
    layer, one slot for each step of the window, and a walk never reads an unwritten slot:
    `n` counts the filled slots and the wall is the walk itself."""

    twin: QuantizedTransformer
    h: np.ndarray  # [walks, d], Q16 in int32
    kc: np.ndarray  # [walks, layers, slots, d], Q12 int16
    vc: np.ndarray
    position: int  # one forward for each step, thus this counts the steps as well
    states: np.ndarray  # [walks], the generator of each walk

    @property
    def d(self):
        return self.twin.d


def engine(twin, seeds):
    """the origin of a batch of walks: an empty ring, no residual, and the generator at the
    SEED AS IT STANDS, which is the board's SEED cell rule.

    The lead-in is not here. It is the first steps of the walk itself, thus a caller that
    counts steps counts the steps the float sampler counts."""
    check_shape(twin)
    walks, d = len(seeds), twin.d
    rings = (walks, len(twin.layers), twin.context, d)
    return Engine(
        twin=twin,
        h=np.zeros((walks, d), np.int64),
        kc=np.zeros(rings, np.int64),
        vc=np.zeros(rings, np.int64),
        position=0,
        states=nn.engine_states(seeds),
    )


def attend(twin, kc, vc, *, layer, cur, filled, query):
    """Attention of one layer over the newest [filled] steps of the rings: the merged
    context of [query], head by head.

    Age a reads slot (cur - a) & (slots - 1), thus the ALiBi distance is the age itself and
    the causal wall is the walk."""
    d, heads, slots = twin.d, twin.heads, twin.context
    head_d = d // heads
    ages = np.arange(filled)
    rows = (cur - ages) & (slots - 1)
    keys = kc[:, layer, rows, :]  # [walks, filled, d]
    values = vc[:, layer, rows, :]
    context = np.zeros((len(query), d), np.int64)
    shift = (2 * KV_Q) - Y_Q + ((head_d.bit_length() - 1) // 2)
    for head in range(heads):
        band = slice(head * head_d, (head + 1) * head_d)
        slope = (twin.slope_span * (head + 1)) // heads
        raw = (query[:, None, band] * keys[:, :, band]).sum(axis=-1)
        scores = (raw >> shift) - (ages << (Y_Q - slope))
        peak = scores.max(axis=-1, keepdims=True)
        weights = exp2_q(apply_scale(LOG2E, scores - peak))
        total = weights.sum(axis=-1, keepdims=True)
        merged = (weights[:, :, None] * values[:, :, band]).sum(axis=1)
        context[:, band] = clamp16(truncated(merged, total))
    return context


def forward(e, classes, phase):
    """one step through the engine: the engine after it"""
    twin = e.twin
    d, slots = twin.d, twin.context
    cur = e.position & (slots - 1)
    filled = min(e.position + 1, slots)
    kc, vc = e.kc.copy(), e.vc.copy()
    h = twin.head.embed(classes, phase)
    for at, layer in enumerate(twin.layers):
        y = rms_norm(h, d)

        query, key, value = (
            projection(y, getattr(layer, name)) for name in ("wq", "wk", "wv")
        )
        # THE RING KEEPS THE TOP BYTE of a Q12 row: the circuit stores eight bits and
        # restores eight zero low bits at the read, thus the granularity is 2^-4 and the
        # format stays Q12. The query does not pass here -- only the stored rows coarsen.
        kc[:, at, cur, :] = (key >> 8) << 8
        vc[:, at, cur, :] = (value >> 8) << 8
        context = attend(twin, kc, vc, layer=at, cur=cur, filled=filled, query=query)
        h = join(h, layer.wo, values=context, at=KV_Q)
        y = rms_norm(h, d)
        hidden = clamp16(
            np.maximum(rescale(y @ layer.w1.values, at=Y_Q + layer.w1.e, to=HID_Q), 0)
        )
        h = join(h, layer.w2, values=hidden, at=HID_Q)
    return e._replace(h=h, kc=kc, vc=vc, position=e.position + 1)


def projection(y, weight):
    """one of the three projections of a step: one matvec column, Q12 in int16. The circuit
    runs the three separately, on one MAC path."""
    return clamp16(rescale(y @ weight.values, at=Y_Q + weight.e, to=KV_Q))


def join(h, weight, *, values, at):
    """a residual join: [values] times the weight lands on the stream; the exponent of the
    weight folds into the shift with [at], the format of [values]"""
    accumulated = values @ weight.values
    return h + rescale(accumulated, at=at + weight.e, to=H_Q)


def tempered_weights(twin, logits):
    """the Q15 weight of every class of one seat, and the min-p floor over it.

    The peak weighs 2^15, thus the floor is a plain share of it; a class the floor refuses
    weighs nothing and the pick cannot land on it."""
    peak = logits.max(axis=-1, keepdims=True)
    weights = exp2_q(apply_scale(twin.temper, logits - peak))
    return np.where(weights >= twin.min_weight, weights, 0)


class Draw(NamedTuple):
    """one draw of the chain, over the batch"""

    seat: int
    logits: np.ndarray  # [walks, CLASSES], Q12 -- what the drift report compares
    word: np.ndarray  # [walks], the 24-bit uniform
    drawn: np.ndarray  # [walks], the class


def chain(e):
    """One frame, drawn in a chain from the soprano down: each seat reads the stream that
    the seats above it have written. The draws come back in the order they happened."""
    twin = e.twin
    stream, states, draws = e.h, e.states, []
    everyone = np.ones(len(stream), bool)
    for seat in reversed(range(data.SEATS)):
        logits = twin.head.logits(stream, seat)
        states, word = prng.uniform_word(states, everyone)
        drawn = nn.pick(tempered_weights(twin, logits), word)
        if seat:
            stream = twin.head.add_row(stream, seat, drawn)
        draws.append(Draw(seat, logits, word, drawn))
    return e._replace(states=states), draws


def next_step(e):
    """one step of the walk: the engine after it, the classes of the frame, and the draws.

    THE BOOT IS A LEAD-IN OF SILENCE, one bar of it, drawing nothing and taking no number
    from the generator. The model opens the music itself after it, thus the walk needs no
    pitch and no table to begin."""
    phase = e.position % data.BAR_STEPS
    if e.position < LEAD:
        classes = np.full((len(e.h), data.SEATS), data.SILENCE, np.int64)
        draws = []
    else:
        e, draws = chain(e)
        classes = np.stack([draw.drawn for draw in reversed(draws)], axis=-1)
    return forward(e, classes, phase), classes, draws


def walk(twin, seeds, steps):
    """the classes of each step of the walk, and the draws behind them.

    It is the integer twin of the float sampler, and the lead-in counts inside [steps] as
    it does there."""
    e = engine(twin, seeds)
    played, taken = [], []
    for _ in range(steps):
        e, classes, draws = next_step(e)
        played.append(classes)
        taken.append(draws)
    return np.stack(played, axis=1), taken


# ---------------------------------------------------------------------
# what the quantization costs
# ---------------------------------------------------------------------


class Drift(NamedTuple):
    """What the quantization costs, measured on the walk the board takes."""

    steps: int  # the steps of the walk, the silent lead-in inside
    draws: int  # four for each drawn step: one for each seat of the chain
    same_peak: int  # the draws where both models elect the same class
    same_draw: int  # the draws where both models pick the same class
    mean_cosine: float


def cosine(here, there):
    """the cosine between an integer row and the float row of the same place"""
    here = np.asarray(here, np.float64)
    return float(np.dot(here, there) / np.sqrt(np.dot(here, here) * np.dot(there, there)))


def drift(model, *, context, steps, seed):
    """The quantized walk, scored against the float model draw for draw.

    ONE WEIGHTS SOURCE AND ONE POLICY: the walk quantizes `model` itself, thus the pair
    cannot slip. The float pass is TEACHER-FORCED on the quantized history and on the
    quantized chain -- it reads the classes the engine drew and conditions each seat on the
    classes the engine chose -- thus what the report measures is the quantization and never
    a walk that parted for another reason.

    The same-draw share reads the float draw on the very uniform the engine took, thus a
    difference there is the arithmetic and not the generator."""
    import jax.numpy as jnp

    e = engine(QuantizedTransformer.of(model, context=context), [seed])
    history = []
    counted = same_peak = same_draw = 0
    total = 0.0
    for at in range(steps):
        e, classes, chain_draws = next_step(e)
        # THE HISTORY IS THE TWIN'S, and the float pass reads it: the window the model saw
        # before this step is the window the engine's own ring held.
        window = list(history)
        history.append(classes[0])
        if not chain_draws or not window:
            continue
        low = max(0, at - context)
        rows = jnp.asarray(np.stack(window[low:])[None])
        phases = jnp.asarray((np.arange(low, at) % nn.PHASE_BUCKETS)[None])
        h = np.asarray(model.hidden(rows, phases))[:, -1:, :]
        floated = np.asarray(
            model.head.logits(jnp.asarray(h), jnp.asarray(classes[None]))
        )[0, 0].astype(np.float64)
        for taken in chain_draws:
            row = floated[taken.seat]
            mine = taken.logits[0]
            uniform = np.array([float(taken.word[0]) * 2.0**-prng.UNIFORM_BITS])
            weights = nn.temper(row[None], ELECTED_TEMPERATURE, ELECTED_MIN_P)
            counted += 1
            same_peak += int(np.argmax(mine) == np.argmax(row))
            same_draw += int(nn.pick_share(weights, uniform)[0] == taken.drawn[0])
            total += cosine(mine, row)
    return Drift(
        steps=steps,
        draws=counted,
        same_peak=same_peak,
        same_draw=same_draw,
        mean_cosine=total / max(1, counted),
    )
