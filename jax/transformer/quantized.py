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

The rules that are not this era's come from `fixed.py`: the exponent rule, the rounding,
the int16 rails, the temper, the min-p floor, the shared exp2 table and the integer pick
stand there, where every twin reads them.

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

from typing import NamedTuple

import numpy as np
from flax import nnx
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

import data
import fixed
import measure
import nn
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

# the policy the ear elected, stated once in `fixed`; era four takes it as it stands
ELECTED_TEMPERATURE = fixed.ELECTED_TEMPERATURE
ELECTED_MIN_P = fixed.ELECTED_MIN_P


class QuantizedLayer(nnx.Module):
    """One decoder layer as the machine holds it -- the twin of `model.Layer`, tensor for
    tensor and under the same names. Each of the six takes its OWN exponent; nothing
    forces them together."""

    def __init__(self, weights):
        for name, weight in zip(step.LAYER_TENSORS, weights):
            setattr(self, name, nnx.data(weight))

    @classmethod
    def of(cls, layer):
        """one float [model.Layer] under the exponent rule"""
        return cls([fixed.Weight.of(tensor) for tensor in layer.tensors()])

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
            head=fixed.QuantizedHead.of(model.head),
            layers=[QuantizedLayer.of(layer) for layer in model.layers],
            heads=model.heads,
            context=context,
            slope_span=model.span,
            temper=fixed.Temper.of(temperature),
            min_weight=fixed.min_weight_of(min_p),
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
    """the contract file of `twin`: the module docstring holds the layout and the
    reasons"""
    check_shape(twin)
    image = twin.every_tensor()
    tensors = {
        str(at): np.asarray(weight.values, np.int32) for at, weight in enumerate(image)
    }
    tensors[EXPONENTS] = np.array([weight.e for weight in image], np.int32)
    tensors[HEADS] = np.array(twin.heads, np.int32)
    tensors[CONTEXT] = np.array(twin.context, np.int32)
    tensors[SLOPE_SPAN] = np.array(twin.slope_span, np.int32)
    tensors[TEMPER] = twin.temper.tensor()
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

    def weight_at(at):
        return fixed.Weight(np.asarray(tensors[str(at)], np.int64), int(exponents[at]))

    twin = QuantizedTransformer(
        head=fixed.QuantizedHead(
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
        temper=fixed.Temper.of_file(tensors, metadata, key=TEMPER),
        min_weight=int(tensors[MIN_WEIGHT]),
    )
    check_shape(twin)
    return twin


# ---------------------------------------------------------------------
# the integer engine: one running inference over a batch of seeds
# ---------------------------------------------------------------------

# THE FORMAT THIS ERA NAMES OF ITS OWN. Every other one -- the stream, the normed vector,
# the hidden vector, the epsilon, log2(e), the lead-in, and the shifts and roots that read
# them -- stands in `fixed.py`, where `Nn_quantized.Constants` has its twin.
#
# `KV_Q` is the query, the keys, the values and the context: the rings store these rows,
# Q12 in int16. Era five states a 12 of its own -- `V_Q`, which names a BLOCK's value rows
# as well as an attention ring's and carries its gate at 2 * V_Q -- thus the two are one
# number and not one format; `Model.Constants` keeps them apart on the OCaml side for that
# reason.
KV_Q = 12


class Engine(NamedTuple):
    """One running inference over a batch of walks. Everything is frozen: a step gives the
    engine after it, thus a walk is a fold and no state hides in a mutable field.

    THE RINGS ARE THE CONTEXT. `kc` and `vc` hold the coarsened key and value rows of
    every layer, one slot for each step of the window, and a walk never reads an unwritten
    slot: `n` counts the filled slots and the wall is the walk itself."""

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
    """the origin of a batch of walks: an empty ring, no residual, and the generator at
    the SEED AS IT STANDS, which is the board's SEED cell rule.

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
        states=fixed.engine_states(seeds),
    )


def forward(e, classes, phase):
    """one step through the engine: the engine after it"""
    twin = e.twin
    d, slots = twin.d, twin.context
    cur = e.position & (slots - 1)
    filled = min(e.position + 1, slots)
    kc, vc = e.kc.copy(), e.vc.copy()
    h = twin.head.embed(classes, phase)
    for at, layer in enumerate(twin.layers):
        y = fixed.rms_norm_q(h, at=fixed.H_Q, width=d)

        query, key, value = (
            projection(y, getattr(layer, name)) for name in ("wq", "wk", "wv")
        )
        kc[:, at, cur, :] = fixed.coarse_to_ring(key)
        vc[:, at, cur, :] = fixed.coarse_to_ring(value)
        # the rings of ONE layer: era four's second axis is the layer, and slicing it here
        # is what lets the shared `attend` name no era's axis
        context = fixed.attend(
            kc[:, at],
            vc[:, at],
            query=query,
            cur=cur,
            filled=filled,
            heads=twin.heads,
            span=twin.slope_span,
            row_q=KV_Q,
        )
        h = fixed.join(h, layer.wo, values=context, at=KV_Q)
        y = fixed.rms_norm_q(h, at=fixed.H_Q, width=d)
        hidden = fixed.clamp16(
            np.maximum(
                fixed.rescale(
                    y @ layer.w1.values, at=fixed.Y_Q + layer.w1.e, to=fixed.HID_Q
                ),
                0,
            )
        )
        h = fixed.join(h, layer.w2, values=hidden, at=fixed.HID_Q)
    return e._replace(h=h, kc=kc, vc=vc, position=e.position + 1)


def projection(y, weight):
    """one of the three projections of a step: one matvec column, Q12 in int16. The
    circuit runs the three separately, on one MAC path."""
    return fixed.clamp16(
        fixed.rescale(y @ weight.values, at=fixed.Y_Q + weight.e, to=KV_Q)
    )


def next_step(e):
    """one step of the walk -- `fixed.next_step` over era four's own trunk"""
    return fixed.next_step(e, forward)


def walk(twin, seeds, steps):
    """the classes of each step of the walk, and the draws behind them"""
    return fixed.walk(engine(twin, seeds), steps, forward)


# ---------------------------------------------------------------------
# what the quantization costs
# ---------------------------------------------------------------------


@nnx.jit
def float_row(held, window, phases, drawn, at):
    """The float logits of the seats of ONE step, teacher-forced on the twin's history.

    It takes the model as an ARGUMENT and stands at the module level, thus its compiled
    form is keyed on the shapes and every step of a drift run reuses the first compile.
    [window] and [phases] are padded to the model's context and [at] is the last real
    position, which holds ONE shape over a whole run -- the causal wall keeps [at] from
    seeing the padding."""
    h = held.hidden(window, phases)[:, at, None, :]
    return held.head.logits(h, drawn[None])[0, 0]


class Drift(NamedTuple):
    """What the quantization costs, measured on the walk the board takes."""

    steps: int  # the steps of the walk, the silent lead-in inside
    draws: int  # four for each drawn step: one for each seat of the chain
    same_peak: int  # the draws where both models elect the same class
    same_draw: int  # the draws where both models pick the same class
    mean_cosine: float


def drift(model, *, context, steps, seed):
    """The quantized walk, scored against the float model draw for draw.

    ONE WEIGHTS SOURCE AND ONE POLICY: the walk quantizes `model` itself, thus the pair
    cannot slip. The float pass is TEACHER-FORCED on the quantized history and on the
    quantized chain -- it reads the classes the engine drew and conditions each seat on
    the classes the engine chose -- thus what the report measures is the quantization and
    never a walk that parted for another reason.

    The same-draw share reads the float draw on the very uniform the engine took, thus a
    difference there is the arithmetic and not the generator."""
    e = engine(QuantizedTransformer.of(model, context=context), [seed])
    history = []
    counted = measure.Counted()
    for at in range(steps):
        e, classes, chain_draws = next_step(e)
        # THE HISTORY IS THE TWIN'S, and the float pass reads it: the window the model saw
        # before this step is the window the engine's own ring held.
        window = list(history)
        history.append(classes[0])
        if not chain_draws or not window:
            continue
        # ONE shape for the whole run, as `infer.walk` holds one: the history is
        # right-padded to [context] and read at its last real position.
        low = max(0, at - context)
        length = at - low
        rows = np.zeros((1, context, data.SEATS), dtype=np.int32)
        rows[0, :length] = np.stack(window[low:])
        phases = np.zeros((1, context), dtype=np.int32)
        phases[0, :length] = np.arange(low, at) % nn.PHASE_BUCKETS
        floated = np.asarray(float_row(model, rows, phases, classes, length - 1)).astype(
            np.float64
        )
        counted = measure.count_draws(
            counted,
            floated,
            chain_draws,
            temperature=ELECTED_TEMPERATURE,
            min_p=ELECTED_MIN_P,
        )
    return Drift(
        steps=steps,
        draws=counted.draws,
        same_peak=counted.same_peak,
        same_draw=counted.same_draw,
        mean_cosine=counted.cosine / max(1, counted.draws),
    )
