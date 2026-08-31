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

The rules that are not this era's come from two files. `quantized.py` holds what all
three eras read -- the exponent rule, the rounding, the int16 rails, the temper, the
min-p floor, the shared exp2 table, the integer pick and the CONTRACT FILE -- and
`ar_quantized.py` what this era and era five share: the stream formats, the norm, the
attention over a ring, the chain and the walk.

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

EVERY TENSOR IS INT32 AND THE SCALARS ARE TENSORS -- both facts of the OCaml reader, and
`quantized.py` states them once for the three eras and writes the archive. The metadata
is written as well, for a reader that has a Python tool in hand: the temperature, the
min-p and the checkpoint are PROVENANCE, because the temper and the floor are already
folded.
"""

from typing import NamedTuple

import numpy as np
from flax import nnx

import ar_model
import ar_quantized
import corpus
import measure
from ar_model import TABLES
from quantized import (
    ELECTED_MIN_P,
    ELECTED_TEMPERATURE,
    EXPONENTS,
    MIN_WEIGHT,
    TEMPER,
    Temper,
    clamp16,
    engine_states,
    image_from_tensors,
    image_tensors,
    min_weight_of,
    read_contract,
    scalar_tensor,
    write_contract,
)
from transformer import model as step

# the names of this era's own scalars; the shared ones are `contract`'s
HEADS = "heads"
CONTEXT = "context"
SLOPE_SPAN = "slope_span"
# the tensors the file carries beside its numbered weights
BESIDE_THE_WEIGHTS = (
    EXPONENTS,
    TEMPER,
    MIN_WEIGHT,
    HEADS,
    CONTEXT,
    SLOPE_SPAN,
)

# `ELECTED_TEMPERATURE` and `ELECTED_MIN_P` are imported above and stand as
# attributes of this module for era four's player to read: the policy has one home,
# `quantized.py`, and this era does not re-elect it.


class QuantizedLayer(ar_quantized.QuantizedImage):
    """One decoder layer as the machine holds it -- the twin of `model.Layer`, tensor for
    tensor and under the same names."""

    names = step.LAYER_TENSORS


class QuantizedTransformer:
    """The model as the bitstream carries it.

    The draw stands beside the layers because the bitstream carries it: one quantization
    serves every seed of a batch, as one bitstream serves every seed of the board. The
    heads, the context and the span are in no tensor, and the elaboration reads a file and
    no flag, thus they stand here too.

    IT IS NOT A `model.Trunk` -- `ar_quantized.QuantizedImage` states why no twin of this
    era is a Flax module -- thus [every_tensor] is restated below. THE ATTRIBUTE NAMES
    ARE THE PARITY and not the base class: `held.layers[2].wq` and
    `twin.layers[2].wq` are the same layer under the two arithmetics, which is what a
    reader auditing them needs.
    The rest of `ar_model.Trunk` reads a float forward and no twin ever wanted it."""

    def __init__(self, *, head, layers, heads, context, slope_span, temper, min_weight):
        self.head = head
        self.layers = list(layers)
        self.heads = int(heads)
        self.context = int(context)
        self.slope_span = int(slope_span)
        self.temper = temper
        self.min_weight = int(min_weight)

    @property
    def d(self):
        return self.head.d

    def every_tensor(self):
        """Every tensor of the model in THE ONE ORDER -- the head's tables, then the
        tensors of each layer, which is `ar_model.Trunk`'s order and the ROM's."""
        return self.head.tensors() + [
            tensor for layer in self.layers for tensor in layer.tensors()
        ]

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
            head=ar_quantized.QuantizedHead.of(model.head),
            layers=[QuantizedLayer.of(layer) for layer in model.layers],
            heads=model.heads,
            context=context,
            slope_span=model.span,
            temper=Temper.of(temperature),
            min_weight=min_weight_of(min_p),
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
    if not ar_quantized.is_power_of_four(head_d):
        raise ValueError(f"the head width {head_d} must be a power of four")
    twin.head.check_tables(d)


def save(path, twin):
    """the contract file of `twin`: the module docstring holds the layout and the
    reasons"""
    check_shape(twin)
    tensors = image_tensors(twin.every_tensor())
    tensors[HEADS] = scalar_tensor(twin.heads)
    tensors[CONTEXT] = scalar_tensor(twin.context)
    tensors[SLOPE_SPAN] = scalar_tensor(twin.slope_span)
    tensors[TEMPER] = twin.temper.tensor()
    tensors[MIN_WEIGHT] = scalar_tensor(twin.min_weight)
    write_contract(
        path,
        tensors,
        {
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
    tensors, metadata = read_contract(path)
    count = len(tensors) - len(BESIDE_THE_WEIGHTS)
    layers, spare = divmod(count - len(TABLES), step.PER_LAYER)
    if count < len(TABLES) + step.PER_LAYER or spare:
        raise ValueError(f"{path}: {len(tensors)} tensors is no quantized step model")
    exponents = tensors[EXPONENTS]
    twin = QuantizedTransformer(
        head=ar_quantized.QuantizedHead.of_file(tensors, exponents),
        layers=[
            QuantizedLayer(
                image_from_tensors(
                    tensors,
                    exponents,
                    first=len(TABLES) + step.PER_LAYER * at,
                    count=step.PER_LAYER,
                )
            )
            for at in range(layers)
        ],
        heads=int(tensors[HEADS]),
        context=int(tensors[CONTEXT]),
        slope_span=int(tensors[SLOPE_SPAN]),
        temper=Temper.of_file(tensors, metadata, key=TEMPER),
        min_weight=int(tensors[MIN_WEIGHT]),
    )
    check_shape(twin)
    return twin


# ---------------------------------------------------------------------
# the integer engine: one running inference over a batch of seeds
# ---------------------------------------------------------------------

# THE FORMAT THIS ERA NAMES OF ITS OWN. Every other one -- the stream, the normed vector,
# the hidden vector, the epsilon, log2(e), the lead-in, and the shifts and roots that read
# them -- stands in `ar_quantized.py`, where `Nn_quantized.Constants` has its twin.
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
        states=engine_states(seeds),
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
        y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)

        query, key, value = (
            projection(y, getattr(layer, name)) for name in ("wq", "wk", "wv")
        )
        kc[:, at, cur, :] = ar_quantized.coarse_to_ring(key)
        vc[:, at, cur, :] = ar_quantized.coarse_to_ring(value)
        # the rings of ONE layer: era four's second axis is the layer, and slicing it here
        # is what lets the shared `attend` name no era's axis
        context = ar_quantized.attend(
            kc[:, at],
            vc[:, at],
            query=query,
            cur=cur,
            filled=filled,
            heads=twin.heads,
            span=twin.slope_span,
            row_q=KV_Q,
        )
        h = ar_quantized.join(h, layer.wo, values=context, at=KV_Q)
        y = ar_quantized.rms_norm_q(h, at=ar_quantized.H_Q, width=d)
        hidden = clamp16(
            np.maximum(
                ar_quantized.rescale(
                    y @ layer.w1.values,
                    at=ar_quantized.Y_Q + layer.w1.e,
                    to=ar_quantized.HID_Q,
                ),
                0,
            )
        )
        h = ar_quantized.join(h, layer.w2, values=hidden, at=ar_quantized.HID_Q)
    return e._replace(h=h, kc=kc, vc=vc, position=e.position + 1)


def projection(y, weight):
    """one of the three projections of a step: one matvec column, Q12 in int16. The
    circuit runs the three separately, on one MAC path."""
    return clamp16(
        ar_quantized.rescale(y @ weight.values, at=ar_quantized.Y_Q + weight.e, to=KV_Q)
    )


def next_step(e):
    """one step of the walk -- `ar_quantized.next_step` over era four's own trunk"""
    return ar_quantized.next_step(e, forward)


def walk(twin, seeds, steps):
    """the classes of each step of the walk, and the draws behind them"""
    return ar_quantized.walk(engine(twin, seeds), steps, forward)


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
        rows = np.zeros((1, context, corpus.SEATS), dtype=np.int32)
        rows[0, :length] = np.stack(window[low:])
        phases = np.zeros((1, context), dtype=np.int32)
        phases[0, :length] = np.arange(low, at) % ar_model.PHASE_BUCKETS
        floated = np.asarray(float_row(model, rows, phases, classes, length - 1)).astype(
            np.float64
        )
        counted = measure.count_chain_draws(
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
