"""The integer twin of the step-frame model: the arithmetic the board plays.

The float model of `transformer/model.py` is what the ear elected. This module is the same
network in the arithmetic the board can hold -- int8 weights with a power-of-two exponent
for each tensor, int8 KV rings, int16 activations and the draw in integers -- and the
circuit of era four must equal it operation for operation, not approximately.

THE ORDER OF OPERATIONS IS THE CONTRACT. A rewrite that is algebraically equal and
differently ordered is a different machine.

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

from typing import NamedTuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

import data
import nn
from transformer import model as step

EXPONENTS = "exponents"
TEMPER = "temper"
MIN_WEIGHT = "min_weight"
HEADS = "heads"
CONTEXT = "context"
SLOPE_SPAN = "slope_span"
# the tensors the file carries beside its numbered weights
BESIDE_THE_WEIGHTS = (EXPONENTS, TEMPER, MIN_WEIGHT, HEADS, CONTEXT, SLOPE_SPAN)

# the policy the ear elected, `Mgen_nn.Policy`: the draw the bitstream commits to
ELECTED_TEMPERATURE = 1.0
ELECTED_MIN_P = 0.05


class Quantized(NamedTuple):
    """The model as the bitstream carries it.

    `tensors` holds every weight tensor in the ONE order -- the two tables, then six for
    each layer -- because that order is the checkpoint's, the file's and the ROM's at
    once. The shape numbers stand beside it: the heads, the context and the span are not
    in any tensor, and the elaboration reads a file and no flag."""

    tensors: list  # (q, e) in the construction order
    heads: int
    context: int
    slope_span: int
    temper: nn.Temper
    min_weight: int

    @property
    def d(self):
        """the width of the residual stream: the seat tensor sizes it"""
        return self.tensors[0][0].size // (data.SEATS * data.CLASSES)

    @property
    def layers(self):
        return (len(self.tensors) - len(nn.TABLES)) // step.PER_LAYER

    @classmethod
    def of(
        cls,
        params,
        *,
        heads,
        context,
        slope_span=nn.SLOPE_SPAN,
        temperature=ELECTED_TEMPERATURE,
        min_p=ELECTED_MIN_P,
    ):
        """the float params under the exponent rule of the eras.

        THE SEAT AND PHASE TABLES SHARE ONE EXPONENT and take it from the larger peak:
        their rows ADD -- the embedding sums them and the Embed op of the circuit walks
        them as one tensor -- thus a difference of exponents would be a difference of
        formats inside one sum. Every layer tensor takes its own."""
        flat = flat_tensors(params)
        tables = [np.asarray(flat[at], np.float64) for at in range(len(nn.TABLES))]
        shared = nn.max_exponent(max(float(np.abs(t).max(initial=0.0)) for t in tables))
        quantized = [nn.quantize(t, e=shared) for t in tables]
        quantized += [nn.quantize(t) for t in flat[len(nn.TABLES) :]]
        return cls(
            tensors=quantized,
            heads=heads,
            context=context,
            slope_span=slope_span,
            temper=nn.Temper.of(temperature),
            min_weight=nn.min_weight_of(min_p),
        )


def flat_tensors(params):
    """every tensor of the model in the construction order: the two tables, then the six
    of each layer.

    `Params_data.to_list` is the same order in OCaml, and the checkpoint names its tensors
    by it, thus this is the one statement of the order on this side."""
    flat = [np.asarray(params[name]) for name in nn.TABLES]
    for layer in params["layers"]:
        flat += [np.asarray(layer[name]) for name in step.LAYER_TENSORS]
    return flat


def check_shape(twin):
    """the rules the consumers assume, refused loudly here rather than inside a walk.

    The arithmetic of the circuit is shifts, thus the shape obeys the shift rules: d and
    the context are powers of two, the head width is a power of four, the seat table holds
    one row for each seat and class, and the two tables share one exponent."""
    d, layers = twin.d, twin.layers
    if len(twin.tensors) != len(nn.TABLES) + step.PER_LAYER * layers or layers < 1:
        raise ValueError(f"{len(twin.tensors)} tensors is no step-frame model")
    if d < 1 or d & (d - 1):
        raise ValueError(f"d is {d} and must be a power of two")
    if twin.context < 1 or twin.context & (twin.context - 1):
        raise ValueError(f"the context is {twin.context} and must be a power of two")
    if d % twin.heads:
        raise ValueError(f"{twin.heads} heads do not divide d {d}")
    head_d = d // twin.heads
    if head_d & (head_d - 1) or (head_d.bit_length() - 1) % 2:
        raise ValueError(f"the head width {head_d} must be a power of four")
    if twin.tensors[0][0].size != data.SEATS * data.CLASSES * d:
        raise ValueError("the seat table holds no row for each seat and class")
    if twin.tensors[0][1] != twin.tensors[1][1]:
        raise ValueError("the seat and phase tables must share one exponent")


def save(path, twin):
    """the contract file of `twin`: the module docstring holds the layout and the reasons"""
    check_shape(twin)
    tensors = {str(at): q for at, (q, _) in enumerate(twin.tensors)}
    tensors[EXPONENTS] = np.array([e for _, e in twin.tensors], np.int32)
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
    if count < len(nn.TABLES) + step.PER_LAYER or (count - len(nn.TABLES)) % step.PER_LAYER:
        raise ValueError(f"{path}: {len(tensors)} tensors is no quantized step model")
    exponents = tensors[EXPONENTS]
    q_value, q = (int(value) for value in tensors[TEMPER])
    twin = Quantized(
        tensors=[(tensors[str(at)], int(exponents[at])) for at in range(count)],
        heads=int(tensors[HEADS]),
        context=int(tensors[CONTEXT]),
        slope_span=int(tensors[SLOPE_SPAN]),
        temper=nn.Temper(q_value, q, float(metadata.get("temperature", np.nan))),
        min_weight=int(tensors[MIN_WEIGHT]),
    )
    check_shape(twin)
    return twin
