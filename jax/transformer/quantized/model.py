"""The integer twin of the step-frame model: the weights, the formats and the file.

The float model of `transformer/model.py` is what the ear elected; this is the same
network in the arithmetic the board can hold -- int8 weights with a power-of-two exponent
for each tensor, int8 KV rings, int16 activations and the draw in integers. What RUNS it
is `quantized/infer.py`, and the circuit of era four must equal that walk OPERATION FOR
OPERATION.

It carries the float model's skeleton under the same attribute names at every level, thus
`twin.layers[k].wq` is the quantization of `float.layers[k].wq`. What is not this era's
comes from `quantized.py` and `ar_quantized.py`.

THE CONTRACT FILE is what crosses the seam to the elaboration. `save` writes it and
`Model.of_int8_checkpoint` reads it; `load` reads it back and a round trip is exact. It
carries the quantized model and the numbers the elaboration would otherwise take as
flags:

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

EVERY VALUE TENSOR KEEPS THE SHAPE THE FLOAT TENSOR HAD and the names are the float
checkpoint's own, thus one order -- the construction order -- carries the checkpoint, the
file and the ROM. `d` and the layer count are NOT in the file: the seat tensor sizes d and
the tensor count states the layers. `quantized.py` states why every tensor is int32 and
every scalar a tensor.
"""

from safetensors.numpy import load_file, save_file

import ar_quantized
import quantized as q
from ar_model import TABLES
from transformer import model as step

# the names of this era's own scalars; the shared ones are `contract`'s
HEADS = "heads"
CONTEXT = "context"
SLOPE_SPAN = "slope_span"
# the tensors the file carries beside its numbered weights
BESIDE_THE_WEIGHTS = (
    q.EXPONENTS,
    q.TEMPER,
    q.MIN_WEIGHT,
    HEADS,
    CONTEXT,
    SLOPE_SPAN,
)

# the policy has one home, `quantized.py`, and this era does not re-elect it; era four's
# player reads it through this module
ELECTED_TEMPERATURE = q.ELECTED_TEMPERATURE
ELECTED_MIN_P = q.ELECTED_MIN_P


class Layer(ar_quantized.Weights):
    """One decoder layer as the machine holds it -- the twin of `model.Layer`, tensor for
    tensor and under the same names."""

    names = step.LAYER_TENSORS


class Transformer:
    """The model as the bitstream carries it: the draw, the heads, the context and the
    span stand beside the layers, because one quantization serves every seed as one
    bitstream serves every seed of the board.

    IT IS NOT A `model.Trunk` -- `ar_quantized.Weights` states why no twin of this
    era is a Flax module -- thus [tensors] is restated below. THE ATTRIBUTE NAMES ARE
    THE PARITY and not the base class."""

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

    def tensors(self):
        """Every tensor of the model in THE ONE ORDER -- the head's tables, then the
        tensors of each layer, which is `ar_model.Trunk`'s order and the ROM's."""
        return self.head.tensors() + [
            tensor for layer in self.layers for tensor in layer.tensors()
        ]

    @classmethod
    def from_float(
        cls,
        model,
        *,
        context,
        temperature=ELECTED_TEMPERATURE,
        min_p=ELECTED_MIN_P,
    ):
        """The float model in the arithmetic the board holds. It is the ONE
        quantization of the era -- the drift walk, the audition and the elaboration all
        take their model here -- thus the pair under comparison cannot slip."""
        return cls(
            head=ar_quantized.Head.from_float(model.head),
            layers=[Layer.from_float(layer) for layer in model.layers],
            heads=model.heads,
            context=context,
            slope_span=model.span,
            temper=q.Temper.from_float(temperature),
            min_weight=q.min_weight(min_p),
        )

    def check_shape(self):
        """The rules the consumers assume, refused here rather than inside a walk. The
        arithmetic of the circuit is shifts, thus d and the context are powers of two and
        the head width is a power of four."""
        d = self.d
        if not len(self.layers):
            raise ValueError("a model of no layers is no step-frame model")
        if d < 1 or d & (d - 1):
            raise ValueError(f"d is {d} and must be a power of two")
        if self.context < 1 or self.context & (self.context - 1):
            raise ValueError(f"the context is {self.context} and must be a power of two")
        if d % self.heads:
            raise ValueError(f"{self.heads} heads do not divide d {d}")
        head_d = d // self.heads
        if not ar_quantized.is_power_of_four(head_d):
            raise ValueError(f"the head width {head_d} must be a power of four")
        self.head.check_tables(d)

    def save(self, path):
        """the contract file of this twin: the module docstring holds the layout and the
        reasons"""
        self.check_shape()
        tensors = q.image_tensors(self.tensors())
        tensors[HEADS] = q.scalar_tensor(self.heads)
        tensors[CONTEXT] = q.scalar_tensor(self.context)
        tensors[SLOPE_SPAN] = q.scalar_tensor(self.slope_span)
        tensors[q.TEMPER] = self.temper.tensor()
        tensors[q.MIN_WEIGHT] = q.scalar_tensor(self.min_weight)
        save_file(tensors, str(path))

    @classmethod
    def load(cls, path):
        """the model of one contract file; a round trip through `save` is exact"""
        tensors = load_file(str(path))
        count = len(tensors) - len(BESIDE_THE_WEIGHTS)
        layers, spare = divmod(count - len(TABLES), step.PER_LAYER)
        if count < len(TABLES) + step.PER_LAYER or spare:
            raise ValueError(f"{path}: {len(tensors)} tensors is no quantized step model")
        exponents = tensors[q.EXPONENTS]
        twin = cls(
            head=ar_quantized.Head.from_file(tensors, exponents),
            layers=[
                Layer(
                    q.image_from_tensors(
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
            temper=q.Temper.from_file(tensors, key=q.TEMPER),
            min_weight=int(tensors[q.MIN_WEIGHT]),
        )
        twin.check_shape()
        return twin
