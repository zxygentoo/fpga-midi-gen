"""The step-frame transformer of docs/transformer.md.

One step of music is one position. Four voice classes enter through four tables that sum
and leave through the same four in a chain from the soprano down -- the head is
`ar_model.Head`, which era five reads too. Under it is a decoder with no bias terms,
RMSNorm before each sublayer, ALiBi for the position, and d_ff = 4 d.

THE NET IS A MODULE TREE AND NOT A DICTIONARY OF TENSORS: `quantized.QuantizedTransformer`
carries the same skeleton in integers under the same attribute names, thus a reader can
put `model.layers[k]` beside `twin.layers[k]` and audit one against its quantization.

THE HEADS AND THE SPAN ARE NOT IN A CHECKPOINT -- the heads only split d at run time and
ALiBi holds no position table -- thus a player states them and the contract file carries
them to the elaboration. A checkpoint is tensors "0".."N" in construction order: seats
[4, 48, d], phase [16, d], then per layer wq wk wv wo [d, d], w1 [d, 4d], w2 [4d, d].
"""

import jax
import jax.numpy as jnp
from flax import nnx
from safetensors.numpy import load_file

import ar_model
from ar_model import SLOPE_SPAN, TABLES
from train import save_checkpoint

LAYER_TENSORS = ("wq", "wk", "wv", "wo", "w1", "w2")
PER_LAYER = len(LAYER_TENSORS)


class Layer(nnx.Module):
    """One decoder layer: the attention sublayer, then the feed-forward under it.

    NO TENSOR HERE CARRIES A BIAS. RMSNorm stands before each sublayer and the residual
    join after it, thus a bias would only shift a stream a norm is about to rescale."""

    def __init__(self, d, *, rngs):
        for name, shape in zip(LAYER_TENSORS, self.shapes(d)):
            setattr(self, name, nnx.Param(ar_model.normal_at(rngs.params(), shape)))

    @staticmethod
    def shapes(d):
        """the shape of each tensor of [tensors], in the construction order"""
        return [(d, d)] * 4 + [(d, 4 * d), (4 * d, d)]

    def __call__(self, h, bias, heads, drop):
        """the residual stream after this layer: the attention branch and the feed-forward
        branch, each normed before it and joined through [drop] after it"""
        batch, length, d = h.shape
        head_d = d // heads

        def split_heads(x):
            return x.reshape(batch, length, heads, head_d).transpose(0, 2, 1, 3)

        normed = ar_model.rms_norm(h)
        q = split_heads(normed @ self.wq[...])
        k = split_heads(normed @ self.wk[...])
        v = split_heads(normed @ self.wv[...])
        scores = q @ k.transpose(0, 1, 3, 2) * (1.0 / jnp.sqrt(float(head_d))) + bias
        context = jax.nn.softmax(scores, axis=-1) @ v
        merged = context.transpose(0, 2, 1, 3).reshape(batch, length, d)
        h = h + drop(merged @ self.wo[...])
        hidden = jnp.maximum(ar_model.rms_norm(h) @ self.w1[...], 0.0)
        return h + drop(hidden @ self.w2[...])

    def tensors(self):
        """the six tensors of this layer in the order of the checkpoint and of the ROM"""
        return [getattr(self, name)[...] for name in LAYER_TENSORS]

    def take(self, tensors):
        """the reverse of [tensors]; the two stand together so the layout cannot drift"""
        for name, value in zip(LAYER_TENSORS, tensors):
            getattr(self, name)[...] = jnp.asarray(value)


class Trunk(ar_model.Trunk):
    """The skeleton both models of the era carry: the tied head, then the layers. The
    float tree and the integer twin fill it under the SAME attribute names at every
    level."""


class Transformer(Trunk):
    """The decoder of the era: the tied head, and [layers] identical layers under it."""

    def __init__(self, d, layers, *, heads, span=SLOPE_SPAN, rngs):
        if d % heads:
            raise ValueError(f"{heads} heads do not divide d {d}")
        self.head = ar_model.Head(d, rngs=rngs)
        self.layers = nnx.List([Layer(d, rngs=rngs) for _ in range(layers)])
        # neither is a weight and neither is in a checkpoint: the heads split d at run
        # time and ALiBi holds no position table
        self.heads = int(heads)
        self.span = span

    @property
    def d(self):
        return self.head.d

    def hidden(self, classes, phases, *, dropout=0.0, key=None):
        """classes [batch, length, SEATS] -> [batch, length, d], the residual stream after
        the last layer and before any readout.

        dropout > 0 needs a PRNG [key]; it drops the embedding sum and each residual
        branch."""
        bias = ar_model.attention_bias(self.heads, classes.shape[1], self.span)
        # the embedding sum, then the two residual branches of each layer
        drop = ar_model.dropout(key, dropout, 1 + 2 * len(self.layers))
        h = drop(self.head.embed(classes, phases))
        for layer in self.layers:
            h = layer(h, bias, self.heads, drop)
        return h

    def describe(self):
        """the shape of this model in one phrase, for the head of a training log"""
        return (
            f"d {self.d}, {len(self.layers)} layers, {self.heads} heads, "
            f"ALiBi span {self.span}"
        )

    def save(self, path):
        """the whole model as one flat list, in the construction order"""
        save_checkpoint(path, self.every_tensor())

    @classmethod
    def load(cls, path, *, heads, span=SLOPE_SPAN):
        """The model of one checkpoint: the tensor count states the layers and the seat
        table states d. The heads and the span are the player's."""
        tensors = load_file(str(path))
        count = len(tensors)
        layers, spare = divmod(count - len(TABLES), PER_LAYER)
        if count < len(TABLES) + PER_LAYER or spare:
            raise ValueError(
                f"{path}: {count} tensors is not {TABLES} and {PER_LAYER} for each layer"
            )
        # the draw is thrown away tensor by tensor below; it buys the one constructor
        held = cls(
            tensors["0"].shape[-1], layers, heads=heads, span=span, rngs=nnx.Rngs(0)
        )
        held.take([tensors[str(at)] for at in range(count)])
        return held

    def take(self, tensors):
        """the flat list of [every_tensor], written back in"""
        self.head.take(tensors[: len(TABLES)])
        for at, layer in enumerate(self.layers):
            base = len(TABLES) + PER_LAYER * at
            layer.take(tensors[base : base + PER_LAYER])

    @classmethod
    def drawn(cls, seed, d, layers, *, heads, span=SLOPE_SPAN):
        """A model of DRAWN weights: the trainer's opening, and the shape a gate can
        afford. THE DRAW IS THE MODULE'S AND NOT ITS INITIALIZER'S -- one key for each
        tensor in construction order -- because `tests/test_drift.py` reads this draw."""
        keys = iter(
            jax.random.split(jax.random.key(seed), len(TABLES) + PER_LAYER * layers)
        )

        def drawn_tensors(shapes):
            return [ar_model.normal_at(next(keys), shape) for shape in shapes]

        held = cls(d, layers, heads=heads, span=span, rngs=nnx.Rngs(0))
        held.head.take(drawn_tensors(ar_model.Head.shapes(d)))
        for layer in held.layers:
            layer.take(drawn_tensors(Layer.shapes(d)))
        return held
