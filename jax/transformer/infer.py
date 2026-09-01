"""The sampler and the player of the step-frame model.

One step is one forward pass: the four seats are drawn in a chain from the soprano down,
on the host, between two passes of the network. No mask guards the draw, because no frame
is illegal.

CPU only, and deliberately: every step needs the drawn frame back on the host before the
next forward, thus the loop is latency-bound and a GPU would take the device from the
trainer. The wire side is midi.py and the decode is corpus.py's.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import click
import jax.numpy as jnp
import numpy as np
from flax import nnx

import ar_model
import cli
import corpus
import midi
import prng
from transformer import model
from transformer import quantized as integer


@nnx.jit
def float_window(held, classes, phases):
    """the float stream over one window, jitted.

    IT TAKES THE MODEL AS AN ARGUMENT AT THE MODULE LEVEL, thus its compiled form is keyed
    on the shapes and every walk of the process reuses the first compile; an `nnx.jit`
    built inside `draw` would be a new callable at every call."""
    return held.hidden(classes, phases)


def draw(held, *, seeds, steps, context, temperature, min_p, quantized=False):
    """One batched run: [len(seeds)] independent walks of [steps] steps each.

    The boot is a lead-in of silence: one bar of silent frames, then the draw. The model
    opens the music itself inside one bar of its end, thus the boot needs no pitch, no
    range and no table. The lead-in counts inside [steps].

    [quantized] draws the INTEGER twin of the circuit: the piece the board plays at
    this seed. The two walks open on different generators -- the float walk folds its
    seed and the twin takes it as the SEED cell does -- and a seed inside 32 bits names
    itself under both. SEED 0 IS THE EXCEPTION, where the twin stands still."""
    if quantized:
        twin = integer.Transformer.from_float(
            held, context=context, temperature=temperature, min_p=min_p
        )
        return integer.walk(twin, seeds, steps)[0]
    batch = len(seeds)
    state = prng.states(seeds)
    lead = corpus.BAR_STEPS
    classes = np.zeros((batch, lead, corpus.SEATS), dtype=np.int32)

    # [classes] carries one column for each step drawn so far, thus its width is [step]
    for step in range(lead, steps):
        # ONE shape for the whole run, or every window length compiles its own kernel:
        # right-padded to [context] and read at the last real position, which the causal
        # wall keeps from seeing the padding
        low = max(0, step - context)
        length = step - low
        window = np.zeros((batch, context, corpus.SEATS), dtype=np.int32)
        window[:, :length] = classes[:, low:step]
        # the phase of a position is the position folded into the bar
        table = np.zeros((batch, context), dtype=np.int32)
        table[:, :length] = np.arange(low, step) % ar_model.PHASE_BUCKETS
        h = np.asarray(float_window(held, jnp.asarray(window), jnp.asarray(table)))[
            :, length - 1, :
        ].astype(np.float64)

        state, frame = held.head.draw_frame(h, state, temperature, min_p)
        classes = np.concatenate([classes, frame[:, None, :]], axis=1)
    # [steps] frames and not [max(lead, steps)]: a walk shorter than one bar is that many
    # silent frames, as the integer twin gives
    return classes[:, :steps]


@click.group(help=__doc__)
def main():
    pass


@main.command(help=draw.__doc__)
@cli.sampler_options
@click.option(
    "--context", default=ar_model.TRAINING_WINDOW, help="must match the training run"
)
@click.option("--heads", default=4, help="must match the training run")
@click.option(
    "--alibi-span", default=ar_model.SLOPE_SPAN, help="must match the training run"
)
@midi.playback_options
def sample(ckpt, seeds, context, heads, alibi_span, **flags):
    walks = draw(
        model.Transformer.load(ckpt, heads=heads, span=alibi_span),
        seeds=seeds,
        context=context,
        steps=flags.pop("steps"),
        temperature=flags.pop("temperature"),
        min_p=flags.pop("min_p"),
        quantized=flags.pop("quantized"),
    )
    midi.audition([corpus.decode(walk) for walk in walks], seeds, **flags)


@main.command()
@cli.ckpt_option
@click.option("--out", required=True, type=click.Path(dir_okay=False))
@click.option("--heads", default=4, help="must match the training run")
@click.option(
    "--context",
    default=ar_model.TRAINING_WINDOW,
    help="the attention window of the circuit",
)
@click.option(
    "--alibi-span", default=ar_model.SLOPE_SPAN, help="must match the training run"
)
@click.option("--temperature", default=integer.ELECTED_TEMPERATURE)
@click.option("--min-p", default=integer.ELECTED_MIN_P)
def quantize(ckpt, out, heads, context, alibi_span, temperature, min_p):
    """Write the contract file of one checkpoint: the quantized model, and nothing else.

    It is the only thing that crosses the seam for a build. The heads, the context and
    the span are NOT in the checkpoint, thus they are flags here and named tensors in
    the file. The temperature and the floor bake into the temper and the min-p share."""
    twin = integer.Transformer.from_float(
        model.Transformer.load(ckpt, heads=heads, span=alibi_span),
        context=context,
        temperature=temperature,
        min_p=min_p,
    )
    integer.save(out, twin)
    click.echo(
        f"wrote {out}: d {twin.d}, {len(twin.layers)} layers, {twin.heads} heads, "
        f"context {twin.context}, span {twin.slope_span}"
    )
    click.echo(
        f"temper {twin.temper.q_value} at Q{twin.temper.q}, "
        f"temperature {twin.temper.temperature}, min weight {twin.min_weight}"
    )


if __name__ == "__main__":
    main()
