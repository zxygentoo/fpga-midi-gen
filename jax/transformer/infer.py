"""The sampler and the player of the step-frame model.

One step is one forward pass, always: the four seats are drawn in a chain from the soprano
down, on the host, between two passes of the network. No mask guards the draw, because no
frame is illegal.

CPU only, and deliberately: every step needs the drawn frame back on the host before the
next forward, so the loop is latency-bound and a GPU would take the device from the
trainer.

The decode is a rule of the frame and lives in corpus.py; the player sends what it makes:
raw channel voice bytes on the rawmidi device, with no backend library in the way —
the wire side itself is midi.py, shared by both eras.
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
from transformer import model, quantized


@nnx.jit
def float_window(held, classes, phases):
    """the float stream over one window, jitted.

    IT TAKES THE MODEL AS AN ARGUMENT AND STANDS AT THE MODULE LEVEL, thus its compiled
    form is keyed on the shapes and every step of every walk of the process reuses the
    first compile. A `nnx.jit` built inside `draw` is a new callable at every call, which
    is the rule `quantized.float_row` states and this is the audition's half of it."""
    return held.hidden(classes, phases)


def draw(held, *, seeds, steps, context, temperature, min_p, twin=False):
    """One batched run: [len(seeds)] independent walks of [steps] steps each.

    The boot is a lead-in of silence: one bar of silent frames, then the draw. It is
    measured and settled -- over 12 seeds the model opened the music itself inside one bar
    of the end of the lead-in, always on a multiple of four steps -- thus the boot needs
    no pitch, no range and no table. The lead-in counts inside [steps] and stands at the
    head of the music, because it is silence the walk really plays.

    [twin] draws the INTEGER twin of the circuit -- the piece the board plays at this seed
    -- and the temperature and the floor bake into it as the bitstream carries them. The
    two walks open on different generators: the float walk folds its seed and the twin
    takes it as the SEED cell does. A seed inside 32 bits names itself under both, thus an
    A/B at one seed hears the quantization and nothing else; SEED 0 IS THE EXCEPTION,
    where the twin stands still while the float walk runs from the folded state."""
    if twin:
        engine = quantized.QuantizedTransformer.of(
            held, context=context, temperature=temperature, min_p=min_p
        )
        return quantized.walk(engine, seeds, steps)[0]
    batch = len(seeds)
    state = prng.states(seeds)
    lead = corpus.BAR_STEPS
    classes = np.zeros((batch, lead, corpus.SEATS), dtype=np.int32)

    # [classes] carries one column for each step drawn so far and the loop starts at
    # [lead], thus the width is [step] at the head of every pass.
    for step in range(lead, steps):
        # ONE shape for the whole run, or every window length compiles its own kernel --
        # the history is right-padded to [batch, context] and read at its last real
        # position. The causal wall keeps a real position from seeing the padding.
        low = max(0, step - context)
        length = step - low
        window = np.zeros((batch, context, corpus.SEATS), dtype=np.int32)
        window[:, :length] = classes[:, low:step]
        # the phase of a position is the position folded into the bar, which is the rule
        # the corpus export states; nothing has to be carried beside the frames
        table = np.zeros((batch, context), dtype=np.int32)
        table[:, :length] = np.arange(low, step) % ar_model.PHASE_BUCKETS
        h = np.asarray(float_window(held, jnp.asarray(window), jnp.asarray(table)))[
            :, length - 1, :
        ].astype(np.float64)

        state, frame = held.head.draw_frame(h, state, temperature, min_p)
        classes = np.concatenate([classes, frame[:, None, :]], axis=1)
    # [steps] frames and not [max(lead, steps)]. The lead-in counts inside [steps], thus a
    # walk shorter than one bar is that many silent frames and not a whole bar of them --
    # the loop adds nothing there, and the integer twin gives exactly [steps] in any case.
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
        twin=flags.pop("twin"),
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
@click.option("--temperature", default=quantized.ELECTED_TEMPERATURE)
@click.option("--min-p", default=quantized.ELECTED_MIN_P)
def quantize(ckpt, out, heads, context, alibi_span, temperature, min_p):
    """Write the contract file of one checkpoint: the quantized model, and nothing else.

    It is the only thing that crosses the seam for a build. The heads, the context and the
    span are NOT in the checkpoint -- the heads only split the width at run time, ALiBi
    holds no position table, and the context is a choice of the draw -- thus they are
    flags here and named tensors in the file, where the elaboration reads them. The
    temperature and the floor bake into the temper and the min-p share."""
    twin = quantized.QuantizedTransformer.of(
        model.Transformer.load(ckpt, heads=heads, span=alibi_span),
        context=context,
        temperature=temperature,
        min_p=min_p,
    )
    quantized.save(out, twin)
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
