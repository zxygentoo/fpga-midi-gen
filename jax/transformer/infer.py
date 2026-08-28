"""The sampler and the player of the step-frame model.

One step is one forward pass, always: the four seats are drawn in a chain from the soprano
down, on the host, between two passes of the network. No mask guards the draw, because no
frame is illegal.

CPU only, and deliberately: every step needs the drawn frame back on the host before the
next forward, so the loop is latency-bound and a GPU would take the device from the
trainer.

The decode is a rule of the frame and lives in data.py; the player sends what it makes:
raw channel voice bytes on the rawmidi device, with no backend library in the way —
the wire side itself is midi.py, shared by both eras.
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import click
import jax
import jax.numpy as jnp
import numpy as np

import data
import midi
import prng
from nn import draw_frame
from transformer import model, quantized


def draw(params, *, seeds, steps, context, heads, span, temperature, min_p):
    """One batched run: [len(seeds)] independent walks of [steps] steps each.

    The boot is a lead-in of silence: one bar of silent frames, then the draw. It is
    measured and settled -- over 12 seeds the model opened the music itself inside one bar
    of the end of the lead-in, always on a multiple of four steps -- thus the boot needs no
    pitch, no range and no table. The lead-in counts inside [steps] and stands at the head
    of the music, because it is silence the walk really plays."""
    batch = len(seeds)
    forward = jax.jit(
        lambda classes, phases: model.hidden(
            params, classes, phases, heads=heads, span=span
        )
    )
    state = prng.states(seeds)
    lead = data.BAR_STEPS
    classes = np.zeros((batch, lead, data.SEATS), dtype=np.int32)

    # [classes] carries one column for each step drawn so far and the loop starts at
    # [lead], thus the width is [step] at the head of every pass.
    for step in range(lead, steps):
        # ONE shape for the whole run, or every window length compiles its own kernel --
        # the history is right-padded to [batch, context] and read at its last real
        # position. The causal wall keeps a real position from seeing the padding.
        low = max(0, step - context)
        length = step - low
        window = np.zeros((batch, context, data.SEATS), dtype=np.int32)
        window[:, :length] = classes[:, low:step]
        # the phase of a position is the position folded into the bar, which is the rule
        # the corpus export states; nothing has to be carried beside the frames
        table = np.zeros((batch, context), dtype=np.int32)
        table[:, :length] = np.arange(low, step) % model.PHASE_BUCKETS
        h = np.asarray(forward(jnp.asarray(window), jnp.asarray(table)))[
            :, length - 1, :
        ].astype(np.float64)

        state, frame = draw_frame(params, h, state, temperature, min_p)
        classes = np.concatenate([classes, frame[:, None, :]], axis=1)
    # [steps] frames and not [max(lead, steps)]. The lead-in counts inside [steps], thus a
    # walk shorter than one bar is that many silent frames and not a whole bar of them --
    # the loop adds nothing there, and the integer twin gives exactly [steps] in any case.
    return classes[:, :steps]


@click.group(help=__doc__)
def main():
    pass


@main.command(help=draw.__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--seeds", default="1", callback=midi.parse_seeds, help="a list, or LOW-HIGH")
@click.option("--steps", default=256, help="steps to draw, the silent lead-in inside")
@click.option("--context", default=256, help="must match the training run")
@click.option("--heads", default=4, help="must match the training run")
@click.option(
    "--alibi-span", default=model.SLOPE_SPAN, help="must match the training run"
)
# Elected by ear 2026-08-17 over a sweep of temperature 0.7 to 1.3 against min_p 0.0039
# to 0.15, and the numbers agree with the ear at both edges, which is rare here. Hotter
# draws more from the tail: at 1.2 the onset rate passes the corpus and the chords go
# strange. A higher floor smooths the arrivals and costs the music: min_p 0.15 leaves
# about one and a half classes standing at a draw, and it reads as dull and MORE silent --
# silence 5.83 percent against 4.22, gaps 13.4 steps against 9.8, where the corpus gives
# 4.19 and 9.9.
# `quantized.ELECTED_*` is the one home of these two numbers since the all-era cut took
# the OCaml `Policy` away, thus the audition and the contract file temper alike.
@click.option("--temperature", default=quantized.ELECTED_TEMPERATURE)
@click.option("--min-p", default=quantized.ELECTED_MIN_P)
@click.option("--play", "to_synth", is_flag=True, help=f"send to the synth on {midi.DEVICE}")
@click.option("--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid")
@click.option("--device", default=midi.DEVICE)
@click.option("--step-ms", default=200)
@click.option("--channel", default=2, help="the S-1 factory default, MIDI channel 3")
@click.option("--velocity", default=100)
def sample(
    ckpt,
    seeds,
    steps,
    context,
    heads,
    alibi_span,
    temperature,
    min_p,
    to_synth,
    to_file,
    device,
    step_ms,
    channel,
    velocity,
):
    params = model.load_params(ckpt)
    walks = draw(
        params,
        seeds=seeds,
        steps=steps,
        context=context,
        heads=heads,
        span=alibi_span,
        temperature=temperature,
        min_p=min_p,
    )
    music = [data.decode(walk) for walk in walks]

    if to_synth or to_file:
        if len(seeds) > 1:
            raise click.UsageError("--play and --save take one seed")
        if to_file:
            midi.save(music[0], to_file, step_ms=step_ms, channel=channel, velocity=velocity)
            click.echo(f"wrote {to_file}")
        if to_synth:
            midi.play(
                music[0],
                device=device,
                step_ms=step_ms,
                channel=channel,
                velocity=velocity,
            )
        return
    for seed, walk in zip(seeds, music):
        if len(seeds) > 1:
            click.echo(f"# seed {seed}")
        click.echo("\n".join(midi.step_line(step, events) for step, events in enumerate(walk)))


@main.command()
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", required=True, type=click.Path(dir_okay=False))
@click.option("--heads", default=4, help="must match the training run")
@click.option("--context", default=256, help="the attention window of the circuit")
@click.option(
    "--alibi-span", default=model.SLOPE_SPAN, help="must match the training run"
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
    params = model.load_params(ckpt)
    twin = quantized.Quantized.of(
        params,
        heads=heads,
        context=context,
        slope_span=alibi_span,
        temperature=temperature,
        min_p=min_p,
    )
    quantized.save(out, twin)
    click.echo(
        f"wrote {out}: d {twin.d}, {twin.layers} layers, {twin.heads} heads, "
        f"context {twin.context}, span {twin.slope_span}"
    )
    click.echo(
        f"temper {twin.temper.q_value} at Q{twin.temper.q}, "
        f"temperature {twin.temper.temperature}, min weight {twin.min_weight}"
    )


if __name__ == "__main__":
    main()
