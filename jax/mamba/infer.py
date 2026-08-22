"""The sampler and the player of the state-space model.

One step is one step of the recurrence, always: the state and the convolution taps carry
forward, the four seats are drawn in a chain from the soprano down, on the host, and the
frame the chain drew goes back in. No mask guards the draw, because no frame is illegal.

There is no window here and no context flag. The model has no context length at inference:
what it remembers, it remembers in a state of fixed size.

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
from mamba import model


def sample(params, *, seeds, steps, temperature, min_p, ring=model.ATTN_CONTEXT):
    """One batched run: [len(seeds)] independent walks of [steps] steps each.

    [ring] is the depth of the attention layer's keys and values, and it exists only where
    the plan holds an attention layer -- a trunk of blocks carries a state of fixed size and
    has no window at all. Training attends over the WHOLE window, thus a ring shorter than
    the training window is a truncation, and whether the truncation costs anything depends
    on the ALiBi span: at span 4 the slowest head weighs e^-8 at distance 128 and a ring of
    256 reads the same as one of 512, measured. At a longer span it would not.

    The boot is a lead-in of silence: one bar of silent frames, then the draw. The state
    opens at zero, which is where a training window opens, thus the model meets the
    condition it trained on. The lead-in counts inside [steps] and stands at the head of
    the music, because it is silence the walk really plays."""
    batch = len(seeds)
    shape = model.shape_of(params)
    forward = jax.jit(
        lambda carry, classes, phases: model.forward_step(params, carry, classes, phases)
    )
    rng = prng.states(seeds)
    carry = model.initial_carry(shape, batch, context=ring)
    lead = data.BAR_STEPS
    silence = np.zeros((batch, data.SEATS), dtype=np.int32)
    frames = []
    h = None
    for step in range(steps):
        # through the lead-in nothing is drawn and the generator does not move, exactly as
        # the integer twin and the circuit leave it standing
        if step < lead:
            frame = silence
        else:
            rng, frame = draw_frame(params, h, rng, temperature, min_p)
        frames.append(frame)
        phases = np.full(batch, step % model.PHASE_BUCKETS, dtype=np.int32)
        carry, stream = forward(carry, jnp.asarray(frame), jnp.asarray(phases))
        h = np.asarray(stream).astype(np.float64)
    return np.stack(frames, axis=1)


@click.command(help=__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--seeds", default="1", callback=midi.parse_seeds, help="a list, or LOW-HIGH")
@click.option("--steps", default=256, help="steps to draw, the silent lead-in inside")
# The draw of era four, carried over unmeasured: this era re-elects it by ear, and until
# it does the two eras are auditioned on one policy. The OCaml side states these once, as
# Mamba.elected_temperature and Mamba.elected_min_p; no constant crosses the language
# seam, thus they stand here again and the two must move together.
@click.option("--temperature", default=1.0)
@click.option("--min-p", default=0.05)
@click.option(
    "--ring",
    default=model.ATTN_CONTEXT,
    help="the depth of the attention layer's key and value ring, in steps. It is the "
    "one context this model has, and only where the plan attends at all.",
)
@click.option("--play", "to_synth", is_flag=True, help=f"send to the synth on {midi.DEVICE}")
@click.option("--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid")
@click.option("--device", default=midi.DEVICE)
@click.option("--step-ms", default=200)
@click.option("--channel", default=2, help="the S-1 factory default, MIDI channel 3")
@click.option("--velocity", default=100)
def main(
    ckpt,
    seeds,
    steps,
    temperature,
    min_p,
    ring,
    to_synth,
    to_file,
    device,
    step_ms,
    channel,
    velocity,
):
    params = model.load_params(ckpt)
    walks = sample(
        params, seeds=seeds, steps=steps, temperature=temperature, min_p=min_p,
        ring=ring,
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


if __name__ == "__main__":
    main()
