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
raw channel voice bytes on the rawmidi device, with no backend library in the way.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import click
import jax
import jax.numpy as jnp
import mido
import numpy as np

import data
import measure
import prng
from mamba import model

NOTE_ON, NOTE_OFF = 0x90, 0x80
RELEASE_VELOCITY = 0x40  # lib/core/midi.ml
DEVICE = "/dev/snd/midiC2D0"
JAX_ROOT = Path(__file__).resolve().parent.parent


def rms_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def temper(raw, temperature, min_p):
    """the tempered weight of each class against the peak, then the min-p floor; the peak
    weighs one, thus min_p is a share of the peak"""
    weights = np.exp((raw - raw.max(axis=1, keepdims=True)) / temperature)
    if min_p > 0.0:
        weights = np.where(weights >= min_p, weights, 0.0)
    return weights


def pick(weights, uniform):
    """The class whose running total passes the draw.

    It takes the uniform and not a draw, thus one function owns both sums and the total is
    the last running total -- never a second sum of the same weights. numpy adds pairwise in
    sum() and left to right in cumsum(), thus two sums of one array differ in the last bits,
    and a draw made against the other sum can land above every running total, where no class
    passes at all.

    Against this total the draw is strictly below it, because the uniform falls under 1 by
    2**-24 at the least. Therefore the walk always ends on a class, and that class always
    holds weight the floor left standing."""
    running = np.cumsum(weights, axis=1)
    return (running > (uniform * running[:, -1])[:, None]).argmax(axis=1)


def draw_frame(params, h, state, temperature, min_p):
    """One step of the chained head, on the host: the soprano first, and each seat under it
    reading the stream the seats above have written.

    The chain is the reason a frame is a joint choice and not four independent ones. Seat 0
    is the bass and seat 3 the soprano, thus the loop runs down."""
    seats = np.asarray(params["seats"])
    stream = h
    frame = np.zeros((len(h), data.SEATS), dtype=np.int32)
    for seat in reversed(range(data.SEATS)):
        raw = (rms_norm(stream) @ seats[seat].T).astype(np.float64)
        weights = temper(raw, temperature, min_p)
        state, uniform = prng.uniform(state, True)
        frame[:, seat] = pick(weights, uniform)
        if seat:
            stream = stream + seats[seat][frame[:, seat]]
    return state, frame


def sample(params, *, seeds, steps, temperature, min_p):
    """One batched run: [len(seeds)] independent walks of [steps] steps each.

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
    carry = model.initial_carry(shape, batch)
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


def step_line(step, events):
    """the line format of bin/play_mamba.ml, so that a dump reads back"""
    return f"step {step:3d}  " + (" ".join(f"{k}:{p}" for k, p in events) or "-")


def report_texture(walks, music, *, seeds, span, corpus_path):
    """both questions of measure.py, and the corpus row above each: does the texture hold
    over the windows, and does the walk arrive before it goes quiet?"""
    canonical, corpus = measure.of_canonical_stream(corpus_path)
    for index, row in enumerate(measure.windows(canonical, len(canonical))):
        click.echo(measure.window_line("the packed corpus", index, row))
    for seed, walk in zip(seeds, music):
        for index, row in enumerate(measure.windows(walk, span)):
            click.echo(measure.window_line(f"seed {seed:4d} window", index, row))
    click.echo("")
    click.echo(measure.walk_line("the packed corpus", corpus))
    rows = [measure.of_walk(walk, decoded) for walk, decoded in zip(walks, music)]
    for seed, row in zip(seeds, rows):
        click.echo(measure.walk_line(f"seed {seed:4d}", row))
    if len(seeds) > 1:
        mean = {name: measure.mean_of(rows, name) for name in rows[0]}
        click.echo(measure.walk_line("the mean", mean))


def play(music, *, device, step_ms, channel, velocity):
    """Send one walk to the synthesizer: raw channel voice bytes on the rawmidi device."""
    ringing = set()
    with open(device, "wb", buffering=0) as wire:
        try:
            for step, events in enumerate(music):
                click.echo(step_line(step, events))
                for kind, pitch in events:
                    if kind == "on":
                        wire.write(bytes([NOTE_ON | channel, pitch, velocity]))
                        ringing.add(pitch)
                    else:
                        wire.write(bytes([NOTE_OFF | channel, pitch, RELEASE_VELOCITY]))
                        ringing.discard(pitch)
                time.sleep(step_ms / 1000.0)
        finally:
            # the drain: each open note closes, as the sequencer does at a stop
            for pitch in ringing:
                wire.write(bytes([NOTE_OFF | channel, pitch, RELEASE_VELOCITY]))


def save(music, path, *, step_ms, channel, velocity):
    """one walk as a standard MIDI file: one step is one tick, and the tempo carries the
    step period"""
    track = mido.MidiTrack()
    midi = mido.MidiFile(ticks_per_beat=4)
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=int(step_ms * 1000 * 4)))
    waited = 0
    for events in music:
        for kind, pitch in events:
            track.append(
                mido.Message(
                    "note_on" if kind == "on" else "note_off",
                    note=pitch,
                    velocity=velocity if kind == "on" else RELEASE_VELOCITY,
                    channel=channel,
                    time=waited,
                )
            )
            waited = 0
        waited += 1
    midi.save(path)


def parse_seeds(ctx, param, value):
    """a list, or LOW-HIGH"""
    del ctx, param
    if "-" in value:
        low, high = value.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(seed) for seed in value.split(",")]


@click.command(help=__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--seeds", default="1", callback=parse_seeds, help="a list, or LOW-HIGH")
@click.option("--steps", default=256, help="steps to draw, the silent lead-in inside")
# The draw of era four, carried over unmeasured: this era re-elects it by ear, and until
# it does the two eras are auditioned on one policy. The OCaml side states these once, as
# Mamba.elected_temperature and Mamba.elected_min_p; no constant crosses the language
# seam, thus they stand here again and the two must move together.
@click.option("--temperature", default=1.0)
@click.option("--min-p", default=0.05)
@click.option("--play", "to_synth", is_flag=True, help=f"send to the synth on {DEVICE}")
@click.option("--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid")
@click.option("--device", default=DEVICE)
@click.option("--step-ms", default=200)
@click.option("--channel", default=2, help="the S-1 factory default, MIDI channel 3")
@click.option("--velocity", default=100)
@click.option(
    "--texture",
    "texture_span",
    type=int,
    help="report the windowed texture at this span instead of the step lines",
)
@click.option(
    "--corpus",
    "corpus_path",
    default=str(JAX_ROOT / "_data" / "frames.safetensors"),
    help="the packed corpus that --texture measures against",
)
def main(
    ckpt,
    seeds,
    steps,
    temperature,
    min_p,
    to_synth,
    to_file,
    device,
    step_ms,
    channel,
    velocity,
    texture_span,
    corpus_path,
):
    params = model.load_params(ckpt)
    walks = sample(
        params, seeds=seeds, steps=steps, temperature=temperature, min_p=min_p
    )
    music = [data.decode(walk) for walk in walks]

    if texture_span:
        report_texture(
            walks, music, seeds=seeds, span=texture_span, corpus_path=corpus_path
        )
        return
    if to_synth or to_file:
        if len(seeds) > 1:
            raise click.UsageError("--play and --save take one seed")
        if to_file:
            save(music[0], to_file, step_ms=step_ms, channel=channel, velocity=velocity)
            click.echo(f"wrote {to_file}")
        if to_synth:
            play(
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
        click.echo("\n".join(step_line(step, events) for step, events in enumerate(walk)))


if __name__ == "__main__":
    main()
