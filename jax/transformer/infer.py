"""The sampler and the player of the step-frame model.

One step is one forward pass, always: the four seats are drawn in a chain from the soprano
down, on the host, between two passes of the network. No mask guards the draw, because no
frame is illegal.

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
import prng
from transformer import model

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
    holds weight the floor left standing: to reach the last index is to know that no earlier
    total passed, thus the weight there is the difference of two totals across the draw. No
    fallback is necessary, and none is written."""
    running = np.cumsum(weights, axis=1)
    return (running > (uniform * running[:, -1])[:, None]).argmax(axis=1)


def draw_frame(params, h, state, temperature, min_p):
    """One step of the chained head, on the host: the soprano first, and each seat under it
    reading the stream the seats above have written.

    The chain is the reason a frame is a joint choice and not four independent ones. Seat 0
    is the bass and seat 3 the soprano, thus the loop runs down.

    Every walk of the batch draws. A step is one frame and never a sentence of its own
    length, thus no walk of the batch finishes before another and none has to sit out a
    draw while the rest go on."""
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


def sample(params, *, seeds, steps, context, heads, span, temperature, min_p):
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


def step_line(step, events):
    """the line format of bin/play_transformer.ml, so that a dump reads back"""
    return f"step {step:3d}  " + (" ".join(f"{k}:{p}" for k, p in events) or "-")



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
# The OCaml side states these once, as Transformer.elected_temperature and
# Transformer.elected_min_p; no constant crosses the language seam, thus they stand here
# again and the two must move together.
@click.option("--temperature", default=1.0)
@click.option("--min-p", default=0.05)
@click.option("--play", "to_synth", is_flag=True, help=f"send to the synth on {DEVICE}")
@click.option("--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid")
@click.option("--device", default=DEVICE)
@click.option("--step-ms", default=200)
@click.option("--channel", default=2, help="the S-1 factory default, MIDI channel 3")
@click.option("--velocity", default=100)
def main(
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
    walks = sample(
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
