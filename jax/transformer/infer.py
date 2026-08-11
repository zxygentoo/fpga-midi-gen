"""The batched twin of the OCaml sampler in lib/transformer/transformer.ml.

The OCaml sampler is the reference: it is what the board must agree with. This file
exists for the sweeps -- a diagnostic wants twenty-four seeds, the seeds are independent,
and they belong in one batch. Measured at d 64, context 256, 192 steps: Nx 15.8 s for one
seed, this file 2.9 s for one and 10.8 s for twenty-four.

It is a TWIN, not a second semantics. The draw is [prng], the circuit's xorshift32, and
the grammar is [data.Sounding], so seed 7 here is seed 7 there token for token -- which
is what lets a sweep nominate a seed and an audition then play it. `--gate` proves that
against the OCaml sampler; run it after any change to either side.

CPU only, and deliberately: every token needs the drawn code back on the host to walk the
sounding state before the next forward, so the loop is latency-bound and a GPU would buy
little while taking the device from the trainer.
"""

import os
import subprocess
import sys
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


def pick(weights, draw):
    """The choice of the OCaml sampler: the first code whose running total passes [draw],
    code 255 when no total does, and code 0 when the chosen weight is not positive."""
    running = np.cumsum(weights, axis=1)
    passed = running[:, : data.VOCAB - 1] > draw[:, None]
    chosen = np.where(passed.any(axis=1), passed.argmax(axis=1), data.VOCAB - 1)
    rows = np.arange(len(chosen))
    return np.where(weights[rows, chosen] > 0.0, chosen, 0)


def temper(raw, legal, temperature, min_p):
    """the tempered weight of each legal code against the legal peak, then the min-p
    floor; the peak weighs one, thus min_p is a share of the peak"""
    peak = np.where(legal, raw, -np.inf).max(axis=1, keepdims=True)
    weights = np.where(legal, np.exp((raw - peak) / temperature), 0.0)
    if min_p > 0.0:
        weights = np.where(weights >= min_p, weights, 0.0)
    return weights


def sample(params, *, seeds, steps, context, heads, span, temperature, min_p):
    """One batched run: [len(seeds)] independent walks, each drawing [steps] steps.

    Every element draws one token an iteration, so the histories stay the same length and
    one array holds the window. A finished element appends END and consumes no draw, thus
    its music equals the music of a run of its own."""
    batch = len(seeds)
    forward = jax.jit(
        lambda codes, phases, buckets: model.logits(
            params, codes, phases, buckets, heads=heads, span=span
        )
    )
    state = prng.states(seeds)
    codes = np.full((batch, 1), data.START, dtype=np.int32)
    phases = np.zeros((batch, 1), dtype=np.int32)
    buckets = np.zeros((batch, 1), dtype=np.int32)
    sounding = data.Sounding(batch)
    step_index = np.zeros(batch, dtype=np.int64)
    music = [[[] for _ in range(steps)] for _ in range(batch)]
    sentence = [[] for _ in range(batch)]

    while True:
        active = step_index < steps
        if not active.any():
            break
        # ONE shape for the whole run, or every window length compiles its own kernel --
        # the history is right-padded to [batch, context] and read at its last real
        # position. The causal wall keeps a real token from seeing the padding, thus the
        # row is the row the OCaml sampler computes from the bare window.
        held = codes.shape[1]
        low = max(0, held - context)
        length = held - low
        window = np.zeros((3, batch, context), dtype=np.int32)
        window[0, :, :length] = codes[:, low:held]
        window[1, :, :length] = phases[:, low:held]
        window[2, :, :length] = buckets[:, low:held]
        raw = np.asarray(
            forward(
                jnp.asarray(window[0]), jnp.asarray(window[1]), jnp.asarray(window[2])
            )
        )[:, length - 1, :].astype(np.float64)

        weights = temper(raw, sounding.legal(), temperature, min_p)
        state, draw = prng.uniform(state, active)
        code = np.where(active, pick(weights, draw * weights.sum(axis=1)), data.END)

        for row in np.nonzero(active)[0]:
            drawn = int(code[row])
            if drawn == data.END:
                music[row][step_index[row]] = sentence[row]
                sentence[row] = []
            else:
                kind = "on" if drawn >= 128 else "off"
                sentence[row].append((kind, drawn - 128 if kind == "on" else drawn))

        sounding.step(code, active)
        # the token carries the position of the step it belongs to, thus the END of a
        # step takes that step's phase and the count rises after it
        codes = np.concatenate([codes, code[:, None].astype(np.int32)], axis=1)
        phases = np.concatenate(
            [phases, (step_index % model.PHASE_BUCKETS)[:, None].astype(np.int32)], axis=1
        )
        bucket = step_index // model.PROGRESS_STRIDE % model.PROGRESS_BUCKETS
        buckets = np.concatenate([buckets, bucket[:, None].astype(np.int32)], axis=1)
        step_index = np.where(active & (code == data.END), step_index + 1, step_index)
    return music


def step_line(step, events):
    """the line format of bin/play_transformer.ml, so that a diff is the gate"""
    return f"step {step:3d}  " + (" ".join(f"{k}:{p}" for k, p in events) or "-")


def lines(music):
    return [step_line(step, events) for step, events in enumerate(music)]


def play(music, *, device, step_ms, channel, velocity):
    """Send one walk to the synthesizer, as bin/play_transformer.ml does: raw channel
    voice bytes on the rawmidi device. No backend library sits in the way, thus the wire
    holds exactly what the OCaml player would put there."""
    ringing = set()
    with open(device, "wb", buffering=0) as wire:
        try:
            for step, events in enumerate(music):
                # the player prints as it sends, as bin/play_transformer.ml does, so that
                # the ear and the eye follow the same step
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
    """Write one walk as a standard MIDI file. The step is the grid of the model, thus
    one step is one tick here and the tempo carries [step_ms]."""
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


def gate(checkpoint, music, seeds, options):
    """Gate C: the batched draw must equal the OCaml sampler token for token.

    Batched, because that is the mode the sweeps use and the mode that can go wrong -- a
    finished element must consume no draw, or the walks behind it shift."""
    root = Path(__file__).resolve().parents[2]
    player = root / "_build" / "default" / "bin" / "play_transformer.exe"
    if not player.exists():
        click.echo(f"no {player}: run dune build")
        return 1
    bad = 0
    for seed, drawn in zip(seeds, music):
        argv = [str(player), "-ckpt", str(Path(checkpoint).resolve()), "-seed", str(seed)]
        for flag, value in options.items():
            argv += [flag, str(value)]
        out = subprocess.run(
            argv, capture_output=True, text=True, cwd=root, check=False
        ).stdout
        theirs = [line for line in out.splitlines() if line.startswith("step")]
        ours = lines(drawn)
        if theirs == ours:
            click.echo(f"seed {seed:4d}: {len(ours)} steps identical")
            continue
        bad += 1
        first = next((i for i, (a, b) in enumerate(zip(theirs, ours)) if a != b), None)
        click.echo(f"seed {seed:4d}: DIFFERS first at step {first}")
        if first is not None:
            click.echo(f"   ocaml {theirs[first]}\n   jax   {ours[first]}")
    click.echo("GATE C PASSED" if not bad else f"GATE C FAILED on {bad} of {len(seeds)}")
    return bad


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
@click.option(
    "--steps",
    default=192,
    help="steps to draw; with a progress model this is "
    "also the length of the piece the host asks for",
)
@click.option("--context", default=256, help="must match the training run")
@click.option("--heads", default=4, help="must match the training run")
@click.option(
    "--alibi-span", default=model.SLOPE_SPAN, help="must match the training run"
)
@click.option("--temperature", default=0.9)
@click.option("--min-p", default=1.0 / 256.0)
@click.option("--play", "to_synth", is_flag=True, help=f"send to the synth on {DEVICE}")
@click.option("--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid")
@click.option("--device", default=DEVICE)
@click.option("--step-ms", default=200)
@click.option("--channel", default=2, help="the S-1 factory default, MIDI channel 3")
@click.option("--velocity", default=100)
@click.option("--gate", "run_gate", is_flag=True, help="check against the OCaml sampler")
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
    run_gate,
):
    params = model.load_params(ckpt)
    music = sample(
        params,
        seeds=seeds,
        steps=steps,
        context=context,
        heads=heads,
        span=alibi_span,
        temperature=temperature,
        min_p=min_p,
    )
    if run_gate:
        options = {
            "-steps": steps,
            "-heads": heads,
            "-context": context,
            "-alibi-span": alibi_span,
            "-temperature": temperature,
            "-min-p": min_p,
        }
        sys.exit(1 if gate(ckpt, music, seeds, options) else 0)
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
    for seed, drawn in zip(seeds, music):
        if len(seeds) > 1:
            click.echo(f"# seed {seed}")
        click.echo("\n".join(lines(drawn)))


if __name__ == "__main__":
    main()
