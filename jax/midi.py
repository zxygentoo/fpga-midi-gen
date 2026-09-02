"""The wire side of an audition: one drawn walk to the synthesizer or to a MIDI file.

Both eras' audition tools speak through this module. The music is a list of steps, each a
list of (kind, pitch) events as corpus.decode gives them.
"""

import time

import click
import mido

import cli

NOTE_ON, NOTE_OFF = 0x90, 0x80
RELEASE_VELOCITY = 0x40  # lib/core/midi.ml
DEVICE = "/dev/snd/midiC2D0"


def step_line(step, events):
    """the line format of the OCaml players (bin/play_*.ml), so that a dump reads back"""
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
            # The drain: each open note closes, as the sequencer does at a stop. SORTED
            # is not a taste -- the capture gate compares these bytes to the board's, and
            # an unordered set would put a different tail on the wire at every run.
            for pitch in sorted(ringing):
                wire.write(bytes([NOTE_OFF | channel, pitch, RELEASE_VELOCITY]))


def rest(steps, *, step_ms):
    """The silence between two pieces of an audition, in steps of the grid. [play] drains
    its notes as it leaves, thus this is a wait and not a message.

    The ear set the default at two bars and the board's scheduler carries the same 32,
    thus the reference and the hardware part their sheets one way. It does not fix the
    ending -- a crop stops where the corpus was cut and no silence makes a phrase that
    never closed sound closed."""
    time.sleep(steps * step_ms / 1000.0)


def save(music, path, *, step_ms, channel, velocity):
    """one walk as a standard MIDI file: one step is one tick, and the tempo carries the
    step period"""
    track = mido.MidiTrack()
    midi = mido.MidiFile(ticks_per_beat=4)
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=int(step_ms * 1000 * 4)))
    ringing = set()
    waited = 0
    for step, events in enumerate(music):
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
            if kind == "on":
                ringing.add(pitch)
            else:
                ringing.discard(pitch)
            waited = 0
        waited += 1
    # the drain, as [play] does it: `corpus.decode` writes an "off" only where the NEXT
    # frame drops the pitch, thus with no drain the file ends still sounding
    for pitch in sorted(ringing):
        track.append(
            mido.Message(
                "note_off",
                note=pitch,
                velocity=RELEASE_VELOCITY,
                channel=channel,
                time=waited,
            )
        )
        waited = 0
    midi.save(path)


# The six flags every `sample` command takes: where the music goes and how it is struck.
# One set because the wire is one wire. The shaping of a piece is NOT here -- --gap is
# era six's, which parts a batch into pieces.
playback_options = cli.add_options(
    [
        click.option(
            "--play", "to_synth", is_flag=True, help=f"send to the synth on {DEVICE}"
        ),
        click.option(
            "--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid"
        ),
        click.option("--device", default=DEVICE),
        click.option("--step-ms", default=200),
        click.option(
            "--channel", default=2, help="the S-1 factory default, MIDI channel 3"
        ),
        click.option("--velocity", default=100),
    ]
)


def audition(music, seeds, *, to_synth, to_file, device, step_ms, channel, velocity):
    """The tail of a `sample` command: play one walk, write one walk, or print them all.

    A walk of the autoregressive eras is one endless piece, thus --play and --save take
    one seed. Era six parts its batch into pieces, which is why its tail is its own."""
    if to_synth or to_file:
        if len(seeds) > 1:
            raise click.UsageError("--play and --save take one seed")
        if to_file:
            save(
                music[0], to_file, step_ms=step_ms, channel=channel, velocity=velocity
            )
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
        click.echo(
            "\n".join(step_line(step, events) for step, events in enumerate(walk))
        )
