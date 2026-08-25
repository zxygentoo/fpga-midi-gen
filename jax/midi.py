"""The wire side of an audition: one drawn walk to the synthesizer or to a MIDI file.

Both eras' audition tools speak through this module. The music is a list of steps, each a
list of (kind, pitch) events as data.decode gives them.
"""

import time

import click
import mido

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
            # the drain: each open note closes, as the sequencer does at a stop
            for pitch in ringing:
                wire.write(bytes([NOTE_OFF | channel, pitch, RELEASE_VELOCITY]))


def rest(steps, *, step_ms):
    """The silence between two pieces of an audition, in steps of the grid.

    [play] drains its notes as it leaves, thus this is a wait and not a message. The ear
    asks for it: a batch is several INDEPENDENT draws and each one is a whole piece, so
    back to back the second opens on the first one's last chord with no breath between
    them, which no performance does.

    THE DEFAULT IS ONE BAR, and the ear set it on 2026-08-25: a quarter note helps and a
    bar is what it wanted. IT DOES NOT FIX THE ENDING. A canvas is a crop of eight measures
    and it stops where the corpus was cut -- it does not arrive -- and no silence after a
    phrase that never closed will make it sound closed. That is the deferred round, the
    whole piece with its cadence, and not this wait."""
    time.sleep(steps * step_ms / 1000.0)


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
