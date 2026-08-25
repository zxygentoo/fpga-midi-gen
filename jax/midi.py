"""The wire side of an audition: one drawn walk to the synthesizer or to a MIDI file.

Both eras' audition tools speak through this module. The music is a list of steps, each a
list of (kind, pitch) events as data.decode gives them.
"""

import time

import click
import mido

NOTE_ON, NOTE_OFF = 0x90, 0x80
RELEASE_VELOCITY = 0x40  # lib/core/midi.ml
# The share of its velocity a note keeps at the last step of a fade. It is not zero, and
# not because a fade should end loud: A NOTE-ON OF VELOCITY ZERO IS A NOTE-OFF on the wire,
# and a note the fade silenced would never be released. A quarter is about twelve decibels
# under the full stroke, which the ear reads as an ending and the synth still sounds.
FADE_FLOOR = 0.25
DEVICE = "/dev/snd/midiC2D0"


def step_line(step, events):
    """the line format of the OCaml players (bin/play_*.ml), so that a dump reads back"""
    return f"step {step:3d}  " + (" ".join(f"{k}:{p}" for k, p in events) or "-")


def fading(step, steps, fade):
    """The share of its velocity a note-on keeps at [step] of a piece of [steps], under a
    fade of the last [fade] steps. Outside the fade it is one.

    A FADE ON THIS WIRE REACHES ONLY THE NOTES THAT BEGIN INSIDE IT. Velocity is a fact of
    the onset, and the S-1 states that a control change is audible only on the next note,
    thus neither velocity nor CC 7 can quiet a chord that already rings. Measured over the
    corpus crops on 2026-08-25, a crop's last note has been sounding 4.5 steps in the mean:
    a fade of 4 steps therefore catches two thirds of the final notes and finds NO onset at
    all in 18 percent of crops, where a fade of 16 -- one bar -- catches 99 percent and is
    never empty. That is the reason for the default and not a taste.

    It is the same gesture as [rest] and it has the same limit: it says an ending is
    happening, and it cannot make a phrase that never closed sound closed."""
    # a canvas shorter than the window fades across the whole of itself; without the
    # clamp it would OPEN partway down the ramp and never sound its full stroke
    fade = min(fade, steps)
    left = steps - step
    if fade <= 0 or left > fade:
        return 1.0
    return FADE_FLOOR + (1.0 - FADE_FLOOR) * (left - 1) / max(fade - 1, 1)


def play(music, *, device, step_ms, channel, velocity, fade=0):
    """Send one walk to the synthesizer: raw channel voice bytes on the rawmidi device."""
    ringing = set()
    with open(device, "wb", buffering=0) as wire:
        try:
            for step, events in enumerate(music):
                click.echo(step_line(step, events))
                struck = max(1, round(velocity * fading(step, len(music), fade)))
                for kind, pitch in events:
                    if kind == "on":
                        wire.write(bytes([NOTE_ON | channel, pitch, struck]))
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

    THE DEFAULT IS TWO BARS, and the ear set it on 2026-08-25 in two readings: a quarter
    note helps and a bar was what it wanted, and then [fading] arrived and a bar was short
    again. That follows -- a canvas now ENDS quiet, thus the silence after it has less to
    part from and must run longer to read as a break at all. IT DOES NOT FIX THE ENDING. A canvas is a crop of eight measures
    and it stops where the corpus was cut -- it does not arrive -- and no silence after a
    phrase that never closed will make it sound closed. That is the deferred round, the
    whole piece with its cadence, and not this wait."""
    time.sleep(steps * step_ms / 1000.0)


def save(music, path, *, step_ms, channel, velocity, fade=0):
    """one walk as a standard MIDI file: one step is one tick, and the tempo carries the
    step period"""
    track = mido.MidiTrack()
    midi = mido.MidiFile(ticks_per_beat=4)
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=int(step_ms * 1000 * 4)))
    waited = 0
    for step, events in enumerate(music):
        struck = max(1, round(velocity * fading(step, len(music), fade)))
        for kind, pitch in events:
            track.append(
                mido.Message(
                    "note_on" if kind == "on" else "note_off",
                    note=pitch,
                    velocity=struck if kind == "on" else RELEASE_VELOCITY,
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
