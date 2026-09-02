"""The wire side of an audition: the DRAIN, and nothing else left to pin.

`corpus.decode` writes an "off" only where the NEXT frame drops the pitch, thus a walk
that ends still sounding ends with no release at all. `midi.play` drains on the way out
and `midi.save` writes the tail; the file is what a test can read back. SORTED is not a
taste -- the capture gate compares the player's bytes to the board's, and an unordered
set would put a different tail on the wire at every run.

Velocity is a fact of the onset and the sequencer's alone, thus a note-on carries
`velocity` as given and this module has no ramp to pin: docs/diffusion_rtl.md, "The fade
-- cut", states why."""

import mido

import midi


def notes_of(path):
    """the note messages of a saved walk, in the order the file states them"""
    track = mido.MidiFile(str(path)).tracks[0]
    return [message for message in track if message.type in ("note_on", "note_off")]


def test_the_saved_file_releases_every_note_that_still_rings(tmp_path):
    """A WALK ENDS WHEREVER IT WAS CUT and its last chord is still sounding, thus the file
    owes a release for each pitch of it -- in ascending order, as the sequencer's own
    silent frame closes them."""
    path = tmp_path / "walk.mid"
    music = [[("on", 60)], [("on", 64)]]
    midi.save(music, str(path), step_ms=200, channel=2, velocity=100)
    notes = notes_of(path)
    assert [message.type for message in notes] == [
        "note_on",
        "note_on",
        "note_off",
        "note_off",
    ]
    assert [message.note for message in notes] == [60, 64, 60, 64]
    assert all(
        message.velocity == 100 for message in notes if message.type == "note_on"
    ), "a note-on carries the velocity as given"


def test_the_drain_releases_no_note_the_music_released_itself(tmp_path):
    """The tail is what STILL RINGS and not the whole chord: a pitch the music let go of
    is already released, and a second release would land on whatever note the synth had
    struck at that pitch since."""
    path = tmp_path / "walk.mid"
    music = [[("on", 60)], [("off", 60), ("on", 64)]]
    midi.save(music, str(path), step_ms=200, channel=2, velocity=100)
    notes = notes_of(path)
    assert [(message.type, message.note) for message in notes] == [
        ("note_on", 60),
        ("note_off", 60),
        ("note_on", 64),
        ("note_off", 64),
    ]
