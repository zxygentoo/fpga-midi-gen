"""The wire side of an audition: the FADE, the velocity it strikes, and the DRAIN.

The fade changes no note. A sheet is a crop and it stops where the corpus was cut; the
fade says that an ending is happening. What it must never do is reach zero -- a note-on of
velocity zero is a NOTE-OFF on the wire, and the note it silenced would ring for ever.
`midi.struck_velocity` owns that floor and every test that wants a struck velocity reads
it, because the rule restated is the rule that drifts.

THE DRAIN IS THE SAME FAULT FROM THE OTHER END: `corpus.decode` writes an "off" only where
the NEXT frame drops the pitch, thus a walk that ends still sounding ends with no release
at all. `midi.play` drains on the way out and `midi.save` writes the tail; the file is
what a test can read back.

The other gesture of the pair, `midi.rest`, is a `time.sleep` and has nothing to pin."""

import itertools

import mido
import pytest

import midi


def test_the_fade_leaves_the_music_before_it_alone():
    """the gesture is the last bar of a sheet and nothing else: a step outside the window
    keeps the whole stroke"""
    assert midi.fading(0, 128, 16) == pytest.approx(1.0)
    assert midi.fading(111, 128, 16) == pytest.approx(1.0)
    # the window opens at the step whose distance from the end is the fade itself
    assert midi.fading(112, 128, 16) == pytest.approx(1.0)
    assert midi.fading(113, 128, 16) < 1.0


def test_the_fade_falls_to_the_floor_and_never_below_it():
    """A NOTE-ON OF VELOCITY ZERO IS A NOTE-OFF. A fade that reached zero would state a
    release the player never made, and the note would keep sounding until the drain."""
    assert midi.fading(127, 128, 16) == pytest.approx(midi.FADE_FLOOR)
    assert midi.FADE_FLOOR > 0.0
    velocities = [midi.struck_velocity(100, step, 128, 16) for step in range(112, 128)]
    assert min(velocities) >= 1
    assert velocities[0] == 100 and velocities[-1] == 25


def test_the_fade_only_falls():
    """a diminuendo that rose anywhere would read as a phrase and not as an ending"""
    scale = [midi.fading(step, 128, 16) for step in range(128)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(scale))


def test_no_fade_is_the_full_stroke_everywhere():
    """--fade 0 must be the player as it stood, and not a fade of one step"""
    assert all(midi.fading(step, 128, 0) == 1.0 for step in range(128))


def test_a_short_sheet_still_fades():
    """the window is the last [fade] steps or the whole sheet, whichever is shorter; a
    walk of four steps must not divide by a fade of sixteen"""
    scale = [midi.fading(step, 4, 16) for step in range(4)]
    assert scale[0] == pytest.approx(1.0)
    assert scale[-1] == pytest.approx(midi.FADE_FLOOR)
    assert all(later <= earlier for earlier, later in itertools.pairwise(scale))


def test_the_struck_velocity_holds_a_floor_of_one():
    """THE ROUNDING IS WHERE A FADE REACHES ZERO EVEN THOUGH `fading` DOES NOT. A quiet
    walk struck at 1 or 2 scales to a quarter and a half at the foot of the ramp, and
    Python rounds both of those to nothing; the floor of one is what stands between that
    and a release the player never made."""
    assert round(1 * midi.FADE_FLOOR) == 0 and round(2 * midi.FADE_FLOOR) == 0
    assert midi.struck_velocity(1, 127, 128, 16) == 1
    assert midi.struck_velocity(2, 127, 128, 16) == 1
    # outside the window the stroke passes whole, and at the foot it is the floor's share
    assert midi.struck_velocity(100, 0, 128, 16) == 100
    assert midi.struck_velocity(100, 127, 128, 16) == 25


def notes_of(path):
    """the note messages of a saved walk, in the order the file states them"""
    track = mido.MidiFile(str(path)).tracks[0]
    return [message for message in track if message.type in ("note_on", "note_off")]


def test_the_saved_file_releases_every_note_that_still_rings(tmp_path):
    """A WALK ENDS WHEREVER IT WAS CUT and its last chord is still sounding, thus the file
    owes a release for each pitch of it. The stroke is 1 under a fade that reaches the
    floor, which is where the rounding would send a note-on to zero -- and a note-on of
    zero is a note-off that would release the note here, leaving the drain to release a
    note that no longer sounds."""
    path = tmp_path / "walk.mid"
    music = [[("on", 60)], [("on", 64)]]
    midi.save(music, str(path), step_ms=200, channel=2, velocity=1, fade=2)
    notes = notes_of(path)
    assert [message.type for message in notes] == [
        "note_on",
        "note_on",
        "note_off",
        "note_off",
    ]
    assert [message.note for message in notes] == [60, 64, 60, 64]
    assert all(
        message.velocity > 0 for message in notes if message.type == "note_on"
    ), "a note-on of velocity zero is a note-off on the wire"


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
