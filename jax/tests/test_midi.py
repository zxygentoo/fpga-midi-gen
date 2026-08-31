"""The wire side of an audition: the FADE, and what it must never do.

It changes no note. A sheet is a crop and it stops where the corpus was cut; the fade says
that an ending is happening, and every test here reads `midi.fading`. What it must never
do is reach zero -- a note-on of velocity zero is a NOTE-OFF on the wire, and the note it
silenced would ring for ever.

The other gesture of the pair, `midi.rest`, is a `time.sleep` and has nothing to pin."""

import itertools

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
    velocities = [
        max(1, round(100 * midi.fading(step, 128, 16))) for step in range(112, 128)
    ]
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
