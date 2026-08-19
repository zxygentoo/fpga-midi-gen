# The pink model

## Scope

This document states the model of era one: what it makes, and why it makes
music and not noise. `docs/pink_rtl.md` states the circuit and
`lib/pink/pink.mli` is the reference implementation. The last section names
the reading for the theory, because this document does not repeat it.

## 1/f noise, and why it plays

White noise has no memory: each value forgets the one before it, thus a melody
of white noise jumps and states nothing. A random walk has too much memory: it
wanders away and never comes back. Pink noise is between them — its power
falls as 1/f, thus it fluctuates at every time scale at one time, and it has
correlation without a direction.

Voss and Clarke measured that fall in music and in speech, and then made music
from 1/f noise to hear the difference. The claim this project takes from that
work is small and it is enough: a line drawn from 1/f noise moves at every
scale — from one note to the next, over a phrase, and over a piece — and no
rule in the program states a phrase.

## The algorithm

The Voss-McCartney algorithm makes 1/f noise from rows of random values on a
binary schedule.

- The model holds N rows of one byte. The default is 8.
- At step `i`, the rows 0 to `ctz i` re-roll, in ascending order, with one PRNG
  draw for each row. Therefore row `r` re-rolls every `2^r` steps.
- The value is the sum of the rows.

Each row fluctuates at its own rate, and the rates are one octave apart. The
sum of them approximates the 1/f fall — that is the whole trick, and the
reading below states why it works.

The fast rows give the local movement and the slow rows give the long shape.
The arithmetic is integer, thus the reference, the simulation and the board
give the same sequence for the same seed.

## The register decomposition

The rows are not summed into one line. **They are partitioned into groups, one
group for each voice**, and each voice sums its own rows.

This is the idea of the model, and it has one consequence that pays for
everything: **a voice re-articulates exactly when one of its rows re-rolls.**
A group that starts at row `r` therefore moves every `2^r` steps, thus the
partition of the rows is the rhythm. The note-rate hierarchy is the 1/f
structure made audible, and the program holds no rhythm generator, no bar and
no phrase rule.

The default partition is 2+2+2+2 over 8 rows:

| Voice | rows | register | it moves every |
|---|---|---|---|
| soprano | 0, 1 | A4 to A6 | step |
| alto | 2, 3 | C4 to G4 | 4 steps |
| tenor | 4, 5 | C3 to A3 | 16 steps |
| bass | 6, 7 | A1 to A2 | 64 steps |

The voices share one row set, thus the decomposition does not change the walk:
the four voices are four views of one 1/f line and not four independent lines.

## From a sum to a note

Each voice maps its own sum onto its own register with three constants: the
`root`, the number of `degrees`, and the `stretch`.

- **The scale is one scale**, rotated to the root of each voice, thus every
  voice holds the pitch classes of that scale. The default is C major
  pentatonic, and a root outside the scale is an error.
- **`stretch` answers the middle.** The sum of `n` uniform bytes crowds the
  centre of its range, thus a map of the full range would reach the outer
  degrees rarely and the voice would sit in the middle of its register.
  `stretch` maps the centred `1/n` of the range onto the degrees and clips
  outside it.
- **The registers are disjoint, and this is not a musical choice.** The four
  voices play on one MIDI channel, where a Note Off releases a voice by pitch;
  two voices that hold one pitch would silence each other.

## What the frame costs

One step of music is one frame, and a frame states which pitch each voice
holds — not that the voice struck it. Therefore a voice that re-articulates
onto the pitch it already holds states the same code again, and the note
sustains to its next move.

Era one re-struck such a pitch. The frame socket cannot state it, and the ear
accepted the smoothing: it costs the soprano about 16 percent of its
articulations and the alto about 34 percent. The frame codes `0x01` to `0x7F`
are free for a design that wants the re-strike back.

## The seed

The sequence is a pure function of the seed, thus the same seed replays the
same piece in the reference, in the simulation and on the board.

A seed of 0 is the corner. The PRNG takes the seed with no folding, thus
`xorshift32(0)` stays 0, every row stays 0, each voice clamps to the bottom of
its window and lands on its root: the piece is one chord and it holds.
`docs/seed_switches_rtl.md` states why the board accepts that — the failure is
loud, and all the switches down is the rest position of the panel. The
reference plays it too, thus the corner is the same in both.

## Further reading

- R. F. Voss and J. Clarke, *1/f noise in music: Music from 1/f noise*,
  Journal of the Acoustical Society of America, 1978. The measurement, and the
  first pieces made this way.
- Martin Gardner, *White and brown music, fractal curves and one-over-f
  fluctuations*, Scientific American, 1978. The same result for a reader who
  wants no mathematics.
- Wikipedia, *Pink noise*, and the note of Robin Whittle on the algorithm of
  James McCartney at `firstpr.com.au/dsp/pink-noise/`. Both state the binary
  schedule and why the rows approximate the 1/f fall.
