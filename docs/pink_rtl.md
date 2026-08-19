# The pink model in RTL

## Scope

`docs/pink.md` states the model — the 1/f walk, the register decomposition and
the mapping. This document states the circuit: the blocks of the source, what
each one holds, and what the elaboration requires of the constants.

The board around the source is not here. `docs/board_rtl.md` states the socket,
the run engine and the pins, and `lib/board/sequencer.mli` states the rules the
sequencer keeps. This document carried those sections while pink was the only
model; they went when the board took a document of its own, because a model is
a seat and the seat is not the model.

![The pink source: the PRNG, the row set on its binary schedule, the four
voice lanes and the frame](pink_rtl.svg)

The rows are the whole state, and the picture states the idea of the model in
one place: the partition of the rows is the rhythm, because a voice moves
exactly when one of its own rows re-rolls.

**`Pink.Source` answers the frame socket** since `feat/pink-v2`: one step is one
frame, thus the block keeps no note register, no owed flag and no report walk,
and it answers a step with one strobe. A frame cannot state a re-strike, thus
`Pink.Voice.restrike` and the soprano gate went with `Player`, and the rule that
makes the events of the wire is `Frame.events_of_frames` in the core.
`docs/pink.md` holds what the smoothing costs.

## The constants

The audition froze the model constants. They are elaboration constants: the
bitstream carries them, and the host cannot change them.

| Voice | rows | root | degrees | register | period |
|---|---|---|---|---|---|
| soprano | 2 | 69 | 11 | A4 to A6 | 1 step |
| alto | 2 | 60 | 4 | C4 to G4 | 4 steps |
| tenor | 2 | 48 | 5 | C3 to A3 | 16 steps |
| bass | 2 | 33 | 6 | A1 to A2 | 64 steps |

The scale belongs to the model and not to a voice: `Pink.default` holds C
major pentatonic one time, and each voice takes its offsets from that scale
rotated to its own root. Therefore every voice holds the pitch classes of
the one scale, and a root outside the scale is an error of the elaboration
and not a silent break of the harmony.

| Constant | Value | Home |
|---|---|---|
| voices | 4 | `Pink.default` |
| rows | 8 | the sum of the groups |
| stretch | 2 | each voice |

The RTL elaboration reads the constants from `Pink.default`, as it
reads the control constants from `Control_intf`, and `Pink.degree_offsets` gives
the semitone offset of each degree. One definition serves the reference and
the circuit.

The four registers are disjoint. The four voices share one MIDI channel, and
a Note Off releases a voice by pitch. Therefore two voices must never hold
one pitch, or the Note Off of one voice stops the other.

The constants are of the elaboration and not of the host: the bitstream
carries them and no control cell reaches them. `bin/play_pink.ml` has a `-hold`
flag that makes a voice speak only when it moves; that is a tool for the
audition and the board has no cell for it.

Each voice maps the sum of its two rows. With two rows and stretch 2 the
mapping has no divider and no general multiplier:

```
sum    = row_a + row_b                  (9 bits, 0 to 510)
x      = clamp(sum - 128, 0, 255)       (8 bits; the stretch-2 window)
degree = (x * degrees) >> 8             (degrees is 4, 5, 6 or 11)
note   = root + offsets[degree]         (offsets: a small table)
```

The four constant multipliers are shifts and adds: `x * 4` is one shift,
`x * 5` is `(x << 2) + x`, `x * 6` is `(x << 2) + (x << 1)`, and `x * 11` is
`(x << 3) + (x << 1) + x`.

## The blocks

| Block | It owns |
|---|---|
| `Prng.Rtl` | the xorshift32 state. The draw stream has exactly one consumer, thus no other block can move the sequence |
| `Source` | the rows, the step count, the walk, and the note of each voice |

`Source_intf` holds the socket between a model core and the sequencer. It is a
definitions-only module and has no `.mli`: the records are their own signature.
The pink source states every voice as sounding, because no voice of this model
rests.

## Prng

The software and the circuit are in one file, because both want the name
`Prng`: the top level holds the OCaml recurrence, and `Rtl` holds the
circuit. The vector test in that file drives the two side by side, thus the
definition of the walk has one home and the comparison needs no export.

`lib/core/prng.mli` states the software side, where a draw carries the state
to the next draw. The circuit is one register and the same recurrence:

```ocaml
module Rtl : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; load : 'a (** a strobe: the state takes [seed] *)
      ; seed : 'a (** 32 bits *)
      ; step : 'a (** a strobe: the state advances one time *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { value : 'a (** the 32-bit state; a draw is the low 8 bits *) }
    [@@deriving hardcaml]
  end

  val create : Signal.t I.t -> Signal.t O.t
end
```

One step is the three shift-and-XOR layers of xorshift32, combinational.
`load` wins over `step` in the same cycle. A draw is the low byte of the state
after a step, as `Prng.next` gives it.

The clear puts 1 into the state, and not 0. The state has no use before the
first load, and 1 walks where 0 stands still: 0 is the fixed point of the
recurrence. The seed itself may be 0 — the panel can set it and the board
accepts it — and then the walk stands still by design, which is the one chord
of `docs/seed_switches_rtl.md`.

`Source` instantiates `Prng.Rtl` and nothing else sees it. The draw stream has
exactly one consumer, therefore no other block can move the sequence.

## Source

```ocaml
module I = Source_intf.I
module O = Source_intf.O

val create : model:Pink.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
```

`Source` is the note source of this era. The `voices` argument is the model:
`Pink.default`, whose voices run from the fastest group to the slowest. `seed` is the
live view of the SEED cell; a `rewind` captures its value, puts the step
count at 0, and draws every row in ascending order — the origin of the
model with the seed of that moment. Each `step` gives the next state of
every voice.

### The walk

The walk does not change with the decomposition, because the voices share
one row set. The block holds the eight row bytes and a step count. The walk
is sequential, with one draw in two cycles: the PRNG steps in one cycle, and
the next cycle captures the registered state into one row. A step at count
`i` re-rolls rows 0 to `ctz i`, in ascending order. A walk takes at most
`2 * rows + 3` cycles, against a step budget of at least `clocks_per_ms`
cycles.

The step count is `rows - 1` bits wide, which is 7. This is exact, and not
an approximation: the reference clamps the re-roll count at `rows`, thus
trailing zeros beyond 7 do not matter, and a 7-bit count has the same
trailing zeros as the unbounded count for every value that is not a
multiple of 128. At a multiple of 128 the count reads 0, and the block
re-rolls all rows, which is the clamp.

### The groups

The voices take the rows in order: the soprano takes rows 0 and 1, the alto
rows 2 and 3, the tenor rows 4 and 5, and the bass rows 6 and 7. The voice
number counts down while the list counts up, thus the soprano is voice 3 and
the bass is voice 0.

Each voice sums its own rows and maps the sum with its own constants. The
four mappings are independent combinational blocks. Time-multiplexing one
mapping would save some logic and cost a state machine; the four blocks are
small, and the first build measures the true cost.

### The frame

The block states no strike. A frame states which pitch each voice holds, thus
the pitch of the step before decides nothing and the note of each voice is
combinational from the rows: no note register, no owed flag, and no walk that
reports the notes one at a time. Every voice of this model sounds, thus each
code the block fills carries the sounding flag.

The row order puts the highest voice first and seat 0 of a frame is the lowest,
thus the seats turn the list around. A model with fewer voices leaves the low
seats silent and takes the melody seats.

Era one decided a strike here — a voice was due, and it spoke when it held no
note, or the pitch moved, or the voice re-struck — and `Player` held the same
rule in software. Both went with the frame socket, and `docs/pink.md` states
what the smoothing costs.

## The cost

The four voices cost **84 LUTs and 55 flip-flops** over a one-voice circuit,
which is 0.1 percent of the XC7A100T. Three more mappings are cheap because
each one takes two rows: a one-voice mapping sums eight rows into an 11-bit
value and multiplies by 15, and a voice of the decomposition sums two rows
into a 9-bit value and multiplies by 4, 5, 6 or 11.

There is no block RAM and there is no DSP, and the count of DSPs proves the
claim of "The constants": the four constant multipliers are shifts and adds,
thus no multiplier is in the circuit.

The numbers of a whole build belong to the build that made them, thus
`build-log.md` holds them and this document holds only what the model costs.
