# The pink model in RTL

## Scope

`docs/host_control_rtl.md` left two seats open: the `model` source of
`Midi_merge`, and the `run_toggle` input of `Control_regs`. This document
describes the blocks that fill the two seats: the pink-noise algorithm as a
circuit, the run engine that plays it, and the board button.

The reference is in two parts, and they answer the two blocks of the model.
`Pink` holds the model — the register decomposition, four voices from the
row groups — and `Voss` computes the same integer arithmetic. `Player` holds
the rule that makes note events from the steps of the model, and `Sequencer`
does the same on the wire. Therefore the same seed gives the same notes and
the same messages in the reference, in the simulation and on the board.

## The constants

The audition froze the model constants. They are elaboration constants: the
bitstream carries them, and the host cannot change them.

| Voice | rows | root | degrees | register | period | restrike |
|---|---|---|---|---|---|---|
| soprano | 2 | 69 | 11 | A4 to A6 | 1 step | yes |
| alto | 2 | 60 | 4 | C4 to G4 | 4 steps | yes |
| tenor | 2 | 48 | 5 | C3 to A3 | 16 steps | no |
| bass | 2 | 33 | 6 | A1 to A2 | 64 steps | no |

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
| clocks_per_ms | 100 000 | `lib/board/top.ml` |
| button debounce | 10 ms | `lib/board/top.ml` |

The RTL elaboration reads the constants from `Pink.default`, as it
reads the control constants from `Control_intf`, and `Pink.degree_offsets` gives
the semitone offset of each degree. One definition serves the reference and
the circuit.

The four registers are disjoint. The four voices share one MIDI channel, and
a Note Off releases a voice by pitch. Therefore two voices must never hold
one pitch, or the Note Off of one voice stops the other.

`clocks_per_ms` is a parameter of `Sequencer`, as `clocks_per_bit` is for
the transmitters. The simulation gives a small value and runs fast.

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

```
                 ┌──────────────────────────────────────┐
   params        │                Model                 │
  ───────────────▶  ┌────────────┐     ┌────────────┐   │  midi
  (Control_regs) │  │    Voss    │◀───▶│ Sequencer  │───┼─────────▶ Midi_merge
                 │  │  ┌──────┐  │ the │            │   │◀── midi_ready
                 │  │  │ Prng │  │socket            │   │
                 │  │  └──────┘  │     └────────────┘   │
                 │  └────────────┘                      │
                 └──────────────────────────────────────┘

   btnC ──▶ Button ── run_toggle ──▶ Control_regs
```

| Block | It owns |
|---|---|
| `Prng.Rtl` | the xorshift32 state |
| `Voss` | the rows, the step count, the walk, the note of each voice, and which voices speak |
| `Sequencer` | the millisecond tick, the step and gate times, the open note of each voice, and the message construction |
| `Source` | nothing: it connects a note source to the sequencer, and the top level names the source, thus the top level keeps the same shape for every model |
| `Button` | the synchronizer, the debounce, and the toggle strobe |

`Source_intf` holds the socket between a model core and the sequencer. It
is a definitions-only module and has no `.mli`: the records are their own
signature. `Midi` holds the status constants that the message construction
needs: Note On, Note Off, and the release velocity.

## The socket

```ocaml
(** the number of voices of the synthesizer *)
let voices = 4

module Note = struct
  type 'a t =
    { voice : 'a [@bits Signal.address_bits_for voices]
    (** the voice that sounds it; 0 is the lowest *)
    ; pitch : 'a [@bits 8] (** the MIDI note number *)
    }
  [@@deriving hardcaml]
end

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; rewind : 'a (** a strobe: go to the origin of the sequence — the run start *)
    ; step : 'a (** a strobe: take one step and give the notes that speak *)
    ; ready : 'a (** 1 when the sequencer can take the note *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { note : 'a Note.t (** the note that speaks; it holds while [valid] is 1 *)
    ; valid : 'a (** 1 while [note] holds a note that the sequencer must play *)
    ; idle : 'a (** 1 when the source is at rest and can take a command *)
    }
  [@@deriving hardcaml]
end
```

A source answers `step` with the notes that speak at that step — zero, one,
or up to one for each voice. A note that does not speak never crosses the
socket: a voice that holds its pitch gives nothing, and a source with one
voice never names a voice it does not have. Therefore the sequencer makes no
selection, and it plays what it receives.

The synthesizer has four voices and no more: a fifth note stops the oldest
note. Therefore `voices` is 4, and this is a fact of the hardware and not a
parameter of the model. Voice 0 is the lowest and voice 3 is the highest. A
source with fewer voices takes the high numbers, because the highest voice
is the melody and the sequencer gates that voice only.

Each note carries the voice that sounds it, and the sequencer keeps one open
note for each voice. The voice number is the key of that state: the
sequencer closes the note of a voice before it opens the new one, it gives
each Note Off the channel of its Note On, and it closes the highest voice at
the gate. Therefore no model core holds this state, and a source that makes
one note at a time — an auto-regressive model — needs no buffer and no
knowledge of the step boundary.

The handshake:

- `idle` is 1 when the source is at rest. The sequencer strobes `rewind` or
  `step` only then.
- `idle` falls while the source works. The source holds a note while `valid`
  is 1, and the transfer is the one cycle in which `valid` and `ready` are
  both 1 — the rule of `Midi.Rtl.Message`. After that cycle the source gives
  the next note or goes back to rest.
- `idle` rising is the end of the command. After `step` it means that the
  step has no more notes, thus a step where nothing speaks gives no `valid`
  at all. After `rewind` it means that the source is at the origin.

The transfer rule is necessary because the MIDI line is slow: one message is
960 µs, and a voice that moves sends two. The source must hold each note
until the sequencer has put it on the wire. The sequencer is then between
two streams with one rule — notes in, messages out, `valid` from the source
of the data and `ready` from the sink of it.

`rewind` is in the socket because every source has a position to reset: a
walk, a current note, a hidden state, or at least an address. The rewind at
each run start is what makes the same run play the same sequence.

No configuration crosses the socket. A source takes what it needs by
closure at elaboration: `Voss` takes the model and the live view of the
SEED cell, and a later core takes its weights port the same way.

## Prng

The software and the circuit are in one file, because both want the name
`Prng`: the top level holds the OCaml recurrence, and `Rtl` holds the
circuit. The vector test in that file drives the two side by side, thus the
definition of the walk has one home and the comparison needs no export.

```ocaml
type t

val create : seed:int -> t
val next : t -> t * int

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
`load` wins over `step` in the same cycle. A draw is the low byte of the
state after a step, as `Prng.next` gives it.

The clear puts 1 into the state. The state has no use before the first
load, and 1 keeps the rule that the state is never 0.

`Voss` instantiates `Prng.Rtl` and nothing else sees it. The draw stream has
exactly one consumer, therefore no other block can move the sequence.

## Voss

```ocaml
module I = Source_intf.I
module O = Source_intf.O

val create : model:Pink.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
```

`Voss` is the note source of this era. The `voices` argument is the model:
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
`2 * rows + 2` cycles, against a step budget of at least `clocks_per_ms`
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

### The strike

A voice re-articulates when the walk re-rolls one row of its group. The
group of a voice starts at row `start`, and the walk of one step re-rolls
`count` rows. Therefore:

```
rerolled = count > start
changed  = the new note is not the note of the last step
speaks   = first_step | (rerolled & (restrike | changed))
```

`restrike` is the constant of the voice: the soprano and the alto speak at
every due step, and the tenor and the bass speak only when they move. A
`first_step` flag makes every voice speak at the first step of a run, as
the reference does. The flag is necessary and a comparison of the step
count is not enough: the count is 7 bits, and it reads 1 again at step 129.

`changed` compares the new note against the note register of the voice, one
cycle before the register takes the new value. This costs no register,
because the note of the last step is already there.

### The report

At the end of the walk the block latches the note of each voice and a mask
of the voices that speak. Then it gives them to the sequencer one at a time,
from the lowest voice upward — the order of the wire, and the order of the
reference. A note holds while `valid` is 1, and the mask loses that voice at
the transfer. When the mask is empty the block goes back to rest and `idle`
rises.

Therefore the block, and not the sequencer, decides which notes exist. A
voice that does not speak takes no cycle on the socket.

### The relation to the player

`Player` decides a strike with three parts: the voice is due, and either it
holds no note, or the pitch moved, or the voice re-strikes. The circuit has
no "holds no note" part, because the sequencer holds that state and not the
source. The two rules give the same result for these voices, and
the reason is the gate:

- The sequencer gates the highest voice only. The three lower voices hold a
  note from the first step until their next articulation, thus their "holds
  no note" part is never true after the first step, and the `first_step`
  flag covers the first step.
- The highest voice has `restrike` at 1, thus the third part is already true
  at each due step.

Therefore a voice with `restrike` at 0 must not be the highest voice. The
stream comparison proves the equality for the model of this era.

## Sequencer

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t (** the named views; each one is stable *)
    ; source : 'a Source_intf.O.t (** the outputs of the note source *)
    ; midi_ready : 'a (** from [Midi_merge]: 1 when the MIDI path takes the message *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { midi : 'a Midi.Rtl.Message.t (** the model source *)
    ; source_rewind : 'a (** a strobe: the source goes to its origin — the run start *)
    ; source_step : 'a (** a strobe: take one step and give the notes that speak *)
    ; source_ready : 'a (** 1 in the cycle that takes the note of the source *)
    }
  [@@deriving hardcaml]
end
```

### The tick

A prescaler divides the clock by `clocks_per_ms` and gives a millisecond
strobe. The run start resets the prescaler and the millisecond count.
Therefore a run is the same cycle for cycle in each simulation and on the
board, and the first step has the full length.

### The run

The run state is `params.run`, one bit. `Control_regs` owns the state; the
sequencer only reads it.

- Idle, and the bit goes to 1: the run starts. The sequencer strobes
  `source_rewind`, waits for the source's `idle`, and enters the first
  step.
- At each step boundary the sequencer samples RUN, STEP_MS and GATE_MS.
  This is the rule "a change applies at the next step". A sampled STEP_MS
  of 0 counts as 1. Between boundaries a change has no effect, thus a blip
  of RUN inside one step moves nothing.
- When the boundary sample reads RUN as 0, the sequencer sends a Note Off
  for each open note, from the lowest voice upward, and goes to Idle.

The sequencer strobes `source_rewind` at the run start and at no other
time. For `Voss` the rewind captures the SEED view, thus a run is a pure
function of the seed and of the sampled parameters, and the same seed
replays the same sequence. A write to SEED during a run applies at the next
run start; a capture at each write would tie the sequence to the write
moment.

### The step

At a step boundary with RUN 1, the sequencer strobes `source_step`. The
source then gives the notes of the step one at a time, and the sequencer
does this with each one:

- a Note Off, if the voice of the note already holds one;
- a Note On: the status byte carries `params.channel`, the note is the pitch
  from the source, and the velocity is `params.velocity`;
- `source_ready` at 1 for the cycle that completes the Note On. The source
  holds the note until then, thus the two messages both read a stable pitch.

The step ends when the source has no more notes and its `idle` rises.

Each voice has one open-note register, and the register keeps the note and
the channel of the Note On. A Note Off always takes the stored pair,
therefore a change of CHANNEL during an open note cannot leave the note hang
on the old channel.

The Note Off of a voice goes before its Note On. Therefore the number of
open notes is never more than four, and the S-1 never steals a voice.

### The gate

The gate closes the highest voice and no other. The lower voices sustain to
their next articulation, and this is the sound of the decomposition. At the
gate boundary — the sampled GATE_MS, when it is less than the sampled
STEP_MS — the sequencer sends the Note Off of the highest voice and clears
its register. When GATE_MS is not less than STEP_MS, the gate boundary never
comes, and that voice sends its Note Off at the next step boundary,
immediately before its next Note On.

### The messages

The sequencer is a message source on the `Midi.Rtl.Message` interface. It holds
`valid` until the transfer, as the doorbell does. The merge can stall the
source: the doorbell has the priority, and `Midi_out` takes one message at
a time.

One step sends at most eight messages: a Note Off and a Note On for each of
the four voices. A message is three bytes, and a byte takes 320 µs on the
MIDI line. Therefore a full step needs about 7.7 ms of line time. The
millisecond count does not pause during a stall, therefore the beat does
not drift. If STEP_MS is less than the line time of the step, the step
stretches to fit the messages. The default STEP_MS is 250, and the piece
has a large margin.

## Model

```ocaml
val create
  :  clocks_per_ms:int
  -> source:(Signal.t Source_intf.I.t -> Signal.t Source_intf.O.t)
  -> Signal.t I.t
  -> Signal.t O.t
```

`Source` connects a note source to the sequencer. It holds no logic — it
crosses the three commands of the sequencer to the source and the answer
back — and it does not name a model: the top level seats a model core with the `source`
argument, and the closure carries the configuration of the core. The block
does not change with the four voices.

```ocaml
Source.create
  ~clocks_per_ms
  ~source:(Voss.create ~model:Pink.default ~seed:control_regs.params.seed)
```

This is the one line that names the model of the era. A later era changes
`Voss.create ~model ~seed` to its own core and its own closure, and
nothing outside the line.

## Button

The block does not change. Two flip-flops synchronize the pin. The
debounced level changes when the synchronized input holds the new level for
`debounce_clocks` cycles, an elaboration parameter — 10 ms of the 100 MHz
clock on the board. `toggle` is one strobe at the rising edge of the
debounced level. The button is BTNC, package pin E16, `IOSTANDARD LVCMOS33`.

## The top level

The top level does not change, except for the `model` argument of the seat
line. LED 5 stays the run state.

The handshake paths are chains, not loops, as in the control design. The
`valid` and the `idle` of the source are functions of registers in `Voss`
and they do not read `ready`; the `source_ready` of the sequencer is a
function of its own registers and of `midi_ready`, and it does not read
`valid`. Therefore neither side of the transfer waits on the other in the
same cycle. `midi_ready` comes through the combinational merge from a
register in `Midi_out`, and `params` comes from the live cells.

## Changes to the host control

`docs/host_control.md` carries these corrections.

1. The model has four voices. They share CHANNEL and VELOCITY, and their
   registers are disjoint.
2. GATE_MS is the gate of the highest voice. The three lower voices sustain
   to their next articulation, and GATE_MS does not touch them.
3. A step that cannot fit its messages in STEP_MS stretches. Four voices
   need about 8 ms.

## What this design does not do

- No dynamics stream: VELOCITY is the one velocity for the four voices.
- No new control cells: the voices, the rows, the registers and the strike
  policy are elaboration constants.
- No hold switch. `bin/play_pink.ml` has `-hold`, which makes every voice
  speak only when it moves. That flag is a tool for the audition, and the
  board has no cell for it.
- No change to `Midi_merge`: the doorbell keeps the priority.
- No System Exclusive, and no running status, as before.
- No guard against a SEED of 0: the reference raises, and the board would
  play one chord without end. The spec forbids the value, and the driver is
  the place for a check if one is wanted.

## The cost

The measurement after synthesis: 735 LUTs and 830 flip-flops, against 651
and 775 for the one-voice circuit, and 418 and 522 for the control design
alone. The four voices cost 84 LUTs and 55 flip-flops, which is 0.1 % of
the XC7A100T. Three more mappings are cheap because each one takes two
rows: the one-voice mapping sums eight rows into an 11-bit value and
multiplies by 15, and a voice of the decomposition sums two rows into a
9-bit value and multiplies by 4, 5, 6 or 11.

The stream of the socket is not free of logic, and it is not expensive
either. Against a design that gives the state of the four voices at one
time, it costs 22 LUTs and no flip-flop: the report of the source and the
walk of the sequencer are the same work in one place instead of two.

There is no block RAM and there is no DSP. The count of DSPs proves the
claim of "The constants": the four constant multipliers are shifts and
adds, and no multiplier is in the circuit.

Timing is met, with 3.168 ns of slack on the 100 MHz clock. The build gives
34 warnings, and all of them are the 17 constant-driven ports of the
design: `JD[1]` to `JD[7]`, which the board holds at 1, and the 10 LEDs
with no function. The model era adds no warning.

## The tests

- `Prng`: 1000 steps of the circuit side by side with the software of the
  same module — the state and the draw, both exact — and a waveform of one
  step that also shows the rule that `load` wins over `step`.
- `Voss`, four voices: the reports rebuild the note of every voice, and the
  rebuild equals `Pink.next_step` for 200 steps. A pitch changes only at a
  step where its voice speaks, thus this one comparison proves the
  arithmetic and the strike rule together, and it runs with no millisecond
  waits.
- `Voss`: the report of the first 16 steps — which voices speak, and with
  what note — and the count for each voice in 128 steps. The counts show the
  two rules together: the alto speaks 33 times, which is its due count, and
  the tenor speaks 8 times against 9 due steps — the one step where it was
  due and did not move.
- `Voss`: the rewind repeats the sequence and a new seed changes it, and a
  waveform shows the rewind walk, one step, and the source holding a note
  while the sink is not ready.
- `Sequencer`: message logs with a small `clocks_per_ms` and a stub source
  that gives a programmed list of notes. The logs pin, with cycle
  timestamps: the voices speak in the order that the source gives them, a
  voice that the source does not name stays silent and holds its note, the
  gate closes the highest voice only, a Note Off keeps the stored channel of
  its voice while the next Note On takes a new one, the stop sends a Note
  Off for each open voice, and a STEP_MS write lands at the next boundary.
- `Source`, four voices: the integration. Drive the parameter views, take
  every message, and compare the stream against the messages that `Player`
  composes, byte for byte, with the gate and without it. A second run
  repeats the sequence from the seed.
- `Player`: the events of the first steps with the gate, which show the four
  voices entering at step 1 and the low voices silent until they move.
- `Pink`: the histogram of the soprano at stretch 1 and stretch 2, and the
  articulation grid that proves the due schedule is the trailing zeros of
  the step count.
- `Button`: a waveform of the debounce, with the clock as the ruler: a
  bouncy press gives one strobe, a bouncy release gives none, and a spike
  of one cycle moves nothing.
- `test/test_txn.ml` passes with no change: RUN is 0 at power-on, the model
  is silent, and the wire behavior does not move.
