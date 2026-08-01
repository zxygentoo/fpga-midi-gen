# The pink model in RTL

## Scope

`docs/host_control_rtl.md` left two seats open: the `model` source of
`Midi_merge`, and the `run_toggle` input of `Control_regs`. This document
describes the blocks that fill the two seats: the pink-noise algorithm as a
circuit, the run engine that plays it, and the board button.

The reference implementation is the one-voice model of `lib/pink.ml`, the
`Pink.notes` view: every row in one group. The circuit and the reference
compute the same integer arithmetic, therefore the same seed gives the same
note sequence in the reference, in the simulation and on the board. The host
model has since grown the register decomposition — four voices from the row
groups — and this circuit is its one-voice case; the four-voice circuit is a
later design.

## The constants

The audition of step 1 froze the model constants. They are elaboration
constants: the bitstream carries them, and the host cannot change them.

| Constant | Value | Home |
|---|---|---|
| rows | 8 | `Pink.Params.default` |
| degrees | 15 | `Pink.Params.default` |
| scale | C major pentatonic, root 60 | `Pink.Params.default` |
| stretch | 2 | `Pink.Params.default` |
| clocks_per_ms | 100 000 | `lib/top.ml` |
| button debounce | 10 ms | `lib/top.ml` |

The RTL elaboration reads the constants from `Pink.Params.default`, as it
reads the control constants from `Control`, and `Pink.degree_offsets` gives
the semitone offset of each degree. One definition serves the reference and
the circuit.

`clocks_per_ms` is a parameter of `Sequencer`, as `clocks_per_bit` is for
the transmitters. The simulation gives a small value and runs fast.

With these constants the mapping has no divider and no general multiplier:

```
sum    = row_0 + ... + row_7            (11 bits, 0 to 2040)
x      = clamp(sum - 512, 0, 1023)      (10 bits; the stretch-2 window)
degree = (x * 15) >> 10                 (x * 15 = (x << 4) - x)
note   = 60 + offsets[degree]           (offsets: a 15-entry table)
```

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
| `Prng` | the xorshift32 state |
| `Voss` | the rows, the step count, the walk, and the note mapping |
| `Sequencer` | the millisecond tick, the step and gate times, the run engine, and the message construction |
| `Model` | nothing: it connects a note source to the sequencer, and the top level names the source |
| `Button` | the synchronizer, the debounce, and the toggle strobe |

`Source_intf` holds the socket between a model core and the sequencer. It
is a definitions-only module and has no `.mli`: the records are their own
signature. `Midi` holds the status constants that the message construction
needs: Note On, Note Off, and the release velocity.

## The socket

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; rewind : 'a (** a strobe: go to the origin of the sequence — the run start *)
    ; step : 'a (** a strobe: give the note of one step *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { note : 'a [@bits 8] (** the MIDI note of the step *)
    ; valid : 'a (** a strobe: [note] answers the last [step] *)
    ; ready : 'a (** 1 when the source can take [rewind] or [step] *)
    }
  [@@deriving hardcaml]
end
```

The protocol has two sides. `ready` is a level, and it gates the two
commands: the sequencer strobes `rewind` or `step` only while `ready` is 1,
and `ready` falls while the source works. `valid` strobes when `note`
answers the last `step`. `rewind` returns no data, thus its completion is
`ready` rising again.

`rewind` is in the socket because every source has a position to reset: a
walk, a current note, a hidden state, or at least an address. The rewind at
each run start is what makes the same run play the same sequence.

No configuration crosses the socket. A source takes what it needs by
closure at elaboration: `Voss` takes the model parameters and the live view
of the SEED cell, and a later core takes its weights port the same way.

## Prng

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; load : 'a (** a strobe: the state takes [seed] *)
    ; seed : 'a [@bits 32]
    ; step : 'a (** a strobe: the state advances one time *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { value : 'a [@bits 32] (** the state; a draw is the low 8 bits *) }
  [@@deriving hardcaml]
end
```

One step is the three shift-and-XOR layers of xorshift32, combinational.
`load` wins over `step` in the same cycle. A draw is the low byte of the
state after a step, as `Pink.Prng.next` gives it.

The clear puts 1 into the state. The state has no use before the first
load, and 1 keeps the rule that the state is never 0.

`Voss` instantiates `Prng` and nothing else sees it. The draw stream has
exactly one consumer, therefore no other block can move the sequence.

## Voss

```ocaml
module I = Source_intf.I
module O = Source_intf.O

val create : params:Pink.Params.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
```

`Voss` is the note source of this era. `seed` is the live view of the SEED
cell; a `rewind` captures its value, puts the step count at 0, and draws
every row in ascending order — the origin of `Pink.notes` with the seed of
that moment. Each `step` gives the next note of that sequence.

The block holds the eight row bytes and a step count. The walk is
sequential, with one draw in two cycles: the PRNG steps in one cycle, and
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

After the walk, the sum and the mapping of "The constants" give the note.
The note register takes it one cycle before `valid` strobes, thus `note` is
stable at the strobe.

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
    { midi : 'a Midi.Message.t (** the model source *)
    ; source_rewind : 'a (** a strobe: the source goes to its origin — the run start *)
    ; source_step : 'a (** a strobe: give the note of one step *)
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

The run state is bit 0 of `params.run`. `Control_regs` owns the state; the
sequencer only reads it.

- Idle, and the bit goes to 1: the run starts. The sequencer strobes
  `source_rewind`, waits for the source's `ready`, and enters the first
  step.
- At each step boundary the sequencer samples RUN, STEP_MS and GATE_MS.
  This is the rule "a change applies at the next step". A sampled STEP_MS
  of 0 counts as 1. Between boundaries a change has no effect, thus a blip
  of RUN inside one step moves nothing.
- When the boundary sample reads RUN as 0, the sequencer sends a Note Off
  for the open note if one is open, and goes to Idle.

The sequencer strobes `source_rewind` at the run start and at no other
time. For `Voss` the rewind captures the SEED view, thus a run is a pure
function of the seed and of the sampled parameters, and the same seed
replays the same sequence. A write to SEED during a run applies at the next
run start; a capture at each write would tie the sequence to the write
moment.

### The step

At a step boundary with RUN 1, the sequencer strobes `source_step` and
waits for the source's `valid`. Then it sends the messages of the step:

- If a note is open — the legato case — first a Note Off for it.
- Then a Note On: the status byte carries `params.channel`, the note is the
  one from the source, and the velocity is `params.velocity`.

The open-note register stores the note and the channel of each Note On.
A Note Off always takes the stored pair, therefore a change of CHANNEL
during an open note cannot leave the note hang on the old channel.

At the gate boundary — the sampled GATE_MS, when it is less than the
sampled STEP_MS — the sequencer sends the Note Off and clears the open
note. When GATE_MS is not less than STEP_MS, the gate boundary never
comes, and the legato case above sends the Note Off at the next step
boundary, immediately before the next Note On.

### The messages

The sequencer is a message source on the `Midi.Message` interface. It holds
`valid` until the transfer, as the doorbell does. The merge can stall the
source: the doorbell has the priority, and `Midi_out` takes one message at
a time. The worst stall is two messages on the line, which is less than
2 ms. The millisecond count does not pause during a stall, therefore the
beat does not drift; only the send moment moves.

## Model

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t
    ; midi_ready : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { midi : 'a Midi.Message.t }
  [@@deriving hardcaml]
end

val create
  :  clocks_per_ms:int
  -> source:(Signal.t Source_intf.I.t -> Signal.t Source_intf.O.t)
  -> Signal.t I.t
  -> Signal.t O.t
```

`Model` connects a note source to the sequencer. It holds no logic, and it
does not name a model: the top level seats a model core with the `source`
argument, and the closure carries the configuration of the core.

```ocaml
Model.create
  ~clocks_per_ms
  ~source:(Voss.create ~params:Pink.Params.default ~seed:control_regs.params.seed)
```

This is the one line that names the model of the era. A later era changes
`Voss.create ~params ~seed` to its own core and its own closure, and
nothing outside the line.

## Button

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; button : 'a (** the raw pin *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { toggle : 'a (** a strobe at each push *) }
  [@@deriving hardcaml]
end
```

Two flip-flops synchronize the pin. The debounced level changes when the
synchronized input holds the new level for `debounce_clocks` cycles, an
elaboration parameter — 10 ms of the 100 MHz clock on the board. `toggle`
is one strobe at the rising edge of the debounced level.

The button is BTNC, package pin E16, `IOSTANDARD LVCMOS33`.

## The top level

The new input is `btnC`. The other pins do not change.

- `Button` drives `run_toggle` of `Control_regs`, which was tied to 0.
- `Model` takes `params` and drives the `model` source of `Midi_merge`,
  which was tied to invalid.
- LED 5 is the run state, bit 0 of `params.run`. The board then shows the
  one bit that decides if the model plays, and the button gives visible
  feedback with no synthesizer connected.

The handshake paths are chains, not loops, as in the control design: the
source's `ready` and `valid` come from registers in `Voss`, `midi_ready`
comes through the combinational merge from a register in `Midi_out`, and
`params` comes from the live cells.

## Changes to the host control

Two rules of `docs/host_control.md` changed with this design, and that
document carries the corrections.

1. The SEED rule. The old rule: the PRNG loads the seed at the end of a
   write that covers a SEED cell. The new rule: the PRNG loads the seed at
   the run start. A write to SEED during a run applies at the next run
   start. One run therefore plays one sequence, and the same seed replays
   it.
2. The Note Off channel. A Note Off uses the channel of its Note On, and
   not the current CHANNEL. This closes a hole: without it, a CHANNEL write
   during an open note leaves the note hang.

## What this design does not do

- No dynamics stream: VELOCITY is the one velocity, per the step-1 cut.
- No new control cells: rows, stretch, root, degrees and the scale are
  elaboration constants.
- One voice. The S-1 has four; this era uses one.
- No change to `Midi_merge`: the doorbell keeps the priority.
- No System Exclusive, and no running status, as before.
- No guard against a SEED of 0: the reference raises, and the board would
  play one note without end. The spec forbids the value, and the driver is
  the place for a check if one is wanted.

## The cost

The measurement after synthesis: 651 LUTs and 775 flip-flops, against 418
and 522 for the control design alone. The model era costs 233 LUTs and 253
flip-flops, about 1 % of the XC7A100T. There is no block RAM, timing is
met, and the build gives no warning.

## The tests

- `Prng`: 1000 steps side by side with `Pink.Prng` — the state and the
  draw, both exact — and a waveform of one step that also shows the rule
  that `load` wins over `step`.
- `Voss`: the stream comparison — 200 notes against `Pink.notes` with the
  same seed. This is the exactness proof of the model, and it runs with no
  millisecond waits. The seed of a test circuit is an elaboration constant,
  as the closure carries it in the top level. Also: the rewind repeats the
  sequence and a new seed changes it, and a waveform shows the rewind walk
  and one step.
- `Sequencer`: message logs with a small `clocks_per_ms` and a stub note
  source. The logs pin, with cycle timestamps: the gate closes each note,
  the legato Note Off goes immediately before the next Note On, a Note Off
  keeps the stored channel while the next Note On takes a new one, the stop
  with an open note sends its Note Off, and a STEP_MS write lands at the
  next boundary.
- `Model`: the integration. Drive the parameter views, take every message,
  and compare the stream against the messages that the reference composes:
  byte for byte in the staccato and the legato case. A second run repeats
  the sequence from the seed.
- `Button`: a waveform of the debounce, with the clock as the ruler: a
  bouncy press gives one strobe, a bouncy release gives none, and a spike
  of one cycle moves nothing.
- `test/test_txn.ml` passes with no change: RUN is 0 at power-on, the model
  is silent, and the wire behavior does not move.
