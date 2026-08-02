# Build log

This log keeps the build milestones. One entry for each milestone, in time
order. New entries go at the end.

## 2026-07-31 — first sound (feat/first-sound)

The board makes a note on the Roland S-1. The full chain is proven: the
board powers on into the design from the QSPI flash, the mgt tool speaks
the host control over the USB UART, the doorbell sends the test message,
and the MIDI current loop carries it to the synth. The branch merged into
develop.

## 2026-07-31 — the control redesign (feat/regfile-redesign)

The control section was a byte memory with one address port. A memory of
that shape permits one reader at a time, thus the doorbell kept its own
copy of the message cells, the write decode was written two times, and the
MIDI sender had to live inside the wire-protocol engine. The model needs to
read the same cells and to send its own MIDI, therefore the shape had to
change before the model era starts.

The design is in `docs/host_control_rtl.md`. There are four blocks:
`Control_port` for the wire protocol, `Control_regs` for the cells and the
doorbell, `Midi_merge` for the priority, and `Midi_out` for the line. The
cells give named views with the natural width of each value, thus a
consumer never sees a byte and needs no address. `Regfile` is deleted.

What changed for the host:

- A write applies at one time. The port fills a shadow copy and one commit
  strobe moves the whole burst, therefore a value of more than one byte
  does not tear.
- The cells carry their power-on value in the bitstream, thus there is no
  init walk and the first request after configuration gets a correct
  answer.
- ADDR is one byte, and the registers are at `00` to `0F`. The two-byte
  address was the last part of a global memory map that this project never
  had.
- MIDI_GO reads 1 until the transmitter takes the message, and no longer
  until the last byte is on the line. `Control_regs` copies the message at
  the ring, thus a write after MIDI_GO reads 0 cannot damage it.

`test/test_txn.ml` passed with no change at each step. It drives the board
top level at the true UART rates and examines the MIDI line, thus it shows
that the behavior on the wire and on the MIDI line stays the same while
each block below it moves.

On the board: timing is met, 418 LUTs and 522 flip-flops, no block RAM.
`mgt dump` gives the same 16 bytes as the simulation, and the doorbell
sends a Note On, a Note Off and a one-byte real-time message. The bitstream
is in the QSPI flash.

## 2026-08-01 — the pink model (feat/pink-model)

The board composes. The first model is pink noise — the Voss-McCartney
1/f generator — chosen because it is small and it does not learn: no
trainer, no weight table. The reference is `lib/pink.ml`, in the integer
arithmetic of the circuit; `play_pink` plays it on the S-1 through USB,
and the ear settled the constants: 8 rows, stretch 2.

The design is in `docs/pink_rtl.md`. `Prng`, `Voss`, `Sequencer` and
`Button` fill the seats that the control redesign left open, and `Model`
is the one seat that the top level sees. The socket between a model core
and the sequencer is `Source_intf`: `rewind` and `step` in, `note`,
`valid` and `ready` out. No configuration crosses it — a core takes what
it needs by closure, and one line of `top.ml` names the model of the era.

The seed loads at the run start, thus the same seed replays the same
sequence; a Note Off uses the channel of its Note On; power-on is silent,
and one push of BTNC plays. The exactness holds at three levels: the
stream comparisons in Cyclesim, and 32 notes captured on the board
through the thru port of the S-1, byte for byte the reference.

On the board: timing is met, 651 LUTs and 775 flip-flops, no block RAM.
The bitstream is in the QSPI flash, and the board powers on into it.

## 2026-08-02 — the four voices (feat/pink-voices, feat/pink-voices-rtl)

The pink model grows from one voice to four, on the host and then in the
circuit. The shipped model summed all eight rows into one pitch; the
decomposition splits them into voices — soprano, alto, tenor, bass — and
maps each group's sum onto its own register. The partition is the rhythm: a
group that starts at row r re-articulates every 2^r steps, thus the
note-rate hierarchy is the 1/f structure made audible, with no rhythm
generator. Three partitions went through the ear test, and 2+2+2+2 won: the
periods 1, 4, 16 and 64 steps. The low voices speak only when they move.

The experiment measured the S-1 on the way, with the speaker off, through
USB MIDI in and USB audio capture out: the synth has four true voices, the
fifth note steals the oldest, and a Note Off releases a voice by pitch —
thus two voices must never hold one pitch, and the registers are disjoint.
Each root is a pitch class of one scale, and a voice takes its offsets from
that scale rotated to its own root. Therefore the harmony holds by
construction, and a root outside the scale is an error of the elaboration
and not a silent break.

The circuit needed a new socket. `Source_intf` gave one note for one step,
which four voices cannot use. The first answer gave the state of all four
voices at each step, and it was wrong for a reason that is worth keeping:
an interface of that shape makes every source describe four voices, thus a
model that makes one note at a time must invent three silent ones for the
sequencer to discard. The socket is now a stream. A source answers `step`
with the notes that speak, one at a time and from the lowest voice upward,
and a note that will not sound never crosses. Each note carries the voice
that sounds it, because the sequencer keeps one open note for each voice
and the voice number is the key of that state — the key that lets a Note
Off take the channel of its Note On. The transfer is the rule of
`Midi.Rtl.Message`, thus the design has one flow control and not two.

The reference is now in two parts, and they answer the two blocks: `Pink`
holds the model and `Voss` computes the same arithmetic; `Player` holds the
rule that makes note events and `Sequencer` does the same on the wire. The
review that found that split also added three rules to `AGENT.md`: what a
top-level export is for, when an export belongs in a `For_test` module, and
where the software and the circuit of one name live — in one file, with the
OCaml at the top and an `Rtl` module below it. `Prng` is the first of those.

The exactness holds at four levels. The circuit of `Prng` walks with the
software beside it for 1000 steps; `Voss` rebuilds the note of every voice
from its own reports and equals `Pink.next_step` for 200 steps; `Model`
gives the messages that `Player` composes, byte for byte; and the board,
captured through the thru port of the S-1, gave 330 bytes that agree with
the reference and then close each open voice at the stop. The same 330
bytes came from both socket designs, thus the stream changed the interface
and not the music.

On the board: timing is met with 3.168 ns of slack, 735 LUTs and 830
flip-flops, no block RAM and no DSP. The count of DSPs is the proof that
the four constant multipliers are shifts and adds. The stream costs 22 LUTs
against the design that gave all four voices at one time, because the
report of the source and the walk of the sequencer are the same work in one
place instead of two. The bitstream is in the QSPI flash, and the board
powers on into it.
