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

## 2026-08-14 — the transformer (feat/transformer, feat/transformer-rtl)

The board does inference. A small decoder-only transformer takes the model
seat, and it learns the Bach chorales. `feat/transformer` merged into
develop at `0b3dc56` and `feat/transformer-rtl` at `dd2264e`.

**The design of that model, which this log now keeps.** One token was one
byte: bit 7 gave the type, 1 for Note On and 0 for Note Off, and bits 6:0
gave the MIDI pitch. The code `0x00` was END and closed the sentence of a
step; the code `0xFF` was START and opened the walk. Each took a pitch that
no music holds — 0 is C-1 at 8.18 Hz and 127 is G9 at 12544 Hz — and the
tokenizer moved a corpus note off those two pitches. The zero word made a
cleared memory read as silence, made the padding of a batch mean silence,
and made the idle test in the circuit a compare with zero.

One step of `step_ms` asked for one sentence: zero or more events, then
END. A rest was a bare END, a held note was a note with no OFF, and a
repeated note was the OFF and then the ON of one pitch. The sentence had
one canonical order. The OFF events came first and climbed, and the ON
events followed and fell. Each direction earned its place: the fall is the
melody leading, because the top voice is chosen before the voices under it
and conditions on none of them, as the music is written; the climb then
made the two runs meet in the middle, so that the release of the top moving
voice sat beside its attack. **The reorder was the first change the ear
ever accepted, and the loss did not see it.**

A legality mask held the grammar before the softmax, in the training loss
and at the draw. An OFF was legal when its pitch sounded, the sentence held
no ON yet, and the pitch stood above the last OFF. An ON was legal when its
pitch was silent, a seat of the four was open, and the pitch fell below the
last ON. END was always legal and START never. Therefore every sentence was
valid MIDI, four voices sounded at the most, and two seats never held one
pitch — the disjoint-register rule of the S-1 by construction.

The network was `d` 64, RMSNorm before each sublayer, no bias terms,
`d_ff = 4 d`, ALiBi for the position, and one 256-row embedding tied with
the output head. Two small tables added to the token embedding: the bar
phase, 16 rows over the position in the bar, and the piece position, 16
rows over the position in the piece. **The piece-position table was the
second change the ear accepted, and the loss saw 0.0001 of it.** Training
divided by the length of a piece; the draw counted `step / 16 mod 16`,
because a draw has no length and the board plays for ever. The two rules
agree exactly at 256 steps.

The pattern of the era is worth more than the two tables: **both changes
the ear ever accepted were invisible to the loss, and both changed what the
model is conditioned on, not how much of it there is.** Every change that
moved the loss — capacity, depth, context, dropout, weight decay, budget,
ALiBi four times, head count — was rejected or null. Look for what the
model is conditioned on, and not for more of it.

The trainer moved to JAX on the host, and an OCaml referee held it honest:
the two forward functions had to give the same loss on the same batch.
`Prng`, the xorshift32 of the circuit, made every random number — the
initial parameters, the dropout masks and the sampler — thus one seed named
one walk in the software, in the simulation and on the board.

**The circuit.** `Quantized` is the integer twin: int8 weights, power-of-two
exponents for each tensor, int8 KV rings, and the sampler constants folded
into integers. The circuit must equal it bit for bit, and it does. The walk
became data — an `Op` schedule compiled at elaboration into one FSM over
one multiplier — and `Mac` took the walk behind that multiplier down to one
term for each cycle, which made the whole step 3.53 times faster than the
prototype. `Op.cycles` predicts the count of cycles exactly.

On the board, six layers: 127 block RAM tiles of 135 (94.07%), timing met
at +0.110 ns, 2,999 LUTs and 2 DSPs. The worst step takes 61 ms of a 200 ms
period. The two-layer model takes 47 tiles and meets timing at +0.300 ns.
The proof chain ran end to end: the circuit equals the reference in
Cyclesim, event for event, and the board — captured through the thru port
of the S-1 — gave 438, then 92, then 64 messages that agree with the
reference exactly.

Two faults of the era are on the record. The endless walk decayed, and the
answer that shipped was a mechanism and not music: every 256 steps the
source releases the sounding pitches, clears its context and feeds START.
And the QSPI flash holds a bitstream older than that correction. Era four
takes both of them up.

## 2026-08-19 — the frame model (feat/transformer-v2)

One step of music is one 32-bit word. Era three spoke a sentence of note
events for each step and held a legality mask before the softmax to keep it
valid; era four states the whole sonority at one time — four voice bytes,
bit 7 for SOUNDS and bits 6:0 for the pitch, seat 0 in the low byte, and a
cleared word is silence. The grammar goes with the sentence, because a frame
cannot name five voices or hold one pitch two times, thus the mask, the END
token and the ordering rule are all unnecessary. The sequencer holds the set
of pitches that sound and the frame states the set that must sound: it sends
the releases and then the strikes, and it composes nothing. The re-anchor of
era three goes too — this walk does not decay. The model is
`_train/d64-frame-do03-96k-s6-l6-nopos-span4.ckpt` at valid 1.6282, elected
by ear and not the lowest number of the sweep, because the finding of the
era is the variance and not the mean: ALiBi span 4 holds a standard
deviation of 0.0016 against 0.0108 at span 8, replicated at two step
budgets, because steep slopes confine every head to the bar while gentle
ones let each seed latch onto whatever distant structure its init favours.
The design is `docs/transformer_v2.md`.

On the board, six layers: timing met at +0.059 ns, 3,061 LUTs, 126 block RAM
tiles of 135, 2 DSPs. Two measurements paid for that margin. A `Switch` in
the `Always` DSL becomes one parallel case only when each match is a
constant, and a chain of guards becomes one mux level for each statement
that writes a target — the slot guards of the sequencer and a serial fold
cost 1.49 ns together. And a combinational ROM address let synthesis retime
each bank's data register onto the block RAM address pins and rebuild the
op-dispatch address cone inside every leaf primitive, which was the whole
layer scaling of the source; one registered address stage before the bank
tree took it from 3,466 to 2,352 LUTs at six layers, with the latency
unchanged. The bitstream is in the QSPI flash. One musical fault stays open:
only 67 to 73 percent of the silences follow a sonority held six steps or
more, against 99.2 percent in the corpus, thus the music stops without
arriving.

## 2026-08-19 — the board simplification (feat/board-simp)

The doorbell goes. It was the bring-up tool of the first entry of this log,
when the board could not otherwise make a note; a model that plays on one
push of the button is the better test. `Midi_merge` then holds one source
and no rule to apply, and `Midi_out` then holds one producer and one sink,
thus the transmitter moves into `Socket`: the seat takes the parameter views
and gives the line, the message interface stays inside it, and the top level
wires no handshake for MIDI. The cells MIDI_MSG, MIDI_LEN and MIDI_GO go,
and the two reserved bytes of era one's GATE_MS go with them; the section
compacts to nine cells, thus each cell has a view and no cell is a strobe.
The board shows two things: the run state on led0 and MIDI activity on led1.
The heartbeat showed only that the clock ran, and the two UART lamps
answered "is the wire alive", which the host answers better and answers with
a reason. The run lamp stays beside the MIDI lamp, because the model is
silent through the lead-in of one bar.

One test lost its subject. `test/test_txn.ml` proved a byte of the host on
the MIDI line at 31250 baud, and the model cannot replace it, because the
first note comes a lead-in and a draw after the run start, which is millions
of cycles of the board clock. The test now states the inverse: RUN rests at
0, thus the line carries no byte and JD keeps its no-current level.
`Midi_out` proves the line format, and `test_socket` proves the message
stream — it decodes the true transmitter now, and its expect file did not
change. On the board: timing met at +0.113 ns from +0.059, 2,991 LUTs from
3,061, 1,501 flip-flops from 1,646, block RAM unchanged at 126, and 262
setup endpoints fewer. Both `phys_opt_design` passes now find no setup
violation and change no netlist, thus the pass that era four needed is
insurance; `build.tcl` keeps it and carries both measurements. The bitstream
is in the QSPI flash, and a dump from the flash-booted image is the nine
bytes of the simulation.
