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
`_train/transformer/d64-frame-do03-96k-s6-l6-nopos-span4.ckpt` at valid
1.6282, elected by ear and not the lowest number of the sweep, because the
finding of the
era is the variance and not the mean: ALiBi span 4 holds a standard
deviation of 0.0016 against 0.0108 at span 8, replicated at two step
budgets, because steep slopes confine every head to the bar while gentle
ones let each seed latch onto whatever distant structure its init favours.
The design is `docs/transformer.md`.

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

## 2026-08-19 — the seed panel (feat/seed-switches)

The seed decides the piece, and only the host could set it, thus the board
alone played one piece for ever. The 16 slide switches now write SEED and the
eight seven-segment digits show the cell. SEED takes the rule that RUN already
keeps: two writers, and the last one wins. A host write stands until a switch
moves, and a switch move applies at once, running or stopped. The panel wins a
cycle that carries a host commit and a switch move together, because the panel
is the writer a person can see and the host tool is mostly a method to debug.
The cell keeps its one home — the block gives a strobe and a value and holds
no seed of its own. The design is `docs/seed_switches_rtl.md`.

Nothing in the panel has a power-on value, and that is the whole of the
power-on rule. The synchroniser and the register of the value read last are 0,
thus a panel that is not at zero disagrees with 0 and writes the cell three
cycles after the power-on; a panel at zero writes nothing and the cell keeps
0, which is the value of that panel. There is no first-time flag, no counter
and no init walk, and a clear behaves as a power-on. SEED therefore lost its
power-on value of 42 and its range: the panel can set 0 and the design accepts
it, because a seed of 0 holds the PRNG at 0 and the piece is one chord that
does not move. That is a loud failure and not silence, and all the switches
down is the rest position of the panel.

The digits show the cell and never the switches, because a person who sets 16
switches and cannot see what the board took is setting it blind. All eight
digits show the 32 bits: the host writes the same cell, thus four digits could
report a value that is not the seed. The display is a scan of 5.24 ms, one
digit each 655 us, and a slice of the free counter makes the sequence, thus
there is no divisor and no state machine.

Two rules the design document did not state, and the circuit needs both. The
panel writes the synchronised value and not the pins, because the strobe says
that the synchroniser moved, thus the cell must take what it moved to. And a
bounce needs no debounce, which is not the rule of the run button: each edge
writes a whole seed and the last one stands, thus the cell ends at the value a
person set.

On the board: the cell answers the switches at the power-on, a host write of
DEADBEEF stands on the digits — the high half, which is the reason there are
eight of them — and the image booted from the flash comes back holding the
switches and not the host value. The bitstream is in the QSPI flash.

The entry before this one states that both `phys_opt_design` passes find no
setup violation. That is not true today, and the panel is not the cause. Three
builds of one day and one toolchain: the design before the panel meets at
+0.018 ns with 2,975 LUTs, the design with the panel at +0.041 with 3,028, and
the same netlist with `place -directive Explore` at +0.030 with 3,039. The
panel is 0.023 better than its absence, and it adds no timed path worth the
name: its pins carry no delay constraint at either end, and its own paths are
the synchroniser, the counter and the write into the cells. The +0.113 of the
board simplification does not reproduce, and this measurement does not say
why; the one difference of the netlist is the power-on value of SEED, but the
utilization of a build is the count after physical synthesis, which replicates
registers by the congestion it finds, thus that constant is a candidate and
not a cause. The route ends negative in each of the three builds — -0.260,
-0.660 and -0.467 — and the pass after it recovers all of them, thus the
post-route pass is load-bearing and not insurance.

The three builds give three different worst paths, at 6, 12 and 15 logic
levels. Therefore this design has no critical path to correct: it has a group
of paths that the congestion of 126 block RAM tiles of 135 holds inside about
0.05 ns, and the placement decides which one wins. A pipeline stage on one of
them would promote the next, and a directive is not free margin. The signed
build is +0.041 and it stands. `build.tcl` keeps the routed checkpoint now,
thus the next question of this kind costs an `open_checkpoint` and not a
build.

## 2026-08-20 — the state-space model (feat/mamba-proto)

Era five, and the whole chain in one branch: the trainer, the reference, the
integer twin, the circuit and the bitstream, each one gated against the one
before it. Nothing is auditioned and nothing is flashed; the board and the ear
wait for a person, and era four stays where it is.

The elected checkpoint is `_train/mamba/d64-mamba-n16-l6-do03-96k-s6.ckpt`:
d 64, inner 128, 4 heads, state 16, 6 layers, **178,504 parameters against era
four's 308,224**. Valid loss **1.6482** against era four's **1.6282** — the
first number that compares across two eras of this project, because both speak
the frame and cut the windows the same way. A state of 24 KB carries most of
what a window of 192 KB carried, for 0.020 nats.

The six-layer build meets at **+0.202 ns** with **3,043 LUTs** and **57.5
block RAM tiles of 135 — 42.6 percent**, where era four stands at +0.059,
3,061 and 126 tiles of 135. The design document estimated 55 tiles and 41
percent. The block RAM the era set out to buy back is bought back, and the
timing margin that comes with it is three times era four's on a design of the
same fabric size; both `phys_opt_design` passes are in the script and the
build meets without needing what they give. A drawn step costs 292,684 cycles
— 2.93 ms at 100 MHz, against era four's 7 — and it costs the same at every
step of the walk, because nothing here fills.

**The frame gate is blunt, and this era measured how blunt.** Weights of scale
0.02 put the classes so near each other that a pick is almost the quantile of
its uniform alone, thus a datapath can be wrong by tens of percent and still
draw the same frames for a dozen steps. Four faults were found by comparing
the residual stream write for write instead, and none of them moved a frame at
first: a weight addressed by a concatenation whose stride was not the tensor's
width, a convolution channel block read at the gate's offset, an operand
selected on the address side of a two-cycle read, and a tap ring whose layer
stride ran the top layer off the end of its memory. The last of those needs
three layers to appear — one hides the field and two round in its favour —
thus the gates run at three. `streams_agree` is in the tree and it is the
lesson.

The drift is measured and the trap of the era did not fire: over the gate
shape the cosine reads 0.9993 at 64 steps, 0.9994 at 256 and 0.9993 at 1,024,
flat to four decimals over many decay lifetimes. On the elected checkpoint at
the shape of the board, 512 steps read top-1 90.3 percent and cosine 0.9836.
One format decision was wrong and the drift report found it: truncating the
gate product back to the working class before the gated norm cost 0.10 of the
cosine on its own, and it threw away twelve bits immediately before the one
operation that would have used them. Of the clamps, `dt` and the state never
fire; `beta` fires on 0.0092 percent of the draws.

Two expectations of the design document are contradicted, and both are
findings. The dropout optimum is 0.3 — the same as era four's — where a model
42 percent smaller was expected to want less: 0.1 reads 1.7697, 0.2 reads
1.6711, 0.3 reads 1.6502 and a fourth run at 0.4 reads 1.7159, thus 0.3 is an
optimum and not the edge of a sweep. And the silence-arrival share, which the
state was a new lever on, went the **wrong way**: 47.2 percent against era
four's 67 to 73 and the corpus's 99.2. The walks are more silent and their
gaps are longer. That is the open musical question of the era, unimproved.

## 2026-08-20 — the quality round (feat/mamba-proto)

The ear heard the prototype of era five and called it jittery and
unmelodic. This round asked where the deficit lives, re-elected the draw
for this model's own logits, and swept the two levers the design reserved.
Nine training runs, one build, no board and no flash: era four stays where
it is.

**The deficit is in the moving steps and in the melody, and nowhere else.**
Over the canonical valid windows, 4,745 of 19,200 steps move two voices or
more. Read over three 96,000-step seeds for each era, era five is behind by
**0.0747 nats** over those and by **0.0001** over the other 14,455 — the
still-step losses are 0.6089 and 0.6090, identical to the fourth decimal. A
state of 24 KB and a window of 192 KB are the same instrument where the
music holds; the window is better only where it moves. Of the four seats
the soprano carries the largest gap, +0.0089 at five standard errors. The
mean gap of 0.02 nats the prototype recorded is the residue of the two.
`jax/diagnose.py` is the instrument and it reads both eras.

**No lever of the model moved it.** Five runs at 48,000 steps: the
baseline, K 16, N 64, both, and the baseline with `dt` opened on a
log-spaced ladder of phrase-scale half-lives. The baseline wins every
number. The ladder is the clean comparison — same shape, same 178,504
parameters, one tensor drawn differently — and it is the sharpest result of
the round: the ladder HOLDS through training, the upper layers keep
half-lives of 34 to 3,809 steps where the baseline keeps nothing above 9,
and the model is worse on the mean, on the moving steps and on the still
steps. K 16, N 64 and both together are confounded with capacity at a fixed
dropout of 0.3 and each overfits; the ladder is not.

**A diagnostic died in the round, and it was one of the round's own.** The
half-lives of the trained state look collapsed on the prototype checkpoint
— nothing above 7 steps in the upper half — and two more seeds of the same
configuration grow heads of 50.7 and 50.0. The three do not rank by it:
seed 7 takes the moving loss, seed 8 the mean, seed 6 sits between with no
long memory at all. **The time constants of a trained state are the seed's
and they predict nothing.**

**The draw did not transfer between the eras, and that is the round's one
actionable finding.** Era five's logits are sharper than era four's by
about 0.2 of temperature: at the shared policy of T 1.0 it holds 82.7 ±
0.4 percent of voice slots against era four's 80.8 ± 0.2 and plays 0.64 ±
0.01 onsets a step against 0.72 ± 0.01, over 32 walks. Era five at T 1.1
reproduces era four's ear-elected row almost exactly. **The ear auditioned
era five at a texture era four would have had near T 0.9.** At T 1.2 the
elected checkpoint reads hold 78.15 and 0.81 onsets against the corpus's
78.17 and 0.81 — it lands on both to the second decimal — and the offset
holds on all three seeds. `Mamba.elected_temperature` stands at 1.0 until
the ear moves it, and moving it means a new bitstream.

**One recorded finding of the prototype does not survive.** The
silence-arrival share of 47.2 percent, called "the honest result of the
era", was a six-walk mean. CADENCED divides by the silences of a walk, a
walk of 512 steps holds three or four, and some hold none: the standard
error is 14 points over eight walks and 6.5 over 32. At 32 walks era five
reads 81.1 ± 6.1 against era four's 72.6 ± 6.5. The instrument does not
part the two eras and never could at that sample size. The warning is now
in `measure.py` beside the number.

**`K` is a field of the configuration.** It was the constant
`Mamba.conv_taps`, and the sweep needed it to move: `Config.of_checkpoint`
now reads it from the kernel tensor, and the tap ring, the age mux, the
conv op and the cost model size themselves from it. The Verilog
regenerated at K 4 is **byte-identical** to the Verilog the prototype build
was made from — 10,767,495 bytes, `cmp` clean — thus the refactor is proven
a no-op on the board shape; a new stream gate at K 16 and N 32 over three
layers proves it correct where it is not.

The elected checkpoint of the round is
`_train/mamba/d64-mamba-n16-l6-do03-96k-s7.ckpt`, by the rule that elects
on the moving-steps loss. It wins by 0.005 nats against a seed spread of
0.018, seed 8 takes the mean and the still steps, and the three seeds are
one model: all three are kept and the ear decides. The six-layer build on
seed 7 meets at **+0.197 ns with 3,144 LUTs and 57.5 of 135 block RAM
tiles**, against seed 6's +0.202 and 3,043 at the same 57.5. **A new
checkpoint of the same shape costs 101 LUTs and 0.005 ns**, which is the
timing lottery of this project seen from the smallest possible distance.
The bitstream is in `board/_build` and it stays there.

Ten `.mid` files are rendered in `_train/audition/round-2026-08-20/` —
three checkpoints against three temperatures, and era four's king as the
control. The ear takes it from there.

## 2026-08-21 — the trainer speed order, refuted (feat/mamba-proto)

A work order of 2026-08-21 held that the mamba trainer was overhead-bound at
56–68 ms for each step against the transformer's 27–29, blamed the
`[batch, T, T, heads]` chain of `selective_window`, and ordered the chunked
semiseparable form of Mamba-2 to remove it. **Its phase 1 refuted its own
premise and the rewrite is shelved.**

**Measured solo on the device, which the order asked for first:**

| shape | ms for each step |
|---|---|
| `MMMMMM` N16 | **25.9** |
| `MMMMMM` N64 | 26.4 |
| `MMMMMMZF` N64 | 30.0 |

The 56–68 ms figures were runs SHARING the card. Solo the mamba trainer is
faster than the transformer on the same shapes, and the two-times gap the
order was written to close does not exist. One arithmetic claim was also
wrong: the `[batch, T, T, heads]` arrays are 16 MiB each at batch 16 and
T 256, not the 67 MB stated — though `selective_window` builds about six of
them for each layer and direction, so the traffic finding stood.

**The rewrite was built anyway, and it underdelivers.** Chunked against the
quadratic form it replaces:

| context | quadratic | chunk 32 | chunk 64 | chunk 128 | best gain |
|---|---|---|---|---|---|
| 256 | 25.9 | **21.5** | 22.3 | 22.7 | 1.17x |
| 512 | 59.8 | — | **44.4** | 47.0 | 1.35x |

Both under the order's own 1.5x bar. The sanctioned fallback — the
`[batch, heads, t, s]` layout reorder — measured 25.1 ms against 25.9, which
is nothing, and it was reverted. **The bottleneck was misidentified**: at
25.9 ms for 9.5 GFLOP the trainer runs at 3 percent of the device, and that
is kernel-launch overhead on a model of width 64 where every kernel is tiny,
not the traffic. Cutting the traffic four to eight times cuts the time by 17
percent, which is how one knows.

**Concurrency is the lever, and it was already in use:**

| | ms each | runs for each hour |
|---|---|---|
| 1-way | 26.7 | 135 |
| 2-way | 40.6 | 177 |
| 3-way | 54.2 | **199** |

Three at once gives 1.48x the throughput of one — more than the rewrite
would have.

`selective_window_chunked` stays in the tree DORMANT with its gate: twelve
cases against the quadratic oracle over four window lengths and three chunk
sizes, including lengths no chunk divides. Writing a sign the wrong way
round there gave errors of 6.6e7 and the gate caught it at once, which is
the argument for widening a gate before a rewrite and not after. It is not
wired in: a change of summation order would put every new run outside the
seed spread of the thirty checkpoints this era compares against, and 17
percent does not buy that. The gain rises with context, thus it becomes
right at an era boundary where the baselines retrain anyway.

## 2026-08-22 — the elected model in RTL (feat/mamba-proto)

The trunk of era five was a circuit already. The model the ear elected is
not a trunk: `MMMMMMZF` — six Mamba blocks, then the Zamba attention head,
then the feed-forward. This round carries that plan through all three
layers of the era — the float reference, the integer twin and the circuit —
and every gate of the era passes on it.

**The plan is a fact of the checkpoint.** A layer is a block, the Zamba
head or the feed-forward, and the first tensor of a group names its kind:
`[d, projection]`, `[2 d, d]`, `[d, 4 d]`. The ALiBi span stands after the
last group as one number. Era four carried its span as a flag that had to
match the training run, and a span played back wrong is silently wrong
music; nothing of this model now stands outside its file except the ring
depth, which is a choice of the player at inference.

**Half a Zamba.** The query and the key read the original embedding beside
the residual stream — `wq` and `wk` are `[2 d, d]` — and the value reads
the stream alone. The circuit carries that as one memory of `d` by 16 bits:
the normed embedding is written once, after the embed, and the head of the
last layer still reads it, where every other vector of the machine dies
inside its layer. The matvec that walks it is the fourth operand in this
design to follow the DATA and not the address.

**Two rules of the machine survived the head, and one did not.**

- NO WALK STALLS survived, and it was in danger. Era four merged the lanes
  of a head in one walk and froze that walk while the divide of a finished
  row ran, thus every read register and every tag of that machine carried a
  freeze enable. `Attend` here merges ONE LANE A WALK and waits on the
  divide with the walk already retired. It costs the drain of each lane and
  it buys back the enable on every read register of the design, thus `Mac`
  still takes its hold at ground.
- The slot and the fill need no registers: the newest slot is the low bits
  of the step counter and the ring is full once that counter passes the
  depth.
- THE COST STOPS BEING A CONSTANT OF THE SHAPE, and this document had
  claimed it would not. `Op.cycles` takes `n` again, and `Attend` is the
  one op that reads it.

**The cost, out of the model at the elected shape:**

| | cycles a drawn step | ms at 100 MHz |
|---|---|---|
| the trunk, `MMMMMM` | 292,684 | 2.93 |
| **the elected plan, `MMMMMMZF`** | **401,410** | **4.01** |

The wire's 8 ms floor stands over both. The step grows until the ring fills
at step 256 and is constant after that. The schedule test prints this
number out of `Op.cycles`, thus the design document and the model cannot
part.

**The two faults this round found were both in the REFERENCE**, and the
write-for-write stream gate found both in one run:

- the ring coarsening was written `v asr 8 lsl 8`, which OCaml reads as
  `v asr (8 lsl 8)` and the machine reads as no shift at all, thus the
  reference coarsened nothing;
- the exp2 argument of the softmax negated before the scale where the
  circuit negates after it, which parts the two by one unit whenever the
  scale does not divide exactly.

Neither moved a frame. A gate that compared frames alone would have shipped
both, which is the same lesson the trunk round recorded and the second time
this era has paid for it.

**What the gates read.**

| gate | result |
|---|---|
| Gate A, the loss across the JAX seam | **1.640810**, against the trainer's 1.6408 |
| Gate C, the identical walk, seeds 1 and 7 | pass |
| the circuit against the twin, frame for frame | pass at one block, and at two blocks with two heads |
| the circuit against the twin, WRITE FOR WRITE | pass at one block, at three blocks with two heads, and at a wide state and kernel |
| the cycle bench against `Op.cycles` | 0 disagree over 18 steps |
| the drift floors | hold |

**The drift on the trained checkpoint, and the coarse ring is not the
problem the test shape said it was:**

| model | top-1 | cosine | same draw |
|---|---|---|---|
| `MMMMMM` 96k s7 | 88.7% | 0.9825 | 85.6% |
| `MMMMMMZ` 48k s7 | 90.2% | 0.9800 | 87.6% |
| **`MMMMMMZF` 48k s7** | **92.7%** | **0.9842** | **90.1%** |

No clamp fires anywhere. On drawn weights at the shape a test can afford
the head does drift more than the trunk — the floors moved from 0.938,
0.974 and 0.9987 to 0.875, 0.979 and 0.9972 — and on the model the board
will really carry it drifts LESS. The key and value ring keeps era four's
coarse byte on that evidence; widening it to int16 would cost about 3.5
tiles of 135 and the measurement above is what that round would be buying.

**The build ran, and it MISSES TIMING by 0.081 ns.**

| | the elected plan | era five's trunk | era four |
|---|---|---|---|
| block RAM tiles | **80.5 of 135 — 59.6%** | 57.5 — 42.6% | 126 — 93% |
| RAMB36 / RAMB18 | 75 / 11 | 53 / 9 | — |
| slice LUTs | 3,447 | 3,144 | 3,061 |
| slice registers | 1,814 | 1,652 | — |
| DSPs | 2 of 240 | 2 | — |
| WNS | **−0.081 ns** | +0.197 | +0.059 |
| WHS | +0.040 ns | +0.073 | — |

The block RAM estimate said about 80 tiles and 59 percent; the build says
80.5 and 59.63. Hold is met everywhere. **Nine setup endpoints fail of
5,685**, and eight of the nine are noise:

| slack | endpoint |
|---|---|
| **−0.081 ns** | the divider's `m`, bit 34 |
| −0.045, −0.041 | the same register, bits 38 and 37 |
| −0.021 to −0.020, six of them | the reset pin of `Mac`'s 14-bit counters |

**The critical path is the divider's input, and the head put it there.**
The path runs from the program counter, through the op-dispatch mux that
chooses `div_num`, into the 40-bit NEGATE at the divider's input, and lands
in `m`: 20 logic levels, eleven of them CARRY4, 10.040 ns of which 5.748 is
route. The trunk had ONE writer of `div_num` — the divide chain of
`rms_norm`. `Attend` is the second, thus the mux in front of a 40-bit carry
chain grew a pc-selected input. It is the same shape of fault era four
recorded and named: every case of the program counter that writes a
register widens that register's parallel case.

**THE TIMING LOTTERY DOES NOT APPLY TO A RE-RUN, and this build proves it.**
The same script ran a second time over the same Verilog and gave a
BYTE-IDENTICAL timing report and the same bitstream MD5. Vivado is
deterministic for one netlist, one part, one set of constraints and one set
of directives. The lottery this project measured is a lottery of CONTENT —
a new checkpoint of one shape moved 101 LUTs and 0.005 ns because the tools
placed a different design — and re-running identical inputs cannot sample
it. A miss must be fixed, not re-rolled.

**The directives close it, by eight picoseconds.** The same netlist under
`place_design -directive ExtraTimingOpt`, `route_design -directive
AggressiveExplore` and the two `phys_opt_design` passes at
`AggressiveExplore`:

| | default directives | aggressive directives |
|---|---|---|
| WNS | **−0.081 ns** | **+0.008 ns — MET** |
| failing setup endpoints | 9 of 5,685 | 0 of 5,632 |
| WHS | +0.040 ns | +0.071 ns |
| slice LUTs | 3,447 | 3,474 |
| block RAM tiles | 80.5 | 80.5 |

**The critical path does not move.** It is the same program counter into
the same 40-bit negate at the divider's input, still 20 logic levels and
still 56 percent route; the tools took 0.089 ns out of it and no more. The
margin that buys is +0.008 ns against era five's trunk at +0.197 and era
four's at +0.059 — an order tighter than either, and well inside the 0.1 ns
band this project has measured between builds of one shape. A re-trained
checkpoint of the same plan would be a coin toss.

The board has NOT been programmed and no bitstream of the aggressive run
was written. The fix that would give this design a real margin is
structural and the diagnosis names it: a travel stage on `div_num` and
`div_den`, which takes the pc mux out of the negate's path and costs one
cycle for each divide — about 1,700 cycles of 401,410, or 0.4 percent.

## 2026-08-23 — the magnitude stage (feat/mamba-proto)

**The miss is fixed, structurally, and the fix is not the one the last
entry named.** That entry closed on a travel stage over `div_num` and
`div_den`. The design round found the trap in it: both callers test
`~div_busy` exactly one cycle after the strobe — `rms_norm` through
`by_tick`, `Attend` through its stage counter — and the divider's busy
rises in the cycle after start, thus the wait holds today by one cycle
exactly. A plain travel register moves the rise one cycle later, both
waits read busy low, and both land a garbage quotient. The naive travel
stage is a correctness fault that no frame gate would catch before the
board.

**The fix that went in: the magnitude moves INSIDE the divider's walk.**
Era four's unit negates in the start cycle, thus a 40-bit carry chain
stands between the caller's operand mux and the first register; the start
cycle now latches the raw numerator and its sign, and the FIRST BUSY
CYCLE takes the magnitude, register to register. `busy_cycles` is 41.
Busy still rises in the cycle after start, thus the contract keeps its
shape and NEITHER CALLER CHANGED A LINE. The cost models read
`Divider.busy_cycles`, thus they moved by themselves, and the cycle bench
held the measured circuit to them at delta 0 on every step — the proof
the wait contract survived at both callers.

**The unit is a copy, on purpose.** `lib/mamba/divider.ml` is era four's
file with the magnitude stage and nothing else; `lib/transformer/` is
untouched and era four's chapter does not re-record. The common parts of
the neural-net sources — `Mac`, `Divider`, the tables — take one home in
their own round, and the two dividers merge there. `Divider` joins `Mac`
as the second unit that could not come whole, and `docs/mamba_rtl.md`
carries both reasons.

**The cost, re-measured out of `Op.cycles`:** one cycle for each divide,
1,664 of them a drawn step.

| | cycles | ms at 100 MHz |
|---|---|---|
| the steady drawn step | 403,074 | 4.03 |
| the first drawn step, 16 ages | 365,634 | 3.66 |
| the trunk `MMMMMM`, for the record | 294,090 | 2.94 |

**The build MEETS at default directives, with the widest margin of any
full design this project has built:**

| | the miss (default) | directives (aggressive) | **this build (default)** |
|---|---|---|---|
| WNS | −0.081 ns | +0.008 ns | **+0.278 ns — MET** |
| failing setup endpoints | 9 of 5,685 | 0 of 5,632 | **0 of 5,640** |
| WHS | +0.040 | +0.071 | +0.096 |
| slice LUTs | 3,447 | 3,474 | 3,461 |
| block RAM tiles | 80.5 | 80.5 | 80.5 |

The structural fix took 0.359 ns out of the path the directives could
only shave by 0.089, and it costs fourteen LUTs. Era five's trunk met at
+0.197 and era four at +0.059; this margin clears the 0.1 ns content
lottery with room, thus a re-trained checkpoint of the shape is no longer
a coin toss. The six `Mac` counter-reset endpoints of the miss closed on
their own under the new placement.

**The divider has left the report.** The critical path is now the pc
one-hot decode into a register clock-enable: 12 logic levels, 9.536 ns,
71 percent route — the route-dominated fan-out shape the placer can
always work with, and the shape every healthy build of this project has
shown.

**The board runs this build since 2026-08-23**, programmed through the
JTAG — volatile, thus era four holds the flash and returns on a power
cycle. The ear has not heard era five yet.

## 2026-08-23 — the common home (feat/nn-uni)

**The unification round the divider copy promised.** Four layers, two
eras, one rule each: `lib/nn` (`mgen_nn`) and `jax/nn.py` now hold what
was one thing written twice, and the era libraries hold what is theirs
alone. Both era libraries depend on `mgen_nn`; neither depends on the
other any more — `mgen_mamba` dropped `mgen_transformer` from its
libraries, and `jax/mamba` stopped importing `transformer.model`.

**What moved, by layer:**

- **RTL.** `Divider`, `Isqrt`, `Exp2`, `Sigmoid`, `Softplus` moved whole
  with their gates. `Mac` became a FUNCTOR over its walk width — era four
  instantiates nine bits, era five fourteen — which is the parameter its
  own file asked for. THE ONE DIVIDER is era five's, with the magnitude
  inside the walk: era four's copy is gone, its cost model followed
  `Divider.busy_cycles` to 41 by itself, and its cycle bench re-recorded
  at delta 0 on every step — 354,696 → 356,808 over the 18-step bench,
  one cycle for each divide and the proof that its waits survived.
- **The integer rules.** `Mgen_nn.Quantized`: the formats, the three
  tables and their index rules, `quantize` and the exponent rule, the
  scalar oracles (`isqrt`, `exp2`, `sigmoid_q`, `softplus`, `silu`,
  `clamp16`), the integer `policy` and the 24-bit `draw`. The era
  `Constants` are now an `include` plus their own formats; the negation
  gate that held the two exp2 readings to one table is gone because one
  definition holds them now.
- **The float rules.** `Mgen_nn.Policy` (the tempered draw, its bounds,
  the elected numbers) and `Mgen_nn.Checkpoint` (the seam writer, the
  refusal scrubber, `numel`). Their gates moved with them.
- **JAX.** `jax/nn.py`: the precision pin, ALiBi and its span, the
  four-table embed, the chained head, the host draw chain, the trainer
  skeleton (Adam step, the schedule, the eval sums, the seam writer).
  `jax/midi.py`: the wire side of an audition, out of both `infer.py`.

**The op and schedule layer did not move**, by standing rule: the
abstraction is a discussion, not a side effect of a branch.

**The proofs.** The era-five board netlist is BYTE-IDENTICAL through the
whole round — `gen_verilog` gives md5 `a648db22…` before and after, thus
the design on the board today is provably untouched. Era four's netlist
changes by design (the 41-cycle divider, the nine-bit functor `Mac`);
its drift floors, its socket gate and every other expect held without a
re-record. The full battery: `dune runtest` clean, all 62 JAX tests
green — the parity seam still reads 1.640810 — and `dune fmt` silent.

The eras now differ only where their models differ, and the next model
starts from `lib/nn` instead of a copy.

**Era four VERIFIED in silicon terms, not only in simulation.** The seat
moved to `mgen_transformer` by the one dune line the top level promises,
era four's `gen_verilog` came back from `4a4814b` with its checkpoint
path corrected to `_train/transformer/`, and the full board top
elaborated from HEAD's libraries — the common home serves the closed era
whole. The default flow, in a scratch directory so `board/_build` keeps
era five's programmed bitstream:

| | era four at 4a4814b | era four at HEAD, one divider |
|---|---|---|
| WNS | +0.059 ns | **+0.005 ns — MET** |
| failing endpoints | 0 | 0 of 6,536 |
| WHS | — | +0.034 |
| slice LUTs | 3,061 | 2,996 |
| block RAM tiles | 126 | 126 |

The critical path is the pc decode into a register — 13 levels, one
CARRY4, 75 percent route — thus the divider stands off era four's
critical path exactly as the magnitude stage predicts. The margin moved
from +0.059 to +0.005, which is inside the content-lottery band a new
netlist rolls; the number that matters is MET at default directives, on
a closed era that ships from the flash and is not rebuilt. No bitstream
was written and the tree carries the era-five seat, restored and proven
byte-identical after the switch.

## 2026-08-25 — era six, the masked sheet (feat/diffusion-jax)

**The model that completes instead of inventing.** Round one, a Gaussian
DDPM over a piano roll, was dead on the ear and stays on
`feat/diffusion-proto` with its autopsy. Round two took Coconet's shape:
a convolutional net over a sheet of T steps by 48 classes with pitch as
an AXIS and silence as a class, the orderless-NADE loss over hidden
cells, and independent blocked Gibbs under Yao et al.'s annealed mask.
Every draw of the walk comes from the shared xorshift32 of `jax/prng.py`
in one consumption order — the opening, each mask, each redraw — thus
one seed names one sheet here, in OCaml and on the board.

**The referee compares outside the repository.** Algorithm 1, framewise,
five orderings in probability space: 0.5884 ± 0.017 nats against the
paper's 0.57 ± 0.01. The ladder said depth is reach and width is a
floor, and the ear elected `l48-h20-100k` — valid 0.4422, framewise
0.6139 ± 0.0151, triads on the corpus. The walk opens on a seeded sheet
inside each seat's register and not on silence, measured the same
instrument over 256 sheets. Merged as `fcb749a`.

**The OCaml reference round** (`f86b96b`, 2026-08-26): the float
reference and the int8 twin in `lib/diffusion`. Gate A agreed to the
sixth decimal (0.189632), Gate C printed the same step lines, and the
drift of the golden candidate at int8 read top-1 97.2 %, cosine 0.9998,
same-draw 95.1 %, zero clamps. The finding: Q12 activations were WRONG
— a trained residual trunk grows its activations to a peak of 313 on
the seeded openings — and Q6 is the measured format, with a margin of
1.6.

## 2026-08-28 — era six on the board: the column engine (feat/diffusion-rtl)

**One machine, no op layer.** The elaboration reads the contract file's
own tensor shapes; the circuit is a column engine of 48 rows by G lanes
— weight-stationary, the DSP accumulating over the 9·Cin dwell, a
three-column window fed under the running dwell — an epilogue that
clamps twice where a pair closes, era four's draw pipeline over one
class counter, a sheet with three faces (the walk, the stem, the score)
and the walk FSM around them. Six stages, each with its instruments;
the sheet agreement compares PER PHASE and the stream gate WRITE FOR
WRITE, because era five's faults moved no frame.

**Three traps, each convicted on silicon.** Ring 3 failed at −6.125 on
broadcast nets with one driver; the replica banks per eight-row slice
took it to +0.010 MET, the board played on 2026-08-27, and the capture
of 268 messages at seed 47872 was byte-exact in order — Gate B whole at
rung 1. Vivado padded the rung-2 ROM to 65 536 words and demoted EVERY
ROM to fabric with no warning; banking by powers of two put rung 1 at
96 tiles / +0.007 and rung 2 (`l64-h16`) at 124 tiles / +0.008, and
that build went to the flash. The service cut's build met by +0.004
only because `phys_opt` moved clock skew inside the uniform's shift
register — three different sheets at one seed — and STA met by a
picosecond is not met; three replicas of the uniform, one for each
consumer, gave +0.018 / +0.029 and two byte-exact captures.

**The golden candidate fits.** The stores bank like the ROM (108 tiles
at the rung-3 shape, +0.122), the fused pair keeps Y as a four-column
ring with B two columns behind A, and the timing cuts close it:
`l48-h20` at T 128, G 5, N 512 — +0.147 MET on one tree, 108.5 tiles,
236 messages byte-exact — IN THE FLASH since 2026-08-28.

## 2026-08-28 — the OCaml cut and the Flax shape (feat/diffusion-ocaml-cut)

**The welds go.** The OCaml float model and the OCaml int8 twin were
only the seam between JAX and the RTL; both are deleted, the integer
twin is `jax/diffusion/quantized.py`, and the RTL gates run under
`uv run pytest` driving `bin/gate_diffusion.exe` — the driver prints
what the circuit did, Python states what it must have done, and neither
side can pass by agreeing with itself. `top.v` stayed md5-identical
(`4e367cef`) through every round below, thus the flash never moved.

**The net is a tree.** `Coconet` is a Flax NNX module — a stem, the
residual pairs, a head — and the twin carries the same skeleton under
the same names; an odd layer count became unrepresentable. The trainer
is optax's AdamW, held equal to `nn.adamw` leaf for leaf and step for
step, thus no retrain; the three JAX packages moved to 0.11.1 together
and the card reads 370 ms/step against 432. G0 pinned the float model
at 0.193459 before and after.

**Then the small rounds:** two simplify passes and a comment pass; the
placement family became `Mgen_nn.Placement`; the int16 rails are named
once and the clamp's circuit is `Mgen_nn.Quantized.Rtl.clamp16`, with a
48-bit gate the frozen eras' own clamps would fail; the twin takes P,
and the stream gate runs at P 8 and P 48 — coverage, not time: the RTL
gates cost 9.9 s before and 10.6 s after. 148 tests became 162.

What waits for the all-era branch: the same cut for eras four and five,
the frozen eras' adoption of the shared clamp and placement, the alpha
ROM as `Signal.rom`, the frames as interfaces, one safetensors reader,
the Exp2 fork's backport, and the walk under `lax.scan`. The branch
merged into develop with this entry.

## 2026-08-29 — the all-era cut (feat/all-era-cut)

**Eras four and five took era six's shape.** The OCaml float models and
the OCaml int8 twins of the two frozen eras are gone with their tools
and their drift gates — 7,431 lines out over 54 files — and each era
keeps ONE `Model`, the module the circuit reads. The twins are
`jax/transformer/quantized.py` and `jax/mamba/quantized.py`, the
quantizers write the contract files, and the RTL gates run under
`uv run pytest` through `bin/gate_<era>.exe`, as era six's do. `Top`
takes its source as a parameter, thus one pytest gate elaborates each
era's netlist and the three md5 pins are checked by machine and not by
hand.

**Nothing moved that must not move.** The three netlist md5s stand at
`4e367cef` (six), `a648db22` (five) and `4ab6e292` (four) — every pin
re-measured at the head of the round and unmoved at its end. Both
frozen G0 losses are bit-identical through the Flax rewrite: 1.628177
for era four and 1.640810 for era five, the same six decimals the
deleted OCaml reference read. The two drift tables were RE-MEASURED on
a JAX draw rather than copied, because the parameter draw order is not
the OCaml one; at the old gates' own shapes the cosine holds between
0.9962 and 0.9981 against the old 0.9960 to 0.9987, and era five's long
walk reads 0.927 top-1 at 1024 steps with every clamp share zero.

**Then the consolidation.** The step-frame eras are ONE RECIPE:
`jax/frames.py` holds the loop, the evaluation, the checkpoint policy
and the audition tail that `transformer/train.py` and `mamba/train.py`
had carried as two copies differing in twelve lines; each trainer is
now its CLI and its draw. `jax/fixed.py` takes the integer rules of the
twins out of `nn.py` — the rails, the exponent rule, the sampling
policy, the tables and the integer draw, 420 lines — so that the Python
side is laid out as `lib/nn` is, where `quantized.ml` is its own
module. Era six's cell order is one `model.cell_order(steps)` with
seven readers where six loops had restated it. The two corpus export
paths are `data.FRAMES` and `data.PIECES`, stated once where nine sites
had spelt them. `midi.playback_options` is the six audition flags of
all three eras.

**The levers nothing named went**: `--train-on` and `--average-top`
from both frozen trainers with the top-K sort and the `-avg` write;
`Block.selective_window_chunked` with its twelve-case gate, whose 1.17×
at T256 never cleared the 1.5× bar and whose reason expired with the
freeze; and eight dead OCaml exports — `Prng.bernoullis` deleted
outright, the rest un-exported. E501 is selected in `pyproject.toml`
and the 343 lines over 90 columns are rewrapped: the Python side lints
where the OCaml side formats, and `ruff format` stays unadopted.
