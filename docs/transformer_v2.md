# The transformer v2

## Scope

This document gives the design of the second transformer of the project.
One step of music becomes one position of the network and one 32-bit
word on the wire — the step frame — and the four voices of that step
keep their names from the corpus to the synthesizer.

The design has two halves, and one of them is built:

- **The packed stream, built.** The corpus is one stream for each split:
  piece, seam, piece. The walk has no boundary, the clock rolls, and the
  boot is a lead-in of silence. `Jsb.pack`, `Jsb.rotation`, `Texture` and
  the sampler of `Transformer` carry it now. This document keeps the
  design that `docs/improviser.md` states and does not repeat its
  reasoning; where the two disagree, this document is the later one.
- **The step frame, not built.** The frame replaces the token sentence
  and the note-at-a-time socket. Everything below that is new is here.

When the code lands, these sections of the other documents become wrong
and a change must correct them:

| Document | Section |
|---|---|
| `docs/transformer_model.md` | The token, The step sentence, The socket, The legality mask |
| `docs/transformer_model.md` | The tokenizer — the order and the escape rules only |
| `docs/transformer_rtl.md` | The socket, The piece boundary |
| `docs/pink_rtl.md` | The socket |
| `docs/improviser.md` | The token vocabulary — the frame has no token |

The RTL block design of the model comes in a later document, as before.

## Why the frame

The socket carries one note at a time. A step gives zero notes or up to
four, and the sequencer takes them in a handshake loop. The transformer
above it gives a sentence: the OFF events, the ON events, then END. The
count of tokens in a sentence is not constant.

Three costs come from that.

**The worst case sizes the design.** One token is one pass of the
network. The corpus needs 2.673 passes for each step at the mean, and 9
at the worst. The worst case must hold inside `step_ms`. The frame makes
the work of a step constant: one pass, always.

**The window holds too little music.** 256 tokens hold 96 steps — 6
bars. One chorale of the 382 fits in that window, and it fits by two
tokens. 256 steps hold 16 bars, and 247 of the 382 chorales fit whole.

**The reader throws the voices away.** The corpus file separates the
four voices, and the reader merges each step into a sorted pitch set.
The frame keeps the voice. The model then learns the soprano, the alto,
the tenor and the bass as four lines, and voice leading is the craft of
a chorale.

## The step frame

One voice code is one byte:

| Bits | Field | Values |
|---|---|---|
| 7 | sounds | 1 = the voice sounds a pitch, 0 = the voice is silent |
| 6:0 | pitch | 0 to 127, the MIDI pitch |

The frame is four voice codes in one 32-bit word. Seat 0 takes the low
byte. Seat 0 is the lowest voice and seat 3 is the highest. The corpus
file gives the soprano first, thus the reader turns the order around:
the soprano takes seat 3 and the bass takes seat 0.

The flag gives three properties:

- A silent voice is the code `0x00`, thus a cleared frame is silence and
  a cleared context memory reads as silence.
- The pitch field holds all of 0 to 127. No pitch is reserved, thus the
  reader moves no note and the MIDI range is the MIDI range.
- The circuit reads the flag as a bit and not as a compare.

The codes `0x01` to `0x7F` have the flag clear, thus they are silence
with a pitch field that no reader writes. They are 127 spare codes.

They keep one door open. This frame states which pitch a voice holds,
and it cannot state that the voice strikes the same pitch again: the
code of a held note and the code of a repeated note are the same. The
JSB corpus does not mark a repeated note, thus nothing is lost today. A
later corpus that marks one can take a spare code for "strike the pitch
of the step before". This design does not use them.

There is no START code and no END code. The context starts at silence,
and the frame of a step is one word: no code must open a walk, and no
code must close a sentence.

The channel, the velocity and the time stay out of the frame. The
control registers hold the channel and the velocity. The step clock
holds the time.

**The wire is general and the model is not.** The frame states any MIDI
pitch. The vocabulary of the model is sized to the corpus, below. The
two are different questions and the code must keep them apart.

## The decode

The sequencer holds the set of pitches that sound. Each step gives a new
frame. Let `sounding` be the set that sounds now, and let `wanted` be
the set of pitches of the frame. Then:

- The releases are `sounding` minus `wanted`.
- The strikes are `wanted` minus `sounding`.
- The sequencer sends **all releases, then all strikes**.

The sequencer sends each message in seat order inside its pass. The two
passes are the rule; the order inside a pass is free.

The sets are small. Four slots hold the sounding pitches, and the frame
holds four pitches. Therefore each membership test is a compare of one
pitch against four, and the whole decode is 38 compares of eight bits:
sixteen for the releases, sixteen for the strikes, and six that find a
unison inside the frame.

### Why the sets, and not the seats

A rule that walks seat by seat, and closes the note of a seat before it
opens the new one, is wrong here. Two cases break it.

**The exchange.** Seat 1 moves to pitch 64 while seat 2 leaves pitch 64.
A seat walk sends the Note On of 64 before the Note Off of 64. The synth
stops the new note, because the four voices share one MIDI channel and a
Note Off releases a note by pitch. The corpus holds this case at 2.77
percent of the step boundaries, and every one of the 382 chorales holds
at least one. Two passes make the case safe: the release of 64 goes
first, thus the strike of 64 finds the pitch free.

**The unison.** Two voices hold one pitch. The corpus holds a unison at
7.97 percent of the steps. A set holds each pitch one time, thus the
decode sends one Note On and one Note Off for it. A seat walk sends two
of each, and the second Note Off stops nothing while the second Note On
is a strike of a pitch that already sounds.

The set rule also gives the safety rules for nothing:

- The decode never strikes a pitch that sounds, because a strike comes
  from `wanted` minus `sounding`.
- The decode never releases a pitch that must stay, because a release
  comes from `sounding` minus `wanted`.
- Five notes never sound at the same time. The releases only make the
  set smaller, and the strikes then fill it to four or less, because the
  frame has four slots.

The legality mask held the first two rules and the seat count held the
third. The decode holds all three now, thus **the mask leaves the
design**. `Sounding_state` and its registers — the sounding vector, the
last ON, the last OFF and the seat count — leave the RTL with it, and
the model draws no illegal code because no frame is illegal.

The sequencer keeps the set and does not compare the frame against the
frame before. The two rules agree while nothing else moves the notes,
and the set is the rule that stays true if something does. The stop
sweep needs the set in any case: it releases what sounds.

`Pink` keeps its four disjoint registers. That is a rule of the pink
model and it stays. The decode no longer needs it.

## The socket

```ocaml
(** the number of voices of the synthesizer *)
let voices = 4

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; rewind : 'a  (** a strobe: go to the origin of the sequence *)
    ; step : 'a    (** a strobe: take one step and give its frame *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { frame : 'a [@bits 8 * voices]
      (** the four voice codes; seat 0 is the low byte *)
    ; valid : 'a  (** a strobe: [frame] holds the frame of the step *)
    ; idle : 'a   (** 1 when the source is at rest and can take a command *)
    }
  [@@deriving hardcaml]
end
```

`Note` leaves the interface, and `ready` leaves it with the note. The
sequencer strobes `step` and then waits, thus it is always ready and no
handshake is necessary. `valid` answers `step`, one time for each step.
`idle` answers `rewind`.

Every step gives a frame. A step where nothing sounds gives the frame of
four zeros; it does not give silence on the socket. Therefore `valid`
comes one time for each `step`, and the sequencer counts on it.

`Pink.Source` becomes smaller. It holds the four notes in registers
already, and its report state, its owed flags and its lowest-voice fold
exist only to give one note at a time. The frame takes the four
registers as they are.

## The corpus

### What does not change

The packed stream is built and this design keeps it whole:

- One stream for each split: piece, seam, piece, seam. The splits never
  mix.
- The seam is the smallest count of empty steps that puts the next
  downbeat on the clock, and it is never zero. Every piece is a whole
  number of quarter notes and every rotation is one, thus a seam is 4,
  8, 12 or 16 steps. No rule states this; the arithmetic gives it.
- One rolling clock of 16 steps. `Jsb.rotation` votes the pickup of a
  piece at 16 steps, and the bar length left the corpus: 71.9 percent of
  the cadences of the corpus land on a downbeat under one clock, and to
  remove the 27 odd-metre pieces buys 1.9 points and costs 8.1 percent
  of the steps.
- Eight streams for each split. Stream zero is canonical — every piece
  at shift zero, in the order of the corpus — and the referee reads it
  alone, thus Gate A and Gate B stay deterministic. The other streams
  take a uniform permutation and a uniform legal shift for each piece.
- `positions` holds the coordinate of each position, and the reader
  takes the bar phase from its low four bits and the frame from its high
  four.
- A shift is legal while each voice stays inside the observed range of
  that voice, and the transposition happens for each piece, before the
  packing.

### What changes

The reader stops translating. One step of the file is four cells indexed
by voice, and one frame is four codes indexed by seat. The whole
transform is: turn the order around, send −1 to `0x00` and a pitch p to
`0x80 lor p`, then pack the four bytes.

The seam becomes simpler than it is today. Under the token sentence the
tokenizer had to make the release of the last chord, because a piece
ends with its four voices still sounding. A seam of frames states
silence, and the decode makes the release for nothing. The count of
steps in a seam does not change.

The parts that made a representation leave:

| Function | Why it goes |
|---|---|
| `escape_reserved` | no code is reserved now |
| `flatten_cells` | the frame keeps the voices; it does not merge them |
| `tokenize` | there is no sentence to build |
| `step_of_each` | one step is one position; no map is necessary |
| the mask words of `Evaluation` | no frame is illegal |

The parts that measure the music stay, and they do not change:
`cadential_holds`, `vote` and `rotation`, because the file does not say
where the bar lines are; and `voice_ranges` with the transposition
policy.

`Token` becomes a pair of small maps: the class index of a wire code,
and the wire code of a class index.

### The boot

The walk boots from a lead-in of silence: one bar of silent frames, then
the draw. This is measured and settled. On the first packed model, over
12 seeds, the first ON fell at step 16 for six seeds, 20 for two, 24 for
two and 28 for two — every seed inside one bar of the end of the
lead-in, and every opening on a multiple of four steps. **The model
opens the music itself**, thus the boot needs no pitch, no range and no
table.

The frame changes nothing here. A silent step is a silent frame.

## The model

### The input: one table for each seat

Four codes must become one vector of the residual stream. Each seat
reads its own table, and the four rows sum:

```
e = E[3][c3] + E[2][c2] + E[1][c1] + E[0][c0] + phase + frame
```

**A shared table with a voice tag cannot work, and the reason is
arithmetic and not capacity.** Every step carries all four seats, thus
the sum of the four tags is the same vector at every position — a
constant, which is a bias and carries nothing. What remains is
symmetric in the four codes: a soprano on 72 with a bass on 48 and a
soprano on 48 with a bass on 72 give the identical vector. The design
would throw the voices away on the way in, which is one of the three
reasons for the frame. Four tables break the symmetry, and the voice
tag is then not necessary anywhere.

**Each table is tied to the head that draws its seat.** The table that
reads a voice is the table that writes it, which is the argument for the
tied table of today, one time for each seat.

### The vocabulary

The head draws over 48 classes. Class 0 is silence, and class 1 + i is
the pitch 36 + i. The corpus states the pitches 36 to 81, and the shift
rule holds each voice inside its own observed range, thus 47 classes
cover the music and the 48th is spare.

The class index and the wire code are not the same number. The flag
makes the wire code cheap for the circuit, and the class index makes the
table small for the model. The map is one subtraction.

The table shapes are fixed when the bitstream is elaborated, as the
weights are. A corpus with a wider range takes a wider table and a new
bitstream; nothing at run time reads the size.

**The size is what sets 48.** Four tables of 129 rows put the six-layer
model at 99 percent of the block RAM of the device, which does not fit.
Four tables of 48 rows put it at 93 percent, under the six-layer design
that plays on the board today.

### The head

One step gives four voice codes. The head draws them in a **chain**, and
not in parallel:

```
h3 = h                   logits(seat 3) = E[3] · rms(h3)
h2 = h3 + E[3][c3]       logits(seat 2) = E[2] · rms(h2)
h1 = h2 + E[2][c2]       logits(seat 1) = E[1] · rms(h1)
h0 = h1 + E[1][c1]       logits(seat 0) = E[0] · rms(h0)
```

`h` is the output of the last layer at that step. `c3` to `c1` are the
codes the chain has drawn already, and the training pass takes the true
codes, thus all four heads run in one pass with no sampling.

The chain runs from seat 3 down to seat 0: soprano, alto, tenor, bass.
The order keeps the one decision the ear accepted: the top voice is
chosen first, and it conditions on no voice under it, as the music is
written.

The chain is necessary. Four heads that draw in parallel make the voices
conditionally independent, and a chord is a joint choice. The cost of
independence is 1.263 bits for each step, held out, and 2.464 bits over
the quarter of the boundaries where two or more voices move. The chain
removes that cost for **no parameters at all** — parallel heads need the
same four tables — and three adds of a vector.

The chain is exact and not an approximation. Any joint distribution is a
product of conditionals, in any order.

What the chain adds is also what the next position reads. Write
`a[v] = E[v][c[v]]`; then the input embedding of step t+1 is
`a3 + a2 + a1 + a0`, and the stream at the bass head is `h + a3 + a2 +
a1`. The chain assembles the next frame's embedding, one voice at a
time, and needs no table of its own to do it.

**The one restriction, stated:** each conditional is an additive step
into the residual stream, with a normalization and a projection and no
non-linearity between the seats. The factorization is exact; this form
of it is not free to represent any conditional. If the measured joint
gain does not appear, the fallback is a small map for each seat on the
accumulated context, before the readout.

### The position signal

One rolling coordinate, modulo 256, in the factored form the hardware
already has: the bar phase is `step mod 16` and the frame is
`step / 16 mod 16`. Two tables of 16 rows add row for row. One table of
256 rows would cost eight times the block RAM for the same information.

The coordinate is unique inside a 256-step window, thus it is a ruler.
Attention returns a weighted sum of values and not a distance, therefore
a model that finds the seam in its context cannot read how far back it
is unless the position carries a coordinate. The content gives the
origin and the coordinate gives the measure. This is the reason to keep
it, and the ablation below is the test.

### ALiBi

The slopes do not carry over. `slope_span` 8 was settled when a position
was one token, and a token is 1/2.673 of a step at the mean. A position
is now a whole step, thus every head reaches about 2.7 times further in
music at the same span. No single span rescales the four slopes by one
factor, because the set is geometric in the head index. **Re-test the
span; do not convert it.**

### The rest of the network

Nothing else changes. The model stays a decoder with no bias terms,
RMSNorm before each sublayer, ALiBi for the position, and
`d_ff = 4 * d`. The floor of two layers stays: a transformer repeats a
motif with an induction circuit, and that circuit needs two attention
layers.

## The sizes

int8 weights, `T` 256 positions, `d` 64. A RAMB36 tile holds 4,096
bytes at eight bits wide, and the device holds 135 tiles. The model
below is validated against the routed builds of today: the same
arithmetic gives 47 tiles for the two-layer king and 127 for the
six-layer model, which is what the tools report.

| | two layers | six layers |
|---|---|---|
| the layers | 98,304 B | 294,912 B |
| four voice tables, 48 rows | 12,288 B | 12,288 B |
| bar phase and frame | 2,048 B | 2,048 B |
| **the weights** | **112,640 B — 27.5 tiles** | **309,248 B — 75.5 tiles** |
| the KV rings, int8 | 65,536 B — 16 tiles | 196,608 B — 48 tiles |
| the small RAMs | 2.5 tiles | 2.5 tiles |
| **total** | **46.0 tiles — 34%** | **126.0 tiles — 93%** |
| today, for comparison | 47 tiles | 127 tiles |

The six-layer model therefore fits one tile under the design that plays
now, and that build meets the clock at +0.110 ns.

The compute of one step:

| | today | with the frame |
|---|---|---|
| mean | 482 K multiplies | 176 K multiplies |
| worst | 1,622 K multiplies | 176 K multiplies |

The mean falls 2.7 times. The worst falls 9.2 times, and the worst is
the number the step period must hold. At six layers the worst step of
today measures 61 ms of a 200 ms period; the frame takes it near 7 ms.
Compute stops being the constraint on the tempo, and the wire becomes
it.

The wire does not change. The worst step sends four Note Offs and four
Note Ons: 24 bytes, 7.68 ms at 31250 baud. The floor of `step_ms` stays
8. The worst case is common and not rare — all four voices move at 7.6
percent of the step boundaries.

## The measurements

The numbers come from `corpus/JSB-Chorales-dataset/Jsb16thSeparated.json`:
382 chorales, 92,536 steps. The train split is 229 chorales and 55,228
steps.

| Measure | Value |
|---|---|
| tokens for each step, the encoding of today | 2.673 mean, 9 worst |
| steps in a piece | 228 median, 100 to 640 |
| pieces inside 256 steps | 247 of 382 = 65% |
| pieces inside 256 tokens | 1 of 382 |
| steps with a unison | 7,375 = 7.97% |
| boundaries with an exchange | 2,552 = 2.77%, in all 382 pieces |
| boundaries where all four voices move | 6,992 = 7.6% |
| voice slots that repeat the step before | 77.91% |
| pitches in the corpus | 36 to 81 |
| pieces that vote a 16-step bar | 226; 27 vote 12; 129 cannot vote |
| cadences on a downbeat, one rolling clock | 71.9% |

### The cost of independent voices

Two count models give the number. Both fit on the train split and score
on the valid split. Both take the same context — the step before — and
the same smoothing. The only difference is the independence.

Cross entropy on the valid split, in bits for each step:

| Context | Joint | Independent | Cost |
|---|---|---|---|
| none | 10.892 | 15.065 | 4.173 |
| one step | 5.361 | 6.624 | 1.263 |
| one step, contexts seen 20 times or more | 3.474 | 4.369 | 0.895 |
| one step, two or more voices move | 10.774 | 13.238 | 2.464 |
| one step, one or zero voices move | 3.550 | 4.411 | 0.861 |

The levels are high because a count model is weak. The difference is the
measure, and not the level.

The cost stands where the music is. Two or more voices move at 25
percent of the boundaries, and the cost there is 2.464 bits. That is the
part a listener hears, thus a small mean would not make the parallel
head safe.

## The traps

**The loss does not carry across the encoding.** The frame gives four
predictions for each step and the sentence gives 2.673 tokens, thus a
loss for each prediction divides against a different count. Report
**nats for each step** and never nats for each prediction.

Even nats for each step do not compare the two encodings. The frame
predicts which voice holds which pitch, and the sentence predicts only
the set, thus the frame answers a harder question by a small amount.
Use the loss to rank inside one encoding. The ear ranks across.

**77.91 percent of the predictions repeat the step before.** They are
easy, they dominate the mean, and they invite a second failure: a model
that holds its chord for ever scores well on a mean and plays a drone.
Report a second number over the steps where two or more voices move, and
read `Texture` for the movement, not the loss.

**The frame does not cure the decay.** The packed stream and the seam
do, and they are already built. The frame changes what a step is; it
does not change what the walk converges to.

## The tests

The rules of the project hold: expect tests in the module, waveform
tests where a waveform shows the behaviour, Cyclesim block by block, and
the comparison of the reference against the circuit.

The decode needs one expect test for each case of the rule:

| Case | Frame before | Frame now | Messages |
|---|---|---|---|
| hold | p | p | none |
| strike | silent | p | On p |
| release | p | silent | Off p |
| move | p | q | Off p, On q |
| exchange | seat 1 = p, seat 2 = q | seat 1 = q, seat 2 = p | none |
| unison arrives | seat 1 = p, seat 2 = q | seat 1 = p, seat 2 = p | Off q |
| unison leaves | seat 1 = p, seat 2 = p | seat 1 = p, seat 2 = q | On q |
| seam | a chord | four silent codes | four Note Offs |

Three properties hold over any stream of frames, and a test must state
them:

- No Note On names a pitch that sounds.
- No Note Off names a pitch that does not sound.
- Four notes sound at the most, at every cycle.

The corpus reader needs an expect test of a small chorale: the frames,
the seam and the coordinate over the seam into the downbeat.

`Texture` measures events and not frames, thus the frame walk reaches it
through the decode. The instrument does not change.

The stream comparison stays. The reference and the Cyclesim sequencer
must give the same messages, byte for byte, from one seed.

## The open questions

1. **The input.** The four tables are settled. The open part is the
   chain: if the measured joint gain does not appear, add a map for each
   seat before the readout.
2. **The window position.** Keep it or drop it. The prediction, written
   before the run: the loss sees nothing either way, and the ear prefers
   the model that keeps it, most audibly in how a piece paces itself
   toward its close. The bar phase is settled and is not part of this
   question.
3. **`T` 512 against `T` 256, in steps.** The old result — a longer
   window made the music worse — was measured at 512 tokens, which is
   191 steps and less than one chorale. It does not carry over, and it
   does not carry against.
4. **The sampler.** The temperature and `min_p` were tuned against a
   masked distribution, and the model put 84 percent of its raw mass
   outside the legal set. No frame is illegal, thus the numbers do not
   carry over. Tune them again.
5. **ALiBi.** The span, per the section above.
6. **The repeated note.** The spare codes hold the door open. Nothing in
   this corpus asks for it.

## The plan

The ear decides, and it decides on the host. The RTL follows the ear and
not the other way around.

1. **The JAX prototype.** The reader gives frames from the packed
   stream; the model takes the four tables, the chained head and the 48
   classes; the loss reports nats for each step. Train the recipe of the
   era, two seeds, because one seed cannot settle an ear comparison.
2. **The measurement.** `Texture` over a long walk — 8192 steps or more,
   twelve seeds — against the same numbers over the canonical packed
   stream. The question is one question: do the windows hold their first
   values?
3. **The ear.** Against the model that plays today. The prototype needs
   a play path of its own: the frames and the decode rule make MIDI
   messages, thus a small sender is enough and the OCaml player can
   wait.
4. **The reference.** `Token`, the reader, the decode and the model in
   OCaml, with the gates of the project: the expect tests, the drift
   report and the stream comparison.
5. **The socket, the sequencer and `Pink`.** This part stands alone and
   carries no risk to the model. It can happen at any time after step 1,
   and the existing Cyclesim comparison tests it.
6. **The circuit and the board.** The block design comes in its own
   document, as before.
