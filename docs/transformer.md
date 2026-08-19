# The transformer model

## Scope

This document gives the design of the model of era four. One step of
music is one position of the network and one 32-bit word on the wire —
the step frame — and the four voices of that step keep their names from
the corpus to the synthesizer.

The design has two halves. The corpus and the model are built, trained
and elected by ear; the OCaml reference, the socket and the circuit are
not. Each section states which of the two it is in.

The design keeps the project rules. The board does the inference. The
bitstream initializes the weights. The host trains the model. At MIDI
rates the compute is never the limit; the block RAM is. Therefore the
design spends its care on memory and on simplicity, not on throughput.

The RTL block design of era four comes in a later document.
`docs/transformer_rtl.md` gives the circuit of era three, which plays on
the board today.

## Why the frame

Era three carried one note at a time. A step gave zero notes or up to
four, and the sequencer took them in a handshake loop. The model above
it gave a sentence of tokens: the OFF events, the ON events, then END.
The count of tokens in a sentence was not constant.

Three costs came from that.

**The worst case sized the design.** One token was one pass of the
network. The corpus needs 2.673 passes for each step at the mean, and 9
at the worst. The worst case must hold inside `step_ms`. The frame makes
the work of a step constant: one pass, always.

**The window held too little music.** 256 tokens hold 96 steps — 6 bars.
One chorale of the 382 fits in that window, and it fits by two tokens.
256 steps hold 16 bars, and 247 of the 382 chorales fit whole.

**The reader threw the voices away.** The corpus file separates the four
voices, and the reader merged each step into a sorted pitch set. The
frame keeps the voice. The model then learns the soprano, the alto, the
tenor and the bass as four lines, and voice leading is the craft of a
chorale.

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

Era three held the first two rules with a legality mask, and the third
with a seat count. The decode holds all three now, thus **the mask
leaves the design**. `Sounding_state` and its registers — the sounding
vector, the last ON, the last OFF and the seat count — leave the RTL
with it, and the model draws no illegal code because no frame is
illegal.

The sequencer keeps the set and does not compare the frame against the
frame before. The two rules agree while nothing else moves the notes,
and the set is the rule that stays true if something does. The stop
sweep needs the set in any case: it releases what sounds.

`Pink` keeps its four disjoint registers. That is a rule of the pink
model and it stays. The decode no longer needs it.

## The socket

Not built. It lands with the circuit of era four — step 5 of the plan —
because one interface serves every model and the source of era three
sits on this one today.

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

The wire is the constraint on the tempo. The worst step sends four Note
Offs and four Note Ons: 24 bytes, 7.68 ms at 31250 baud. Therefore the
floor of `step_ms` is 8. The worst case is common and not rare — all
four voices move at 7.6 percent of the step boundaries.

## The corpus

Built. `Jsb.pack` and `corpus_tool export` carry it.

### The packed stream

The corpus is one stream for each split: piece, seam, piece, seam, and
so on. The splits never mix. There are no pieces in the stream, and
there is no boundary in the walk.

**The packing is the change that carries the design, and the reason is
the quiet.** A chorale is full of movement toward quiet: the cadences,
the fermatas, the wind-down at the end. The movement back — silence,
then a new piece — is at the boundary between two pieces, and a training
window of a corpus of separate pieces never crosses one. Such a corpus
teaches the roads into the quiet and no road out. The packed stream
states what follows the quiet, thus the model learns the transition it
has never seen, and the walk needs no counter to start the next piece.

### The seam

The seam is the count of empty steps between the last step of a piece
and the first step of the next. It is the smallest count that puts the
next piece on its rotation, and it is never zero.

The arithmetic gives its size: a piece is a whole number of quarter
notes and a rotation is a whole number of quarter notes, thus the seam
is 4, 8, 12 or 16 steps. The quiet of a seam is never shorter than a
quarter note and never longer than a bar. No rule states this.

A seam of frames states silence, and the decode makes the release of the
last chord for nothing. Era three had to write that release into the
corpus, because a piece ends with its four voices still sounding.

### The metre of the corpus

The measurement, over all 382 pieces and 92,536 steps:

| | pieces | steps |
|---|---|---|
| a 16-step bar | 226 | 64.4% |
| a 12-step bar | 27 | 8.1% |
| too few holds to vote | 129 | 27.5% |
| a pickup | 211 | — |

Every rotation is 0, 4, 8 or 12 steps, and every piece is a whole number
of quarter notes.

One rolling clock of 16 steps holds this corpus. Each piece takes the
rotation that puts the most of its cadences on a downbeat, and the seam
places it there. The cadences on a downbeat are then 73.8 percent of
those of the 16-vote pieces, 44.6 percent of those of the 12-vote
pieces, and 71.9 percent of all. To remove the 27 odd-metre pieces moves
the last number to 73.8 percent and costs 8.1 percent of the steps. That
is a bad trade, and the votes it would act on are weak: 8 of the 27 win
by a lift of 4 and 4 more by a lift of 16, near-ties on 2 to 4 holds.
**The corpus needs no rule for the odd metre.**

The bar length is therefore not in the corpus. `Jsb.rotation` votes the
pickup of a piece at one bar length of 16 steps.

### The streams

A piece takes one of its legal shifts each time it enters a stream, thus
one stream is one draw and a split needs more than one. The export
writes eight streams for each split: a piece has 7.4 legal shifts at the
mean, thus eight streams hold the step count of the corpus.

Stream zero is the canonical one — every piece at shift zero, in the
order of the corpus. The referee reads it alone, thus a gate between two
trainers stays deterministic. The other streams take a uniform
permutation of the pieces and a uniform legal shift for each.

The draw of a training row is: a uniform stream, then a uniform window.
The piece draw is not uniform — a long piece holds more windows than a
short one — and this is the objective the endless walk asks for.

A shift is legal while each voice stays inside the observed range of
that voice in the corpus: soprano 60 to 81, alto 52 to 74, tenor 46 to
69, bass 36 to 66. The transposition happens for each piece, before the
packing.

### The reader

One module turns a corpus into frames. The trainer, the reference model
and the tests all read that one module: one definition of the walk.

| Policy | Rule |
|---|---|
| tracks | note events only; drop channel 10 |
| tempo | drop the tempo meta events; `step_ms` owns tempo |
| grid | one step is a sixteenth: PPQ / 4 ticks |
| quantize | each event moves to the nearest grid line |
| short notes | a note below half a step becomes one step |
| voices | four voices, one seat for each |
| silence | a voice that does not sound gives the code `0x00` |

The JSB corpus reads from the voice-separated file,
`Jsb16thSeparated.json`. One step of that file is four cells indexed by
voice, and one frame is four codes indexed by seat. The whole transform
is: turn the order around, send −1 to `0x00` and a pitch p to
`0x80 lor p`, then pack the four bytes.

The gate restores a part of the articulation that the grid removes, thus
the round-up of short notes is safe.

The parts that measure the music stay: `cadential_holds`, `vote` and
`rotation`, because the file does not say where the bar lines are; and
`voice_ranges` with the transposition policy above.

### The boot

The walk boots from a lead-in of silence: one bar of silent frames, then
the draw. Attention needs one position, and a silent one is a position
the model trained on at every seam.

This is measured and settled. Over 12 seeds, the first ON fell at step
16 for six seeds, 20 for two, 24 for two and 28 for two — every seed
inside one bar of the end of the lead-in, and every opening on a
multiple of four steps, which is the grid the seams teach. Half of them
open on the downbeat itself. **The model opens the music itself**, thus
the boot needs no pitch, no range and no table.

## The model

Built, in `jax/transformer`.

### The input: one table for each seat

Four codes must become one vector of the residual stream. Each seat
reads its own table, and the four rows sum:

```
e = E[3][c3] + E[2][c2] + E[1][c1] + E[0][c0] + phase
```

**A shared table with a voice tag cannot work, and the reason is
arithmetic and not capacity.** Every step carries all four seats, thus
the sum of the four tags is the same vector at every position — a
constant, which is a bias and carries nothing. What remains is symmetric
in the four codes: a soprano on 72 with a bass on 48 and a soprano on 48
with a bass on 72 give the identical vector. The design would throw the
voices away on the way in, which is one of the three reasons for the
frame. Four tables break the symmetry, and the voice tag is then not
necessary anywhere.

**Each table is tied to the head that draws its seat.** The table that
reads a voice is the table that writes it, which is the argument for the
tied table of era three, one time for each seat.

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

**The window belongs to the corpus and not to this model.** The corpus
states the pitches, and the shift rule holds each voice inside them;
this model states only how many tables read the window, and a later
model of another kind reads the same one. Therefore `Vocab` sits in the
corpus library beside `Jsb`, and an expect test there holds the observed
range of the corpus inside the window.

**The size is what sets 48.** Four tables of 129 rows put the six-layer
model at 99 percent of the block RAM of the device, which does not fit.
Four tables of 48 rows put it at 93 percent, under the six-layer design
of era three.

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
The order keeps the one decision the ear accepted in era three: the top
voice is chosen first, and it conditions on no voice under it, as the
music is written.

The chain is necessary, and it is measured. Four heads that draw in
parallel make the voices conditionally independent, and a chord is a
joint choice. **Parallel heads cost 0.3157 nats for each step — 0.456
bits, sixteen times the seed spread.** The chain removes that cost for
**no parameters at all** — parallel heads need the same four tables —
and three adds of a vector.

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
of it is not free to represent any conditional. The measured gain
appeared in full, thus the restriction costs nothing that can be seen
today. If a later corpus asks for more, the fallback is a small map for
each seat on the accumulated context, before the readout.

### The position signal

One table of 16 rows holds the bar phase, `step mod 16`. The row adds to
the sum of the four seat rows.

The window position — a second table of 16 rows over `step / 16 mod 16`
— was designed, built and **removed**. It is free on the loss and the
ear prefers the model without it. The export still writes one rolling
coordinate modulo 256, and the reader takes the bar phase from its low
four bits; the high four bits are the window position and no reader
takes them. That costs nothing, and it keeps one export honest for a
model that wants either.

### ALiBi

The slope of head k is `2 ** -(span * (k + 1) / heads)`, and the span is
**4**.

The span did not carry over from era three, where 8 was settled when a
position was one token. A token is 1/2.673 of a step at the mean, thus a
position now reaches about 2.7 times further in music at the same span.
No single span rescales the four slopes by one factor, because the set
is geometric in the head index.

The measurement elected 4, and it is the most robust number of the era.
The means of 4 and 8 are a dead heat; **the variance is the finding**:

| | mean | sd | worst |
|---|---|---|---|
| 48k, span 8 | 1.6404 | 0.0100 | 1.6496 |
| 48k, span 4 | 1.6381 | **0.0019** | 1.6394 |
| 96k, span 8 | 1.6272 | 0.0108 | 1.6377 |
| 96k, span 4 | 1.6276 | **0.0016** | 1.6288 |

Six seeds at each budget, and the result replicates at both. The spread
is 5 to 7 times tighter.

The mechanism agrees with the ear. The slopes of span 4 are 1/2, 1/4,
1/8 and 1/16, thus every head is local: the gentlest of them still costs
16 logits at 256 steps. A gentle slope lets each seed hold on to
whatever distant structure its initial values favour, and a steep one
holds every head near the bar, thus the seeds agree. The ear reads span
4 as more balanced and smooth.

A gentler span is worse, and flat is worst: span 16 costs 0.023, span 24
costs 0.044, and span 64 — which is no ALiBi — costs 0.053. Era three
paid 0.105 for the same flattening. The frame retires a job its steep
head used to do: the position of a token in a sentence was a voice rank,
and a frame states the voice.

### The rest of the network

The model is a decoder with no bias terms, RMSNorm before each sublayer,
ALiBi for the position, and `d_ff = 4 * d`. The trainer folds the
RMSNorm scale into the next matrix. The floor of two layers stays: a
transformer repeats a motif with an induction circuit, and that circuit
needs two attention layers. Music lives on repetition.

The context is `T` positions. The KV cache is `2 * L * T * d` bytes, and
the step period is long, thus the recompute datapath with no cache is
also legal. That choice belongs to the RTL document.

## The draw

Built.

One generator makes every random number of the model: `Prng`, the
xorshift32 of the circuit. The initial parameters, the dropout masks and
the sampler all draw from it. Therefore one seed names one sequence in
the software, in the simulation and on the board, and that seed is the
value of the SEED cell of the host control.

One step of the walk gives eight bits, and one uniform takes three
steps. The grid of `2 ** -24` holds the tail of a draw that one byte
would cut at 3.3 sigma. A seed from a flag or from a stream folds into
the 32 bits of the state; a seed already inside the range names itself,
so seed 7 is the walk of the board's seed 7.

A part of the model that must draw on its own takes an independent walk.
Each dropout block takes one. Therefore the block holds no place in the
order of the parent, and the parent can gain or lose draws while the
masks stay.

The sampler tempers the logits, applies a min-p floor as a share of the
peak, and walks the cumulative weights. **Temperature 1.0 and min_p
0.05**, elected by ear over a sweep of temperature 0.7 to 1.3 against
min_p 0.0039 to 0.15. No mask stands before the draw, because no frame
is illegal.

The numbers of era three do not carry over. That sampler drew from a
masked distribution, and the model put 84 percent of its raw mass
outside the legal set.

## The sizes

int8 weights, `T` 256 positions, `d` 64. A RAMB36 tile holds 4,096 bytes
at eight bits wide, and the device holds 135 tiles. The model below is
validated against the routed builds of era three: the same arithmetic
gives 47 tiles for its two-layer model and 127 for its six-layer model,
which is what the tools report.

| | two layers | six layers |
|---|---|---|
| the layers | 98,304 B | 294,912 B |
| four voice tables, 48 rows | 12,288 B | 12,288 B |
| bar phase | 1,024 B | 1,024 B |
| **the weights** | **111,616 B — 27.5 tiles** | **308,224 B — 75.5 tiles** |
| the KV rings, int8 | 65,536 B — 16 tiles | 196,608 B — 48 tiles |
| the small RAMs | 2.5 tiles | 2.5 tiles |
| **total** | **46.0 tiles — 34%** | **126.0 tiles — 93%** |
| era three, for comparison | 47 tiles | 127 tiles |

The six-layer model therefore fits one tile under the design that plays
now, and that build meets the clock at +0.110 ns. The 1,024 bytes that
the window position gave back do not buy a tile: they buy margin inside
one.

The compute of one step:

| | era three | with the frame |
|---|---|---|
| mean | 482 K multiplies | 176 K multiplies |
| worst | 1,622 K multiplies | 176 K multiplies |

The mean falls 2.7 times. The worst falls 9.2 times, and the worst is
the number the step period must hold. At six layers the worst step of
era three measures 61 ms of a 200 ms period; the frame takes it near 7
ms. Compute stops being the constraint on the tempo, and the wire
becomes it.

## The measurements

The numbers come from `corpus/JSB-Chorales-dataset/Jsb16thSeparated.json`:
382 chorales, 92,536 steps. The train split is 229 chorales and 55,228
steps, and one packed train stream holds 57,546 steps.

| Measure | Value |
|---|---|
| positions for each step, the encoding of era three | 2.673 mean, 9 worst |
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
| silence in a packed stream | 4.19%, and 97% of it is seam |

### The cost of independent voices

Two count models gave the first estimate. Both fit on the train split
and score on the valid split. Both take the same context — the step
before — and the same smoothing. The only difference is the
independence.

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

**The transformer measures the cost at 0.456 bits, and not at 1.263.**
The count model sees one step of context; the transformer sees 256, and
that context absorbs two thirds of the cost. The count model gave the
sign and the order of size, and the model itself gave the number. Use
0.456.

The cost stands where the music is. Two or more voices move at 25
percent of the boundaries, and the count model reads 2.464 bits there.
That is the part a listener hears, thus a small mean would not make the
parallel head safe.

## What the era measured

The model of the board is
`_train/d64-frame-do03-96k-s6-l6-nopos-span4.ckpt`: **d 64, 6 layers, 4
heads, `T` 256, ALiBi span 4, dropout 0.3, 96,000 steps, seed 6, the
tied chain, no window position. 308,224 parameters, valid 1.6282 nats
for each step.** The ear elected it 2026-08-18. Those are the defaults
of `jax/transformer/train.py`.

It is not the lowest number of the table. A span-8 run of the same
recipe reached 1.6162, and that is the lucky tail of a wide
distribution: the ear likes its character less, and the span-4 spread is
the reason to trust it.

| Question | Answer |
|---|---|
| the chain against parallel heads | the chain, by 0.3157 nats for each step |
| the untied readout | null: 1.6470 against 1.6496, inside the noise, and it costs three tiles |
| the window position | free on the loss, and the ear prefers it gone |
| dropout at 6 layers | 0.3 is a real optimum: 0.1 gives 1.7683, 0.2 gives 1.6975, 0.3 gives 1.6404, 0.4 gives 1.6638 |
| the ALiBi span | 4, for the variance |
| 96,000 steps against 48,000 | 0.010 to 0.013, at both spans, 3 seeds of 3 |
| `T` 512 against `T` 256 | nothing: 1.6440 against 1.6404 |
| the draw | temperature 1.0, min_p 0.05 |

Dropout 0.3 at six layers settles a confound that era three could not:
depth and regularization are not interchangeable. The distance of about
0.42 between the train loss and the valid loss is benign — the valid
loss descends to the last step — and its cause is arithmetic. One packed
train stream holds 57,546 steps against about 152,000 tokens of era
three, thus 48,000 steps is 2.64 times the epochs the same budget bought
before.

**A retraction, kept because the wrong reading is attractive.** An
earlier measurement said the six-layer model of era three drones — its
onset rate falling from 0.72 to 0.20 over 8192 steps — and it named the
packed corpus as the cure. That was a fault in the JAX reader, which
permitted the code `On 0`. Code 0 was END, thus a drawn pitch 0 could
never be released and it held a voice for the whole walk. The OCaml side
and the RTL corrected it at `12c43b8`; the JAX twin did not. With the
correction, the two-layer and the six-layer models of era three both
hold their texture. **The frame does not cure a drone. What it wins on
texture is consistency**: the spread of the onset rate over the seeds is
0.13 against 0.52.

## The traps

**The loss does not carry across the encoding.** The frame gives four
predictions for each step and the sentence gave 2.673 tokens, thus a
loss for each prediction divides against a different count. Report
**nats for each step** and never nats for each prediction.

Even nats for each step do not compare the two encodings. The frame
predicts which voice holds which pitch, and the sentence predicted only
the set, thus the frame answers a harder question by a small amount. Use
the loss to rank inside one encoding. The ear ranks across.

**77.91 percent of the predictions repeat the step before.** They are
easy, they dominate the mean, and they invite a second failure: a model
that holds its chord for ever scores well on a mean and plays a drone.
Report a second number over the steps where two or more voices move, and
read the windowed texture of `jax/measure.py` for the movement, not the
loss.

**Numbers nominate; the ear elects.** Ten times in this project a metric
has ranked a model against the ear. Two of the levers of this era — the
window position and the ALiBi span — were settled by the ear over a loss
that saw nothing.

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

The decode has two implementations and they must agree: `Core.Frame` in
OCaml and `data.decode` in JAX. The eight cases above are the expect
test of the first and the unit test of the second.

The stream comparison stays. The reference and the Cyclesim sequencer
must give the same messages, byte for byte, from one seed.

## The open question

**The model stops without arriving.** Only 67 to 73 percent of its
silences follow a sonority held six steps or more; the corpus is 99.2
percent. A chorale never stops without an arrival first, and a model
that goes quiet in the middle of a phrase reads as fractured however
short the gap is.

Ruled out by measurement:

- the rate — 253 steps between silences, against 236 in the corpus;
- the placement — as even over the 256 coordinate as the corpus, and
  there is no re-anchor in this design;
- the timing — the walk resumes on a beat 100 percent of the time, and
  on a downbeat 43.6 percent against the corpus 42.0.

The draw cannot correct it. The best cell of the sampler sweep reaches
89 percent, and the ear rejects that character as dull.

**One lever is untried: hold the final chord of a piece into the seam,
and release it late.** The cadence cue is then longer and the true
silence shorter. This is a change of the corpus, thus it needs a change
of this document first. Note that a seam is 4, 8, 12 or 16 steps with a
mean of 10.2, and that 51 of the 230 seams of a stream are a forced full
bar whose reason is vestigial under the frame.

## The plan

The ear decides, and it decides on the host. The RTL follows the ear and
not the other way around.

1. **The JAX prototype.** DONE. `jax/transformer` holds the model, the
   trainer and the sampler; `jax/data.py` holds the corpus and the
   decode, and `jax/measure.py` the instruments.
2. **The measurement.** DONE. The windowed texture over long walks,
   twelve seeds, against the canonical packed stream.
3. **The ear.** DONE, 2026-08-18. The frame wins.
4. **The reference.** The decode is DONE: `Core.Frame` holds the wire
   word and the rule, with the eight cases as its expect test. The
   reader and the model in OCaml follow, with the gates of the project.

   **Era three leaves the build at the head of this step, and it stays
   in the tree.** `Quantized` reads the vocabulary of the token, and the
   circuit reads `Quantized`, thus neither one can stand while the model
   of the frame takes their place. Each part comes back when its
   era-four version is written, and the old code stands beside the new
   one while a person writes it. The model seat of the board holds
   `Pink` until the circuit of era four exists.

   The gates are a chain, and each link is an equivalence and not a
   judgement of the music: the OCaml float model against the JAX one, by
   the drawn walk compared line for line and by the loss on a fixed batch
   inside a tolerance; the quantized model against the OCaml float one,
   by the drift report; and the circuit against the quantized model, by
   the stream comparison, event for event. The ear elected the model
   already, thus what these steps owe is that they did not change it.

   The walk comparison is not exact by construction — two float
   implementations differ in the last bits, and a draw parts from its
   twin when a uniform lands near a boundary of the cumulative sum. It
   detects well and it diagnoses badly; the loss gate beside it is what
   says whether a mismatch is a fault or that straddle.

   **Measured 2026-08-18: the first two links hold exactly.** The OCaml
   model gives **1.6282 nats for each step** over the 75 canonical valid
   windows, which is the `best valid 1.6282` of the training log over the
   same windows. The two samplers draw **identical walks** at seeds 1 and
   7, 256 steps each: 480 drawn steps, 1,920 draws, and every event of
   the decode agrees, line for line. The straddle above is a real risk
   and it did not happen — `Nx` and XLA hold the same float32 pass
   closely enough that no uniform found a boundary.

   One rule of the draw earned a part of that, and it belongs in this
   document because both samplers must state it. **The draw takes the
   uniform, and never a number the caller multiplied**: the total is the
   last running total, and never a second sum of the same weights. Two
   sums of one array differ in the last bits — the twin adds pairwise in
   `sum` and left to right in `cumsum` — thus a draw made against the
   other sum can land above every running total, where no class passes at
   all. Against this total the draw is strictly below it, because the
   uniform falls under 1 by 2 ** -24 at the least. Therefore a class
   always passes, that class always holds the weight the floor left it,
   and the fallback that each sampler carried for the case is gone. The
   integer twin never had the case: its arithmetic is exact.
5. **The circuit, the board and the socket.** The block design comes in
   its own document, as before, and **the socket changes here and not
   before it**. The new source gives a frame, thus `Source_intf` takes
   the frame word and loses `Note` and `ready`, and the sequencer takes
   the decode in the place of its note handshake.

   **The socket cannot move first, and an earlier draft of this plan was
   wrong to say that it stands alone.** One interface serves every
   model. The source of era three sat on it, and `Pink` sits on it now:
   to move the socket before this step would mean that the pink source
   and the pink player change while no model of era four can play. The
   source of era three is **never ported**. It left the build at step 4,
   because the integer twin under it could not stand, and the source of
   era four takes its seat here.

6. **`Pink` on the frame socket.** Last, and it is the only step that
   touches music the ear has already elected.

   `Pink.Source` becomes smaller, and `Player` becomes smaller still:
   the report state, the owed flags and the lowest-voice fold exist only
   to give one note at a time, and the rule of the player — a voice
   speaks when it is due and it holds no note, or its pitch moved, or
   its policy re-strikes a held pitch — becomes "each voice holds its
   pitch", with `Frame.events_of_frames` making the events.

   **One decision is owed, and it is musical.** The soprano and the alto
   of `Pink.default` take `restrike`: they articulate the same pitch
   again at every due step, and that is where the rhythm of the model
   comes from — the partition of the rows is the rhythm. A frame states
   which pitch a voice holds and cannot state that the voice strikes it
   again, thus those articulations have no encoding. Measured over 512
   steps at seed 1: the player strikes 667 notes and the frame strikes
   546, thus **the frame loses 18 percent of them**.

   Three answers, and the ear picks one. Accept the smoothing, for which
   there is a precedent — the gate left the design and the melody of the
   pink model already sustains to its next articulation. Or spend one of
   the spare codes on "strike the pitch of the step before", which is
   the door the vocabulary keeps open. Or leave `Pink` on an interface
   of its own, which makes the sequencer carry two and gives up what the
   socket is for.

   **DONE on `feat/pink-v2`: the ear accepted the smoothing.** The
   measurement of 18 percent is a mean over the voices and it hides where
   the loss falls. Over 2048 steps at seed 1, with `Pink.default`: the
   soprano loses 320 of 2048 articulations (15.6 percent) and its longest
   note becomes 5 steps; the **alto loses 173 of 513 (33.7 percent)** and
   its longest note becomes 24 steps, which is 4.8 s at the power-on
   tempo. The tenor and the bass lose nothing, because they never had
   `restrike`. The alto is the voice to listen to: its register is 4
   degrees wide, thus two re-rolls land on one degree often.

   `Pink.Voice.restrike` goes, and `Player` goes with it: the rule of the
   player is `Frame.events_of_frames` in the core, thus `Pink.next_frame`
   is the whole interface of the model. `Pink.state.due` stays, because it
   describes the walk and not the player, and its expect test is the proof
   of the periods. `play_pink` loses `-gate-ms` and `-hold`, thus the
   audition plays what the board plays. Nothing is spent: the frame codes
   `0x01` to `0x7F` stay free, and the second answer is still open if the
   ear changes its mind.

What steps 4 to 6 remove from the machine:

- `Token` goes. Its two maps — the class index of a wire code, and the
  wire code of a class index — are `Vocab` in the corpus library, and
  `Token.Start`, the START rule and the grammar go with the file.
- `Sounding_state` and `Sounding_state.Rtl`, because no frame is
  illegal.
- The mask words of the export, and the mask of the training loss.
- `Quantized.Model.piece_steps`, `Engine.boot_context`,
  `Engine.reanchor` and the boundary of `Engine.next_step`.
- `Transformer.piece_steps`, the `piece_steps` argument of
  `Transformer.sample`, and the boundary of the sampler loop.
- The source of era three, whole: its `State.Release`, its
  `at_boundary`, its release scan over the seats, the extra term of its
  `valid` and its `Op.boundary_cycles` go with the file, and none of
  them is ported.
- `-piece-steps` of `play_transformer`.
