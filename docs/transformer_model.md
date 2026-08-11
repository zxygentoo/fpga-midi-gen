# The transformer model

## Scope

Era three puts a small transformer in the model seat. This document gives
the design of the model: the token, the step sentence, the sequencer
socket, the legality mask, the network, the sizes, the tokenizer and the
plan. The RTL block design comes in a later document, after the audition
on the host.

The design keeps the project rules. The board does the inference. The
bitstream initializes the weights. The host trains the model. At MIDI
rates the compute is never the limit; the block RAM is. Therefore the
design spends its care on memory and on simplicity, not on throughput.

## The token

One token is one byte:

| Bits | Field | Values |
|---|---|---|
| 7 | type | 1 = Note On, 0 = Note Off |
| 6:0 | pitch | 0 to 127, the MIDI pitch |

The code `0x00` is END. END closes the sentence of a step. END takes the
code of "Note Off, pitch 0". Pitch 0 is C-1, 8.18 Hz, and no music holds
that pitch. If a corpus holds pitch 0, the tokenizer moves the note up
one semitone.

The code `0xFF` is START. START opens the walk: it is the first token
of every training sequence, and the source writes it one time at
rewind, before the first draw. The model never draws START, because
the mask forbids it. START takes the code of "Note On, pitch 127".
Pitch 127 is G9, 12544 Hz, and no music holds that pitch. If a corpus
holds pitch 127, the tokenizer moves the note down one semitone.

The zero word gives three properties for free:

- A cleared context memory reads as a run of ENDs, and that history is
  silence: a safe value for every slot that the boot does not write.
- The padding of a training batch is `0x00`, and it means silence.
- The idle test in the circuit is a compare with zero.

The channel, the velocity and the time are not in the token. The control
registers hold the channel and the velocity. The step clock holds the
time. A voice identity is not in the token: in MIDI the channel is the
voice identity, and the channel is a constant of this system.

## The step sentence

The sequencer is the clock of the model. One step of `step_ms` asks for
one sentence: zero or more event tokens, then END.

- A rest is a bare END.
- A held note is a note without an OFF. The vocabulary has no hold
  token, because MIDI has none.
- A repeated note is the OFF and then the ON of one pitch, in one
  sentence.

The sentence has one canonical order. The OFF events come first and
climb: each OFF pitch is greater than the OFF pitch before it. The ON
events follow and fall: each ON pitch is less than the ON pitch before
it. Therefore one chord has exactly one sentence and not a permutation
family, and the position of an ON in the sentence acts as a voice rank.

Each direction earns its place. The fall is the melody leading: the top
voice is chosen before the voices under it, and conditions on none of
them, as the music is written. The climb then makes the two runs meet in
the middle, so the release of the top moving voice sits beside its
attack and one melodic step is two adjacent tokens.

The mask holds both directions, so they are rules of the instrument and
not conventions of the tokenizer. A convention would leave the
permutations of a chord inside the softmax, where the model must spend
mass to learn an order that the mask can refuse for nothing.

Tempo is not in the sentence. The host changes `step_ms` and the music
changes speed, with no new training. A groove is a clock-side pattern:
an alternation of `step_ms` gives swing, and the model does not see it.

## The socket

The model is a source of `Sequencer`, behind the same interface as the
pink model: `rewind`, `step`, then `valid`, `idle` and `ready`. The
sequencer waits in `Take`, so a slow source is legal. The contract is
one sentence inside one step period.

At rewind the source starts from an empty context and writes START.
The first sentence follows inside the first step: power on, music on.
The entry draw does not see a bar position — the composer writes now —
and the bar counter takes its boot value at rewind, a free RTL choice.

The source decodes its own END: on token `0x00` it raises `idle` and
sends no event. One field is new on the interface: each event carries
its type bit, On or Off, because the model states the releases and the
seat compare of the sequencer does not. `gate_ms` stays a clock-side
trim of the articulation.

The worst sentence is four OFFs, four ONs and END: nine tokens, eight
messages, 24 bytes on the wire, 7.68 ms at 31250 baud. A `step_ms`
below 8 cannot carry that burst. If the register ranges become strict,
the floor of `step_ms` is 8.

## The legality mask

The sampler masks the logits before the softmax. A masked token has
probability zero.

| Token | Legal when |
|---|---|
| OFF(p) | p sounds now, the sentence holds no ON yet, p above the last OFF |
| ON(p) | p does not sound, open seats < 4, p below the last ON of the sentence |
| END | always |
| START | never |

Therefore every sentence is valid MIDI, at most four voices sound, and
two seats never hold one pitch. The disjoint-register rule of the S-1
holds by construction. The same mask sits inside the softmax of the
training loss, thus the composer spends no mass on a code that the
sampler would refuse. Therefore its raw mass outside the legal set stays
untrained, and the mask must guard every draw.

## The network

The model is a decoder-only transformer with integer weights.

| Part | Design |
|---|---|
| embedding | one 256-row table, tied with the output head |
| position | ALiBi: a constant slope per head, a power of two |
| bar phase | a 16-row table, indexed by the bar counter of the sequencer |
| norm | RMSNorm; the trainer folds the scale into the next matrix |
| bias | none |
| feed-forward | `d_ff = 4 * d` |
| context | `T` tokens, KV cache `2 * L * T * d` bytes, or recompute |

ALiBi makes the position bias a shift, so the model has no position
table. The step period is long, so the recompute datapath with no KV
cache is also legal; the choice is simplicity against 64 KB and it can
wait for the RTL document.

The integer ladder has three rungs. Int8 weights from post-training
quantization are the base, with no training risk. Int4 and ternary
weights need quantization-aware training. Ternary weights remove the
multipliers: the MAC becomes add, subtract or skip, and the DSP blocks
leave the design.

## The piece position

The bar phase gives the position in the bar. It does not give the
position in the piece. The tokens hold no other measure of time.
Therefore the model cannot know that it is near the end, and it cannot
name the part of the piece that it must state again.

A second table gives that position. The table has 16 rows, and the row
adds to the token embedding beside the bar phase. The two tables have
the same shape and the same use: one says where the step is in the bar,
the other says where the step is in the piece.

Training and the draw index it differently, because training knows the
length of a piece and a draw does not. The corpus divides:

    piece_phase = 16 * step / steps_in_piece

The ratio is the correct measure, and the corpus shows it. A repeated
bar of the top line comes again after a distance. That distance is more
constant as a part of the piece than as a count of steps: the
coefficient of variation is 0.48 against 0.58 for the exact repeat, and
0.49 against 0.59 for the contour. Therefore a ratio names a repeat
better than a step count.

The count of rows is a balance. With 16 rows, one offset holds 44
percent of the repeats. With 32 rows it holds 31 percent, and the other
repeats divide between 25 offsets that each get less data. With 8 rows
it holds 52 percent, but the last row is then 1.8 bars, and the model
cannot tell "prepare the cadence" from "make the cadence". 16 rows keep
both properties.

The draw counts; it does not divide. The corpus divides each piece by
its length, but a draw has no length: the board plays for ever, and a
long draw would stretch one arc of sixteen buckets over a span that no
training piece ever had. Therefore the bucket of step `i` is

    piece_phase = i / 16 mod 16

One bucket is 16 steps, one arc is 256 steps, and the arc repeats. A
chorale runs 228 steps at the median, thus one arc is about one piece
and the walk gives chorale-shaped arcs, one after another.

Both numbers are powers of two, thus the circuit takes a bit-slice of
the step counter. **The piece length is not an input**: no control
register holds it, the host states nothing, and the counter never
stops. A draw of exactly 256 steps gives the buckets that a division by
the length would give, so that case fixes the two rules together.

The table costs 16 rows of `d`: 1 KB at `d` 64 with int8 weights, which
is 0.9 percent of the parameters.

## What the piece position gave

The table was tested at the recipe of the best model: `d` 64, 2 layers,
`T` 256, dropout 0.1, seed 4, 48000 steps. One variable changed.

The loss did not move. The valid loss is 0.6298, and the model without
the table gives 0.6299. A test at `T` 512 gave the same answer: the
table moved the loss by 0.0001 or less at every evaluation, and the
sign changed from one evaluation to the next.

The ear moved. The music keeps the same craft, and it becomes more
dynamic and more intentional. Therefore the table stays. This is the
second change of the era that the ear accepted, and both were invisible
to the loss: the other is the order of the sentence. Both change what
the model is conditioned on. No change of capacity, depth, context,
dropout, weight decay or ALiBi has ever passed the ear.

The one measurable effect of the table is a smaller distance between
the train loss and the valid loss: 0.0698 against 0.0750. The train
loss is worse, not the valid loss better. Thus the table acts as a weak
regularizer, and that is not the reason to keep it.

**The crop dilution is not a reason for a longer window.** A repeat
pair 80 steps apart is in about one training window in five at `T` 256.
The model without the table has the same dilution, and the ear prefers
it to every model with a longer context. Therefore the dilution is a
condition of this corpus and not a fault. A `T` 512 test moved the loss
by 0.005, moved the effect of the table by nothing, and the ear found
the music worse. Test a change against the best model, and change one
thing.

## The sizes

The budget is the block RAM: 607.5 KB, with about ten percent reserved
for the rest of the design. Sizes below are int8, `T = 256`.

| Config | Params | Weights | KV | Total | Of 607.5 KB |
|---|---|---|---|---|---|
| d=64, L=2 | 115 K | 113 KB | 64 KB | 177 KB | 29% |
| d=96, L=3 | 358 K | 349 KB | 144 KB | 493 KB | 81% |
| d=128, L=2 | 428 K | 418 KB | 128 KB | 546 KB | 90%, the ceiling |

The floor is near 50 K parameters. A transformer repeats a motif with
an induction circuit, and that circuit needs two attention layers.
Music lives on repetition. Therefore the model has at least two layers,
at any size.

The optimum comes from the data and not from the board. The rule of
thumb is twenty training tokens per parameter. Transposition of a
training piece is legal while each voice stays inside the observed
range of its voice in the corpus: soprano 60 to 81, alto 52 to 74,
tenor 46 to 69, bass 36 to 66. The measured gain on the training pool
is seven to eight, not twelve.

- A chorale-scale corpus gives 100 K to 200 K parameters.
- The ceiling serves a corpus of ten million tokens or more.
- Above the ceiling the board is the limit, not the model.

A model above the optimum of its corpus does not fail loudly: it stores
the corpus and replays it. The audition must listen for that.

## The tokenizer

One module turns a MIDI file into sentences. The trainer, the reference
model and the tests all read that one module: one definition of the
walk.

| Policy | Rule |
|---|---|
| tracks | note events only; drop channel 10 |
| tempo | drop the tempo meta events; `step_ms` owns tempo |
| grid | one step is a sixteenth: PPQ / 4 ticks |
| quantize | each event moves to the nearest grid line |
| short notes | a note below half a step becomes one step |
| voices | at most four sound at once; thin the middle voices |
| order | the canonical sentence order, per step |
| pitch 0 | move to pitch 1 |
| pitch 127 | move to pitch 126 |

The JSB corpus reads from the voice-separated file,
`Jsb16thSeparated.json`. The voices give the transposition rule; the
pitch sets of the walk derive from them: drop the rests, merge the
unisons, sort ascending.

The gate restores a part of the articulation that the grid removes, so
the round-up of short notes is safe.

## The draw

One generator makes every random number of the model: `Prng`, the
xorshift32 of the circuit. The initial parameters, the dropout masks and
the sampler all draw from it. Therefore one seed names one sequence in
the software, in the simulation and on the board, and that seed is the
value of the SEED cell of the host control.

One step of the walk gives eight bits, and one uniform takes three
steps. The grid of `2 ** -24` holds the tail of the Box-Muller draw,
which one byte would cut at 3.3 sigma. A seed from a flag or from a
stream folds into the 32 bits of the state; a seed already inside the
range names itself, so seed 7 is the walk of the board's seed 7.

A part of the model that must draw on its own takes an independent walk.
Each dropout block takes one. Therefore the block holds no place in the
order of the parent, and the parent can gain or lose draws while the
masks stay.

The initial parameters moved when the draw came to this generator. A
checkpoint of an earlier sweep still loads, because the file holds only
tensors, but a run from a given seed does not repeat the numbers of that
sweep.

## The plan and the tests

1. Tokenizer and trainer on the host, in OCaml. Raven carries the
   work: Nx computes, Rune differentiates, Kaun optimizes.
2. The sweep: `d` in {64, 96, 128} and the integer ladder, on the real
   corpus. The validation loss picks the band; the audition through the
   S-1 USB picks the model. The ear decides.
3. Freeze the config. Then the RTL document, the blocks and the board.

Experiments that wait for the fast trainer: a boot phase that the
entry draw can see, a learned silent lead after START, voice
supervision in the loss.

The tests follow the project rules. The reference model is exact
integer arithmetic, so the stream comparison against Cyclesim is a
cheap test, block by block below it: the MAC array, the mask, the
sampler. The seed is an input, and one seed gives one sequence in the
reference, in the simulation and on the board.
