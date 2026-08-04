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

The zero word gives three properties for free:

- A cleared context memory reads as a run of ENDs. That history is
  silence, and silence is the correct state before the first note.
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

The sentence has one canonical order. The OFF events come first. The ON
events follow, and each ON pitch is greater than the ON pitch before it.
Therefore one chord has exactly one sentence and not a permutation
family, and the position of an ON in the sentence acts as a voice rank.
The tokenizer writes this order, and the mask enforces it in the
sampler.

Tempo is not in the sentence. The host changes `step_ms` and the music
changes speed, with no new training. A groove is a clock-side pattern:
an alternation of `step_ms` gives swing, and the model does not see it.

## The socket

The model is a source of `Sequencer`, behind the same interface as the
pink model: `rewind`, `step`, then `valid`, `idle` and `ready`. The
sequencer waits in `Take`, so a slow source is legal. The contract is
one sentence inside one step period.

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
| OFF(p) | pitch p sounds now |
| ON(p) | p does not sound, open seats < 4, p above the last ON of the sentence |
| END | always |

Therefore every sentence is valid MIDI, at most four voices sound, and
two seats never hold one pitch. The disjoint-register rule of the S-1
holds by construction. The trainer applies the same mask inside the
softmax of the loss. A mask in the sampler without the same mask in the
loss moves the distribution; the two must agree.

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
thumb is twenty training tokens per parameter. Transposition of the
corpus over the twelve keys is legal, because the pitch field covers
the full MIDI range; count it as a gain of four to ten, not twelve.

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

The gate restores a part of the articulation that the grid removes, so
the round-up of short notes is safe.

## The plan and the tests

1. Tokenizer and trainer on the host, in OCaml. The backprop is written
   by hand; the model is two matrices deep.
2. The sweep: `d` in {64, 96, 128} and the integer ladder, on the real
   corpus. The validation loss picks the band; the audition through the
   S-1 USB picks the model. The ear decides.
3. Freeze the config. Then the RTL document, the blocks and the board.

The tests follow the project rules. The reference model is exact
integer arithmetic, so the stream comparison against Cyclesim is a
cheap test, block by block below it: the MAC array, the mask, the
sampler. The seed is an input, and one seed gives one sequence in the
reference, in the simulation and on the board.
