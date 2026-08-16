# The improviser

This document is a plan. Nothing in it is built. The machine that plays
today is the one of `transformer_rtl.md`, and it keeps its place until
the ear elects what is planned here.

## The goal

One button, and music that does not end and has no seam. The model
plays as an improviser plays: it holds the last minutes of its own music
in its window, and it makes the next step from them. It does not know
where a piece starts or stops, because there are no pieces. There is one
walk.

## What is wrong today

The endless walk decays. The texture leaves the corpus in minutes and
settles at a quarter of the onset rate, with no note shorter than a
quarter note. The six-layer model at eight bits holds through about 500
steps; the same model in floating point holds through about 2000; the
two-layer king holds through about 3000. No checkpoint and no sampling
knob changes the end state.

The walk therefore plays pieces: every 256 steps the source releases the
sounding pitches, clears its context and feeds START. This holds the
texture, and the board plays it now. It has three costs:

- The seam is a mechanism, not music. A counter decides where a piece
  ends, and the model has no part in the decision.
- One piece in ten opens thin and stays thin until the next boundary.
  The re-anchor gives the model a context of one token, and the model
  never trained on such a context.
- START is scaffolding. It starts a walk and does nothing else.

## Why the walk decays

Two causes, and neither is START.

**The error of each step compounds, and nothing pulls the walk back.**
The learned conditional is the corpus conditional with a small error.
The walk applies it to its own output thousands of times. A chain like
this goes to its own stationary distribution, which owes the corpus
nothing.

**The quiet absorbs the walk, because the corpus teaches the roads in
and no road out.** A chorale is full of movement toward quiet: the
cadences, the fermatas, the wind-down at the end. The movement back —
silence, then a new piece — is at the boundary between two pieces, and a
training window never crosses one. The trainer knows this already: it
drops the padding of a short piece from the loss, because "a padded
label would teach the walk to hold the last chord of the piece and emit
END for ever, the drone".

START is not the cause, and its removal alone changes nothing. START
leaves the window at about step 100 of every walk, and 99.3 percent of
the training windows at context 256 hold no START. The walk is already
in the condition this plan proposes; it decays there. **The packed
corpus is the change that carries the plan. The removal of START is its
result.**

## The design

### The packed corpus

The corpus becomes one stream for each split: piece, seam, piece, seam,
and so on. The seam holds what the re-anchor does today, as data:

1. the last step of the piece,
2. one step that releases the sounding pitches, ascending,
3. silent steps to the next bar boundary,
4. the first step of the next piece.

Step 2 is necessary and is new: `Jsb.tokenize` walks the steps in pairs
and never releases the last chord, thus a piece ends with four voices
still sounding.

The model then learns the transition it has never seen. The quiet stops
absorbing the walk, because the corpus states what follows the quiet.
The release the source performs today, the model performs itself.

The steps 2 and 3 are one thing, and the tokenizer already makes it. A
step with no sounding pitch gives the OFFs of the chord before it,
ascending, then END; the silent steps after it give END alone. Therefore
the seam is a count of empty steps appended to the piece, and the
encoder gains no token rule.

### The gap of a seam

The gap is the count of empty steps between the last step of a piece and
the first step of the next. It is the smallest count that puts the next
piece on its rotation, and it is never zero, because the release needs
one step.

The measurement of the corpus below gives its size: a piece is a whole
number of quarter notes and a rotation is a whole number of quarter
notes, thus the gap is 4, 8, 12 or 16 steps. The quiet of a seam is
never shorter than a quarter note and never longer than a bar. No rule
states this — the arithmetic gives it.

Transposition keeps its place: a piece takes one of its legal shifts
each time it enters a stream.

### The metre of the corpus

The measurement of step 1, over all 382 pieces and 92,536 steps:

| | pieces | steps |
|---|---|---|
| a 16-step bar | 226 | 64.4% |
| a 12-step bar | 27 | 8.1% |
| too few holds to vote | 129 | 27.5% |
| a pickup | 211 | — |

Every rotation is 0, 4, 8 or 12 steps, and every piece is a whole number
of quarter notes.

One rolling clock of 16 steps holds this corpus. Each piece takes the
rotation that puts the most of its cadences on a downbeat, and the pad
of the seam places it there. The cadences on a downbeat are then 73.8
percent of those of the 16-vote pieces, 44.6 percent of those of the
12-vote pieces, and 71.9 percent of all. To remove the 27 odd-metre
pieces moves the last number to 73.8 percent and costs 8.1 percent of
the steps. That is a bad trade, and the votes it would act on are weak:
8 of the 27 win by a lift of 4 and 4 more by a lift of 16, near-ties on
2 to 4 holds. **The corpus needs no rule for the odd metre.**

The bar length therefore leaves the corpus. `Jsb.metre` becomes
`Jsb.rotation`: one vote at 16 steps. `Jsb.vote` loses the lift, which
only made the votes of 12 and 16 compare.

The draw has always run a rolling clock of 16 steps at rotation zero —
`Transformer.sample` takes `step mod 16` and `Source` takes the low four
bits of its step counter. The 12-step bar and the pickup were therefore
signals the machine could never make. The packed corpus closes that
mismatch as well.

### The streams

A piece takes one of its legal shifts each time it enters a stream, thus
one stream is one draw and a split needs more than one. The export
writes eight streams for each split by default: a piece has 7.4 legal
shifts at the mean, thus eight streams hold the token count of the
corpus of today.

Stream zero is the canonical one — every piece at shift zero, in the
order of the corpus. The referee reads it alone, thus Gate A and Gate B
stay deterministic and both trainers can make it. The other streams take
a uniform permutation of the pieces and a uniform legal shift for each.

The draw of a training row becomes: a uniform stream, then a uniform
window. The piece draw is no longer uniform — a long piece holds more
windows than a short one — and this is the objective the endless walk
asks for.

### The position signal

The two tables become one rolling coordinate: the step count modulo 256.
It says where a token stands in the memory window, and not where it
stands in a piece.

The coordinate keeps the factored form the hardware already has: the bar
phase is `step mod 16`, the frame is `step / 16 mod 16`, and the two
16-row tables add row for row. One 256-row table would cost eight times
the block RAM for the same information.

The seam file therefore carries one array and not two. `positions` holds
the coordinate of each token, and the reader takes the phase from its low
four bits and the frame from its high four. `phases` and `progress` leave
the file. The window of a packed stream never spans 256 steps unless it
is silent through all of them, and a silent window wraps the coordinate
exactly once, thus the modulo is always correct.

**The draw already computes this.** `Source` takes the phase from bits 3
to 0 of its step counter and the frame from bits 7 to 4;
`Quantized.Engine` takes the same two. Only the corpus changes: the
frame becomes `step / 16 mod 16` of the packed stream, where today it is
the sixteenth of the piece. The oldest mismatch of the era closes — the
one that makes an audition ask for a multiple of 256 steps.

The frame earns its rows with a job the attention cannot do alone. ALiBi
measures distance in tokens, and a token is not a clock: a dense passage
compresses it and a sparse one stretches it. The frame gives the model
musical time over 16 bars, which is the length of its window.

### The boot

The walk must start from something, because attention needs one token.
Two candidates:

**(a) The silent lead-in.** The boot forwards silent steps — END
tokens — and then draws as usual. The packed stream holds runs of silent
steps at every seam, thus this condition is one the model trained on,
and the model opens the music itself. The seed keeps its whole part: it
drives the sampler, thus a seed still names a walk.

**(b) The manufactured first note.** The boot draws one pitch from the
PRNG and forwards it as an ON. The pitch must come from the range of the
first ON of a piece, which is the top voice: the ONs of a sentence fall,
thus the first pitch bounds the chord under it.

(a) is the smaller machine: it needs no pitch range, no table and no
new rule, and it asks the model for the opening instead of imposing one.
(b) is certain to make a note. **Decide with a measurement: run (a) and
count the steps to the first ON over many seeds.** If the model opens
promptly, (a) wins on subtraction.

**MEASURED, 2026-08-14: (a) wins.** The lead-in is one bar — 16 silent
steps, the longest seam of the corpus, which leaves the first draw on a
downbeat. On the first packed model (L2, valid 0.6424), over 12 seeds,
the first ON falls at step 16 for 6 seeds, 20 for 2, 24 for 2 and 28 for
2. Every seed opens inside one bar of the end of the lead-in, and every
opening sits on a multiple of four steps — the grid the seams taught.
Half of them open at the downbeat itself. The model opens the music, and
the boot needs no pitch, no range and no table.

### The token vocabulary

START retires. Code 255 stays a code that is never legal, thus the
vocabulary stays 256, the tables keep their shapes and the weight ROM
does not move. `Token.t` loses its `Start`, and `Sounding_state` keeps
its rule for code 255 with a new reason: it names no token.

## What the machine loses

- `Jsb.encode`, because there is no encoded piece any more, only a
  stream. `Jsb.pack` takes its place.
- `Jsb.metre` and the vote of the 12-step bar; `Jsb.progress_buckets`.
- The short-piece padding of both trainers, and the loss weights with it:
  a window of a packed stream is always full.
- `Token.Start` and the START rule of the grammar.
- `Quantized.Model.piece_steps`, `Engine.boot_context`, `Engine.reanchor`
  and the boundary of `Engine.next_step`.
- `Transformer.piece_steps`, the `piece_steps` argument of
  `Transformer.sample` and the boundary of the sampler loop.
- The re-anchor of `Drift.walk`.
- `State.Release` of the source, `at_boundary`, the release scan over the
  seats, the extra term of `valid` and `Op.boundary_cycles`.
- `-piece-steps` of `play_transformer`.
- The piece lottery, and the seam that causes it.

## The work, in order

**1. Measure the corpus first.** DONE, 2026-08-14. The seam pads to 16
steps for every piece and the corpus needs no rule for the odd metre.
The numbers and what they cut are in "The metre of the corpus" above.

**2. The corpus side.** `Jsb.pack` takes the place of `Jsb.encode`: it
walks a list of transposed pieces, appends the gap of each seam, and
gives the codes and the positions of one stream. `corpus_tool export`
writes packed streams — the variant machinery of the index carries them,
with a packed stream in the place of a transposed piece. `jax/data.py`
drops `piece_progress` and its short-piece padding, because a window of
a packed stream is always full. The referee windows must move with the
training windows, or Gate B breaks.

The masks are the one cost of the packing. A stream is too long to hold
a mask for each of its tokens in memory, thus the trainers walk
`Sounding_state` from an anchor. A stream states its anchors: the first
token of each piece, where the release of the seam before it leaves the
walk silent.

**3. Train.** The recipe of the era holds: masked loss, dropout 0.1 at
d64, 48k steps, warmup 300, lr 1e-3, batch 16, context 256. Two training
seeds, because one seed cannot settle an ear comparison.

**4. Measure the endless walk.** DONE, 2026-08-14, and the answer
divides the two models. The instrument is `Texture` and
`checkpoint_tool texture`; it did not exist and had to be built. It adds
a fourth measurement to the three named here: the share of notes under a
quarter note. The median alone cannot see the decay, because the median
of the corpus is a quarter note already.

Twelve seeds, 8192 steps, windows of 1024. The corpus reads onsets/step
0.81, single-ON 0.10, median 4.0, under a quarter 0.37.

**The two-layer model holds.** Its onset rate goes 0.74 to 0.63 over
8192 steps, it never leaves the corpus texture, and all twelve seeds end
between 0.47 and 0.75. The median duration stays 4.0 and the short notes
stay near a third. There is no boundary and no decay.

**The six-layer model drones.** Its onset rate falls 0.72 to 0.20 by
step 5120 — a quarter of the corpus, which is the decay of the era
exactly. The median duration doubles to 9, and the short notes fall from
0.31 to 0.06: "no note shorter than a quarter note". The outcome is not
one outcome but two: six of the twelve seeds end at 0.08 to 0.17, and
three hold at 0.55 to 0.79. Half the walks die and the rest live.

**The loss ranks them the other way** — the six-layer model wins by
0.030. This is a measurement, not an ear, and it disagrees with the loss
before the ear is asked.

**A confound stands in the way of naming the cause.** The six-layer run
also takes dropout 0.3, where the two-layer takes 0.1, because it
follows the recipe of the era. Depth and regularization move together,
thus this measurement cannot say which one drones. One run of six layers
at dropout 0.1 settles it.

The float sampler must move before this step and not at step 6. A model
of the packed corpus never saw START, thus `Transformer.sample` and
`jax/transformer/infer.py` must boot from the silent lead-in and lose
the boundary of their loops before they can draw one walk to measure.
Step 6 keeps the quantized reference, the source and the board, which
the ear does not need.

**5. The ear.** Against the board of today, which plays six layers with
the re-anchor.

**6. Only then the machine.** The quantized reference takes the drift
gate, the source loses the parts listed above, and the board takes a
bitstream.

## The gates

- The stream test: the circuit equals `Quantized.Engine`, event for
  event.
- The drift gate: the quantized model against the float model.
- Gate A and Gate B: the JAX trainer against the OCaml referee.
- The windowed measurement of step 4, at 12 seeds or more.
- The ear, which elects.

## The open questions

- Whether the frame earns its rows. The bar phase is settled — the
  ablation without it was rejected long ago. The frame is new in its
  meaning, thus one run without it would say what it is worth.
- The ear must judge the piece position again. Its old form was the one
  conditioning the ear ever approved, and this plan changes what it
  says.

## What stays until the ear elects

Everything that plays today. The re-anchor is committed, gated and on
the board. This plan removes it only if the ear takes the improviser
over the machine that plays pieces.
