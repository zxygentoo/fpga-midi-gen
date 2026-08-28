# The diffusion model

## Scope

This document gives the design of the model of era six: a MASKED SHEET.
Eight measures of four voices stand as one piano roll, some of its cells are
hidden, and the model states a distribution over the pitch of every cell at
once. Nothing here is causal and nothing is written left to right — a piece
is composed knowing its own ending, which is the thesis of the era and the
opposite of every era before it.

**The era is named for the method and not for the paper.** Masked diffusion
is what the field calls this now: corrupt a discrete field by hiding cells,
train a network to restore them, and sample by hiding and restoring in turn.
COCONET (Huang et al., arXiv 1903.07227) is the 2017 instance of it, under
another name and eight years before the name existed. This round replicates
that paper as faithfully as the corpus permits, because its published
likelihood on this same corpus lineage is the one calibration this project
can get from outside itself. "The lineage since the paper" below states what
the field learned in between, and which of it the era takes.

**THE DELIVERABLE IS THE CURVE OF QUALITY AGAINST N.** The board draws one
sheet while the last one plays, thus the passes it can afford are counted
and the curve alone says whether the masked era reaches the RTL. Everything
else here serves that number: the ladder finds the shape, the referees say
what a shape is worth, and the ear elects.

**The round CLIMBS A LADDER OF SIZES.** It starts at the smallest rung that
fits the board and it goes up; it does not start at the top and subtract.
Each rung is trained, refereed and heard, thus the round learns what each
step of size buys while it climbs, and a rung that fails says which axis it
failed on. The paper's size is the last rung and it is the reference: it
never goes on the board, and its number is what says this stack is right.

Not in this round: whole pieces, endings, the length mask, the RTL. The
sheet is 128 sixteenth steps and one chorale in a hundred fits inside it,
thus a sheet is a crop and it stops where the corpus was cut instead of
ARRIVING. That is the open musical problem the era inherits, it is the
reason for the fade and the rest on the audition wire, and the round that
answers it is in "Deferred". The findings of `feat/diffusion-proto` are
built into this one: the piano roll reads pitch, the padded tail starves the
sheet, and the trunk completes where it does not invent.

## Why the masked objective

The Gaussian round failed in ways that belong to its objective, not to the
sheet. A mean-square loss states the mean of the pitches that fit, which is
not a pitch: the drawn cells stood a coin toss from the lattice. The padded
tail gave silence a 53 percent prior, and the loss hedged the sheet thin.
Both terms leave with the objective. A softmax over the pitch rows cannot
state a mean — a sample is a pitch, every time. And blocked Gibbs spends its
walk in the completion regime, which is the regime the proto proved: a trunk
that finishes a noised corpus sheet into real chorale texture, and rewrites
under a growing context what a one-way chain freezes early.

## The sheet

The sixteenth grid, as the paper states. A training example is a crop of
T = 128 steps — eight measures — taken uniformly inside one piece. One piece
of the train split is shorter than 128 and is dropped; 228 train, 76 valid
and 77 test.

A crop carries no padded tail. Silence inside a crop is the real rests of
the music, a fraction of one percent of the cells, thus the tail prior of
the proto round never enters the training.

The roll keeps the proto's 48 rows: rows for the pitches 36 to 81, row 0
for silence, and the spare. The paper has no silence row because its data
always sings; this corpus rests, thus silence stays a class, and the
vocabulary agreement with `jax/data.py` holds across the eras. A cell is
one-hot over the 48 rows, thus the paper's constraint — one row for each
voice at each step — holds with silence as one more row.

## The input

The paper's two-I planes: the four voices of the masked roll, and the four
mask planes, thus 8 by 128 by 48. A masked cell shows zero in the roll and
one in its mask plane.

## The net

The paper's, at the paper's size: 64 layers of three-by-three convolution
over time and pitch, 128 channels, batch normalization with statistics tied
across time and pitch, a residual connection past every second layer, and a
final projection to four channels — the logits of the four voices. About
nine million parameters.

This size never goes near the board. It states the ceiling and it is the
reference of the round; what reaches the board is a rung far below it, and
the ladder below states how far. The pitch axis is the paper's reason:
contrapuntal rules are near-invariant to translation in time and in pitch —
the equivariance the proto's channel sheets had to learn from 228
chorales, and did not.

## The ladder

The rungs, and the block RAM each one needs on the Nexys 4: the weights at
era four's measured 53 percent packing, and the two activation tensors that a
residual pair holds live. The device has 135 tiles of 4.5 KB.

| rung | L | H | parameters | tiles | MAC for one pass |
|---|---|---|---|---|---|
| 1 | 16 | 16 | 33,984 | 57 (42%) | 198 M |
| 2 | 16 | 24 | 75,168 | 95 (70%) | 446 M |
| 3 | 24 | 24 | 116,640 | 112 (83%) | 701 M |
| 4 | 32 | 24 | 158,112 | 129 (95%) | 956 M |
| — | 16 | 32 | 132,480 | 140 (103%) | over the device |
| ceiling | 64 | 128 | 9,172,232 | 4,090 (3030%) | 56.2 G |

WIDTH IS THE WALL AND DEPTH IS CHEAP. A residual pair holds two activation
tensors of T by P by H, thus the activations grow with H where the weights
grow with H squared: at H 16 the activations are three times the weights.
The board therefore wants a deep, narrow trunk, which is the opposite of the
paper's trade at H 128 and L 64. The frontier that follows — the deepest
trunk each width affords, held under 92 percent of the device because era
four shipped at 93 and its timing was tight:

| H | activations | deepest L | parameters | MAC for one pass |
|---|---|---|---|---|
| 12 | 32 tiles | 174 | 224,208 | 1,370 M |
| 16 | 43 tiles | 87 | 197,568 | 1,203 M |
| 20 | 53 tiles | 49 | 171,360 | 1,040 M |
| 24 | 64 tiles | 29 | 142,560 | 860 M |
| 28 | 75 tiles | 18 | 115,920 | 694 M |
| 32 | 85 tiles | 11 | 86,400 | 510 M |
| 40 | 107 tiles | 4 | 33,120 | 177 M |
| 48 | 128 tiles | — | the activations alone take the device |

The most parameters the board can hold therefore stand at the NARROWEST
width. The packing of 53 percent is era four's, measured on a transformer's
odd-width int8 tables; a convolution kernel is a regular block and a RAMB36
at 1024 by 36 holds four int8 words at 8 of 9, thus this frontier is a floor
and a synthesis run may move it outward.

DEPTH IS THE REACH AND WIDTH IS THE RESOLUTION, and the trunk says why before
any measurement does. Three-by-three convolutions with no pooling reach
exactly L steps in time and L rows in pitch at L layers, and a voice's pitch
stands at its own row of the roll — thus a pair of voices further apart than
the reach cannot be seen at all, at any width. Measured 2026-08-24 over the
first three rungs, the excess dissonance of a voice pair is a function of its
span and of nothing else: at L 16 every pair inside 16 semitones reads the
corpus, and the bass against the soprano, which spans 19.6, reads 27.5
percent against the corpus's 11.3. L 24 takes that pair to 16.6, where the
width step at the same depth took it only to 24.6 and fixed the near pairs
instead. The corpus spans 45 semitones, thus the reach argument says harmony
keeps paying to about L 40 and then stops.

Depth buys the time axis too, and this round cannot see it. L 16 reaches one
bar either way and L 64 reaches four, which is a whole chorale phrase; that
is the axis era four's open defect stands on — the silences that stop without
arriving. A crop holds no cadence, thus the next round measures it.

The sheet passes one window affords, against the MAC lanes. One sheet is
25.6 seconds of music at the audition's step, and the next sheet is drawn
while this one plays:

| L | H | 1 lane | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 16 | 16 | **13** | 26 | 52 | 103 | 207 | 413 |
| 16 | 24 | 6 | 11 | 23 | 46 | 92 | 184 |
| 24 | 24 | 4 | 7 | 15 | 29 | 58 | 117 |
| 32 | 24 | 3 | 5 | 11 | 21 | 43 | 86 |

A lane is one DSP48 taking one product each cycle, which is what `Mac`
already is. **Rung 1 reaches thirteen passes on the datapath the board has
today.** Every rung above it buys its passes with lanes, and the device holds
240 DSPs where the shipped designs use 2.

T stays at 128 on every rung. It is eight measures, it is the length the
paper's raters heard, and it is the frame count the likelihood referee
divides by; a rung that moved it would not compare with the rungs beside it.

## The loss

Orderless NADE, the paper's exactly. Draw a crop; draw the MASKED count
|not-C| uniform on 1 to D, with D = 4 times 128 cells; mask that many cells,
chosen uniformly; take the negative log-likelihood of the masked cells under
the softmax, scaled by one over |not-C|. Adam.

The paper's own line reads "|C| ~ U(1, D)" for the context. That is its slip
and not its model: its reweighting term D − d + 1 IS the masked count, and a
context of all D cells divides by zero. The code release settles it —
`OrderlessMaskoutMethod` draws `k = choice(D) + 1` cells to mask. The divisor
is per sheet, because each sheet of a batch drew its own count.

The code release also carries one default that the paper's equation 9 does
not: it puts the context cells inside the loss too. This round follows the
equation and reads the masked cells alone.

THE OPTIMIZER IS PLAIN ADAM AND THE RATE IS THE PAPER'S. The paper is ISMIR
2017 and AdamW is arXiv 1711.05101 of November 2017, thus AdamW postdates it;
the code release calls `AdamOptimizer` and holds no weight decay, no dropout
and no L2 anywhere. Batch norm and the best-by-valid checkpoint are its whole
regularization. The rate is in neither the paper nor a flag — the release
carries `2**-4` and `2**-6` and halves on a plateau of five epochs. The
plateau rule does not carry over: the trainer runs the warmup and cosine
decay every era of this project trains under, and the sweep below stands on
that schedule. The sweep was measured under `nn.schedule`; the trainer states
the same curve through optax now, and `test_train.py` holds the two equal at
every step. Measured 2026-08-24 at two ends of the ladder, over 1,500 steps,
in valid nats for each masked cell:

| shape | 3e-4 | 1e-3 | 3e-3 | 1.6e-2 | 3e-2 | 6e-2 |
|---|---|---|---|---|---|---|
| rung 2, L 16 H 24 | 1.6822 | 1.0003 | 0.8255 | **0.7699** | 0.7740 | 0.7719 |
| ceiling, L 64 H 128 | 1.4604 | 0.6758 | **0.5750** | 0.5934 | — | — |

**THE OPTIMUM MOVES WITH THE RUNG.** The board rung wants `2**-6` = 1.6e-2,
which is the value the release keeps commented out, and it stands on a
plateau up to 6e-2. The ceiling wants 3e-3, one step below it. Both are far
above the modern default of 1e-3, which costs 0.23 nats at the board rung and
0.10 at the ceiling. Batch norm is why a rate this large is safe: it makes
the loss scale-invariant in the weights. Each rung therefore carries its own
rate, and a rate carried up from the rung below is a lever nobody pulled.

A trap of the two numbers, because they are nearly equal and they are not the
same measurement: the ceiling reads 0.5750 valid nats for each MASKED CELL
above, and the paper's Table 1 reads 0.57 nats for each FRAME under Algorithm
1. Only the likelihood referee compares with 0.57.

No transposition augmentation this round: the paper states none, and the
pitch axis carries the equivariance. The elected checkpoint is the best
valid NLL — the memorization guard, because the proto measured 228 pieces
memorizing when nothing guards them.

## The sampling

Independent blocked Gibbs with the annealed schedule of Yao et al.: at step
n of N, each cell masks with probability
alpha_n = max(alpha_min, alpha_max − n (alpha_max − alpha_min) / (eta N)),
one forward pass runs, and every masked cell resamples independently from
its softmax. The paper's rule of thumb is N = I times T = 512 evaluations,
and it states that a lower N costs a little quality. The constants pin from
the code release, `YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7)`: alpha_min
0.1, alpha_max 0.9, eta 0.7. They are levers, not decisions.

EVERY STEP OF THE WALK IS THE SAME STEP, and it took a change of the opening
to keep it that way. The paper starts on "an empty (zero everywhere) piano
roll" and its roll has no silence row, thus an empty cell there states
nothing. THIS roll holds silence as a class, so an empty cell states a REST
with the authority of context, and the corpus rests in 0.35 percent of its
cells; at alpha_max 0.9 the opening Bernoulli leaves a tenth of the sheet
standing and a tenth of it would be a lie. The round paid for that with an
opening step that masked the whole free region — one step unlike every other,
and a branch the RTL would have to carry.

**The walk therefore opens on NOTES and not on rests.** Every cell takes a
pitch drawn from the seed, each voice inside its own register of
`measure.RANGES`, thus a cell the Bernoulli leaves standing states some note
— and four voices sounding is 99.8 percent of the corpus. The release ships
this initialiser itself, as `UniformRandomSampler` beside `ZeroSampler`.

Measured 2026-08-25 on L 48 by H 20 over 256 sheets, THE TWO OPENINGS ARE
THE SAME INSTRUMENT. On the parallels they part by under half an error at
every N above 32, and at N 32 the silent opening leads by 1.6 and 1.3 errors,
which is not significant across ten comparisons and sits at a budget the
board will not run — L 48 by H 20 reaches N 512 on 206 of the 238 free DSPs.
The ear could not tell them apart. The silent opening was removed rather than
kept as a flag nobody would pull, and the curve numbers recorded above were
measured on it and carry forward unchanged.

**The curve of the round is quality against N**, over N in 32, 64, 128, 256
and 512, read by the referees below. The board's silence window affords tens
of sheet passes, thus this curve alone decides whether the masked era
reaches the RTL.

## The referees

1. **The likelihood.** The paper's Algorithm 1: M = 5 random orderings,
   importance-weighted, in nats for each frame. The anchor is the paper's
   Table 1 on this corpus lineage: **0.57 ± 0.01** on the sixteenth grid.
   The protocol pins from the paper exactly — a referee that computes a
   different number reads nothing. Four details carry that:

   - The ordering is over FRAMES, and the four voices take their own
     ordering inside each frame. It is not an ordering over all D cells.
   - A frame conditions on the TRUE frames that stand before it in the
     ordering, and on the model's own answers for the voices before it
     inside the frame. Algorithm 1 restores the ground truth at each frame
     boundary. Therefore the frames are independent given the ordering, and
     the referee costs I forward passes over a stack of T sheets, not I
     times T passes.
   - The model's own answer is written as the ARGMAX. The paper's Algorithm
     1 draws there; its code release takes the argmax, and the code release
     made 0.57.
   - The orderings combine in probability space, one frame at a time:
     logsumexp less log M. A mean of log-likelihoods is an unnormalized
     geometric mean and it wastes probability mass.

   An untrained model must read 4 log 48 = 15.48 nats for each frame. That
   is the arithmetic check of the referee, and it passes.

   **THE TAIL STANDS BESIDE THE MEAN.** The referee also reports the median,
   the 90th and the 99th percentile of the frames and the share of them over
   2 nats, each with a bootstrap error over the PIECES — the frames of one
   chorale are one draw of a composer and not 128 of them. Measured over the
   ladder on 2026-08-25, THE MEDIAN AND THE 90TH ARE THE MEAN IN OTHER
   CLOTHES: they rank the six rungs in the same order and separate them
   about as well (9 to 10 times their own error, against the mean's 9.5).
   The 99th does not move with capacity at all — 1.4 times its error, and
   not monotone. Read as a ratio it moves the wrong way: the 99th over the
   median runs 9.97 at the smallest rung to 11.25 at the largest, thus
   CAPACITY BUYS THE ORDINARY FRAME AND NOT THE EXCEPTIONAL ONE.

   Read the tail against what it measures. These are CORPUS sheets, thus a
   frame of high nats is a frame where Bach surprised the model and not one
   where the model wrote something strange.

   It earns its place on a pair that capacity did not separate. The span arm
   reads 0.7635 ± 0.0202 against rung 1's 0.7599 ± 0.0155 — 0.14 of an error
   apart, one model to the mean — and 5.3 ± 0.4 percent of its frames stand
   over 2 nats against 4.0 ± 0.3, which is 2.6 errors apart, with the BETTER
   median of the two. Wrong less often and more badly is a trade the mean
   cannot see and the ear can.
2. **The structure battery** of the proto round, held to its method: the
   corpus row stands as the referee of every number, and triads count only
   over the steps that carry three voices or more. The corpus row on this
   grid, over one crop of each train chorale: hold 76.9 percent, 0.89
   onsets for each step, 63.9 percent triads, 10.3 percent dissonant pairs,
   97.4 percent of the steps in register order, and all four voices
   sounding at 99.8 percent of them.

   One instrument is new, and it is the one that explains the others: THE
   VOICE PAIRS, each pair's mean span in semitones beside the share of it
   that is dissonant. It reads the pitch reach of the trunk directly, as
   the ladder above states. The corpus row is near 10 percent dissonant at
   every span, from the alto against the soprano at 5.5 semitones to the
   bass against the soprano at 19.7, thus a rung that reads 27 at the
   widest pair has a reach fault and not a taste.

   **PARALLEL FIFTHS AND OCTAVES** are the one instrument of the battery
   that reads across time, and the one that separates this era from the
   corpus where every vertical instrument says it has arrived. THE DIVISOR
   IS THE PAIRS THAT MOVE TOGETHER, and it was the pairs that merely SOUND
   until 2026-08-25. A parallel needs both voices to move, thus a model that
   holds its notes was being paid for holding them — which the span arm was
   caught doing, halving the old number while its onsets fell a fifth below
   the corpus. The share of pairs that move stands on the same line, because
   a rung whose motion has left the corpus is not to be read on the rates
   alone. The corpus reads **1.37 fifths and 1.06 octaves for each thousand
   pairs that move, with 15.2 percent of its pairs moving**.

   A parallel is a rare event. 64 sheets hold a Poisson error near a sixth
   of the count, thus a comparison between two models wants 256.
3. **The ear.** Gibbs draws of eight measures through the audition rig —
   the excerpt length the paper's raters heard. A harmonization audition —
   keep the soprano, write the three voices under it — was heard in the
   ladder round and CUT with its flag on 2026-08-26: the completion regime
   is the whole-piece round's thesis, and the mask planes give it back for
   one flag whenever that round wants it.

   `--gap` puts a silence between two sheets on the wire and `--fade`
   takes the velocity down over the last bar of one, two bars and one bar
   by default. The fade reaches only the notes that BEGIN inside its
   window — velocity is a fact of the onset, and the S-1 makes a control
   change audible only on the next note — thus its length is measured and
   not chosen: a crop's last note has been sounding 4.5 steps in the mean,
   so a bar catches 99 percent of the final notes where four steps catches
   67 and finds no onset at all in 18 percent of crops. The gap doubled
   when the fade arrived, because a sheet that ends quiet has less to
   part from. A batch is several INDEPENDENT draws and each one is a whole
   piece; with nothing between them the second opens on the first one's
   last chord, which no performance does. The ear set the bar on
   2026-08-25 and reported what the silence CANNOT do: a sheet is a crop
   and it stops where the corpus was cut, thus it does not arrive, and no
   silence after a phrase that never closed makes it sound closed. That is
   the deferred whole-piece round below, and it is the open musical
   problem the transformer era logged before this one.

## What the round measured

The design above carries the numbers that settled it where each one stands.
These are the findings that belong to no single decision.

### The golden candidate is L 48 by H 20

169,648 parameters, 90 percent of the device at int8, 31 ms for a training
step. L 48 is the FIRST DEPTH WHOSE REACH COVERS THE WHOLE PITCH AXIS from
any row, and H 20 is the widest that depth affords — H 20 allows L 49, thus
the two walls meet and neither holds slack the other could spend.

| | valid | nats for each frame | triads | octaves |
|---|---|---|---|---|
| L 64 H 16, 147k | 0.4446 | 0.6145 ± 0.0153 | 66.4% | 20.4× |
| L 64 H 18, 185k | 0.4429 | 0.6126 ± 0.0159 | — | — |
| **L 48 H 20, 170k** | **0.4422** | **0.6139 ± 0.0151** | **62.1%** | **19.3×** |
| L 106 H 12, 139k | 0.4391 | 0.6276 ± 0.0159 | 61.9% | 33.6× |
| corpus | — | — | 62.7% | 1.0× |

The three shapes near 170k parameters read the same likelihood inside a
tenth of an error, thus ABOVE L 48 DEPTH HAS STOPPED PAYING. What separates
them is harmonic content: L 48 by H 20 lands on the corpus's triad share
where L 64 by H 16 is markedly over-triadic, and it holds the best octaves
of any board-feasible rung.

### Depth is the reach; width is a floor and not an axis

The cleanest experiment of the round is in the ladder, at the widest voice
pair. The corpus is near 10 percent dissonant at every span:

| rung | ba-te, 8.6 st | ba-al, 14.1 st | ba-so, 19.6 st |
|---|---|---|---|
| L 16 H 16 | 11.8% | 17.0% | **24.7%** |
| L 16 **H 24** | 13.1% | 17.9% | **25.3%** |
| **L 24** H 24 | 9.8% | 13.5% | **16.6%** |

At a fixed L 16, more WIDTH does nothing for the distant pair. At a fixed
H 24, more DEPTH halves its error. Width cannot buy what the receptive field
never saw.

Width was called "the resolution" until 2026-08-25, when the disambiguating
run said otherwise. On parallel octaves H 12 reads 33.6 times the corpus,
H 16 reads 20.4 and **H 20 reads 19.3, which is 0.7 of an error from H 16**.
H 12 is a floor to clear and above H 16 width buys nothing measurable; the
ceiling's 9.9× is its 62 times the parameters and not its H 128.

### The defect is parallel motion, and nothing reaches it

Every vertical instrument says these models have arrived — the ceiling reads
triads 62.6 against the corpus's 62.7, dissonance 10.5 against 10.7, hold
76.5 against 77.3. What happens BETWEEN two chords has not arrived at all.
At N 512 over 256 sheets, for each thousand pairs that move together:

| | 5ths | octaves |
|---|---|---|
| the board rung | 8.9× | 20.4× |
| the ceiling, 62× the parameters | 4.6× | 9.9× |
| **the corpus** | **1.08** | **0.60** |

**Four independent levers do not touch it.** Sixty-two times the parameters
buys two and stops. No board-feasible shape improves it — every rung from
H 16 to H 20 sits at 19 to 20 times the corpus on octaves whatever its
depth. No training mask reaches it: the span lever and its mix are measured
and cut above. And no affordable N reaches it — the rate falls about 0.85
for each doubling with no flattening, thus arriving at Bach would need
N ≈ 660,000, which is a wrong mechanism and not too few passes.

The reason stands in the objective. **The loss scores the MARGINAL of each
cell under its context and the walk draws those marginals INDEPENDENTLY**,
thus a joint configuration of two voices across two steps is evaluated by
neither. A term that stands in no objective cannot be sampled away. What
remains is sampler-side — draw a block jointly instead of independently — or
an objective that scores the joint at all.

### Every model sings sharp in the upper three voices

Register was unmeasured until the instrument existed, because the voice
pairs read DIFFERENCES of pitch and the order instrument reads the stacking:
a texture that slid bodily moves neither. Measured at N 512, every rung of
the ladder reads its tenor, alto and soprano ABOVE the corpus — tenor 59.9
to 61.6 against 59.5, alto 65.8 to 67.2 against 65.1, soprano 71.0 to 72.3
against 70.5 — while the bass straddles 51.0. Depth fixes the MAGNITUDE of
register error and nothing fixes its DIRECTION: not depth, not width, not
complete pitch reach, not the golden candidate. It is an open item beside
the parallels.

### The tail of the likelihood is the mean in other clothes, except once

On the capacity ladder the median and the 90th percentile rank the six rungs
exactly as the mean does and separate them no better. The 99th does not move
with capacity at all: the CEILING — the ear's own first choice — holds the
best mean, median and 90th and **the heaviest 99th of the whole ladder**,
and the ratio of the 99th to the median climbs 9.97 → 11.19 → 14.66 as the
rungs grow. These are CORPUS sheets, thus a frame of high nats is one
where BACH surprised the model and not one where the model wrote something
strange.

It earns its place on a pair that capacity did not separate: the span arm
and rung 1 stand 0.14 of an error apart on the mean and **2.6 errors apart
on the frames above 2 nats**, with the span arm holding the better median.
Wrong less often and more badly is a trade the mean cannot see.

## The traps

- **THE CORPUS ROW IS THE REFEREE OF EVERY NUMBER, and forgetting it cost
  three corrections to one instrument in one day.** Parallel motion divided
  by the pairs that merely SOUND, so a model that held its notes was paid
  for it; it counted contrary motion onto a fifth, which read 53 percent of
  the CORPUS's own fifths as faults; and its absolute gap let a crossed pair
  hold an interval class its interval had turned upside down. Every one was
  found by reading the instrument against the corpus and none by comparing
  two models.
- **Valid nats and framewise nats do not map across shapes.** L 106 by H 12
  holds the best valid of any rung outside the ceiling and is 0.59 of an
  error BEHIND the board rung on Algorithm 1. Elect on the referee.
- **A parallel is a rare event.** At 64 sheets its Poisson error is near a
  sixth of the count; draw 256 before quoting a comparison between two
  models.
- **The bar phase of a training sheet is uniformly randomised.** 65.2
  percent of chorales are a whole number of bars plus a quarter-note
  anacrusis, and `Crops.crop` draws its start uniformly — so a sheet begins
  on a random beat and THE MODEL CAN LEARN NO METRE FROM POSITION. Aligning
  crops to the real downbeats is a cheap untested lever.
- **The likelihood referee at the paper's size writes nothing for 39
  minutes.** Its per-piece progress lines begin with `piece`, which the log
  filters of the round's scripts drop; a 40-second board rung does not need
  them and the ceiling does.

## The cost

Measured 2026-08-24 on the RTX 3060 (12 GB), at the paper's shape:
9,172,232 parameters, T = 128, 64 layers, 128 channels.

| what | cost |
|---|---|
| one training step, batch 8 | 327 ms |
| one training step, batch 8, rematerialised | 432 ms |
| one training step, batch 16, rematerialised | 826 ms |
| one training step, batch 16 | the card runs out of memory |
| 30,000 steps at batch 8 | 2.7 hours |
| the likelihood referee, one test piece | 29 s, thus 37 minutes for all 77 |
| one Gibbs step, four sheets | 47 ms |
| the whole N curve, four sheets | 47 s |

Rematerialisation of each residual pair costs 28 percent of the step and
buys the memory that a batch of 16 needs. A batch of 8 fits without it,
thus it is off by default and it is a flag.

TF32 changes nothing here: batch 8 reads 327.9 ms with it and 327.3 ms
without. The float32 pin of `jax/nn.py` therefore costs this round nothing.

## The files

- `docs/diffusion.md` — this document.
- `lib/corpus/jsb.ml`, `bin/corpus_tool.ml` — the piece export of the proto
  round, taken from `feat/diffusion-proto` and generalized: `Jsb.on_grid
  ~every` takes the grid as a parameter and a grid of 1 is the identity, and
  `corpus_tool pieces -grid N` writes it. With no `-steps` the longest piece
  states the width of a row and no piece is dropped: 229 train, 76 valid and
  77 test chorales, in rows of 640 steps.
- `jax/data.py` — the piece reader, and `Crops`, the uniform crop taken
  inside the true length. It drops the one train chorale that is shorter
  than the crop, thus the round trains on 228.
- `jax/diffusion/model.py`, `train.py`, `infer.py` — the sheet, the trainer
  and the walk. `infer.py` draws and measures nothing itself.
- `jax/measure.py` — THE COMMON HOME of the instruments, as `jax/nn.py` is of
  the network. Everything in it is arithmetic over a `[sheets, steps,
  SEATS]` array of class indices and none of it knows which era drew them: a
  Gibbs sheet, a walk of the packed stream and a corpus crop all read the
  same way, and a single walk is a stack of one.
- `jax/diffusion/measure.py` — what this era measures with its OWN model,
  which is the paper's Algorithm 1 and the tail of it. Its sibling
  `jax/mamba/measure.py` holds era five's forced pass and walk.
- `jax/tests/test_diffusion.py`, `jax/tests/test_midi.py` — the loss
  reweighting, the mask draw, the anneal, the mask planes, the checkpoint,
  the battery, Algorithm 1, and the two gestures of the audition wire.

## Deferred

- **THE STRETCH GOAL, AFTER THE INT8 RTL SHIPS: int4, the 17-bar sheet and
  the mix, which are one round and not three.** The int8 board holds
  L 48 by H 20 at T 128, and T 128 holds one whole chorale in a hundred.
  T 272 is 17 bars and holds 77 percent of them, and at int8 it forces
  L 48 by H 15 — one channel above the H 16 that works and three above the
  H 12 that reads 33.6 times the corpus on parallel octaves. **Int4 is what
  clears that floor**: L 48 by H 24 at T 272, 87 percent of the device. It
  is NOT for more width at T 128, where H 16 to H 20 bought nothing
  measurable and the extra MAC costs more than the parameters return.
  The round then runs at N 256, because one pass of that shape is 3,147 M
  MAC and N 512 would need 296 lanes of the 238 the device has free — and
  N 256 is the regime where the mix above pays for itself. Each of the
  three answers a problem the other two create; none of them is worth
  doing alone.
- Whole pieces on the eighth grid, the length mask as un-noised
  conditioning, and the endings: the thesis of the era returns in the next
  round, on top of this stack.
- MaskGIT confidence-ordered unmasking: the low-N reserve, if the curve
  demands it.
- Transposition by the legal shifts, if the valid NLL asks for more data.
- The RTL, behind all of it.

## The lineage since the paper

Searched 2026-08-24, after the first rungs read well. The finding that
frames the round: THE FIELD REDISCOVERED THE PAPER'S SAMPLER. The modern
masked diffusion models sample by unmasking alone and freeze what they
commit; ReMDM (ICLR 2025) proves that the correction — mask a committed
cell again and rewrite it under a better context — strictly increases the
expressive power of the model. That is the annealed Gibbs of Yao et al.,
stated as a theorem eight years later. On the training side, MDLM and MD4
(both 2024) show the masked-diffusion bound is a weighted orderless-NADE
loss, thus the loss of this round is the modern objective already, and the
draw of the masked count is the one free knob they expose. The families
also split on architecture along the corruption: the masked line — BERT,
MaskGIT, MDLM — keeps the full lattice at every layer, and the Gaussian
line keeps the UNet, because averaging denoises Gaussian noise and
destroys a one-hot row. This round therefore replicates a 2017 paper and
runs the 2025 method under its original name, on the architecture its
corruption asks for.

**SPAN MASKING (MAGNeT, ICLR 2024) WAS ONE OF THESE AND IT IS MEASURED AND
CUT.** Mask contiguous runs of one voice instead of lone cells, because a
held note writes its own answer into the neighbours of a masked cell — at a
masked share of 0.1, where the walk settles, 94.2 percent of masked cells
have their class standing in a live neighbour, and a span of 4 takes that to
35.9. Measured 2026-08-25 at the board rung, 100,000 steps, each arm on its
own probed rate:

- **A pure span of 4 costs 0.149 nats of framewise NLL, six times its error
  and more than the whole climb from rung 1 to the board rung.** Its
  parallels do fall, and it buys that by writing blander music — onsets 0.65
  against the corpus 0.88, hold 83.5 against 77.3, and the frames nobody
  sings growing over the walk to 6.4 percent, ten times the corpus.
- **A mix of half span and half lone cells cures the smoothing and the
  parallel win goes with it** — one error on the fifths and none on the
  octaves at N 512. The two are ONE AXIS.
- What the mix does buy is quality for each Gibbs pass, 12 to 17 percent
  fewer parallel fifths at N 32 to 128. **That is not a currency this board
  is short of**: N is bought with MAC lanes, N 512 costs 75 percent of the
  240 DSPs, and the board rung at N 512 beats the mix at N 128 outright.
- Training on spans buys 1.8 percent on the span task itself, against a
  baseline that never saw one. A trunk that learned the conditional from
  lone cells already fills a span.

The flags and the two draws were removed after the measurement. The leak is
real and the lever does not pay for it AT THIS BUDGET.

**THE ONE CONDITION THAT BRINGS IT BACK, and it is not hypothetical.**
The mix's whole return is quality for each Gibbs pass, thus it is worth
exactly what a pass is worth. Measured on the board rung, halving N from 512
to 256 costs 22 percent on the fifths and 17 on the octaves, and the mix buys
back 13 percent of each — **the mix at N 256 stands level with the lone-cell
draw at N 512**, 0.7 and 0.2 of an error apart. So the lever pays whenever N
is capped at 256, and N is capped whenever the MAC for one pass passes about
1,200 M: the device holds 238 free DSPs, and N 512 inside a 25.6-second
window needs one lane for each 4.7 M MAC. **Int4 at T 128 and int4 at T 272
both land there** — L48/H30 needs 462 lanes at N 512 and L48/H24 at T 272
needs 296. A round that goes to int4 should carry the mix with it.

The draw, so that round need not rediscover it. A cell is masked when any of
the `span` start positions that reach it fired; the start rate
1 − (1 − share)^(1/span) puts the masked share back where the paper's draw
put it. The starts stand on an axis of `steps + span − 1`, thus exactly
`span` of them reach every cell and no cell near an edge is masked less
often, and ONE start is forced because the loss divides by the masked count
and a Bernoulli draw at the smallest share comes up empty three times in
four. The mix is a coin FOR EACH SHEET between that draw and the paper's,
never a mix over cells: a sheet asks one question or the other, and the
masked share of each stays the paper's. The valid probe keeps the lone-cell
draw whatever the run trains on, or two runs are two numbers of two
different tasks.

Three advances remain, and they are levers of a later round:

- **The Halton order** (Besnier et al., 2025). A precomputed
  low-discrepancy sequence states the unmask order, in place of confidence
  or chance. It spreads the unmasking over the sheet, it cuts the error
  propagation, and it needs no retraining. For the RTL it is the strongest
  of the four: the order is a ROM read, there is no sort, and the seed
  keeps its determinism. It joins the MaskGIT reserve of the deferred
  list, and it is the cheaper half of it.
- **Sampler-matched training** ("Revise, Don't Freeze", arXiv 2606.01026;
  "Learn from Your Mistakes", arXiv 2602.11590). Tune the trained model on
  contexts the sampler made, not only on masked ground truth. It closes
  the gap between the training draw and the walk, and it aims where the
  board is narrow: the quality that a low N affords.
- **The masked-count knob** (the MDLM line). The draw of |not-C| need not
  be uniform, and an emphasis toward small counts is reported to help. One
  sweep states it.

Three findings stand as context and stay out:

- **Token-Critic** (ECCV 2022): a second network states what to remask. A
  second model has no seat on the board.
- **The whole-song cascades** (2024): a scaffold of phrases first, the
  surface second. It is the modern form of the deferred fermata scaffold,
  and it waits with it.
- **D3PM on symbolic music** (IJCAI 2023) and **the fixed-grid masked
  harmony model** (Applied Sciences, 2025): the near settings. They
  validate the note-level infill and the painted conditioning roll, thus
  the length round's design has precedent beyond this repository.
