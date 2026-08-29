# The Mamba model

## Scope

This document gives the design of the model of era five: a selective
state-space trunk under the step frame of era four. One step of music is
one step of the recurrence and one 32-bit frame on the wire. The four
voices keep their names from the corpus to the synthesizer.

**This was a prototype design, written before the first training run.
The chain is now built and measured, end to end**: the trainer, the
reference, the twin, the circuit and the bitstream, each one gated
against the one before it. The numbers a run had to replace are replaced
where they stand, and the paragraph that predicted each one keeps its
reasoning beside the measurement. Era four is still on the board and in
the flash; nothing here has been auditioned, and that step belongs to a
person.

**The deliverable of the prototype is the whole chain, end to end.**
The prototype accepts lower musical quality; what it must produce is
every link — the trainer, the reference, the twin, the circuit, the
build — proven against each other on a real checkpoint. Therefore the
plan below runs to its last step on the best checkpoint its small
budget gives, and no measurement of quality stops it: a disappointing
number is a finding to record and carry, never a reason to halt. The
gates of equivalence are the opposite — they hold as hard as ever,
because they prove the chain and not the music. The ear's election is
the last step, with the whole era in view, and it belongs to a person.

**A quality round followed the first audition, on 2026-08-20.** The ear
heard the prototype and called it jittery and unmelodic; the round asked
where the deficit lives, re-elected the draw, and swept the two levers
this document reserved. Its numbers are in "What the round measured", and
they change two paragraphs of this document where they stand.

The design keeps the project rules. The board does the inference. The
bitstream initializes the weights. The host trains the model. At MIDI
rates the compute is never the limit; the block RAM is. Therefore the
design spends its care on memory and on simplicity, not on throughput.

The RTL block design is `docs/mamba_rtl.md`.

## What carries over

Era four settled the whole outside of the model, and this era changes
none of it. The trunk between the embedding and the head is the one
experiment.

| Part | Status |
|---|---|
| the step frame, the wire codes | unchanged, `docs/transformer.md` |
| the decode, the sequencer, the socket | unchanged; `Source_intf` does not move |
| the corpus: the packed streams, the seam, the reader, the shifts | unchanged, `Jsb` |
| the vocabulary: 48 classes, `Vocab` | unchanged |
| the input: four seat tables in one tensor, and the bar phase | unchanged |
| the head: the tied chain from the soprano down | unchanged |
| the draw: temper, min-p, the cumulative walk, `Prng` | unchanged; the policy re-elects by ear |
| the boot: one bar of silent frames | unchanged |

The loss keeps its unit: **nats for each step**, the sum over the four
seats and the mean over the steps, on the windows of `Jsb.windows`. Era
four and era five share one encoding and one window rule, thus **the
loss compares across the two eras for the first time**. The elected
model of era four stands at 1.6282 nats for each step on the canonical
valid windows, and that number is the bar this trunk must approach. The
ear still elects: a loss ranks, and ten times in this project a metric
has ranked a model against the ear.

## Why a state-space trunk

Three reasons, and one honest risk.

**The context memory goes.** The transformer holds a KV ring of
196,608 bytes at six layers — 48 tiles — and reads it back at every
step. A state-space layer holds a fixed state and carries it forward:
24,576 bytes at six layers, written and read in place. The window
disappears as a concept: the model has no context length at inference,
and the training window is a training choice alone. **The election
brought one window back**: the ear took the plan with the Zamba head,
thus one layer of eight carries a ring of 256 again — 32,768 bytes —
and the model has a context length after all. The seven other layers
keep the argument whole.

**The step cost falls and stops growing.** Attention walks the ring at
every step: two passes of `T * d` terms for each layer, 32,768 of the
81,920 multiplies of an era-four layer. The recurrence costs a constant
that does not know `T`. The estimate below puts the step at about 2.4 ms
against era four's 7, on the same one-multiplier machine. **The trunk
measured 2.93 ms, and the elected plan 4.04** — 404,314 cycles, and with
the head the step grows again until the ring fills at step 256, then
holds. The wire's 8 ms floor stands over both numbers, thus the argument
bought margin and the head spent some of it.

**The block RAM falls from 93 percent to about 41.** The six-layer build
of era four closes timing at +0.031 ns and does not close without the
post-route pass, and the reason is routing at 126 of 135 tiles. The
budget below puts this model near 55 tiles. The margin that era four
fights for, this design buys back with room to spare. **The trunk built
at 57.5 tiles — 42.6 percent — and the elected plan at 80.5, 59.6
percent.** The head paid a third of the buy-back, and the margin
argument held: the elected build meets timing at +0.278 ns on the
default directives, where era four stood at +0.059.

**The risk, stated first: music lives on repetition, and exact
repetition is what a state-space model does worst.** The floor of two
transformer layers in era three stood on the induction circuit — a
transformer repeats a motif by attending to the last time it happened. A
fixed-size state cannot look back; it can only have kept what it will
need, in a lossy sum. The literature states this plainly for exact
copying tasks. The counter-argument is that a chorale is more grammar
than quotation: era four's own ALiBi finding was that every head wants
to be local, and the distant structure the seeds latched onto was noise.
Whether a state holds enough of a chorale is the question of the era,
and the experiment answers it. A weak answer does not stop the
prototype: the chain runs to the board on the best checkpoint there is,
the instruments of `jax/measure.py` record what the walk does, and the
verdict comes at the end, with the whole era in view. If the verdict is
no, the era closes as era two closed — after the run, not during it.

## The research round

Lean, 2026-08-19. What it found, and what this design takes from it:

- **The state is the quantization-sensitive part.** Quamba and Quamba2
  measure massive outliers in the selective-scan activations and state
  persistence that per-tensor int8 breaks; their recipe suppresses
  outliers and quantizes the scan I/O with care
  ([Quamba](https://arxiv.org/html/2410.13229v1),
  [Quamba2](https://arxiv.org/html/2503.22879)). This design therefore
  keeps the state in **int16** and never coarsens it to a byte: the ring
  of era four took the top byte because 48 tiles forced it, and this
  model has no such pressure.
- **Power-of-two SSM scales are proven practice.**
  [LightMamba](https://arxiv.org/abs/2502.15260) quantizes the SSM with
  power-of-two scales on an FPGA, which is exactly the shift-only
  dequantization idiom of this project.
- **FPGA precedent exists, at the wrong scale.** eMamba, FastMamba,
  SpecMamba and MARCA all accelerate LLM-scale Mamba for tokens each
  second ([eMamba](https://arxiv.org/pdf/2508.10370),
  [SpecMamba](https://arxiv.org/abs/2509.19873)). None runs on one DSP
  at MIDI rates. They confirm that the recurrent mode maps to hardware
  and that the nonlinearities take table and piecewise forms; nothing
  transfers directly.
- **Symbolic music has no small Mamba.** The literature holds large
  hybrid Mamba-transformer models
  ([MusicMamba](https://arxiv.org/html/2409.02421),
  [Mamba-Diffusion](https://arxiv.org/html/2505.03314)); nothing near
  300 K parameters, nothing on four-voice chorales, and the hybrids
  answer no question this era asks. The comparison that matters is the
  one this project runs itself: the same corpus, the same frame, the
  same budget, era four against era five.
- **Mamba-2 over Mamba-1.** The Mamba-2 form takes one scalar decay for
  each head where Mamba-1 takes one for each channel and state pair
  ([the SSD papers](https://goombalab.github.io/blog/2024/mamba2-part1-model/)).
  On this machine that is the difference between six exponentials a step
  for each layer and two thousand. Mamba-2 also adds a normalization
  before the output projection that stabilizes training, and it affords
  a larger state for the same compute. This design is the Mamba-2 form
  throughout, in its recurrent mode.

## The block

One layer, in the recurrent form. `d` is the residual width, `d_in = 2 d`
the inner width, `H` the heads, `P = d_in / H` the head width, `N` the
state width, `G = 1` the B/C groups, `K = 4` the convolution taps.

```
y   = rms_norm(h)                                   [d]
zxbcdt = y . W_in                                   [2 d_in + 2 N + H]
z   = zxbcdt[0 : d_in]                              the gate
xBC = conv(zxbcdt[d_in : d_in + d_in + 2N])         K taps, causal, for each channel
xBC = silu(xBC)
x   = xBC[0 : d_in]   B = xBC[d_in : d_in + N]   C = xBC[d_in + N :]
dt_h = softplus(zxbcdt[dt slice][h] + dt_bias[h])   one for each head
a_h  = exp(A_log[h])                                a constant of the checkpoint
alpha_h = exp(-dt_h * a_h)                          the decay, in (0, 1]

S_h[p, n] <- alpha_h * S_h[p, n] + x_h[p] * (dt_h * B[n])    the state update
y_out[p]  =  sum_n S_h[p, n] * C[n] + D[h] * x_h[p]          the readout

g   = rms_norm(y_out * silu(z))                     the gated norm of Mamba-2
h   <- h + g . W_out                                the residual join
```

The state `S` is `H` blocks of `P x N`, `d_in x N` values for each
layer. It carries from step to step; it is the whole memory of the walk.
A training window starts it at zero, and the boot of the walk starts it
at zero, thus the seam condition of the corpus is the condition the
model trains on — exactly the argument of the era-four lead-in.

The convolution is depthwise and causal: each of the `d_in + 2 N`
channels holds its last `K - 1` inputs, and a step reads four taps. It
is the short-range half of the model, and the state is the long-range
half.

No bias terms anywhere but `dt_bias`, RMSNorm with no scale — the
trainer folds the scale, as era four's does — and no position table
beyond the bar phase of the embedding: the recurrence is a position
signal by construction.

## The plan, and the Zamba head

**A trunk of blocks is not the model. The plan is.** A layer is one of
three kinds, and the sequence of them is the plan of a model:

| letter | kind | it holds |
|---|---|---|
| `M` | the block above | `w_in, conv, dt_bias, a_log, d_skip, w_out` |
| `Z` | the Zamba attention head | `wq, wk, wv, wo` |
| `F` | the feed-forward | `w1, w2` |

**The elected model is `MMMMMMZF`** — six blocks, then the head, then the
feed-forward. Eleven levers were tried against the six-block trunk and
this is the one that paid: mean −0.0062 over the baseline, the still
steps −0.0135, and the soprano −0.0033 at t = −3.15, which is the seat
that carried the whole deficit of the era and the only effect of the
round to clear three sigma. The ear agrees, and it says more than the
loss does: *the craft becomes better*.

### Half a Zamba

The head is era four's attention sublayer with one addition, and the
addition is where the gain is:

```
e   = rms_norm(embedding)          computed once, live for the whole step
y   = rms_norm(h)
q   = [y ; e] . wq                 wq, wk : [2 d, d]
k   = [y ; e] . wk                 -> the ring
v   =  y . wv                      wv, wo : [d, d]
h  <- h + attend(q, k, v) . wo     H heads, ALiBi at the span, causal
```

**The query and the key read the original embedding beside the residual
stream; the value reads the stream alone.** Six blocks of recurrence smear
which note was really played. The embedding still says it, and a head
needs it to match on. That is the whole of the difference from era four's
layer, and era four's plain attention — a square `wq` over the stream —
measured null in this trunk three times over three shapes.

The feed-forward is era four's, as a layer of its own so that it can be
ablated without touching the head beside it. It measured neutral and
costs 32,768 parameters; it stays because the ear elected the pair.

### The context comes back, for one layer

A block carries a state of one size and knows nothing of how long the
walk has run. An attention layer carries a ring, thus **a hybrid walk has
a context where the trunk had none**, and two numbers come with it:

- **the ALiBi span is 4**, era four's elected value, and it is written
  into the checkpoint after the last layer as one number. Era four
  carried this as a flag that had to match the training run; a span
  played back wrong is silently wrong music, thus this era puts it in the
  file. Span 8 measured null again here, in a regime where the different
  behaviour was expected: every head still wants to be local.
- **the ring is 256 at inference**, which is the training window. At that
  depth a window of the loss reads exactly the attention the trainer
  computed. A ring of 128 was measured against a 512-step window and
  parts from it by 7.6e-05 relative, thus the depth can buy back about
  four tiles of block RAM if a build ever wants them. Span and ring are
  ONE decision: the 128 result holds only while the span is 4.

### The baseline shape

| | value | why |
|---|---|---|
| `d` | 64 | era four's width; the embedding and the head carry over unchanged |
| layers | 6 | era four's depth; one variable at a time |
| `d_in` | 128 | the standard expansion of 2 |
| `H` | 4 | one head for each voice is a coincidence; `P = 32` is the reason |
| `N` | 16 | the Mamba-1 default; **measured against 64 and elected** |
| `K` | 4 | the Mamba default; **measured against 16 and elected** |
| `G` | 1 | shared B and C; the model is small and the groups buy nothing here |
| plan | `MMMMMMZF` | six blocks, the Zamba head, the feed-forward; **elected by the ear** |
| span | 4 | era four's; the file states it, and 8 measured null again here |
| ring | 256 | the training window, at inference; 128 measured sufficient |

The head width `P = 32` is a power of four, thus the shift rules of the
machine hold as they hold in era four. `N` is the one lever this
document expected the sweep to move: the state is the experiment, and 16
against 64 was the first question after the recipe stood.

**The sweep ran and the baseline stands.** K 16, N 64 and both together
all read a worse moving-steps loss than K 4 and N 16 — the table is in
"What the round measured" — thus the shape above is the elected shape and
not merely the first one tried. `K` is a field of the configuration now,
as `N` always was: `Mamba.Config.of_checkpoint` reads it from the kernel
tensor, and the tap ring, the age mux and the conv op size themselves from
it. No width of this model is a constant of the library any more.

### The parameters

For each block: `W_in` is `d x (2 d_in + 2 N + H)` = 64 x 292 = 18,688;
the convolution is `(d_in + 2 N) x K` = 640; `W_out` is `d_in x d` =
8,192; `dt_bias`, `A_log` and `D` are `H` = 4 each. **27,532 for each
block.** The head is four matrices — `2 d x d` twice and `d x d` twice —
24,576; the feed-forward is 32,768. The tables are era four's: the seat
tensor 12,288 and the bar phase 1,024.

The elected plan: **235,848 parameters against era four's 308,224** — 77
percent, at the same width. The state buys its memory at run time, not in
the weights.

**The trunk is what the corpus caps, and it caps at about 170,000.** The
six blocks stand at 165,192 and are healthy; the same trunk at `N` 32 is
178,248 and turns — its valid loss bottoms early and climbs. The head and
the feed-forward do NOT count against that ceiling: `6M+Z+F` at 235,848
total is healthy, and `6M` at `N` 64 with 217,672 total turns. Two models
within 256 parameters of each other, opposite behaviour, and the
difference is where the parameters sit. More dropout does not fix it; it
is a corpus limit.

### The checkpoint

Safetensors, the tensors named "0" upward in the flat order, as era four:
the two tables, then the tensors of each layer in the order of the plan,
then the ALiBi span alone.

**The file states its own plan.** The first tensor of a group names its
kind, thus the reader walks the groups sequentially and reads the kind
before it reads the count:

| the group opens with | the kind | tensors |
|---|---|---|
| `[d, projection]` | a block | 6 |
| `[2 d, d]` | the Zamba head | 4 |
| `[d, 4 d]` | the feed-forward | 2 |
| `[d, d]` | era four's plain attention | **refused** |

The projection is `2 d_in + 2 N + H` and is never `d`, `2 d` or `4 d`,
thus no block head can be read as another kind. A tensor of one value
after the last group is the span; a file without one reads as era four's
4, which is what the trainer defaults to.

`Config.of_checkpoint` takes every width, the plan and the span from the
file. The ring is the one number that cannot be in it: the ring is a
choice of the player at inference and not a fact of the training run. One
`Params_data`-shaped structure states the order once for the float model
and the quantized twin.

## The nonlinearities

Era four had one transcendental function, the exponential of the
softmax, and it became one table: exp2 of a nonpositive Q12 power, 256
entries, with log2(e) folded into the temper. This era has three, and
each takes the same treatment. The float model and the trainer use the
true functions; the quantized twin states the table rules; the drift
report measures the distance. Nothing approximates silently.

- **The decay** `alpha = exp(-dt * a)` is the existing exp2 table with
  log2(e) folded into the constant: `alpha = exp2_q(dt * a * log2(e))`,
  Q15, the domain of the unit exactly. `a` is a constant of the
  checkpoint, thus `a * log2(e)` quantizes at elaboration and the run
  time multiplies by `dt` alone.
- **The sigmoid** serves SiLU twice — the conv branch and the gate:
  `silu(v) = v * sigmoid(v)`. One new table in the style of the exp2
  ROM: 256 entries of 16 bits, Q15, over the signed Q12 input clamped to
  |v| < 8 with the low bits falling away; beyond the clamp the sigmoid
  is within 2^-11 of 0 or 1 and the table's end rows hold those. SiLU is
  then one lookup and one multiply.
- **The softplus** serves `dt` alone, `H` values a layer:
  `softplus(v) = relu(v) + ln(1 + exp(-|v|))`. The correction term is
  the second new table: 256 entries, Q12, over |v| clamped to 16. `dt`
  then clamps to the range of its format; the float model does not
  clamp, and the drift report owns the difference.

There is **no division in the trunk**. The softmax normalization was the
one division of the era-four datapath, and the recurrence has nothing to
normalize. `rms_norm` keeps its isqrt and its per-element division, as
before.

## Training

`jax/mamba/` holds `model.py`, `train.py` and `infer.py`, in the shape
of `jax/transformer/`. The corpus, the windows, the decode and the
measure instruments are `jax/data.py` and `jax/measure.py`, unchanged
and shared.

- **The scan.** A window is `context + 1` steps and the state starts at
  zero at its head. `lax.scan` carries the recurrence; the width is
  small and the window is 256, thus the associative scan is an
  optimization to measure if training drags, not a requirement.

  **It drags. Measured: 203 ms for each training step** against era four's
  61, which would put this closed budget past thirteen hours. The cost is
  not arithmetic — it is nine thousand tiny kernels in a chain 256 deep.
  `jax/mamba/model.py` therefore holds a second form for the trainer: the
  quadratic form of Mamba-2, where a cumulative sum of `dt * a` turns the
  decay between two steps into one subtraction and the whole recurrence
  into one weight matrix for each head. It answers in **59 ms**, and era
  four's rate is 61.

  Two forms of one recurrence is a second thing to keep true, and the
  price is paid in a gate: `jax/tests/test_mamba.py` holds them to each
  other over a whole forward — the convolution, the state, the gated norm
  and the residual joins — and over windows shorter than the tap ring,
  where a pad written the other way round would show. The step form is
  the definition; the integer twin and the circuit compute that one.
- **The recipe opens where era four closed.** The same hand-rolled AdamW
  with a decoupled decay and a global-norm clip, the same batch draw —
  a uniform stream, then a uniform window — the same reporting: nats
  for each step, and the second number over the steps where two or more
  voices move. Dropout on the residual branch after `W_out`; the
  era-four optimum of 0.3 belongs to a 77 percent larger model, thus the
  sweep runs {0.1, 0.2, 0.3} and expects the optimum lower.
- **The initialization.** The Mamba defaults: `A_log` from uniform `a`
  in [1, 16]; `dt_bias` the inverse softplus of a uniform draw in
  [0.001, 0.1]; the matrices normal at scale 0.02 as era four; `D` at
  one.

  **The convolution takes 0.02 too, and it was measured.** The argument
  against was fan-in: four taps at 0.02 pass a fiftieth of their input,
  the SiLU under them sits near its own origin, and B and C open so small
  that the state has little to learn from. `1/sqrt(K)` is the fan-in scale
  and the Mamba reference uses it. Over 4 000 steps it reads 1.7311 valid
  against 0.02's 1.7113: the argument is wrong here, because the gated
  norm rescales the branch in any case. One rule covers every matrix.
- **The prototype budget is small and closed.** The dropout sweep at
  48,000 steps — three runs — then the best of the three at 96,000, and
  **that checkpoint carries the rest of the plan, whatever its number
  says.** Checkpoints and logs in `_train/mamba/`, named
  `d64-mamba-n16-l6-...` in the era's convention; the era-four runs
  move to `_train/transformer/`.

  **Ran 2026-08-20, and the expectation above is wrong.** The valid loss
  at 48 K reads 1.7697 at dropout 0.1, 1.6711 at 0.2 and 1.6502 at 0.3.
  This document expected the optimum BELOW era four's 0.3, because the
  model is 42 percent smaller; it is not below it. A fourth run at 0.4
  answered whether 0.3 was merely the edge of the sweep — it reads 1.7159,
  thus 0.3 is an optimum and not a boundary. **The smaller model wanted
  the same regularization, and that is the finding.**

  The 96 K run at 0.3 elects the checkpoint of the era:
  `d64-mamba-n16-l6-do03-96k-s6.ckpt`, **valid 1.6482**, train 1.1668, and
  4.8528 over the moving steps. The parameter count is 178,504, exactly
  the arithmetic above.

  Three runs share one GPU without contending — 44 ms for each step
  against 59 alone, because one run of this size leaves the device idle.
- **The wider levers wait.** `N` in {32, 64}, layers in {4, 8}, and the
  levers the ear closed in era four reopen only in a second round, and
  only if the ear asks for one. The prototype trains four runs and
  stops training.

  **The ear asked, and the round of 2026-08-20 ran `N` and `K`.** Four
  runs at 48,000 steps, then the winner at 96,000 over three seeds. The
  baseline shape won both levers; the table is in "What the round
  measured". Depth is still closed.

The trainer draws its own randomness and its checkpoints alone cross the
language seam, as in era four. `infer.py` draws through the xorshift32
twin of `jax/prng.py`, thus one seed names one walk in JAX, in OCaml, in
the simulation and on the board, and the identical-walk gate of era four
runs here unchanged.

## The sizes

The estimate, by the arithmetic of the era-four tables — RAMB36 tiles of
4,096 bytes, int8 weights, int16 state. **The build has measured the
total, and it reads 57.5 tiles — 42.6 percent — against the 55 and the 41
this arithmetic gives.** It seats 53 RAMB36 and 9 RAMB18; the byte counts
below are right and the two and a half tiles are what each memory wastes
when it rounds up to a whole primitive.

| | six layers | era four, six layers |
|---|---|---|
| the layer weights | 165,192 B | 294,912 B |
| the tables | 13,312 B | 13,312 B |
| **the weights** | **178,504 B — 43.6 tiles** | **308,224 B — 75.5 tiles** |
| the state, int16 | 24,576 B — 6 tiles | — |
| the conv taps, int16, a ring of `K` | 7,680 B — 1.9 tiles | — |
| the KV rings | — | 196,608 B — 48 tiles |
| the small RAMs and ROMs | ~3 tiles | 2.5 tiles |
| the estimate | ~55 tiles — 41% | 126 tiles — 93% |
| **the build** | **57.5 tiles — 42.6%** | **126 tiles — 93%** |

The compute of one step, at one term a cycle:

| Part | Multiplies |
|---|---|
| `W_in` | 18,688 for each layer |
| the convolution | 640 |
| the state update, two terms an element | 4,096 |
| the readout, `N + 1` terms an element | 2,176 |
| the gate and the activations | ~1,000 |
| `W_out` | 8,192 |
| **one layer** | **~35,000** |
| the chain: four readouts | 12,288 |

Six layers and the chain give about **220,000 multiplies for each
step**. The divisions of `rms_norm` and the control put the step near
300,000 cycles — **about 3 ms at 100 MHz**, against era four's 7 — and
`docs/mamba_rtl.md` carries that model. **The cycle bench measures
292,684 cycles, 2.93 ms**, and the estimate above was right. Both eras
stand far under the 8 ms floor of the wire, thus the wire stays the
constraint on the tempo, which is where era four left it. What the fall
buys is not tempo: it is the 52 percent of the block RAM and the timing
room that comes with it.

## What the prototype measured

Every number the prototype owed, and **none of them gated the plan**: a bad
number is a finding of the era, the chain ran to the board regardless, and
the verdict read them all at the end. The full battery — the many-seed
spreads, a repetition instrument — belonged to a second round if the ear
asked. It asked; "What the round measured" follows this section and
corrects it where a later measurement overturned it.

All of them are measured. The elected checkpoint, over six seeds and 512
steps for the texture:

| | this era | era four | the corpus |
|---|---|---|---|
| the loss on the canonical valid windows | **1.6482** | 1.6282 | — |
| over the moving steps | 4.8528 | — | — |
| parameters | 178,504 | 308,224 | — |
| silence | 9.51% | — | 4.19% |
| **the silence-arrival share** | **47.2%** | 67 to 73% | 99.2% |
| the gap, in steps | 15.8 | — | 9.9 |
| onsets for each step | 0.67 | — | 0.81 |
| the median duration | 4.0 | — | 4.0 |
| four voices sounding | 84.8% | — | 88.1% |

**The loss is 0.020 nats behind era four at 58 percent of the
parameters**, and it is the first number that compares across the two
eras. A state of 24 KB carries most of what a window of 192 KB carried.

**The silence-arrival share reads 47.2 percent here, and a later round
found that number to be noise.** The row above stands as it was measured;
"What the round measured" below shows that CADENCED over six walks carries
a standard error near 14 points, and that the two eras do not part on this
instrument at any sample size the prototype used. Read the row with that
correction.

The repetition check — does a walk restate a motif over a gap a state must
carry — is not built. The risk it was written for is real and the ear is
the instrument that should ask for it first.

## What the round measured

The prototype recorded its numbers and the ear heard the result as jittery
and unmelodic. **The quality round of 2026-08-20** asked where the deficit
lives. It ran four diagnostics, re-elected the draw, and swept the two
levers this document reserved. `jax/diagnose.py` holds the diagnostics;
every number reads on the canonical valid windows, which both eras share.

### The deficit is in the moving steps, and it is in the melody

4,745 of the 19,200 predicted steps move two voices or more. **The whole
deficit is in those, and none of it — none at all — is in the other
14,455.** Each era is read over its three 96,000-step seeds, thus every gap
stands beside the spread that says whether it is a gap.

| | era five, 3 seeds | era four, 3 seeds | gap | t |
|---|---|---|---|---|
| the loss over every step | 1.6459 ± 0.0074 | 1.6276 ± 0.0016 | +0.0183 | 4.2 |
| over the moving steps | 4.8050 ± 0.0185 | 4.7304 ± 0.0131 | **+0.0747** | 5.7 |
| **over the still steps** | **0.6089 ± 0.0109** | **0.6090 ± 0.0037** | **−0.0001** | **0.0** |

**The two eras predict a held chord equally well, to the fourth decimal.**
A state of 24 KB and a window of 192 KB are the same instrument where the
music stands still; the window is better only where the music moves. The
trap this document names — "a recurrence can drone politely" — is what the
numbers show, and the mean gap is its residue and never the story.

The four seat losses put the deficit in seat 3.

| seat | era five, 3 seeds | era four, 3 seeds | gap | t |
|---|---|---|---|---|
| soprano | 0.4120 ± 0.0008 | 0.4031 ± 0.0030 | **+0.0089** | 5.0 |

Of the four seats the soprano carries the largest gap, and the elected
checkpoints put it at twice the next one. The soprano is the melody, and
**"unmelodic" is literal.**

### The time constants are the seed's, and they predict nothing

Over a 1,024-step teacher-forced pass the runtime `dt * a` of each head
gives a half-life in steps. On the prototype checkpoint two heads of layer
2 hold 55 and 42 steps, and no head above layer 2 holds more than 7. Its
own 48,000-step checkpoint finds the same structure.

| layer | h0 | h1 | h2 | h3 |
|---|---|---|---|---|
| 0 | 12.4 | 14.5 | 17.5 | 12.3 |
| 1 | 0.5 | 0.3 | 1.2 | 0.3 |
| 2 | **55.2** | 1.5 | **41.9** | 4.1 |
| 3 | 4.0 | 4.3 | 2.5 | 1.8 |
| 4 | 2.9 | 7.1 | 3.3 | 3.9 |
| 5 | 1.1 | 0.8 | 0.9 | 1.3 |

**That collapse belongs to seed 6 and not to the model, and the round
found it out by training two more seeds.** At the same configuration seed 7
grows a head of 50.7 steps in the upper half and seed 8 one of 50.0, where
seed 6 tops out at 7.1. The three then rank: seed 7 takes the best moving
loss, seed 8 the best mean, and seed 6 — the one with no long memory —
sits between them. **The time constants of a trained state vary by seed and
do not predict the loss.** A3's decision rule fires or does not fire
depending on which seed it reads, and that is the finding about the rule.

**A run that opens the state on a phrase-scale ladder keeps it and plays
worse.** The C5 run of the sweep draws `dt_bias` so that the four heads of
each layer start on half-lives of 4, 16, 64 and 256 steps, and it draws
every other tensor exactly as the baseline does — the parameter count is
the same 178,504 and one tensor is different. The ladder holds through
48,000 steps: the upper half then reads 16.8, 75.6, 42.5, 34.1, 131.7 and
3,809 steps where the baseline reads nothing above 9. The loss is worse on
every number — the mean by 0.0152, the moving steps by 0.0345, the still
steps by 0.0089.

Training gives the state a long memory when it is asked to, and the model
is worse for it. **The memory of the state is not the lever, in either
direction**: a seed that grows one is no better, a seed that does not is no
worse, and a run forced to hold one is worse. This is the best-controlled
comparison of the round, because it is the only one that moves one tensor
and holds everything else.

### The elected checkpoint of the round

Three seeds at 96,000 steps, at the elected shape:

| seed | mean | **moving** | still | soprano | longest half-life, upper half |
|---|---|---|---|---|---|
| 6 | 1.6482 | 4.8262 | 0.6050 | 0.4120 | 7.1 |
| **7** | 1.6519 | **4.7919** | 0.6212 | 0.4128 | 50.7 |
| 8 | **1.6376** | 4.7970 | **0.6005** | 0.4113 | 50.0 |

The rule of the round elects on the moving-steps loss, thus **seed 7**, and
the rule is followed. It should be read for what it is: 0.005 nats against
a seed spread of 0.018 is not a separation, seed 8 takes the mean by 0.014
and the still steps as well, and the three are one model. All three are
kept. The ear decides among them, and it may hear what the numbers cannot.

### Under teacher forcing the two eras draw alike, and it means nothing

The round also read the draw where the corpus holds the context: the
probability each model gives the class a seat already holds, and the count
of classes that survive the elected floor at each of the 76,800 draws of
the valid windows.

| | era five | era four | the truth |
|---|---|---|---|
| the predicted hold | 78.69% | 79.04% | 78.80% |
| survivors, median | 1 | 1 | — |
| survivors, mean | 1.72 | 1.70 | — |
| survivors, p90 | 4 | 3 | — |

Both are calibrated within a quarter of a point, and the survivor counts
are the same but for the upper decile. **The instrument said the two models
draw alike and the walk says they do not** — over 32 walks their hold rates
part by four standard errors. The reason is that a walk visits its own
states and a forced pass visits the corpus's. **Measure the draw on the
walk, not on the windows**; that is what this diagnostic is worth, and the
next section is the measurement that works.

### The draw did not transfer between the eras

The elected draw — temperature 1.0, min-p 0.05 — was elected on era four's
logit shape. A grid of temperature {0.7 to 1.0} against min-p {0.05, 0.1,
0.15} put its best cell at the hot corner: every colder cell and every
higher floor moves the walk away from the corpus. Therefore the boundary
was tested, as the dropout sweep tested 0.4.

The instrument is the hold — the share of voice slots that repeat the step
before, which `jax/measure.py` now records beside the arrival. The corpus
reads 78.17 percent. Over 32 walks of 512 steps:

| | hold | onsets each step | gap |
|---|---|---|---|
| the corpus | 78.17 | 0.81 | 9.9 |
| era five at T 1.0 | 82.71 ± 0.37 | 0.64 ± 0.01 | 12.4 |
| era five at T 1.2 | **78.97 ± 0.41** | **0.78 ± 0.02** | 10.3 |
| era four at T 1.0 | 80.75 ± 0.23 | 0.72 ± 0.01 | 10.4 |

**This model's logits are sharper than era four's by about 0.2 of
temperature.** At the shared policy era five holds 2.0 points more than era
four — four standard errors — and plays 0.08 fewer onsets for each step,
eight standard errors. Era five at T 1.1 reproduces era four's ear-elected
row: hold 81.16 against 81.07, onsets 0.70 against 0.71. The ear therefore
heard era five at a texture era four would have had near T 0.9.

**The offset holds on all three 96,000-step seeds**, over 16 walks each:

| checkpoint | T 1.0 | T 1.2 |
|---|---|---|
| seed 7 | hold 82.13, onsets 0.66 | **hold 78.15, onsets 0.81** |
| seed 8 | hold 82.28, onsets 0.66 | hold 79.14, onsets 0.78 |
| seed 6 | hold 82.71, onsets 0.64 | hold 78.97, onsets 0.78 |

The corpus reads hold 78.17 and onsets 0.81. **Seed 7 at T 1.2 lands on
both to the second decimal.** The ear elects the policy;
`jax/mamba/quantized.py`'s `ELECTED_TEMPERATURE` stands at 1.0 until it does,
and a change there is a change of the bitstream.

**The silence-arrival share does not rank models at these sample sizes.**
CADENCED divides by the silences of a walk; a walk of 512 steps holds three
or four, and two walks in 32 hold none at all. Over 32 walks the standard
error is 6.5 points and over eight it is 14. Era five reads 81.1 ± 6.1 at
T 1.0 against era four's 72.6 ± 6.5. The two eras are not parted by this
instrument, and the 47.2 percent recorded above was inside its own noise.
The instrument needs many more walks, or much longer ones, before it ranks
anything.

### No lever pays

The sweep, at 48,000 steps, dropout 0.3, seed 6, elected on the
moving-steps loss:

| Run | K | N | dt | parameters | mean | **moving** | still | soprano |
|---|---|---|---|---|---|---|---|---|
| **C1** | 4 | 16 | Mamba | 178,504 | **1.6502** | **4.8315** | 0.6059 | 0.4132 |
| C2 | 16 | 16 | Mamba | 190,024 | 1.7008 | 4.9971 | 0.6188 | 0.4183 |
| C3 | 4 | 64 | Mamba | 217,672 | 1.6911 | 4.9024 | 0.6370 | 0.4160 |
| C4 | 16 | 64 | Mamba | 236,104 | 1.8284 | 5.4840 | 0.6284 | 0.4496 |
| C5 | 4 | 16 | ladder | 178,504 | 1.6654 | 4.8660 | 0.6148 | 0.4140 |

**The baseline wins every number against every lever.** A wider kernel, a
wider state, both together and a phrase-scale state all read a worse
moving-steps loss than K 4, N 16 and the Mamba draw.

Two readings, and they are not equally strong. **C5 is clean**: same
shape, same parameter count, one tensor drawn differently, and it loses.
**C2 to C4 are confounded**: each adds parameters at a fixed dropout of
0.3, each reads a lower train loss and a higher valid loss, thus the
sweep tests "more capacity at this regularization" and not "more
capacity". To re-tune the dropout for each shape was outside the closed
budget. What survives the confound is the observation beside the loss:
N 64 grows a phrase-scale head in the upper half — 48.2 steps at layer 4,
where every other Mamba-draw run tops out under 8 — and the music does not
follow it.

### The chain, at the elected shape

`K` left the library as a constant and entered `Mamba.Config` as a field;
`Config.of_checkpoint` reads it from the kernel tensor, and the tap ring,
the age mux, the conv op and the cost model take it from there. The Verilog
regenerated at K 4 is **byte-identical** to the Verilog the prototype build
was made from, thus the refactor is proven a no-op at the shape the board
runs; a new stream gate at K 16 and N 32 over three layers proves it
correct where it is not.

Every gate was re-run at the elected checkpoint: the JAX forward against
the OCaml reference, the two walks, the drift of the twin over 1,024 steps
— top-1 85.2 percent, cosine 0.9894, `dt` and the state never clamped and
`beta` at 0.0117 percent — the frame and stream gates at one, two and three
layers and at seed 0, and the cycle bench. **The six-layer build meets at
+0.197 ns with 3,144 LUTs and the same 57.5 tiles**, and its bitstream is in
`board/_build`. It is not programmed and not flashed.

### What the round did not build, and it is a finding

The diagnostics say WHERE the deficit is. They do not say why, and the
instrument that would separate the two candidate causes was not built,
under the rule that a round records the instrument it wanted instead of
growing one: **era four's advantage as a function of the position inside
the window.** If era four's moving-step advantage grows with the history
behind the position, the advantage is a look-back advantage and a hybrid
attention layer is the indicated lever; if it is flat, the advantage is
capacity or optimization, and attention buys nothing. It is one forward
pass over the windows the diagnostics already cut, and it should open the
next round.

**What the numbers argue for, as they stand.** The deficit is in the
moving steps and in the melody — where a chorale restates its motifs, and
where a look-back mechanism would pay. Four levers over the state and the
kernel all failed to move it, and the one that gave the state a real
phrase-scale memory made it worse. That is evidence against more state and
for a different mechanism, and it is not proof: the gap is 0.096 nats over
a quarter of the steps, era five holds its own everywhere else, and a
hybrid could lose what the state won. The measurement above is what would
settle it.

## The traps

**The state is the sensitive tensor.** The research round says so and
the design answers it — int16, never a coarse byte — but the drift
report is the proof, and a drift that era four's floors would pass can
still hide in the recurrence: an error carries forward here, where an
attention error died with its step. The drift walk must therefore run
long, past many state lifetimes, not just past one window.

**The loss compares across eras four and five, and only there.** Both
speak the frame; era three spoke tokens and its numbers compare to
neither. Rank models inside the encoding by loss, across encodings by
ear, as always.

**A recurrence can drone politely.** 77.91 percent of the voice slots
repeat the step before, and a state that decays too fast degenerates to
exactly that predictor. Watch the moving-steps loss and the texture, not
the mean.

**`dt` saturation.** A trained `dt` that rides its clamp is a silent
disagreement between the float model and the twin. The drift report
prints the share of clamped `dt` draws. A share above noise is a
finding for the record: it reopens the format in a later round, and the
prototype proceeds on the format it has.

## The plan

**Every step runs. The plan has one human step, and it is the last.**
The gates inside the plan are gates of equivalence — one link against
the one before it — and they must pass; the numbers of quality are
recorded beside them and gate nothing. A step that meets a real fault
fixes it or records it and continues; only a broken equivalence stops
the chain, because a chain with a broken link proves nothing.

1. **The JAX prototype.** `jax/mamba/`: the block above, the trainer,
   the sampler. The gate: the loss on the canonical windows prints, the
   recipe of era four runs unchanged, a 48 K run converges — converges
   means the valid loss descends and levels, not that it reaches any
   particular number.
2. **The training round.** The closed budget above: three dropout runs
   at 48 K, the best at 96 K. Record the loss, the moving-steps loss and
   the texture numbers of the elected checkpoint against era four's.
   **Whatever they read, this checkpoint carries the plan.**
3. **The reference.** `lib/mamba/mamba.ml`: the float model, the loss
   and the sampler, on `Nx`, with the gates of era four — the loss on
   the same windows to the printed digit, the identical walk from one
   seed. The sampler and the player come with it; the trainer does not.
4. **The twin.** `lib/mamba/quantized.ml`: int8 weights with power-of-two
   exponents, the formats of `docs/mamba_rtl.md`, the two new tables,
   and the drift report over long walks — the seed-0 walk and the
   dt-saturation share inside it. The drift floors calibrate on this
   model's own first measured minima, as era four's did; they hold the
   scheme thereafter, and the measured level itself is a finding.
5. **The circuit.** `lib/mamba/source.ml` on the unchanged socket, by
   `docs/mamba_rtl.md`. The frame gate holds it to the twin — at two
   layers as well as one, from the first day. This gate is exact and
   stays exact.

   **STEPS 3 AND 4 WERE CUT 2026-08-29 by the all-era cut.** The OCaml
   float model and the OCaml integer twin both went; the JAX model is the
   float reference and `jax/mamba/quantized.py` is the twin. The gates of
   steps 3 to 5 read the same way from above the seam: `test_parity.py`
   holds the float model to a pinned loss and the JAX quantizer to a
   pinned netlist md5, `test_drift.py` carries the drift sweep with its
   floors, and `test_rtl_mamba.py` holds the circuit to the twin — frame
   for frame and write for write — through `gate_mamba.exe`.
6. **The build.** `gen_verilog` seats the mamba source with its
   checkpoint constant; the six-layer Vivado build runs, and the timing
   and utilization reports land beside the estimates of this document.
   A build that misses timing is recorded with its slack and the chain
   is still whole; the fix is its own round.
7. **The ear, and the board.** The one human step. A person programs
   the board, runs the amidi capture against the reference, and
   auditions the era against era four. The ear elects, or it closes the
   era with the whole chain in view — and either answer is a result the
   six steps above have already paid for. Era four stays in the tree
   and in the flash until this step says otherwise.
