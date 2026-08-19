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
and the training window is a training choice alone.

**The step cost falls and stops growing.** Attention walks the ring at
every step: two passes of `T * d` terms for each layer, 32,768 of the
81,920 multiplies of an era-four layer. The recurrence costs a constant
that does not know `T`. The estimate below puts the step at about 2.4 ms
against era four's 7, on the same one-multiplier machine.

**The block RAM falls from 93 percent to about 41.** The six-layer build
of era four closes timing at +0.031 ns and does not close without the
post-route pass, and the reason is routing at 126 of 135 tiles. The
budget below puts this model near 55 tiles. The margin that era four
fights for, this design buys back with room to spare.

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
signal by construction, thus ALiBi has nothing to bias and the design
carries none.

### The baseline shape

| | value | why |
|---|---|---|
| `d` | 64 | era four's width; the embedding and the head carry over unchanged |
| layers | 6 | era four's depth; one variable at a time |
| `d_in` | 128 | the standard expansion of 2 |
| `H` | 4 | one head for each voice is a coincidence; `P = 32` is the reason |
| `N` | 16 | the Mamba-1 default; the budget affords 32 and 64, and the sweep asks |
| `K` | 4 | the Mamba default |
| `G` | 1 | shared B and C; the model is small and the groups buy nothing here |

The head width `P = 32` is a power of four, thus the shift rules of the
machine hold as they hold in era four. `N` is the one lever this
document expects the sweep to move: the state is the experiment, and 16
against 32 against 64 is the first question after the recipe stands.

### The parameters

For each layer: `W_in` is `d x (2 d_in + 2 N + H)` = 64 x 292 = 18,688;
the convolution is `(d_in + 2 N) x K` = 640; `W_out` is `d_in x d` =
8,192; `dt_bias`, `A_log` and `D` are `H` = 4 each. **27,532 for each
layer.** The tables are era four's: the seat tensor 12,288 and the bar
phase 1,024.

Six layers: **178,504 parameters against era four's 308,224** — 58
percent, at the same width and depth. The state buys its memory at run
time, not in the weights.

### The checkpoint

Safetensors, the tensors named "0" upward in the flat order, as era
four: the two tables, then six tensors for each layer —
`w_in, conv, dt_bias, a_log, d_skip, w_out`. The reader of
`Config.of_checkpoint` takes the width and the layer count from the
shapes, as before. One `Params_data`-shaped structure states the order
once for the float model and the quantized twin.

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
  the definition; the OCaml reference and the circuit compute that one.
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

The trainer draws its own randomness and its checkpoints alone cross the
language seam, as in era four. `infer.py` draws through the xorshift32
twin of `jax/prng.py`, thus one seed names one walk in JAX, in OCaml, in
the simulation and on the board, and the identical-walk gate of era four
runs here unchanged.

## The sizes

The estimate, by the arithmetic of the era-four tables — RAMB36 tiles of
4,096 bytes, int8 weights, int16 state. **These are estimates until a
build measures them.**

| | six layers | era four, six layers |
|---|---|---|
| the layer weights | 165,192 B | 294,912 B |
| the tables | 13,312 B | 13,312 B |
| **the weights** | **178,504 B — 43.6 tiles** | **308,224 B — 75.5 tiles** |
| the state, int16 | 24,576 B — 6 tiles | — |
| the conv taps, int16, a ring of 4 | 7,680 B — 1.9 tiles | — |
| the KV rings | — | 196,608 B — 48 tiles |
| the small RAMs and ROMs | ~3 tiles | 2.5 tiles |
| **total** | **~55 tiles — 41%** | **126 tiles — 93%** |

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
`docs/mamba_rtl.md` carries that model. Both eras stand far under the
8 ms floor of the wire, thus the wire stays
the constraint on the tempo, which is where era four left it. What the
fall buys is not tempo: it is the 52 percent of the block RAM and the
timing room that comes with it.

## The measurements owed

Every number a run must produce and **record. None of them gates the
plan**: a bad number is a finding of the era, the chain runs to the
board regardless, and the verdict reads them all at the end. The full
battery — the many-seed spreads, a repetition instrument — belongs to a
second round if the ear asks; the prototype records these on its one
elected checkpoint:

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

**The silence-arrival share went the wrong way, and that is the honest
result of the era.** The state was a new lever on exactly era four's open
question — whether a walk arrives somewhere before it goes quiet — and it
reads 47.2 percent where era four read 67 to 73 and the corpus reads 99.2.
The walks are also more silent and their gaps are longer. The design
document said this measurement was not a hope; it was not.

The repetition check — does a walk restate a motif over a gap a state must
carry — is not built. The risk it was written for is real and the ear is
the instrument that should ask for it first.

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
