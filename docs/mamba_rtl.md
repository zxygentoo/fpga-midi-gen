# The Mamba source in RTL

## Scope

The state-space model of era five on the board: one step of music is one
step of the recurrence and one 32-bit frame on the socket. The model is
`docs/mamba.md`, and the schematic is `docs/mamba_rtl.svg`.

**The circuit is built.** This document was written before it, and the
paragraphs that a measurement has since replaced say so where they stand:
each one keeps what the design decided and adds what the implementation
found. The build numbers are in `build-log.md`.

The design keeps the rules of era four. The reference of the circuit is
exact integer arithmetic — the twin, `jax/mamba/quantized.py` — and the
circuit must match it bit for bit: `jax/tests/test_rtl_mamba.py` states
what the circuit must do, and `bin/gate_mamba.ml` prints what it did. The
float model is not the reference of the circuit; the drift report
measures what the quantization costs.

The modules of the era:

| Module | It owns |
|---|---|
| `jax/mamba/model.py` | the float model: the plan, the block, the head, the loss, the sampler |
| `jax/mamba/quantized.py` | the quantizer of the checkpoint, and the integer twin: the recurrence, the chain and the sampler |
| `Mamba.Model` (`lib/mamba/model.ml`) | the model as the circuit reads it: the formats, the plan, the contract file and the ROM image |
| `Mamba.Source` (`lib/mamba/source.ml`) | the same integers as a circuit: the schedule, the datapath and the socket machine |
| from `mgen_nn` (`lib/nn/`) | the common home of the sources: the units — `Mac`, `Divider`, `Isqrt`, `Exp2`, `Sigmoid`, `Softplus` — the draw of the chain (`Sampler`), and the shared integer rules the circuits read. The quantizer and the sampling policy stand above the seam, in `jax/fixed.py` |

**The units live in `lib/nn`, and the unification round put them there.**
The prototype imported era four's units as they stood and copied the two
that could not come whole; the promised common-home round then moved all
of them, and the history of the two copies is the design content of two
units:

- **`Mac` takes its walk width as a functor argument.** Era four's
  longest walk ran 256 rows and takes nine bits; the state update here
  walks `d_in * N` rows — 2 048 at the baseline, and 8 192 if the state
  sweep ever reaches 64 — and takes fourteen. Each source instantiates
  its own width, thus both netlists stand as their boards proved them.
- **`Divider` takes the magnitude inside the walk.** Era four's original
  negated in the start cycle, which put a 40-bit carry chain between the
  caller's operand mux and the first register. One writer of the
  numerator closed; the head is a second writer, and the build read the
  program counter's mux in front of that carry chain as the critical
  path of the whole design, at −0.081 ns. The magnitude stage cut it —
  `busy_cycles` 41, one cycle more for each divide — and the one unit
  now serves both eras, thus era four pays the cycle too and its cost
  model follows `Divider.busy_cycles` as it always did.

The op and schedule layer is **restated** in `lib/mamba/source.ml`, not
shared: the op vocabulary is different, and the abstraction of era four
is an open question by standing rule — an improvement to it is a
discussion, not a side effect of a branch. The unification round left it
out on that rule.

## What the state changes

**The state arrives**, and it is the one thing this machine holds that
era four's did not: memory that survives the step. Its rules — the
formats, the zero origin, the read-modify-write — are the new content of
this document.

**Two tables arrive**: the sigmoid and the softplus correction — a
registered read, no start and no busy, the caller holds the input two
cycles. That was `Exp2`'s idiom too until the backport of 2026-08-29 gave
it a registered magnitude and a weight every cycle; these two keep the
older shape, because no caller here wants one a cycle.

**The KV rings and `Attend` do NOT leave, and this document said they
would.** The trunk needs none of them: the state RAM takes their seat,
24,576 bytes against 196,608, written and read in place, never windowed,
never wrapped. The elected model is not a trunk. One layer of it is the
Zamba head, thus one ring stands — 32,768 bytes at ring 256 — with the
slot arithmetic, the fill count `n`, the age walk and the causal wall
that come with it. The paragraphs this replaces are kept in the git
history; what they got right is that a recurrence has no ages, and what
they got wrong is that the model would be a recurrence alone.

Three things follow from the head, and each one is a rule of the machine
that had to be restated:

- **`Divider` serves the head as well as the two norms.** The softmax
  denominator divides every merged lane. The head is thus a second writer
  of the numerator, and that is what moved the magnitude into the
  divider's walk — the import section above carries the path.
- **The step cost takes the fill again.** `Op.cycles` takes `n`, and
  `Attend` is the one op that reads it: the trunk's cost is a constant of
  the shape, and a step grows until the ring is full and is constant
  after that.
- **No walk stalls, and that rule survives.** Era four merged the lanes
  of a head in ONE walk and froze that walk while the divide of a
  finished row ran, thus every read register and every tag of that
  machine carried a freeze enable. This machine merges ONE LANE A WALK
  and waits on the divide with the walk already retired. It costs the
  drain of each lane — 16 of them a head — and it buys back the enable on
  every read register of the design, thus `Mac` still takes its hold at
  ground.

**The normed embedding stands for the whole step**, in a memory of its
own of `d` by 16 bits. Every other vector of this machine dies inside its
layer; this one is written once, after the embed, and the head of the last
layer still reads it. That is what the Zamba query and key are.

**A layer's place in the plan is not its place in a memory.** The state
RAM and the tap ring hold one region for each BLOCK and the rings one for
each HEAD, thus an op carries the ordinal of its own kind and never the
index of the layer: the seventh layer of the elected plan owns ring 0.
`Mamba.Config.ordinals` states the map, and the reference and the circuit
address their memories through the one definition.

## The socket

`Source_intf` is unchanged, to the field. `step` answers with a frame
the source has already drawn; `rewind` is the one reset and loads the
PRNG from SEED; `valid` and `idle` keep their contracts. The sequencer,
the decode and the board around the socket do not know the era changed.

## The integer model

### The weights

Int8 with a per-tensor power-of-two exponent, `w ~ q * 2^-e`, the
largest `e` that keeps `round(max|w| * 2^e)` at 127 or less — the rule
of era four, unchanged, and the same quantizer above the seam and the
same `Model` reader below it. The seat tensor and the bar phase share one
exponent because their rows add; that rule and its check carry over.

`a_log`, `dt_bias` and `d_skip` are `H` values a layer. They quantize at
elaboration into the constants the ops carry — `a * log2(e)` folds into
one Q constant for each head, as era four folded log2(e) into the
temper — thus the run time never sees them as tensors. An int8 tensor
could not hold them in any case: the bias enters a softplus, thus a step
of one part in 127 of its range moves `dt` by more than a small `dt` is.

**Therefore the ROM image is not the checkpoint.** A checkpoint layer
holds six tensors and three of them never reach the image; the image holds
the two tables and then `w_in`, `conv` and `w_out` of each layer. The two
orders are two structures and neither is implied by the other.

**And `w_in` is stored transposed.** The circuit reaches a weight by
CONCATENATING the two walk counters, which costs nothing and is the
row-major address only when the dimension under the outer counter is a
power of two. `d` is one; the projection width — 292 at the baseline — is
not. Storing that one tensor the other way round puts `d` under the outer
counter and the concatenation is right again. The alternative was a
constant multiply on the ROM address, and era four's measurement says not
to: that address cone is the path the whole layer scaling of that block
turned on.

### The formats

The starting rules of the prototype. They are chosen with margin and
the build proceeds on them; a later round meters them on the trained
checkpoint as era four metered its own, and until then a clamp that
fires or a signal that runs hot is a finding for the drift report to
record, not a bar to the build:

| Signal | Format | Why |
|---|---|---|
| residual `h` | int32, Q16 | era four's, unchanged |
| normed `y` | int16, Q12 | era four's, unchanged |
| `x`, `z`, `B`, `C` after conv and SiLU | int16, Q12 | the working class of the datapath |
| `dt` | int16, Q12, clamped at the format | softplus output; the drift report prints the clamped share |
| the decay `alpha` | uint16, Q15 | the exp2 output, peak 2^15 at dt 0 |
| `beta = dt * B` | int16, Q15 | the state-inject operand; see the state update |
| **the state `S`** | **int16, Q12, clamped** | **the sensitive tensor; never a coarse byte** |
| **the gate product** | **int32, Q24, whole** | **measured; see below** |
| sampler weights | uint16, Q15 | era four's, unchanged |

**The gate product is the one format this document got wrong, and the
drift report found it.** The plan truncated `y_out * silu(z)` back to the
working class before the gated norm. Both operands are Q12 values well
under one, thus the truncation kept about five bits of a product that
holds seventeen — and it threw them away immediately before the one
operation that would have used them, because a norm divides by the size of
its vector and does not care what scale it arrives in. Measured on the
gate shape, the truncation cost 0.10 of the cosine on its own: 0.87
against 0.999. The product now stays whole into the norm, which is the
format the era-four stream already had — Q16 in an int32, normalized down
to Q12 — one axis further in.

**Nothing else clamped.** The drift walk counts every clamp of `dt`, of
`beta` and of the state, and over the gate shape, over 1 024 steps, and on
a trained checkpoint at the shape of the board, every one of them reads
zero. The margins this document chose with no measurement behind them
hold, thus the metering round it defers is deferred on evidence.

The state keeps full int16 where the era-four ring kept a top byte. The
research round says the state is where SSM quantization breaks, the
budget has 52 percent of the block RAM free, and an error in the state
carries forward where a ring error died with its window. The drift walk
must run long — past many decay lifetimes — because state error is
cumulative in a way era four never had.

### The operations

Each operation is one definition in `jax/mamba/quantized.py`, and the
circuit computes the same integers. Every product fits one DSP48, 25 by 18
signed. `rms_norm`, the embed, the chain and the sampler are era four's
operations unchanged. The new ones:

- **conv**: for each of the `d_in + 2 N` channels, a row of `K` terms —
  the taps against the channel's kernel — then the SiLU chain on the
  sum. The taps live in a small ring of `K` for each channel and layer;
  tap `k` reads zero while `position < k`, thus the origin needs no
  clearing and the rule is a mux, as the era-four fill count was. `K` is
  4 at the elected shape and it is a **field of the configuration**, read
  from the kernel tensor: the ring depth, the age mux and this op all
  size themselves from it. The address rule wants a power of two, and
  `check_shape` asserts it. A gate at `K` 16 over three layers holds the
  circuit to the reference stream write for stream write.
- **silu**: one sigmoid table read, one multiply, one shift:
  `(v * sigmoid_q(v)) >> 15`. A bespoke chain in the idiom of the
  exp-weight chain, walked over the conv outputs and again over the
  gate `z`.
- **decay**: for each head, `dt = relu(raw + bias) + softplus_table(|raw + bias|)`,
  then `alpha = exp2_q(dt * a_log2e_const)` — one table read, one DSP
  multiply, one table read. Six values a layer for each step.
- **state update**: for each element of the head's `P x N` block, one
  row of **two terms** on the walk:
  `S' = (alpha * S + x_p * beta_n) >> 15`, clamp16. `alpha` is Q15 on
  the 18-bit port with `S` Q12 on the 25-bit port; `beta` is Q15 with
  `x` Q12; both products land Q27, thus one row sums them and one shift
  lands Q12. The `beta` row — `N` products of `dt` against `B` — is a
  bespoke chain into the shared RAM before the walk, as the temper
  writes its weights.
- **readout**: for each `p`, a row of `N + 1` terms:
  `y_out[p] = (sum_n S[p, n] * C[n] + x[p] * d_skip_q) >> 12` — the
  skip folds into the walk as the last term, with `d_skip` quantized to
  Q12 so every product of the row lands Q24. Reads the state RAM the
  update just wrote.
- **gate**: the SiLU chain over `z`, then one multiply against `y_out`,
  then `rms_norm` — the gated norm of Mamba-2 — into the join.

The joins to `h` — `W_out` and the chain's accumulate — are era four's
`join_to_h`, unchanged.

### The state update in place

The state RAM is the one memory this machine modifies rather than
rewrites, and the hazard is stated here so the implementation does not
rediscover it. The update walk reads `S[i]` at cycle `t` through the
two-register read and writes `S'[i]` back at about `t + 4`; the walk
advances one element a cycle, thus the write of `i` lands while the
reads are at `i + 4`, and no read of the walk sees a stale row —
**provided the walk never revisits an address**, which a linear walk
does not. The readout op runs after the update op retires, on the op
boundary, thus it reads only finished state. A hold freezes the read
registers and the walk together, as every era-four memory does.

**The origin is a mux, not a clearing walk.** At `position = 0` the
update reads the state as zero — one mux on the read data, driven by
the step counter — and the conv taps read zero by their age rule.
`rewind` therefore stays what era four made it: load the PRNG, clear the
counters, run nothing. `idle` never falls for it.

## The machine

The five layers stand:

| Layer | What it is |
|---|---|
| L0 | the units of `mgen_nn` — `Divider` with the magnitude inside the walk, `Isqrt`, `Exp2`, `Sigmoid`, `Softplus` — and `Prng.Rtl` from the core |
| L1 | the datapath: the RAMs, the state RAM, the tap rings, the banked weight ROM, `Mac` |
| L2 | the schedule: the step as a list of operations, built from the config |
| L3 | the compiler, `Mgen_nn.Program`: the list folds into the cases of a program counter. The four draw ops it compiles are `Mgen_nn.Sampler` |
| L4 | the outer machine: the step strobe, the lead-in, the held frame |

L3 carries over LITERALLY and not by convention: since the op/schedule
round it is one text, `Mgen_nn.Program`, which both eras call — the
op-finish-runs-next-entry rule, the single `switch` on the pc, the tick
counter that steps itself, the seat register and the four-times-one-seat
chain. L4 carries over structurally and stays each era's own: the rewind
and the step strobe are ten lines, and era four clears a ring slot where
this era has none. The lead-in that draws nothing and moves no PRNG is
`Program`'s too. The forward program changes its op list; the chain
program is era four's seven ops, restated.

One thing carried over that this machine turns out not to need:

- **The hold is gone.** Era four's attention stalled its merge walk while
  a pending divide finished, thus every read register and every tag
  carried a freeze enable. No walk here ever stalls: the two norms divide
  in a bespoke chain that waits on its own tick, and the head merges one
  lane a walk and waits after the walk retires. The enables are not built,
  and `Mac` takes its hold at ground.

**The slot and the fill need no registers, and that is what the step
counter buys.** Era four held a slot register and a filled flag; here the
newest slot is the low bits of the step counter and the ring is full once
that counter passes the depth, thus the two facts are slices of a register
the machine already had.

One thing this machine needs that era four never did. **Three ops choose
which memory feeds the multiplier by the position inside the row** —
the state update alternates two terms, the readout folds the skip in as
its last, and the inject walk selects a head's step size. Era four's ops
each read one memory. A choice like that must follow the DATA and not the
address: the counters carried forward by the read latency are what the
operand mux reads, and using the live counter there selects the wrong
memory two cycles early.

The op vocabulary follows era four's closed-and-concrete rule: when a
field's meaning would depend on another field, write a new op. The list as
built: `Embed`, `Rms_norm`, `Matvec`, `Conv`, `Silu_over` (a range),
`Decay`, `State_update`, `Readout`, `Gate`, `Attend`, and the chain's
`Temper`, `Draw`, `Threshold`, `Pick`, `Accumulate`. The schedule prints
as data and an expect test pins it, as before.

`Rms_norm` and `Matvec` each carry one field that names a whole set of
facts that move together, and neither field's meaning depends on another:

- `over` says which vector a norm reads, at what format, over what width,
  and into which memory: `Stream` is Q16 over `d` into the y RAM,
  `Embedding` the same vector into the embedding RAM, `Gated` the gate
  product at Q24 over `d_in`.
- `src` says which memory feeds the multiplier: `Y` the normed vector,
  `Hidden` the feed-forward hidden, and `Joined` the PAIR — the normed
  stream then the normed embedding, `2 d` terms, which is the Zamba query
  and key. `Joined` is the fourth operand in this machine to follow the
  DATA and not the address: the top bit of the inner counter carried
  forward by the read latency selects the memory, and the live counter
  would select the wrong one two cycles early.
- `landing` says where a finished row goes: `To_v`, `To_q`, `To_ring`
  (the top byte, into the newest slot), `To_hidden` (a ReLU at Q10),
  `To_logits` and `Add_to_h`.

`Attend` runs one head after another in four stages: the scores of the
ages, the exp2 weight of each age over its own score, then one merged lane
a walk and the divide that lands it. Its context lands in the y RAM, thus
the output projection is an ordinary `Matvec` over `Y` and needs no
landing of its own.

The timing rules of era four are inherited as rules, not re-derived:
every read two cycles from address to data, the ROM's first cycle on the
address side (the register is load-bearing — the retiming trap is
measured and recorded in era four), the DSP a two-register multiply with
the fabric accumulator behind it, the banked ROM under RAM_STYLE with a
gated-off write port. At 41 percent of the block RAM the placement
pressure that forced era four's travel stages may relax; **remove
nothing until a build measures it** — the stages cost fill latency only,
and the era-four numbers say what removing them risks.

The dormant debt of era four — tick positions hand-encoded against the
two-cycle product latency — is inherited by every bespoke chain here.
The same rule: if the pipe deepens, replace ticks with a wait on a
valid bit; do not renumber.

## The memories

| Memory | Size | Content |
|---|---|---|
| weight ROM | 235,776 x 8 at the elected plan | the image, flat order |
| exp2 ROM | 256 x 16 | era four's table, from `Mgen_nn.Quantized.Constants` |
| sigmoid ROM | 256 x 16 | Q15 over signed Q12 in, clamped at |v| = 8 |
| softplus ROM | 256 x 16 | the correction term, Q12 over |v| up to 16 |
| **state RAM** | **6 x 128 x 16 x 16 b = 24,576 B** | the recurrence; int16, in place |
| tap rings | 6 x 160 x `K` x 16 b = 7,680 B at `K` 4 | the conv inputs, a ring of `K` |
| **the key and value rings** | **2 x 256 x 64 x 8 b = 32,768 B** | the head's context, a coarse byte |
| `h` | 64 x 32 | the residual stream, and the stream of the chain |
| `y` | 64 x 16 | the normed vector, and the head's merged context |
| `e` | 64 x 16 | the normed embedding, live for the whole step |
| `q` | 64 x 16 | the head's query |
| shared RAM | 512 x 32 | `zxbcdt`, the SiLU outputs, the scores and age weights, the feed-forward hidden, and the logits and sampler weights |

**The estimate was about 80 tiles and the build reads 80.5.** The
arithmetic of this table said 59 percent and the build says 59.63 — the
one number of this document that a build has never yet moved.

| | the elected plan | era five's trunk | era four |
|---|---|---|---|
| block RAM tiles | **80.5 of 135 — 59.6%** | 57.5 — 42.6% | 126 — 93% |
| RAMB36 / RAMB18 | 75 / 11 | 53 / 9 | — |
| slice LUTs | 3,447 | 3,144 | 3,061 |
| slice registers | 1,814 | 1,652 | — |
| DSPs | 2 of 240 | 2 | — |
| WNS | **−0.081 ns — MISSED** | +0.197 | +0.059 |
| WHS | +0.040 ns | +0.073 | — |

The quality round of 2026-08-20 swept `K` and `N` and kept this shape. What
the other shapes would have cost, by the arithmetic of this table — none of
them was built, because none of them won:

| K | N | weights | state RAM | tap rings | total |
|---|---|---|---|---|---|
| **4** | **16** | 43.6 t | 6 t | 1.9 t | **~55 t** (built: 57.5) |
| 16 | 16 | 46.4 t | 6 t | 7.5 t | ~63 t |
| 4 | 64 | 53.1 t | 24 t | 3 t | ~83 t |
| 16 | 64 | 57.6 t | 24 t | 12 t | ~97 t |

`N` is what costs: the state RAM grows with it and the state RAM is the one
memory this design writes every step. Every shape still fits under era
four's 126 tiles, thus the block RAM was never the reason to keep the
baseline — the loss was.

The shared RAM's depth is set by `zxbcdt` at 292 entries, rounded to the
memory the tools infer; the scores of a head, the feed-forward hidden, the
logits and the sampler weights all reuse it, and none of them is wider —
256 slots, 256 hidden lanes, 48 classes. It is 32 bits wide because the
gate product is. The tap rings may infer as distributed RAM at this size;
either is fine, and the build reports which. Two small RAMs the plan did
not name are in the tree: the readout, 128 by 16, and the inject operands,
64 by 16. The readout wants a memory of its own because the gate reads it
and the shared RAM in one cycle.

**The key and value ring keeps era four's coarse byte**, and the argument
of this document is the one that says why: the state must not coarsen
because a state error carries forward, and a RING error dies with its
window. Era four shipped six such rings. The block RAM is there to widen
this one, and the drift sweep of `jax/tests/test_drift.py` records what
widening it would buy — on a trained checkpoint the whole model reads
92.7 percent top-1 against the float twin at a cosine of 0.984, which is
BETTER than the trunk alone read at 88.7 and 0.982.

**The three addresses that carry a region take two different rules**, and
the reason is which fields are powers of two:

- the STATE address and the RING address pack — `d_in`, `N`, `d` and the
  ring depth are all powers of two, thus the region stands above the row
  and the whole address is a concatenation;
- the TAP address adds — the channel count `d_in + 2N` is 160 at the
  baseline and no power of two, thus a concatenated block field would
  stride by the rounded-up power, every block above the first would sit at
  the wrong base, and the top block's region would run off the end of the
  memory. One constant add puts the circuit on the reference's own
  address and wastes no row.

This is the fault a one-layer or two-layer simulation cannot see, and it
is why the gates run at three blocks and at two heads.

## The cost

The analytic model, at one term a cycle, at the elected plan `MMMMMMZF`.
`Op.cycles` states it exactly per op, and the cycle bench holds it to
the measured circuit; this table is the sum to one significant figure:

These are the numbers `Op.cycles` states, and the cycle bench holds the
measured circuit to every one of them.

| Part | Cycles for each layer | the estimate this replaces |
|---|---|---|
| `rms_norm` at `d` | 2,906 | — |
| `W_in` matvec | 18,692 | 18,688 |
| conv, then SiLU over 160 channels | 808 + 960 | ~2,000 |
| decay chains | 36 | ~50 |
| state update: the inject walk, then 2-term rows | 4,168 | ~4,100 |
| readout, 17-term rows | 2,180 | ~2,200 |
| SiLU over the gate, then the gate | 768 + 132 | ~1,300 |
| the gated norm at `d_in` | 5,786 | — |
| `W_out` matvec + join | 8,197 | ~8,200 |
| **one block** | **44,633** | ~45,000 |
| the embed | 324 | — |
| the embedding norm | 2,906 | — |
| one seat of the chain | 6,492 | — |

The head and the feed-forward, at a FULL ring of 256:

| Part | Cycles |
|---|---|
| `rms_norm` at `d` | 2,906 |
| `wq` and `wk`, `2 d` terms each | 8,196 + 8,196 |
| `wv` | 4,100 |
| `Attend`, four heads | 42,896 |
| `wo` matvec + join | 4,101 |
| **the Zamba head** | **70,395** |
| the feed-forward: a norm, then the two matvecs | 2,906 + 32,777 |
| **the feed-forward layer** | **35,683** |

**404,314 cycles a drawn step — 4.04 ms at 100 MHz**, against the trunk
alone at 294,090. The schedule test prints this number out of `Op.cycles`
at the elected shape, thus the document and the model cannot part. It read
403,074 until 2026-08-29, when the Exp2 backport gave the unit a registered
magnitude and the two chains that read it — the temper and the decay —
each waited one tick more.

**It is no longer constant in the walk, and the head is why.** `Attend`
walks the ages the ring holds, thus a step grows until the ring fills at
step 256 and every step after that costs the same. The first drawn step
holds 16 ages and costs 365,914; the steady step costs 404,314. The wire's
8 ms floor still stands over both, thus the source is never the tempo and
the growth is not audible.

## The board

- `gen_verilog` on this branch seats the mamba source: the checkpoint
  path is its constant, the elaboration loads and quantizes, the
  bitstream carries the weights. The transformer source and its
  `gen_verilog` stay in the tree untouched; the branch decides the seat.
- No new host-control cells: SEED, STEP_MS, CHANNEL and VELOCITY serve
  as they stand. The seed panel of the switches reads through SEED
  unchanged, and seed 0 is a legal walk here as everywhere.
- **The build is the deliverable of the branch; the board is the human
  step.** `build.tcl` runs and its timing and utilization reports land
  beside the estimates of this document — a miss of timing is recorded
  with its slack, and the chain is whole with it. Programming the board
  and `flash.tcl` wait for a person: the hardware and the ear both sit
  there. Era four stays in the flash until that person says otherwise.

**The six-layer build, 2026-08-20.** The prototype built the checkpoint of
seed 6; the quality round of the same day re-elected the shape, kept it,
and built the checkpoint of seed 7. Both are here, because the pair
measures what a checkpoint alone moves:

| | era five, seed 7 | era five, seed 6 | era four |
|---|---|---|---|
| WNS | **+0.197 ns** | +0.202 ns | +0.059 ns |
| WHS | +0.073 ns | — | — |
| slice LUTs | 3,144 | 3,043 | 3,061 |
| slice registers | 1,652 | 1,658 | — |
| block RAM tiles | **57.5 of 135, 42.6%** | 57.5 of 135, 42.6% | 126 of 135, 93% |
| DSPs | 2 of 240 | 2 of 240 | — |

The estimate of this document was about 55 tiles and 41 percent; the build
reads 57.5 and 42.6, and it reads the same 53 RAMB36 and 9 RAMB18 for both
checkpoints — the shape sizes the memories and the weights do not. **The
block RAM the design set out to buy back is bought back**, and the timing
margin that came with it is three times era four's on a design of the same
fabric size. Both `phys_opt_design` passes are in the script and the build
meets without needing what they give.

**A new checkpoint of the same shape costs 101 LUTs and 0.005 ns**, and
that is the timing lottery of this project seen from the smallest possible
distance: nothing but the ROM contents changed, and the tools placed a
different design. Read a slack difference of this size as noise.

## The tests

The gates of era four, inherited with the lessons of its test review
already applied. **A gate of equivalence must pass; a number of quality
is recorded and gates nothing** — the prototype accepts the music it
gets, and only a broken link stops the chain.

The all-era cut of 2026-08-29 moved the OCaml float model and the OCaml
integer twin above the seam. The gates below are the same gates; where
each RUNS is stated with it, and no gate has an oracle in the language it
tests any more:

- **The units.** `Sigmoid` and `Softplus` against the reference tables,
  **exhaustively** — 256 entries under the input rules is a few thousand
  readings, the `Exp2` precedent. Waveform tests where the two-cycle
  hold is the contract. Every unit keeps its own gates beside it, in
  `mgen_nn` since the unification round.
- **The twin against the float model.** `test_drift.py`, teacher-forced
  on the twin's own walk, with the fixed sweep and the property floors —
  and **long walks**: the state carries error forward, thus a walk of a
  few windows' length proves less than it proved in era four. The report
  adds the clamped-`dt` share. The floors calibrate on this model's own
  first measured minima, as era four's calibrated on theirs: measure,
  set the floors far under the minima, and pin both in the test. A low
  measured level is a finding of the era, not a failure of the gate.
- **The circuit against the twin.** `test_rtl_mamba.py` states what the
  circuit must do and `gate_mamba.exe walk` states what it did: at one
  block **and at two blocks with two heads** — the region field of the
  state address, of the tap address and of the ring address is EMPTY at
  one of a kind, thus a gate that ran one of each would elaborate none of
  them and the board runs six blocks. The era-four review found that gap
  late. At seed 0, and across the lead-in.
- **The circuit against the twin, WRITE FOR WRITE.** The frame gate above
  is blunt at the shape a test can afford, and this era learned how blunt:
  weights of scale 0.02 put the classes so near each other that a pick is
  almost the quantile of its uniform alone, thus a datapath can be wrong
  by tens of percent and still draw the same frames for a dozen steps.
  `gate_mamba.exe stream` walks the circuit's h RAM instead — the embed
  and each layer's join write the whole stream, in that order — and
  `test_rtl_mamba.py` holds every element of it against the twin's own
  per-layer streams.

  **Four faults were found through it and none of them moved a frame at
  first**: a weight addressed by a concatenation whose stride was not the
  tensor's width, a convolution channel block read at the gate's offset,
  an operand selected on the address side of a two-cycle read, and a tap
  ring whose layer stride ran the top layer off the end of its memory. A
  gate that only compared frames would have shipped all four. It runs at
  one block, at three blocks with two heads, and at a wide state and
  kernel.

  **The head round found two more, and both were in the REFERENCE and not
  in the circuit.** The ring coarsening was written `v asr 8 lsl 8`, which
  OCaml reads as `v asr (8 lsl 8)` and the machine reads as no shift at
  all, thus the reference coarsened nothing; and the exp2 argument of the
  softmax negated before the scale where the circuit negates after it,
  which parts the two by one unit whenever the scale does not divide
  exactly. Neither moved a frame. This gate found both in one run.
- **The schedule prints; the cycle bench pins the cost model.** The
  schedule test also prints the cycles of a drawn step at the ELECTED
  shape, which no simulation can afford, thus the cost in this document
  and the cost model cannot part. The memory geometry prints beside them:
  the three memories that carry a layer field, at the four plans the RTL
  gates walk.
- **The quantizer through the netlist.** `test_parity.py` quantizes the
  elected checkpoint in JAX, elaborates this era's top through
  `gate_mamba.exe verilog`, and holds the md5 of the Verilog against its
  pin. The per-head `decay` reads the libm's exponential, thus one ulp
  there would move a ROM byte and this gate is what would say so.
- The sequencer and the decode have their gates already; nothing on
  that side moves.
- **The board.** The amidi thru capture against the twin's events,
  as every era has proved its stream — **the one gate that waits for a
  person and the hardware**; everything above it runs without either.

## What it does not do

- No chunked SSD algorithm, no parallel scan in hardware: the recurrent
  mode is the design, one step a step.
- No era-four attention layer: a square query over the stream alone
  measured null in this trunk three times, thus the circuit knows the
  Zamba head and refuses a checkpoint that holds the other.
- No int4, no ternary; the ladder waits, as it waited in era four.
- No runtime configuration: one checkpoint, one bitstream, one seat.
- No shared op/schedule library with era four; the restatement is the
  prototype's price, and the abstraction round — if the era earns one —
  is a discussion first, by standing rule.
