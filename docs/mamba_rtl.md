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
exact integer arithmetic in OCaml — `lib/mamba/quantized.ml` — and the
circuit must match it bit for bit. The float model is not the reference
of the circuit; the drift report measures what the quantization costs.

The modules of the era:

| Module | It owns |
|---|---|
| `Mamba` (`lib/mamba/mamba.ml`) | the float reference: the block, the loss, the sampler |
| `Mamba.Quantized` (`lib/mamba/quantized.ml`) | the quantization of the checkpoint, and the integer twin: the recurrence, the chain and the sampler |
| `Mamba.Source` (`lib/mamba/source.ml`) | the same integers as a circuit: the schedule, the datapath and the socket machine |
| `Sigmoid`, `Softplus` (`lib/mamba/`) | the two new tables, in the idiom of `Exp2` |
| from `mgen_transformer`: `Mac`, `Divider`, `Isqrt`, `Exp2` | the units of era four, reused as they stand |

**The reuse is a dependency, not a refactor.** `mgen_mamba` depends on
`mgen_transformer` for the units and the exp2 table. The units are
model-free already — their interfaces speak widths and strobes, not
transformers — thus the prototype imports them and moves nothing. If the
era survives the ear, the units and the shared tables take a common home
in their own round; a prototype does not pay for one.

**`Mac` is the one that could not come.** The import failed on a single
number: era four's walks never ran past 256 rows, thus its counters are
nine bits, and the state update here walks `d_in * N` rows — 2 048 at the
baseline, and 8 192 if the state sweep ever reaches 64. `lib/mamba/mac.ml`
is that file with fourteen-bit counters and nothing else changed. The
claim that those units are model-free is therefore ALMOST true, and the
walk length is the one place a bigger model shows through; the round that
gives them a common home should make it a parameter. The op and
schedule layer is **restated** in `lib/mamba/source.ml`, not shared: the
op vocabulary is different, and the abstraction of era four is an open
question by standing rule — an improvement to it is a discussion, not a
side effect of this branch.

## What the state changes

Four things leave the machine of era four, and two arrive.

**The KV rings leave.** The state RAM takes their seat: 24,576 bytes
against 196,608, written and read in place, never windowed, never
wrapped. The slot arithmetic, the fill count `n`, the age walk and the
causal wall all go with the rings — the recurrence has no ages.

**`Attend` leaves, and the trunk's division with it.** The softmax
denominator was the one division of the era-four datapath; the
recurrence normalizes nothing. `Divider` now serves `rms_norm` alone.
The pending-divide machinery of the merge walk goes.

**The context parameter leaves.** No `slots`, no window: the machine
holds a step counter for the bar phase and the lead-in, and nothing
else counts time.

**The step cost stops depending on the walk.** Era four's cost model
took the fill `n`; this one is a constant of the shape. The cycle bench
gets simpler than its predecessor.

**The state arrives**, and it is the one thing this machine holds that
era four's did not: memory that survives the step. Its rules — the
formats, the zero origin, the read-modify-write — are the new content of
this document.

**Two tables arrive**: the sigmoid and the softplus correction, in the
idiom of `Exp2` — a registered read, no start and no busy, the caller
holds the input two cycles.

## The socket

`Source_intf` is unchanged, to the field. `step` answers with a frame
the source has already drawn; `rewind` is the one reset and loads the
PRNG from SEED; `valid` and `idle` keep their contracts. The sequencer,
the decode and the board around the socket do not know the era changed.

## The integer model

### The weights

Int8 with a per-tensor power-of-two exponent, `w ~ q * 2^-e`, the
largest `e` that keeps `round(max|w| * 2^e)` at 127 or less — the rule
of era four, unchanged, and the same `Quantized.Model` machinery
pattern. The seat tensor and the bar phase share one exponent because
their rows add; that rule and its check carry over.

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

Each operation is one definition in `quantized.ml`, and the circuit
computes the same integers. Every product fits one DSP48, 25 by 18
signed. `rms_norm`, the embed, the chain and the sampler are era four's
operations unchanged. The new ones:

- **conv**: for each of the `d_in + 2 N` channels, a row of `K = 4`
  terms — the taps against the channel's kernel — then the SiLU chain
  on the sum. The taps live in a small ring of 4 for each channel and
  layer; tap `k` reads zero while `position < k`, thus the origin needs
  no clearing and the rule is a mux, as the era-four fill count was.
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
| L0 | `Divider`, `Isqrt`, `Exp2`, `Prng.Rtl` from era four; `Sigmoid` and `Softplus` new |
| L1 | the datapath: the RAMs, the state RAM, the tap rings, the banked weight ROM, `Mac` |
| L2 | the schedule: the step as a list of operations, built from the config |
| L3 | the compiler: the list folds into the cases of a program counter |
| L4 | the outer machine: the step strobe, the lead-in, the held frame |

L3 and L4 carry over structurally whole: the op-finish-runs-next-entry
convention, the single `switch` on the pc, the tick counter that steps
itself, the seat register and the four-times-one-seat chain, the
lead-in that draws nothing and moves no PRNG. The forward program
changes its op list; the chain program is era four's seven ops,
restated.

Two things carried over that this machine turns out not to need:

- **The hold is gone.** Era four's attention stalled its merge walk while
  a pending divide finished, thus every read register and every tag
  carried a freeze enable. No walk here ever stalls — the two divides live
  in a bespoke chain that waits on its own tick, not inside a walk — thus
  the enables are not built and `Mac` takes its hold at ground.
- **The slot count and the fill are gone with the rings.** Nothing counts
  ages, thus `Op.cycles` takes no `n` and every step of the walk costs the
  same number.

One thing this machine needs that era four never did. **Three ops choose
which memory feeds the multiplier by the position inside the row** —
the state update alternates two terms, the readout folds the skip in as
its last, and the inject walk selects a head's step size. Era four's ops
each read one memory. A choice like that must follow the DATA and not the
address: the counters carried forward by the read latency are what the
operand mux reads, and using the live counter there selects the wrong
memory two cycles early.

The op vocabulary follows era four's closed-and-concrete rule: when a
field's meaning would depend on another field, write a new op. The
expected list: `Embed`, `Rms_norm`, `Matvec` (sources `Y`, `Gated`, and
the state readout), `Conv`, `Silu_over` (a range), `Decay`,
`State_update`, `Gate`, and the chain's `Temper`, `Draw`, `Threshold`,
`Pick`, `Accumulate`. The schedule prints as data and an expect test
pins it, as before.

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
| weight ROM | 178,504 x 8 at six layers | the checkpoint, flat order |
| exp2 ROM | 256 x 16 | era four's table, from `Quantized.Constants` |
| sigmoid ROM | 256 x 16 | Q15 over signed Q12 in, clamped at |v| = 8 |
| softplus ROM | 256 x 16 | the correction term, Q12 over |v| up to 16 |
| **state RAM** | **6 x 128 x 16 x 16 b = 24,576 B** | the recurrence; int16, in place |
| tap rings | 6 x 160 x 4 x 16 b = 7,680 B | the conv inputs, a ring of 4 |
| `h` | 64 x 32 | the residual stream, and the stream of the chain |
| `y` | 64 x 16 | the normed vector |
| shared RAM | 512 x 32 | `zxbcdt`, then the SiLU outputs, then the logits and the sampler weights |
| **total** | **~55 tiles — 41 percent** | against era four's 126 — 93 |

The shared RAM's depth is set by `zxbcdt` at 292 entries, rounded to the
memory the tools infer; the gate product, the logits and the sampler
weights reuse it as era four's did — it is 32 bits wide because the gate
product is. The tap rings may infer as distributed RAM at this size;
either is fine, and the build reports which. Two small RAMs the plan did
not name are in the tree: the readout, 128 by 16, and the inject operands,
64 by 16. The readout wants a memory of its own because the gate reads it
and the shared RAM in one cycle.

**The two addresses that carry a layer take two different rules**, and the
reason is which fields are powers of two:

- the STATE address packs — `d_in` and `N` are both powers of two, thus
  the layer stands above the lane and the lane above the state index and
  the whole address is a concatenation, as era four's ring was;
- the TAP address adds — the channel count `d_in + 2N` is 160 at the
  baseline and no power of two, thus a concatenated layer field would
  stride by the rounded-up power, every layer above the first would sit at
  the wrong base, and the top layer's region would run off the end of the
  memory. One constant add puts the circuit on the reference's own
  address and wastes no row.

This is the fault a one-layer or two-layer simulation cannot see, and it
is why the gates run at three.

## The cost

The analytic model, at one term a cycle, six layers, the baseline shape.
`Op.cycles` states it exactly per op, and the cycle bench holds it to
the measured circuit; this table is the sum to one significant figure:

These are the numbers `Op.cycles` states, and the cycle bench holds the
measured circuit to every one of them.

| Part | Cycles for each layer | the estimate this replaces |
|---|---|---|
| `rms_norm` at `d` | 2,842 | — |
| `W_in` matvec | 18,692 | 18,688 |
| conv, then SiLU over 160 channels | 808 + 960 | ~2,000 |
| decay chains | 36 | ~50 |
| state update: the inject walk, then 2-term rows | 4,168 | ~4,100 |
| readout, 17-term rows | 2,180 | ~2,200 |
| SiLU over the gate, then the gate | 768 + 132 | ~1,300 |
| the gated norm at `d_in` | 5,658 | — |
| `W_out` matvec + join | 8,197 | ~8,200 |
| **one layer** | **44,441** | ~45,000 |
| the embed | 324 | — |
| one seat of the chain | 6,428 | — |

**292,684 cycles a drawn step — 2.93 ms at 100 MHz**, against the 3 ms
this document estimated. It is constant in the walk: the lead-in steps
cost the forward alone at 266,972, and every drawn step costs the same as
every other, because nothing here fills. The wire's 8 ms floor stands;
the source is never the tempo.

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

**The six-layer build, 2026-08-20**, on the elected checkpoint:

| | era five | era four |
|---|---|---|
| WNS | **+0.202 ns** | +0.059 ns |
| slice LUTs | 3,043 | 3,061 |
| slice registers | 1,658 | — |
| block RAM tiles | **57.5 of 135, 42.6%** | 126 of 135, 93% |
| DSPs | 2 of 240 | — |

The estimate of this document was about 55 tiles and 41 percent; the
build reads 57.5 and 42.6. **The block RAM the design set out to buy back
is bought back**, and the timing margin that came with it is three times
era four's on a design of the same fabric size. Both `phys_opt_design`
passes are in the script and the build meets without needing what they
give.

## The tests

The gates of era four, inherited with the lessons of its test review
already applied. **A gate of equivalence must pass; a number of quality
is recorded and gates nothing** — the prototype accepts the music it
gets, and only a broken link stops the chain:

- **The units.** `Sigmoid` and `Softplus` against the reference tables,
  **exhaustively** — 256 entries under the input rules is a few thousand
  readings, the `Exp2` precedent. Waveform tests where the two-cycle
  hold is the contract. The reused units keep their own gates in
  `mgen_transformer`.
- **The twin against the float model.** `Drift.walk`, teacher-forced on
  the twin's own walk, with the fixed sweep and the QCheck floors — and
  **long walks**: the state carries error forward, thus a walk of a few
  windows' length proves less than it proved in era four. The report
  adds the clamped-`dt` share. The floors calibrate on this model's own
  first measured minima, as era four's calibrated on theirs: measure,
  set the floors far under the minima, and pin both in the test. A low
  measured level is a finding of the era, not a failure of the gate.
- **The circuit against the twin.** `frames_agree`, from the first
  version: at one layer **and at two** — the state RAM's layer field and
  the per-layer bases must elaborate in simulation, the era-four review
  found that gap late — at seed 0, and across the lead-in.
- **The circuit against the twin, WRITE FOR WRITE.** The frame gate above
  is blunt at the shape a test can afford, and this era learned how blunt:
  weights of scale 0.02 put the classes so near each other that a pick is
  almost the quantile of its uniform alone, thus a datapath can be wrong
  by tens of percent and still draw the same frames for a dozen steps.
  `streams_agree` walks the circuit's h RAM instead — the embed and each
  layer's join write the whole stream, in that order — and holds every
  element of it against the reference's own per-layer streams.

  **Four faults were found through it and none of them moved a frame at
  first**: a weight addressed by a concatenation whose stride was not the
  tensor's width, a convolution channel block read at the gate's offset,
  an operand selected on the address side of a two-cycle read, and a tap
  ring whose layer stride ran the top layer off the end of its memory. A
  gate that only compared frames would have shipped all four. It runs at
  one layer and at three.
- **The schedule prints; the cycle bench pins the cost model.** Simpler
  than era four's: no fill `n` in the model.
- **The identical-walk gate** between `jax/mamba/infer.py` and
  `lib/mamba/mamba.ml`, from one seed through the shared xorshift32, as
  era four proved its samplers.
- The sequencer and the decode have their gates already; nothing on
  that side moves.
- **The board.** The amidi thru capture against the reference's events,
  as every era has proved its stream — **the one gate that waits for a
  person and the hardware**; everything above it runs without either.

## What it does not do

- No chunked SSD algorithm, no parallel scan in hardware: the recurrent
  mode is the design, one step a step.
- No hybrid attention layers; the era tests the state, not a blend.
- No int4, no ternary; the ladder waits, as it waited in era four.
- No runtime configuration: one checkpoint, one bitstream, one seat.
- No shared op/schedule library with era four; the restatement is the
  prototype's price, and the abstraction round — if the era earns one —
  is a discussion first, by standing rule.
