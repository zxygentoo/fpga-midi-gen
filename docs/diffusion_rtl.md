# The diffusion machine

## Scope

Era six on the board: the RTL round. The model is `docs/diffusion.md` — the
masked canvas, blocked Gibbs over an annealed schedule. The board draws one
canvas and the sequencer plays it.

**The contract of the round is QUANTIZED-RTL EXACTNESS.** The integer twin
(`lib/diffusion/quantized.ml`) is the specification. The circuit must equal
`Quantized.Engine` operation for operation: the same seed gives the same
canvas, bit for bit. This is Gate B. The circuit takes its ROM image and
bases from the twin (`rom_bits`, `rom_bases`) — the twin is the authority
on every value, and the dwell-order packing is the elaboration's
permutation of it, as "The weight ROM" states. The float
reference is not the specification of the circuit; the drift report already
measured what the quantization costs.

**The round CLIMBS A LADDER OF CHECKPOINTS, as the model round climbed its
ladder of sizes.** First make it work, then make it good: the machine
elaborates from the checkpoint (see "The iteration strategy"), and the
climb is `l16-h16` first, then `l64-h16`, then the golden candidate
`l48-h20` behind a storage optimization this document defers. Every rung
is the same machine at H 16 — the ladder proved depth is the cheap axis in
training, and it is the cheap axis in hardware for the same reason: L sets
only the layer count and the weight ROM; H sets the geometry and the
activation memories.

Out of scope, in order behind this round:

- **Int4, the 17-bar canvas and the mix** — one stretch round, possibly,
  after the int8 board ships. Pinned out 2026-08-26.
- Whole pieces, the length mask and the endings — the thesis round, on top
  of this stack.
- Harmonization, anywhere.

### The references, as built

The reference round built the two software layers below the JAX seam and
its record is the code. What the machine round stands on:

- **The float reference** (`lib/diffusion/diffusion.ml`). Gate A holds it
  to the JAX forward: one masked-NLL number over a deterministic corpus
  set, inside a pinned tolerance. Gate C holds the walk: a small-N canvas
  prints as the same text, character for character, from JAX and from the
  reference.
- **The integer twin** (`lib/diffusion/quantized.ml`). The drift report
  pins it to the float reference. The report is a measurement and gates
  nothing; Gates A, B and C are equivalences and must pass.

The drift lines of the climb, measured 2026-08-26 at seed 42, T 128,
32 passes:

| checkpoint | top-1 | cosine | same draw | clamps | hottest write |
|---|---|---|---|---|---|
| `l16-h16-100k` | 96.1% | 0.9997 | 92.0% | 0 | 106.8 of 512 |
| `l64-h16-100k` | 96.7% | 0.9995 | 94.5% | 0 | 100.9 of 512 |
| `l48-h20-100k` | 97.2% | 0.9998 | 95.1% | 0 | (313 on seeded openings) |

**The deep, narrow trunks peak near 107 where the golden candidate reached
313**, thus Q6 holds the whole climb with about a five-times margin and
THE TWIN STAYS FROZEN: no format moves, and Gate B's target is the same
module at every rung.

### The consumption order

Every uniform of a walk comes from `Prng`, the xorshift32 of the circuit,
and THE CONSUMPTION ORDER IS THE CONTRACT — the full statement is in
`lib/diffusion/diffusion.mli` and `quantized.mli`, and the machine obeys it
as written:

1. One canvas is one seed. The board takes the SEED cell as it stands
   (`Prng.create`). Seed 0 is the walk that stands still: every uniform is
   0 and the twin plays silence. That is the design, not a fault.
2. The cell order is step-major, seat-minor.
3. The opening: one uniform for each cell; the class is
   `low + floor(u * width)` over the register of the cell's seat.
4. Each pass n of N: the masks (one uniform for each cell, hidden when
   `u * 2^24 < floor(alpha_n * 2^24)`), then one forward pass, then the
   draws (one uniform for each hidden cell, `Policy.draw_class` over its
   48 logits).
5. The anneal is `alpha_n = max(0.1, 0.9 - 0.8 n / (0.7 N))`; the
   temperature is 1.0, baked.

### The formats

The formats of the twin, which the datapath holds:

| what | format |
|---|---|
| weights | int8, one power-of-two exponent for each tensor |
| activations | Q6 in int16, clamped and counted |
| accumulator | int32; at most 216 products, no overflow, any tap order |
| folded norm | gain as `Constants.scale`, bias Q6 int16 — the activation format — then ReLU, then the clamp |
| logits | Q6 in int16 like every activation, no ReLU; the draw shifts the differences to Q12 |
| the draw | era four's pipeline: shift to Q12, temper, `exp2` to Q15, pick on a 24-bit uniform |

The order-free accumulator is the twin's gift to this round: the schedule
may take the taps in any order, thus the geometry below is legal by
construction and Gate B's exactness lives in the drain alone.

## The machine

**THE OP AND SCHEDULE PROGRAM LAYER OF ERAS FOUR AND FIVE IS RETIRED FOR
THIS ERA**, elected 2026-08-26. Those machines sequenced a heterogeneous
chain of ops on one or two lanes at under two percent duty; this one runs
ONE op shape — the three-by-three convolution layer — for 99.5 percent of
its cycles, at near-full utilization of about two hundred lanes. The
program collapses to a layer table and a counter; the complexity moves
inside the one engine. What survives from the eras: the `mgen_nn` units
(`Prng`, `Exp2`, the pick policy, the clamp and rescale rules), the outer
FSM idiom for the serial walk, and the discipline — the schedule prints,
and the cycle bench pins the cost model.

The machine is four parts: the column engine, the memories, the walk FSM,
and the score port to the sequencer.

### The column engine

The pitch axis is the natural lane axis. One memory word is ONE PITCH
COLUMN: the 48 rows of one time step and one channel, 48 by int16, 768
bits. From that one choice the whole geometry follows:

- **The lanes are 48 rows by G output channels.** One column read feeds
  all 48 rows; every output-channel group reuses the SAME broadcast
  column, and its price is G weight bytes each cycle. G is 4 on the H 16
  rungs: 192 lanes, and 4 divides 16 with no ragged group.
- **The taps and the input channels stay serial** as the accumulation
  dwell: `9 * Cin` cycles for each (column, group), one (tap, cin) pair
  broadcast each cycle. The sum accumulates in the DSP48's own P register
  — no accumulator memory exists.
- **A three-column sliding window** holds the time taps in registers:
  three column reads for each nine tap cycles. The three pitch taps are
  wire shifts of the registered columns; zeros shift in at rows 0 and 47,
  and the columns beyond t 0 and t 127 are the zero column, muxed.
- **The drain** applies the folded norm (the gain multiply and the Q12
  bias), the ReLU where the layer table states it, the Q6 clamp, and the
  residual add on the pair-closing layers. All 48 G accumulators finish in
  the SAME cycle, thus one cycle captures them into a chain of 48 stages and
  a G-lane epilogue empties the chain one row a cycle, beside the array. "The
  circuit" chapter states the chain, its rule and its reasons.

The cycle count of one layer is exact and the schedule test prints it:

```
cycles = T * ceil(Cout / G) * 9 * Cin
```

The drain adds one tail of P at the end of a layer. The count is EXACT and not
a bound, because the elaboration refuses a layer whose dwell is shorter than
its drain — the rule of "The circuit" chapter.

**The weight word is the canonical block-RAM word.** At G 4 one word is
four int8 weights — 32 of 36 bits, the eight-of-nine packing the model
round predicted for a regular kernel. The elaboration packs the twin's ROM
image (`rom_bits`, in the checkpoint order) into G-byte words; the twin
stays the authority on every value, and the order of the dwell is free.

**The timing risk is new in kind and the array is all of it.** The eras'
critical paths were single carry chains; this design is a wide DSP array,
768-bit buses and a 48-way broadcast. The mitigations are structural: the
broadcast tree is pipelined, the ROM outputs are registered (the era-four
retiming trap), and the array is 48 identical row slices — regularity the
placer can use. The first build at the smallest rung already exercises the
full array, which is the point of starting there.

**THE GUESS WAS MEASURED ON 2026-08-26 AND IT NAMED THE WRONG THING.** The
array, its 768-bit buses and its broadcast are all cheap; the critical path is
ONE CONTROL NET — the drain chain's capture select, reaching every one of its
registers from one flop. "The array, measured" below states the numbers.

### The memories

- **Two activation tensors, X and Y**, each `T * Cin` columns deep by 768
  bits wide. The layers alternate X to Y and Y to X; the pair-closing
  layer reads its residual column and writes its output column IN PLACE —
  the read of column t stands before the write of column t, thus two
  tensors are the whole activation store.
- **The stem stores nothing.** Its input planes are 0 or 1, decoded on
  the fly from the canvas RAM and the mask bits. No input tensor exists.
- **The head stores nothing.** Its output columns ARE the logit columns —
  the 48 rows of seat v at step t are exactly the 48 logits of that cell —
  and they stream to the draw pipeline in the step-major order the PRNG
  contract already demands. No logit tensor exists; the circuit holds ONE
  STEP, and "The circuit" chapter states where.
- **The weight ROM** is one linear memory of G-byte words, read once each
  cycle, initialized by the bitstream. The gains and biases are small
  per-channel constants beside it.
- **The small state**: the canvas RAM (`T` by four classes), the mask
  bits, the alpha threshold ROM of N entries, the seat registers.

### The activation budget, and the fused pair

**THE LADDER'S TILE TABLE UNDERCOUNTS THE ACTIVATIONS BY TWO, and the
golden candidate does not fit the simple machine.** The table of
`docs/diffusion.md` was computed before the reference round elected Q6 in
int16; its activation numbers correspond to one byte per element. At two
bytes, two live tensors of the golden candidate are 107 tiles, its weights
42, and the sum is about 152 of the device's 135. No lane geometry fixes
storage. The H 16 rungs fit — that is why the climb runs on them.

**The answer for the golden candidate is THE FUSED PAIR, and it is the
"make it good" round, deferred.** Conv2 of a residual pair needs only a
three-column band of conv1's output, thus the intermediate tensor never
exists in full: the engine streams a pair for each column, holds a
five-column band of X and a three-column band of Y, and the pair output
overwrites X in place, trailing the reads by two columns. One full tensor
plus bands is about 57 tiles, and the golden candidate lands near 104 of
135 at G 5 — with the same cycle count, because fusion moves memory, not
work. The rejected alternative — int8 activations with per-layer exponents
— respins the twin, re-runs the drift election and moves Gate B's target;
it returns only if fusion fails or the stretch round wants the tiles.

### The walk and the layer table

The outer FSM is the proto's four states grown to five: OPEN, then N
rounds of MASK, FORWARD, DRAW, then PLAY. OPEN, MASK and DRAW are small
serial machinery in the pinned PRNG order — the masks are one uniform for
each of the 512 cells, the draws ride era four's pipeline over the head's
streamed logit columns — and together they cost about two percent of a
pass: three cycles for each cell of the mask draw, and about 2 P cycles
for each hidden cell that draws, which the cycle bench settles. FORWARD walks the
layer table: one record for each layer, stating Cin, Cout, the source and
destination tensors, the ReLU and residual flags, and the weight and
constant bases. A counter walks it; no program, no op vocabulary.

### The prior art

Searched 2026-08-26, before any RTL. **THE FIELD HAS BUILT THIS MACHINE
MANY TIMES AND EVERY COMMITMENT ABOVE HAS A NAMED PRECEDENT**; the round
changed nothing and validated one escalation path.

- **The single looped engine is one of the field's two canonical classes.**
  The toolflow survey of Venieris et al. (arXiv 1803.05900) splits CNN
  accelerators into single computation engines — one array, layers in turn
  — and streaming architectures with one stage per layer (fpgaConvNet,
  FINN). Phase I is a textbook single engine, and the fused pair is a
  bounded step toward the streaming class: both rounds sit on the
  taxonomy's own axis.
- **All-on-chip weights are FINN's defining move** (arXiv 1612.07119), and
  FINN — a sliding-window unit feeding a folded MAC array — is the nearest
  published neighbor of this design. A reading filter follows: the
  loop-tiling literature (Zhang et al., FPGA 2015) optimizes DDR traffic
  this board does not have.
- **The geometry is a standard folding point.** DDR-era designs unroll
  input and output channels with adder trees because they tile the spatial
  axes; all-on-chip designs unroll spatially with wide words. The column
  engine's reasons hold against both: the DSP-internal accumulator needs
  no adder tree and keeps the sum order-free — which Gate B leans on — and
  the column word is the bandwidth answer. The tap-parallel alternative
  (nine multipliers for each kernel) needs nine operands each cycle and an
  external accumulator: strictly worse here.
- **The fused pair is Alwani et al., "Fused-Layer CNN Accelerators"
  (MICRO 2016)**, almost verbatim: fuse adjacent convolutions, hold only a
  band of the intermediate, 95 percent of the transfer cut on VGG-E's
  early layers. The line is still active (fusion-aware mappers, 2025–26).
- **One MAC for each DSP is confirmed at the part number.** The
  two-INT8-MACs trick (Xilinx WP486) needs the DSP48E2's 27 by 18
  multiplier — UltraScale+ only. The Artix-7's DSP48E1 is 25 by 18, and
  the int8 by int16 operands rule the packing out regardless.
- **DSP double-pumping is published practice** — arrays at two to four
  times the fabric clock behind CDC FIFOs (arXiv 2407.19449 runs 100 MHz
  fabric under 400 MHz DSPs). The lane escalation path is prior art, not
  a hope; the climb does not need it.
- **Large DSP-array timing has its own tooling literature** (DSPlacer,
  DAC 2025): the risk is real, regularity is the accepted answer, and
  100 MHz on 7-series is a conservative target by the field's standards.
- **Winograd is rejected on purpose, not missed**: its transforms change
  the arithmetic and break Gate B against the plain-MAC twin, int8
  Winograd costs precision, and MACs are not the binding constraint.

## The circuit

The chapter above is the geometry. This one is the circuit that holds it: the
shape of the code, the dwell, the drain, the memories and their ports, the
walk, and the seam to the sequencer. `lib/diffusion/source.ml` holds the
design and its reasons, as the era before it did.

### The shape of the code

Five layers, and the middle one is where the retired program stood:

- **L0, the shared units**: `Prng.Rtl`, `Exp2`, the pick rules of
  `Mgen_nn.Quantized`, and `Vocab.Rtl` for the score port. Era four and era
  five built them and this era changes none of them.
- **L1, the elaboration**: a value, computed from `Quantized.Model.t`. It
  gives the layer table, the weight ROM image, the constant ROMs, the alpha
  ROM, the address maps and the cycle cost. It states no signal, thus an
  expect test prints it and the cost model cannot rot.
- **L2, the column array**: the pitch shifts, the two broadcast trees, the
  P by G lanes, AND THE DRAIN CHAIN. The chain is the array's output port:
  without it that port is P by G by 32 bits, which is neither wireable nor
  testable alone.
- **L3, the epilogue**: the folded norm, the ReLU, the clamp and the residual
  add — one drained row into one row of activations, and nothing wider than
  a row.
- **L4, the walk**: the outer FSM, the two activation stores and the three
  bands that cache them — the column window, the residual columns and the
  output columns — the canvas, the draw, and the score port.

**EVERY WIDE BUFFER THAT TOUCHES THE STORE LIVES WITH THE STORE.** The
three-column window is a read cache for the column port, a residual column is
a read cache for the join, and an output column is a write buffer. What fills
each one, when a slot is free, and what the zero column is beyond the ends of
the roll are questions of the memory and of the walk that reads it; registers
in one unit and the policy in another put a load strobe on one side of an
interface and the dwell it must be timed against on the other. The rule costs
nothing to follow — the path is the same either way, thus no logic and no
stage — and it is what holds the array's interface at eight fields and the
epilogue's at rows in and rows out.

### The dwell

The loop order is the column, then the output group, then the input channel,
then the tap:

```
for t   in 0 .. T - 1
  for g   in 0 .. ceil (Cout / G) - 1
    for cin in 0 .. Cin - 1
      read the columns (t - 1, cin), (t, cin), (t + 1, cin)
      for tap in 0 .. 8                              -- nine cycles of work
```

Thus the memory gives one column each three cycles and the tap cycles are the
work. The three pitch taps are wire shifts of the registered column — zeros
shift in at row 0 and row P - 1 — and the columns before t 0 and after t T - 1
are the zero column, muxed.

**THE WINDOW STANDS WITH THE STORE AND NOT INSIDE THE ARRAY**, under the rule
of "The shape of the code": a term names the column it takes, and the array
shifts it. The path is the same either way — the window register, the time
mux, the pitch mux, the operand register — thus the cut takes three fields and
two rules out of the array's interface and costs nothing for them.

One tap cycle broadcasts one operand pair. Lane (r, c) takes the activation of
row `r + dx - 1` of the column of `dy`, and the weight byte of
`(tap, cin, g * G + c)`. **One activation serves the G lanes of a row and one
weight serves the P lanes of a channel**, thus the broadcast is two trees and
never a mesh. Both trees are pipelined, and **the ROM address is registered
before the memory and the data is registered after it** — era four's trap,
where a combinational address lets the tools retime the data register onto the
address pins of every primitive and rebuild the address cone inside each one.

### The weight ROM

One linear memory of G-byte words, one read each cycle, and the address is one
counter. **The image is packed in the DWELL order and not the checkpoint
order: the group, then the input channel, then the tap.** The twin stays the
authority on every value; the permutation belongs to the elaboration, and it
buys an address that only counts — one column's dwell walks a layer's whole
range straight through, thus the address reloads one time for each column.

**ANY OTHER ORDER MAKES THE ADDRESS A STRIDE AND NOT A COUNT.** With the group
innermost the address steps by the group count at each tap cycle, which is a
valid permutation of the same weights and a worse walk. The elaboration's gate
therefore walks the image with a plain counter and demands the weight that
step of the dwell needs, because a bijection test alone passes on either
order.

Where G does not divide Cout, the elaboration pads each `(cin, tap)` row of
Cout channels to a whole number of words with zeros: the padded lanes multiply
by zero and the drain does not read them. Each layer has its base.

### The drain

At the end of a dwell **all P by G accumulators finish in the SAME cycle**,
and the array must start the next dwell on the next cycle. The DSP48E1 holds
one P register and no shadow, thus the sums leave the array in one cycle or
they are lost.

They leave into a chain. One cycle captures the array into P stages of G by 32
bits. Each cycle after it, every stage gives its G values to the stage below,
and the epilogue takes the bottom stage. **No value crosses a mux and no
register reaches farther than its neighbour**, thus the chain lays out as a
column beside the array — the regularity the timing risk asks for. The
alternative, a central mux, costs the same registers and makes a P-source star
of the routing; this design refuses it.

Two rules follow, and the elaboration holds the first one:

- **The chain must empty before the next capture: `9 * Cin >= P`.** The
  elaboration REFUSES a layer that breaks the rule and names the layer. It is
  a check and not a comment, and the cost model prints it. Every real rung
  stands far inside it — a trunk layer is 144 against 48, and the stem is 72 —
  and so does each test shape, thus the refusal is a guard and not a limit.
- **The rows leave the chain in row order, which IS the column order.** The
  epilogue packs the G output words as the rows come out and writes each
  column one time. The drain's memory traffic is whole columns, as every other
  traffic of this machine is.

### The epilogue

G lanes, one for each output channel of the group, thus a group drains in P
cycles. Not fewer lanes: at G / 2 the drain is 2 P cycles, and the stem —
whose dwell is `9 * 2 * voices`, 72 cycles — would then stall.

One lane is the tail of the twin's `layer_forward`, operation for operation:

1. the gain multiply, int32 by the channel's int16 scale;
2. the arithmetic shift by that channel's own q — a VARIABLE shift, because a
   scale carries its own q and the q is per channel;
3. the bias;
4. the ReLU, where the layer table states it;
5. the clamp to int16.

**A pair-closing layer clamps TWICE.** The twin writes the convolution result
through the counted clamp, and then writes the sum through it again:
`clamp16 (max 0 (x + clamp16 second))`. A circuit that clamps one time is
silently wrong on each write that rides the clamp, and Gate B is bit for bit.

**THE ARRAY OWNS THE DSPS; EVERYTHING OUTSIDE IT IS LUTS** — the epilogue
gain, the draw's temper, and the pick's product. The reason is the top of the
ladder and not today's headroom: G follows H and not L, thus `l64-h16` runs on
the same 192 lanes as `l16-h16` and costs no DSP, but the golden candidate
fused at G 5 is 48 by 5, which is 240 of the device's 240. An epilogue that
took DSPs would have to move at the moment the design is tightest.

The accumulator stays int32. The measured peaks stand far under it, but the
twin guarantees int32 exactness and Gate B is bit for bit, thus a narrower
accumulator is a guess that one rung happens to survive.

### The memories and their ports

| memory | shape | traffic |
|---|---|---|
| X and Y | `T * H` columns by `P * 16` bits | one column read each three cycles; G column writes each group |
| the weight ROM | the packed image, G bytes each word | one word each cycle |
| the constants | gain and bias, one entry for each output channel | G entries each group |
| the canvas | `T` by `voices` classes | registers |
| the mask | `T * voices` bits | registers |
| the alpha ROM | N entries of 24 bits | one entry each pass |
| the logit file | `voices` files of P by 16 bits | the head's drain writes it, the draw reads it |

**The stem decodes and does not read.** Its input column for `(t, plane)`
comes from the canvas and the mask: a class plane is one-hot at
`canvas[t][v]`, in activation units, when the cell stands, and zero when the
cell hides; a mask plane is all ones when the cell hides and zero when it
stands. No input tensor exists.

**The pair-closing layer reads Y and X and writes X.** Y gives the three taps,
X gives the residual column `(t, c)`, and the output goes back to `(t, c)`.
The read stands before the write, thus X and Y are the whole activation store.

**The head writes one step and not a tensor.** At the head Cout is `voices`,
thus the drain's G columns are the logit columns of the cells of step t, and
they go to the logit file. That is what "the head stores nothing" means
exactly: it stores one step. The draw empties the file before the head goes to
step t + 1.

### The walk

OPEN, then N rounds of MASK, FORWARD, DRAW, then PLAY. Every uniform comes
from `Prng.Rtl` in the consumption order of the Scope chapter.

- **OPEN** takes one uniform for each cell in the cell order. The class is
  `low + ((k * width) >> 24)` over the register of the cell's seat, where k is
  the 24-bit uniform: one multiply and no divide. Three PRNG steps a cell.
- **MASK** takes one uniform for each cell and hides the cell when k stands
  under the pass's alpha entry. Three PRNG steps a cell.
- **FORWARD** walks the layer table with a counter. One record states Cin,
  Cout, the source and destination tensors, the ReLU and residual flags, and
  the weight and constant bases. There is no program and no op vocabulary.
- **DRAW rides the head.** The head's drain fills the logit file of step t,
  and the draw then takes the hidden cells of that step in the seat order the
  group order already gives. The peak of each cell costs nothing — the
  epilogue tracks it as the rows come out. Then one walk of the file gives the
  exp2 weights and their total, and a second walk picks on the 24-bit uniform:
  about 2 P cycles for each hidden cell.
- **PLAY** answers the socket.

**The canvas is written IN PLACE during the head, and it is exact.** The head
reads the trunk tensor, and the forward computed that tensor from the canvas
as it stood at the start of the pass. Nothing after the stem reads the canvas,
thus a draw at step t cannot reach a later column. The twin copies the canvas
because a value engine must; the circuit does not have to.

### The seam to the sequencer

`Source_intf`, unchanged, and the top level does not move: `Top` names
`Source.create ~model ~seed` and nothing narrower, thus one line of
`board/nexys-4/dune` seats era six.

- **`rewind`** is the run start. It captures SEED, drops `idle`, and runs OPEN
  and the N passes. `idle` rises when the canvas stands. The sequencer's
  `WaitRewind` state waits for exactly this already, thus the seconds of the
  draw need no rule of their own.
- **`step`** reads `canvas[t]`, maps each class through `Vocab.Rtl`, packs the
  frame and answers `valid`. The map is the vocabulary's rule and this era
  does not restate it.
- **Past step T - 1 the frame is silence**, for ever, until the next
  `rewind`. The sequencer plays the canvas one time and the reset button gives
  the next run, as Phase I states.

The score port stands behind its own interface, thus Phase II's second canvas
memory is a local change.

### What the elaboration refuses

The elaboration reads the checkpoint and every dimension follows it. It
refuses loudly, and the message names what it refused:

- `Model.check_shape` first: the chain of the channels, the accumulator bound,
  the kernel counts and the constant rows.
- `9 * Cin >= P` for each layer: the drain rule above.
- a walk of no passes, a canvas of no steps, and a group of no lanes: N, T
  and G each stand at 1 or above.

## The cost model

A lane is one DSP48 taking one product each cycle at 100 MHz. The device
has 240; this board is a new top level and carries no other era's units,
thus all 240 are the model's to spend. At 192 lanes, 48 stand free — and
they STAY free: the epilogue and the draw are LUTs, because the fused rung
wants all 240 for the array. "The circuit" chapter gives the reason.

The climb, at T 128, N 512, inside the 25.6-second playback window
(2.56 G cycles):

| rung | params | weights | activations | tiles (of 135) | cycles / pass | canvas at N 512 |
|---|---|---|---|---|---|---|
| `l16-h16` | 34 k | ~9 | ~88 | ~100 (74%) | 1.09 M | 5.6 s |
| `l64-h16` | 147 k | ~36 | ~88 | ~127 (94%) | 4.63 M | 23.7 s |
| `l48-h20` unfused | 170 k | ~42 | ~107 | ~152 — OVER | — | — |
| `l48-h20` fused, G 5 | 170 k | ~42 | ~57 | ~104 (77%) | 4.30 M | 22.0 s |

The cycle numbers are the formula's ideal at 192 lanes (240 for the fused
G 5 row) and land within a percent of `MAC / lanes`, because the geometry
divides the H 16 shapes exactly. The real overheads — the window loads,
the group drains, the layer turns — are what the cycle bench measures, and
`l64-h16` holds 8 percent of slack against the window for them. Even 48
lanes (G 1) plays rung 1 at N 512 inside the window: the climb has rungs
in lanes as well as layers, if the first build fights.

### The array, measured

The column array alone, through Vivado out of context on the part, at 100 MHz
and P 48. It needs no checkpoint: the unit takes its weights and its columns
as ports, thus the shape is the whole input. `board/nexys-4/gen_probe.ml`
writes it and `probe.tcl` reads it.

| G | lanes | DSP48E1 | LUTs | registers | WNS | WHS |
|---|---|---|---|---|---|---|
| 1 | 48 | 48 (20%) | 1 501 (2.4%) | 1 559 | +5.390 | +0.118 |
| 4 | 192 | 192 (80%) | 3 769 (5.9%) | 6 167 | +2.950 | +0.132 |
| 5 | 240 | 240 (100%) | 4 505 (7.1%) | 7 707 | +2.761 | +0.124 |

Every geometry MEETS with no failing endpoint. Three readings agree with the
cost model above:

- **One DSP for each lane, exactly** — 48, 192 and 240. No adder tree and no
  extra multiplier stands anywhere, and G 5 lands on 240 of the device's 240,
  which is the fused rung's whole budget and the reason the epilogue is LUTs.
- **The DSP absorbs all four registers.** The synthesis log states it for the
  operand pair, the product and the sum, at mode `((C:0x0) or P)+(A2*B2)` —
  the primitive's full pipeline. That is what the free-running operand
  registers buy, thus the register count is the drain chain and almost nothing
  else: 6 167 against the chain's own 6 144 at G 4.
- **No block RAM**, as the unit holds none.

**THE CRITICAL PATH IS NOT THE ARRAY. IT IS ONE CONTROL NET.** At G 4 it runs
from one flop to the chain's mux, fanout 6 019, and it is 6.415 ns of ROUTE
against 0.580 ns of logic — 92 percent route, one logic level. That flop is
the capture select, and it reaches every register of the chain. The same shape
stands at G 5 (fanout 7 523, 6.606 ns) and shrinks at G 1 (fanout 144,
0.847 ns), thus the net scales with the chain and with nothing else.

The design does not move for it: it meets, and this project measures before it
optimizes. What the reading buys is the KNOWN WEAK POINT and its answer — one
copy of the capture register for each chain stage, 48 drivers of 128 registers
where one drives 6 144 — held until a build asks for it. The slack of the
whole machine will go here first when the memories arrive and take the routing
this probe had to itself.

The reserve is also the variance fix, not only the slack fix: a net at this
fanout and 92 percent route is exactly the kind whose length moves from build
to build — the 0.1 ns lottery the seed round measured. When the full build
first wobbles, the replication comes BEFORE a re-roll of the lottery.

Two rules for reading this table:

- **An out-of-context number is optimistic by construction** — an empty
  device, no congestion, a flat 2 ns charged at the ports. The finding to
  carry forward is that the array is not the problem; never that 2.9 ns
  exist for the machine to spend.
- **The probe is a ladder, as the drift is.** The next ring stands where the
  risk moved: the weight ROM through the broadcast trees into the array —
  era four's retiming trap on its home ground — and after it the column
  store's cascade read. Each ring is measured before the machine around it
  exists, at probe cost and not build cost.

## The iteration strategy

**THE MACHINE ELABORATES FROM THE CHECKPOINT, as the twin loads from it.**
The elaboration reads the shape the way `Config.of_checkpoint` does — no
flag states a dimension — and the layer table, the ROM image, the memory
depths and the group count all follow. The weights were never runtime
state in this project; the bitstream initializes them. That standing rule
becomes the iteration loop:

```
elect a checkpoint
  -> the drift line (check_diffusion drift, minutes, on the host)
  -> dune build: the netlist, and the Cyclesim gates at a tiny shape
  -> Vivado -> the board -> the capture gate
```

Each rung of the climb is this loop with new constants. Two rules keep it
honest:

- **Trained checkpoints elect music; drawn weights measure builds only.**
  `Params.init ?norm_scale` elaborates a shape that has no checkpoint, for
  a timing or utilization reading — never for a drift number (the drift of
  drawn weights reads the format floor, a known trap) and never for a
  rung.
- **Every elaboration parameter has a test shape.** P — the 48 pitch rows
  — is a parameter of the CIRCUIT like the rest, pinned to 48 by the board.
  The twin holds P at `Diffusion.rows`, thus Gate B compares at P 48 and the
  tiny shapes come from T, L, H, G and N. **Parameterizing the twin is
  DEFERRED and not refused**: it moves a frozen reference, and nothing moves
  the reference before a circuit works. When one works, the twin takes P as
  well and the test shapes get smaller.

## The two phases

**Phase I — the machine.** The working network on the board: one canvas
from the panel seed on run, and the simplest circuit that hands it to the
sequencer to play. Then measure: the Vivado build, the real generation
cycles, the utilization against the cost model. Phase I holds every
unknown of the round.

**Phase II — the performance.** Many canvases: draw the next while this
one plays, the gap, the fade, the buffer. Phase II reuses semantics the
software side already pinned.

## Phase I: the locked design

- **The climb starts at `l16-h16-100k`** — first make it work, then make
  it good. The engine, the memories and the walk are the machine above;
  `l64-h16-100k` is the same elaboration with a longer table and a larger
  ROM. The golden candidate waits behind the fused pair.
- **One canvas on run.** Reset releases, the machine draws one canvas from
  the SEED cell, the sequencer plays it once, the machine stops. The reset
  button gives the next run. Continuation belongs to phase II.
- **The canvas is the score.** The engine draws in one canvas memory and
  the sequencer reads the same memory: no copy, no second buffer. The
  handoff is a read port, the step timer, the `Vocab` decode and the
  releases-before-strikes rule, into the socket the board already has. The
  sequencer and the decode have their gates already; nothing on that side
  moves.
- **Lanes and N are elaboration parameters.** N is the depth of the alpha
  threshold ROM and nothing else. Bring-up runs at a small N; the drift
  report's own walk is N 32.
- **Gate B, the machine gate.** The instruments, inherited from the era
  four and five pattern:
  1. the unit gates — the draw pipeline and the tables, expect and
     waveform tests beside the units;
  2. the canvas agreement — the circuit against `Quantized.Engine` in
     Cyclesim, end to end at a small shape, and at MORE THAN ONE shape, so
     that no address region field elaborates empty;
  3. the stream gate, WRITE FOR WRITE — every per-layer activation write
     against the twin's, because era five proved that four real datapath
     faults move no frame at a test-sized shape;
  4. the cycle bench — the schedule prints its cycles at the elected
     shape, thus the cost model above and the machine cannot part;
  5. the board rung — the amidi capture of the board's events against
     `play_diffusion -quantized` at the panel seed, the one gate that
     waits for a person and the hardware.
- **The measurements of phase I**, the numbers the phase must report: the
  build (WNS, LUTs, block RAM, DSPs), the measured cycles of one pass, the
  utilization against the cost model, and the N the board affords inside
  the playback window.

## Phase II, in short

Nothing here is designed; the chapter waits for phase I's numbers. The
items, so the seams stay clean:

- **The buffer.** Gibbs rewrites the canvas in place, thus the playing
  canvas must be its own copy: two canvas memories in ping-pong. Phase I
  keeps the score read port behind its own interface so the doubling stays
  local.
- **The scheduling.** Draw the next canvas while this one plays; the lane
  count and N come from phase I's measurement.
- **The gap and the fade**, as the software states them: `Player`'s
  `velocity_at ~step` is the fade's one point of variation, and velocity
  is a fact of the onset.
- **The seed succession.** The rule that names the seed of canvas k is a
  contract to pin with `play_diffusion -seeds` and the JAX handoff before
  phase II elaborates.
