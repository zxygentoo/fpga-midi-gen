# The diffusion machine

## Scope

Era six on the board: the RTL round. The model is `docs/diffusion.md` — the
masked canvas, blocked Gibbs over an annealed schedule. The board draws one
canvas and the sequencer plays it.

**The contract of the round is QUANTIZED-RTL EXACTNESS.** The integer twin
(`lib/diffusion/quantized.ml`) is the specification. The circuit must equal
`Quantized.Engine` operation for operation: the same seed gives the same
canvas, bit for bit. This is Gate B. The circuit reads its memory layout
from the twin (`rom_bits`, `rom_bases`) and restates none of it. The float
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
| folded norm | gain as `Constants.scale`, bias Q12 int16, then ReLU, then the clamp |
| logits | Q12 int16, no ReLU |
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
  residual add on the pair-closing layers. It runs at `48 G / (9 Cin)`
  results each cycle — 1.33 at the H 16 geometry — through a small
  pipelined epilogue beside the array, and packs the results back into
  columns.

The cycle count of one layer is exact and the schedule test prints it:

```
cycles = T * ceil(Cout / G) * 9 * Cin
```

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
  contract already demands. No logit tensor exists.
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
streamed logit columns — and together they cost about one percent of a
pass. FORWARD walks the layer table: one record for each layer, stating
Cin, Cout, the source and destination tensors, the ReLU and residual
flags, and the weight and constant bases. A counter walks it; no program,
no op vocabulary.

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

## The cost model

A lane is one DSP48 taking one product each cycle at 100 MHz. The device
has 240; this board is a new top level and carries no other era's units,
thus all 240 are the model's to spend. At 192 lanes, 48 stay free for the
epilogue and the draw.

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
  — is a parameter like the rest, pinned to 48 by the board and left free
  so Cyclesim runs Gate B end to end at tiny shapes.

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
