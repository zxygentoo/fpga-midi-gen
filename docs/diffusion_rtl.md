# The diffusion machine

## Scope

Era six on the board: the RTL round. The model is `docs/diffusion.md` — the
masked sheet, blocked Gibbs over an annealed schedule. The board draws one
sheet and the sequencer plays it.

**The contract of the round is QUANTIZED-RTL EXACTNESS.** The integer twin
(`jax/diffusion/quantized.py`) is the specification. The circuit must equal
it operation for operation: the same seed gives the same sheet, bit for
bit. This is Gate B, and it runs under `uv run pytest` — Python states what
the machine must do and `bin/gate_diffusion.exe` states what it did. The
circuit takes its ROM image and bases from the CONTRACT FILE the twin
writes (`rom_bits`, `rom_bases` over `Model.of_int8_checkpoint`) — the twin
is the authority on every
value, and the dwell-order packing is the elaboration's permutation of it,
as "The weight ROM" states. The float model is not the specification of the
circuit; the drift report already measured what the quantization costs.

**The round CLIMBS A LADDER OF CHECKPOINTS, as the model round climbed its
ladder of sizes.** First make it work, then make it good: the machine
elaborates from the checkpoint (see "The iteration strategy"), and the
climb is `l16-h16` first, then `l64-h16`, then the golden candidate
`l48-h20` behind the fused pair, which "The fused pair" below delivered. Every rung
is the same machine at H 16 — the ladder proved depth is the cheap axis in
training, and it is the cheap axis in hardware for the same reason: L sets
only the layer count and the weight ROM; H sets the geometry and the
activation memories.

Out of scope, in order behind this round:

- **Int4, the 17-bar sheet and the mix** — one stretch round, possibly,
  after the int8 board ships. Pinned out 2026-08-26.
- Whole pieces, the length mask and the endings — the thesis round, on top
  of this stack.
- Harmonization, anywhere.

### The references, as built

THE TWO SOFTWARE LAYERS ARE ONE LAYER, AND IT IS IN JAX. The reference
round built a float model and an integer twin in OCaml to weld the JAX
model to the circuit; the cut of 2026-08-28 (commit 5ed90f8) removed them —
both had become welds between the JAX model and the circuit, and a weld
needs no second copy of what it welds — and the chain is now:

```
JAX float --(drift, in one framework)--> JAX int8 --(exact, pytest drives Cyclesim)--> RTL
```

What the machine round stands on:

- **The float model and the integer twin** (`jax/diffusion/model.py` and
  `jax/diffusion/quantized.py`). The drift report pins the twin to the
  float model in one framework, on the walk the board takes:
  `uv run python -m diffusion.infer drift --ckpt C`. The report is a
  measurement and gates nothing.
- **The contract file**, the only thing that crosses the seam for a build:
  `infer.py quantize --ckpt C --out C.int8` states the int8 image and the
  folded norm, and `Model.of_int8_checkpoint` reads it into the
  elaboration. Its own gate is the NETLIST — the Verilog of `gen_verilog`
  must stay md5-identical to the one the flash carries.
- **The gates of the circuit**, in `jax/tests/test_rtl.py`: Python states
  what the machine must do and `bin/gate_diffusion.exe` states what it
  did. The walk gate holds every write of the cell port, phase for phase;
  the stream gate holds every column the stores take. Neither side can
  pass by agreeing with itself.
- **What stays in OCaml below the seam** is what the CIRCUIT reads, and it
  is ONE MODULE: `lib/diffusion/model.ml` — the roll, the formats, the
  contract file as data, the walk (the cell order, the registers of the
  seats, the opening, the masks, the anneal) and the frames. The two
  software halves that a unit must equal stand beside that unit instead:
  the stem's decode in `sheet.ml`, the draw in `draw.ml`. Each of those is
  a rule the RTL must equal rather than restate.

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
`lib/diffusion/model.mli` and `jax/diffusion/quantized.py`, and the
machine obeys it as written:

1. One sheet is one seed. The board takes the SEED cell as it stands
   (`Prng.create`). Seed 0 is the walk that stands still: every uniform is
   0 and the twin plays silence. That is the design, not a fault.
2. The cell order is step-major, seat-minor.
3. The opening: one uniform for each cell; the class is
   `low + floor(u * width)` over the register of the cell's seat.
4. Each pass n of N: the masks (one uniform for each cell, hidden when
   `u * 2^24 < floor(alpha_n * 2^24)`), then one forward pass, then the
   draws (one uniform for each hidden cell, the tempered pick over its
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
a bound, because the elaboration refuses a layer whose dwell is too short to
cover its drain and the band loads behind it — the rule "What the elaboration
refuses" states in full.

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
  the fly from the sheet RAM and the mask bits. No input tensor exists.
- **The head stores nothing.** Its output columns ARE the logit columns —
  the 48 rows of seat v at step t are exactly the 48 logits of that cell —
  and they stream to the draw pipeline in the step-major order the PRNG
  contract already demands. No logit tensor exists; the circuit holds ONE
  STEP, and "The circuit" chapter states where.
- **The weight ROM** is one linear memory of G-byte words, read once each
  cycle, initialized by the bitstream. The gains and biases are small
  per-channel constants beside it.
- **The small state**: the sheet RAM (`T` by four classes), the mask
  bits, the alpha threshold ROM of N entries, the seat registers.

### The activation budget, and the fused pair

**THE LADDER'S TILE TABLE UNDERCOUNTS THE ACTIVATIONS BY TWO, and the
golden candidate does not fit the simple machine.** The table of
`docs/diffusion.md` was computed before the reference round elected Q6 in
int16; its activation numbers correspond to one byte per element. At two
bytes, two live tensors of the golden candidate are 107 tiles, its weights
42, and the sum is about 152 of the device's 135. No lane geometry fixes
storage. The H 16 rungs fit — that is why the climb runs on them.

**The answer for the golden candidate is THE FUSED PAIR, and it is
built.** Conv2 of a residual pair needs only a three-column band of conv1's
output, thus the intermediate tensor never exists in full. Y IS A RING OF
FOUR COLUMNS. The pair output overwrites X in place, and the ring costs the
WIDTH of a column and not the length of the sheet: eleven tiles at any T,
because a 768-bit word fills eleven `512x72` tiles at any depth.

| memory | unfused | fused |
|---|---|---|
| X, `T * H` columns, banked 2 048 + 512 | 54 | 54 |
| Y | 54 | a ring of `4 * H` columns, one bank of 512: **11** |
| the weight ROM, `32768x40` + `1024x40` | 42 | 42 |
| the norms, the anneal table, the exp2 table | 2 | 2 |
| **total** | **152 — OVER** | **109 of 135** |

**THE SCHEDULE, AND WHY B TRAILS A BY TWO.** Write A for the pair's opening
layer and B for its closing one. A at column `c` reads X at `c - 1`, `c`,
`c + 1` and writes Y at `c`; B at `c` reads Y at those three and the
residual X at `c`, and writes X at `c`. With `s` the pair's step counter
from 0 to `T + 1` the turn runs A at column `s` while `s < T`, then B at
column `s - 2` while `s >= 2`: **A0, A1, A2 B0, A3 B1, ..., B(T-2),
B(T-1)**. Three facts fix the lag at two:

- **B at `c` reads Y at `c + 1`, which A wrote two blocks earlier.** A flush
  lands one epilogue behind its drain, thus one WHOLE block must stand
  between the write and the read. At a lag of one every column would wait
  for a flush and "the same cycle count" would be false.
- **B at `c` overwrites X at `c` in place.** A at `c + 1` was the last
  reader of X at `c` and it ran before B at `c`; B's own residual read
  happens at its band load, before its flush.
- **The ring holds exactly four columns.** When B at `c` runs, Y at
  `c - 1`, `c` and `c + 1` are live and A at `c + 2` has just written
  `c + 2`; Y at `c - 2` died with B at `c - 1`. Four is a power of two, thus
  the ring's step is the low two bits of the semantic column and no modulo
  stands anywhere.

The stem and the head are not pairs: they walk column by column, one layer
each. The rejected alternative — int8 activations with per-layer exponents —
respins the twin, re-runs the drift election and moves Gate B's target; it
returns only if the stretch round wants the tiles.

**A STORE PADS AS A ROM PADS, AND THE BANKING IS WHAT KEEPS THESE NUMBERS
TRUE.** The rung-3 measurement build of 2026-08-27 mapped each store of
1 280 columns as `2048x768`. At T 128 and H 20 a store is 2 560 columns:
one memory rounds to 4 096 and costs 86 tiles alone — more than the whole
activation column of the table above — while the ROM's plan banks it as
2 048 + 512 for 43 + 11 = 54. The stores are banked, thus the fused pair
inherits a store that costs its columns and not its address space.

**BANKING DOES NOT MAKE THE UNFUSED CANDIDATE FIT.** Two stores at 54, the
weight image at 42 and the small state at 1 are 151 of 135, which is the
same refusal the table states. The fused pair is what buys the candidate,
and this round only makes the tile it inherits an honest one.

### The walk and the layer table

The outer FSM is the proto's four states grown to five: OPEN, then N
rounds of MASK, FORWARD, DRAW, then PLAY. OPEN, MASK and DRAW are small
serial machinery in the pinned PRNG order — the masks are one uniform for
each of the 512 cells, the draws ride era four's pipeline over the head's
streamed logit columns — and together they cost about three percent of a
pass: three cycles for each cell of the mask draw, and 3 P + 11 cycles for
each hidden cell that draws, which the cycle bench settles. FORWARD
walks the layer table: one record for each layer, stating Cin, Cout, the
source and destination tensors, the ReLU and residual flags, and the weight
and constant bases. A counter walks it; no program, no op vocabulary.

**A TURN IS THE UNIT OF THE WALK, AND NOT A LAYER.** The stem is a turn,
each pair is a turn, the head is a turn: one preamble at each, and one drain
tail behind the last block of each. Inside a pair the two layers INTERLEAVE
block by block, thus neither of them is a unit the walk can name — and the
cycle model counts turns. A turn's blocks are `blocks_of_turn`, and
`Rtl.next_block` is the same rule as a circuit; a gate walks every turn of
every shape through both and demands the same sequence, because the walk
cannot be free to drift from the order the cost model counts.

**THE PHASE TRAVELS IN THE FRAMES.** A layer used to end before the next
began, thus one register named it and every fact of the table muxed by it.
Inside a pair the lead frame can be in B while the now frame is still in A,
and the flush trails both — so what the machine holds is the TURN, and each
frame carries its own phase: the lead frame walks and addresses, the now
frame runs the terms, the DRAIN frame states the ReLU and the residual, the
band load carries the phase it was fired for, and the flush nest walks the
block order a second time. `Elaboration.Rtl.layer_of` turns a turn and a
phase into the table's index, and the table's mux is the one it always was.

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

![The diffusion source: the elaboration, the walk and its one generator, the
sheet with three faces, the column engine with its memories, bands, array
and epilogue, the draw, and the five broadcast nets where ring 3 first
missed timing](diffusion_rtl.svg)

### The shape of the code

Five layers, and the middle one is where the retired program stood:

- **L0, the shared units**: `Prng.Rtl`, `Exp2`, the pick rules of
  `Mgen_nn.Quantized`, and `Vocab.Rtl` for the score port. Era four and era
  five built them and this era changes none of them.
- **L1, the elaboration**: a value, computed from `Model.t`. It
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
- **L4, the walk**: the outer FSM, the sheet, the draw and the score port —
  and `Forward`, the unit that runs the FORWARD state. `Forward` holds the two
  activation stores, the three bands that cache them — the column window, the
  residual columns and the output columns — the weight and norm ROMs, the
  counters and the layer turn, and one instance each of the array and the
  epilogue. It takes the elaboration itself as its functor argument, thus
  every width, depth and base has one authority.

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

**THE WINDOW IS AN IN-PLACE ROTATION OF THREE REGISTERS, AND THE TAP ORDER IS
WELDED TO IT.** Three column registers and no bank. The taps run dy-major and
`row_shift` is dx: the array's interface keeps the freedom to take the taps in
any order, and this caller spends it, because the rotation leans on it.

- Slot A holds `(t - 1, c)`, slot B holds `(t, c)` and slot C holds
  `(t + 1, c)`. Their last reads stand at the dwell cycles 2, 5 and 8.
- The fetches of the next dwell go out at the cycles 0, 3 and 6 and land at 2,
  5 and 8, through the registered read. Each slot takes its new column on the
  edge of its own last-use cycle: the operand register takes the old value on
  that edge as the slot takes the new one.

Thus every turn — the input channel, the group and the column — hides its
fetches under the running dwell, and the only cost that stays is ONE SHORT
PREAMBLE FOR EACH LAYER, at its first dwell. The cycle bench prints it, and if
a fetch fails to hide at some turn the bench says so and this design moves.

**The zero column and the stem enter at the slot load.** Beyond the ends of
the roll the slot loads zero; on the stem it loads the sheet's plane column.
One mux, one place, and the array never knows. The residual band and the norm
bank load in the port slack of the same dwell: the taps take three read slots
of nine, and on a pair-closing layer the taps read Y while the residual rides
the free port of X.

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

**AND THE MEMORY STANDS IN BANKS, BECAUSE VIVADO PADS AN INFERRED ROM TO ITS
FULL ADDRESS SPACE AND SAYS NOTHING.** A ROM of 8 496 words takes the tiles of
16 384. At rung 2 an image of 36 144 words asks 64 tiles against 49 free, and
the mapper answers by demoting EVERY ROM of the design to fabric — the
weights, the norms, the anneal table and the exp2 table — with no warning that
names it. The elaboration therefore cuts the image into banks whose depth is a
power of two: a bank has no address space above its own depth, thus the pad
becomes the elaboration's own, bounded and printed. The plan is the top bit of
the word count and one tail, taken only where it costs less than one bank —
8 192 + 512 at rung 1 for 208 words of pad, and 32 768 + 4 096 at rung 2 for
720. A third bank would save half a tile and buy a second level of mux. A bank
is never below 512 words, because a 512 by 36 RAMB18 is the smallest tile a
word of this width fills.

**THE READ DOES NOT MOVE.** One counter feeds every bank as it stands; each
bank keeps ITS OWN data register, era four's rule for each of them and each
one absorbable as its BRAM's output register; and one mux behind them selects
by the top address bits, carried two cycles behind as the data is. Nothing
stands between the counter and the memories, and nothing between a bank and
its data register. The mux stands BEFORE the operand replicas, thus the
replica bank still breaks the broadcast and the mux's own fanout is the
replica count and no more. The banking is a permutation of ADDRESS SPACE and
never of values: the concatenation of the banks is the flat image in the dwell
order, and the elaboration's gate walks that concatenation with the circuit's
own counter, its own bank decode and its own offset.

**THE PLAN IS THE ROM'S BUT IT IS NOT THE ROM'S ALONE, AND ONE PORT BUILDS
BOTH.** The two activation stores bank by the same rule and the same code —
`bank_plan` states a plan, `Rtl.bank_at` decodes one, and `block_memory` builds
one — because the mapper rounds a RAM's depth as it rounds a ROM's. "The
memories and their ports" holds the measurement and the tiles. The two classes
differ in one thing: a ROM's banks carry an image and a store's banks are
written. Everything else is one statement — the address held once at the pins,
the data held inside each bank, the mux behind those holds, the select held
twice. Where the mux is a whole column wide the select rides one `dont_touch`
replica for each slice of the column, as every array-scale take does. No cycle
is added at either class.

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

- **The chain must empty before the next capture: `9 * Cin >= P`.** It is a
  check and not a comment, and the cost model prints it. Every real rung
  stands far inside it — a trunk layer is 144 against 48, and the stem is 72 —
  and so does each test shape, thus it is a guard and not a limit. **THE
  REFUSAL IS WIDER THAN THIS RULE**: the band loads behind the drain want
  `P + G + 2`, and "What the elaboration refuses" states the whole of it.
  A shape that met this rule alone and broke that one would read a half-loaded
  residual band.
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
| X | `T * H` columns by `P * 16` bits, in banks of a power of two | one column read each three cycles; G column writes each group |
| the Y ring | `4 * H` columns by `P * 16` bits, one bank | the same traffic; the address is the low bits of the semantic column |
| the weight ROM | the packed image in banks of a power of two, G bytes each word | one word each cycle |
| the constants | gain and bias, one entry for each output channel | G entries each group |
| the sheet | `T` by `voices` classes | registers |
| the mask | `T * voices` bits | registers |
| the alpha ROM | N entries of 24 bits | one entry each pass |
| the logit file | `voices` files of P by 16 bits | the head's drain writes it, the draw reads it |

**The stem decodes and does not read.** Its input column for `(t, plane)`
comes from the sheet and the mask: a class plane is one-hot at
`sheet[t][v]`, in activation units, when the cell stands, and zero when the
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

**A ROM PAYS FOR ITS WHOLE ADDRESS SPACE, AND NO WARNING SAYS SO.** The
synthesis log's `ROM: Preliminary Mapping Report` is the only table that ever
states a padded depth, and it states it as the tool's INTENT: rung 2's first
build lists `65536x32 | Block RAM` for an image of 36 144 words, and then, in
the SAME table, lists `65536x32 | LUT` again — the demoted copy. Reading the
two rows together is what names the demotion; there is no message that does.

**AND A RAM PAYS THE SAME, WHICH THE RUNG-3 MEASUREMENT BUILD IS WHAT SAYS.**
`report_ram_utilization -detail` names each store by its mapped dimension, and
at T 64 and H 20 — 1 280 columns — both read `2048x768`, the very 43 tiles
rung 2 pays for 2 048 columns. The mapper rounds the depth of a RAM up to a
power of two as it rounds a ROM's. THE STORES ARE THEREFORE BANKED BY THE ROM'S
OWN PLAN, `Elaboration.store_banks`, and the tiles follow the columns instead
of the address space: 2 560 columns are 43 + 11 = 54 tiles banked as
2 048 + 512, against 86 for one memory the mapper rounds to 4 096.

**BUT SILENCE IN THAT TABLE PROVES NOTHING.** The same weight ROM at rung 1
appears in NO mapping table at all — not the ROM one and not the block RAM
one — in any build of this design, banked or not, and the norm ROM never
appears at either rung. Only the two activation stores are named every time,
in the block RAM reports, because they are the only memories here that are
written. What is extracted as a ROM and what is inferred straight into block
RAM follow different paths through the tool, and a path can go unreported.

**THUS THE INSTRUMENT IS THE CENSUS AND NOT A REPORT**: the RAMB36 and RAMB18
counts of `report_utilization`, the `Block RAM Tile` total, and the LUT count
beside them. Every tile of this design is accountable to a memory by name, and
a demotion costs about twenty thousand LUTs, which is unmissable. Read all
three at every build. "The weight ROM" above states what the elaboration does
about the padding itself.

**THE ELABORATION OWNS THE STORE ADDRESS MAP.** `column_address` states where
a store holds a column — `step * store_channels + channel` — and the map is
t-major, thus the G writes of a group land consecutive. Nothing else
distinguishes the orders. The circuit's ports and the stream instrument slice
ONE rule, which is the argument `norm_word` already makes: a consumer that
computes its own address disagrees with a gate and not with a board.

### The walk

`Source` is the walk: **Idle, then OPEN, then N rounds of MASK and SERVE,
then PLAY.** One unit holds the outer FSM, the generator, the uniform shift
register, the opening multiply, the alpha ROM, the draw service and the
socket answers; it instantiates `Sheet`, `Forward`, `Draw` and `Prng.Rtl`
and wires the plane face straight across. Every uniform comes from that one
generator, in the consumption order of the Scope chapter.

**THE BYTE ORDER OF A UNIFORM IS HIGH FIRST.** `Prng.uniform` takes the
first of its three bytes as the top of the 24: `((high * 256 + middle) * 256)
+ low`. The shift register therefore shifts UP — `u <- (u << 8) | byte` — and
it takes a byte ONE CYCLE BEHIND each step, because a step states its byte in
the cycle that follows it. One capture rule serves every phase: take a byte
exactly when the cycle before stepped. An assembly that shifts the other way
is wrong in every phase at one time, which the walk gate reads on the first
mask.

- **OPEN and MASK are ONE CELL WALK with two write faces.** Each walks the
  cells in the cell order and spends three cycles on each of them — three
  steps of the generator, one uniform. OPEN writes the class
  `low + ((k * width) >> 24)` over the register of the cell's own seat, where
  k is the 24-bit uniform: one multiply and no divide, and the multiply is
  LUTs, because the array owns the DSPs. MASK writes the bit `k < alpha`, the
  24-bit compare against the pass's entry of the alpha ROM.

  **THE FRAME THAT WRITES STANDS TWO CYCLES BEHIND THE FRAME THAT DRAWS**,
  which is the engine's own discipline. The third byte of a cell lands one
  cycle behind its step, and the shift register states the whole uniform one
  cycle behind that. A phase therefore costs its uniforms and two cycles
  more, and the generator steps three times for each cell and no more: the
  step rides the LEAD frame, thus a tail that wrote without drawing cannot
  take a fourth.

  The alpha entry is read at the start of the pass through the registered
  read of era four's rule — the address before the memory and the data behind
  it — thus it stands two cycles later and the first mask write cannot want
  it before that.

- **FORWARD** walks the layer table with a counter. One record states Cin,
  Cout, the source and destination tensors, the ReLU and residual flags, and
  the weight and constant bases. There is no program and no op vocabulary.

- **SERVE rides the head, and the service is SEQUENTIAL AND SIMPLE.** The
  walk strobes `start` and then rides `step_ready`. At each level it takes
  the seats of the offered step in order: one cycle reads `hidden` at the
  cell port, and A STANDING CELL COSTS NOTHING MORE. A hidden cell spends
  five cycles to assemble its uniform whole — three steps of the generator,
  and the two behind them in which the last byte lands and the whole 24
  stands — then `start` to `Draw` over that seat's `logits`, and the drawn
  class writes back through the same cell port in the cycle `busy` falls. After seat 3 the walk strobes
  `step_taken` and waits for the next level; the pass ends when `busy` falls.

  The overlap of the uniform under the draw's own peak walk would save three
  cycles of a draw that is 2.8 percent of a pass, measured. It is not taken:
  the walk holds one draw at a time and states it in one order, and that is
  what the per-phase gate reads.

  **THE DRAW TAKES ITS OWN PEAK**, thus a cell is three walks of the file:
  the peak, then the exp2 weights and their total, then the pick on the
  24-bit uniform. The head's drain could track the peak for nothing and save
  one walk, and it does not, because a peak handed in is a precondition the
  unit cannot check: a caller that states one that is not the peak states
  another distribution, and nothing says so.

  **THE TABLE IS A FORK, AND THE FORK IS WHY A CELL COSTS 3 P AND NOT 5 P.**
  The shared `Exp2` registers its table entry but takes the shift and the zero
  test from its magnitude as it stands, thus it asks a caller to hold that
  magnitude for two cycles — a walk of 48 classes would pay it twice over, and
  a magnitude a cycle reads one class's entry under another class's shift. The
  draw's fuzz read that fault as 58 disagreements of 60 before the fork stood.
  `lib/diffusion/exp2.ml` registers the shift beside the entry: two flip-flops,
  and a gate that states what a holding caller reads does not move. **WHETHER
  TO BACKPORT IT TO `lib/nn` IS A DECISION FOR WHEN ERA SIX SETTLES** — a unit
  two shipped eras carry does not move for a round that has not shipped — and
  that gate is the evidence for it.

  **THE SEAM TO THE DRAW IS A LEVEL AND A STROBE, AND NO TAG CROSSES IT.**
  `step_ready` is a LEVEL: "the file stands whole" is a state of the forward's
  own registers, and the wire exports that state. A strobe would export the
  event and ask every consumer to rebuild the state with a latch of its own —
  the same flip-flop, on the wrong side of the seam. The level rises the cycle
  after the last drain row of the step lands, and at a ragged shape after the
  band of the LAST group fills; it falls on the edge after `step_taken`. **The
  head offers the steps in order, 0 to T - 1, each one exactly one time**, and
  the level falls before it rises again, thus THE WALK'S COUNT OF ITS OWN
  ACKNOWLEDGEMENTS IS THE STEP it names at the cell port. The tag-travels rule
  earns its keep where a caller must model the depth of a pipe; this seam is a
  full interlock and has no depth to model. `logits` stands still exactly
  while the level stands, which is the draw's own precondition, satisfied by
  construction.

  **Phase I holds the head and the draws apart.** The dwells of step t + 1 do
  not start before `step_taken`, because a capture would write over the file.
  The overlap is safe on the data — the draws write the sheet and the head
  reads X — thus it stays a local cut, for when the cycle bench prices the
  wait.

- **PLAY answers the socket, and it is what the Idle state does.** The walk
  has one rest, thus `rewind` is read where `idle` stands and the generator's
  `load` rides that one condition — the idiom of the eras. A step counter
  walks the sheet: `step` reads its frame through the score face, `valid`
  answers ONE CYCLE BEHIND the strobe with the frame the face stated, and the
  counter then advances. **Past step T - 1 the frame is four zero bytes, for
  ever**, until the next `rewind` puts the counter back at 0. The face is
  combinational from the cells, thus PLAY holds no copy of the sheet and the
  answer costs one register.

**One cell port, three users, and no contention BY STATE.** OPEN writes
classes, MASK writes bits, and SERVE reads `hidden` and writes classes, in
disjoint states of one FSM. The frame face is the sequencer's own and never
the walk's, thus a piece plays while nothing writes.

**The sheet is written IN PLACE during the head, and it is exact.** The head
reads the trunk tensor, and the forward computed that tensor from the sheet
as it stood at the start of the pass. Nothing after the stem reads the sheet,
thus a draw at step t cannot reach a later column. The twin copies the sheet
because a value engine must; the circuit does not have to.

### The seam to the sequencer

`Source_intf`, unchanged, and the top level does not move: `Top` holds the
elaboration of the elected checkpoint and names `Source.create ~e ~seed` and
nothing narrower, thus one line of `board/nexys-4/dune` seats era six. The
elaboration is the one argument here as it is at `Forward` — every width,
every depth, every base and the register of every seat has one authority —
thus a new checkpoint moves that line and nothing else.

- **`rewind`** is the run start. It captures SEED, drops `idle`, and runs OPEN
  and the N passes. `idle` rises when the sheet stands. The sequencer's
  `WaitRewind` state waits for exactly this already, thus the seconds of the
  draw need no rule of their own.
- **`step`** reads `sheet[t]`, maps each class through `Vocab.Rtl`, packs the
  frame and answers `valid` one cycle behind the strobe. The map is the
  vocabulary's rule and this era does not restate it.
- **Past step T - 1 the frame is silence**, for ever, until the next
  `rewind`. The sequencer plays the sheet one time and the reset button gives
  the next run, as Phase I states.

The score port stands behind its own interface, thus Phase II's second sheet
memory is a local change.

### What the elaboration refuses

The elaboration reads the checkpoint and every dimension follows it. It
refuses loudly, and the message names what it refused:

- `Model.check_shape` first: the chain of the channels, the accumulator bound,
  the kernel counts and the constant rows.
- `9 * Cin >= P + G + 2` for each layer: the drain rule above AND THE BAND
  LOADS BEHIND IT. The chain of P stages must empty before the next capture,
  and behind that the residual columns and the norm words of the next group
  are fetched the moment the drain has read its last residual row — one
  address for each lane, two cycles of read latency behind them — because one
  band serves every group and nothing is doubled. **The array's rule alone
  leaves a gap one lane wide**: H 6 at G 5 dwells 54 against 55, the chain of
  48 empties inside that, and the engine would read a half-loaded band with
  nothing below it saying so. G 5 is the fused rung's geometry, thus the gap
  is a shape the ladder could really elaborate; the elaboration states the
  whole rule and a test stands on the gap.
- a walk of no passes, a sheet of no steps, and a group of no lanes: N, T
  and G each stand at 1 or above.

## The cost model

A lane is one DSP48 taking one product each cycle at 100 MHz. The device
has 240; this board is a new top level and carries no other era's units,
thus all 240 are the model's to spend. At 192 lanes, 48 stand free — and
they STAY free: the epilogue and the draw are LUTs, because the fused rung
wants all 240 for the array. "The circuit" chapter gives the reason.

The climb, at T 128, N 512, inside the 25.6-second playback window
(2.56 G cycles):

| rung | params | weights | activations | tiles (of 135) | cycles / pass | sheet at N 512 |
|---|---|---|---|---|---|---|
| `l16-h16` | 34 k | ~9 | ~88 | ~100 (74%) | 1.09 M | 5.6 s |
| `l64-h16` | 147 k | ~36 | ~88 | ~127 (94%) | 4.63 M | 23.7 s |
| `l48-h20` unfused | 170 k | ~42 | ~108 | ~151 — OVER | — | — |
| `l48-h20` fused, G 5 | 170 k | 42 | 65 | **109 (81%)** | 4.30 M | 22.0 s |

The cycle numbers are the formula's ideal at 192 lanes (240 for the fused
G 5 row) and land within a percent of `MAC / lanes`, because the geometry
divides the H 16 shapes exactly. `l64-h16` holds 8 percent of slack against
the window for the real overheads.

**THE WEIGHT COLUMN IS THE UNPADDED TRUTH, AND IT WAS NOT WHAT A BUILD PAID
UNTIL THE ROM ROUND.** Vivado pads an inferred ROM to its full address space:
rung 1 paid 16 tiles for the ~9 this table states, and rung 2 asked 64 for
~36 and lost every ROM of the design to fabric for it. The banking of "The
weight ROM" takes the padding back, and "The ROM round" below measures both
rungs against this table.

**THE OVERHEADS ARE MEASURED NOW, and the window loads are not among them.**
The cycle bench beside `Forward` counts one forward against `forward_cycles`,
and the walk bench beside `Source` counts everything around it:

- **Nine cycles at each layer, for the preamble, and nothing else grows.**
  The number does not follow the columns, the groups or the channels, thus
  every fetch of the rotation but the first of a layer hides under a running
  dwell. That is the claim of "The dwell", measured.
- **The layer turn stands at or under the drain tails the model already
  counts.** It is not an overhead on top of them; it IS them.
- **The head's wait is about `P + 10` cycles at every step** — 58 at P 48,
  one of them the epilogue's fourth stage — and it grows with T and with
  nothing else. This is PHASE I'S
  SERIALIZATION, priced: the head does not open step `t + 1` until the draw
  has taken step `t`, because a capture would write over the logit file.
- **One pass at the rung, measured whole: 1 175 164 cycles** for the opening
  and pass 0 at T 128, H 16, G 4 — the opening 1 538 and the mask 1 538
  against a cell walk of 1 536 each, the engine 1 096 246 against
  `forward_cycles` 1 088 256, and the service 75 840 for the 470 cells that
  pass 0 redraws of 512. Every part stands inside a percent of its model.
  PASS 0 IS THE HOTTEST and not the mean: the anneal opens at alpha 0.9,
  thus the mean pass redraws 194 and costs the 1 121 558 below.
- **The walk's own machinery is 2.8 percent of a pass**, and the walk bench
  beside `Source` measures the constants it stands on: a cell walk costs its
  uniforms and two cycles more, a STANDING cell costs one cycle, and a HIDDEN
  cell costs 161 at P 48 — one for the seat, five for the uniform and the
  draw's own 154 with the cycle that writes the class it drew. The rung's
  anneal table hides 194 cells of 512 in the mean pass, thus the service is
  31 766 cycles against 1 088 256. The same bench reads the ENGINE inside the
  walk at 10 210 cycles a pass where S3's bench read 10 211 for that shape
  alone: the walk adds nothing to the engine, and every cycle of the service
  is its own.

At the elected rung that is 144 cycles of preamble and 7 424 of head wait
against 1 088 256, thus a pass measures about 0.7 percent over the model, and
the service above it takes the whole to about 1 121 558 — the playback window
does not move. What Phase II's overlap buys back is the head's wait and the
service, and not the preamble. Even 48
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
- **The probe is a ladder, as the drift is.** Each ring is measured before
  the machine around it exists, at probe cost and not build cost. The next
  ring is below.

### The column engine, measured

Ring 2: the whole of `Forward` at the elected `l16-h16`, T 128 and G 4,
through Vivado out of context on the part at 100 MHz. The weights are DRAWN
under `Model.For_test.drawn` and not a checkpoint's — a timing reading needs no
correct data, and every width of this design follows a RULE and not a
model's own peak, which is the argument `shift_bits` already makes — thus a
drawn model elaborates the netlist a trained one does.

| | measured | against |
|---|---|---|
| WNS | **+0.050** | MET, 0 failing of 54 405 endpoints |
| WHS | +0.096 | MET |
| DSP48E1 | 192 (80%) | exactly one for each lane |
| block RAM | 103 (76%) | the cost model's ~100 (74%) |
| LUTs | 13 498 (21%) | — |
| registers | 18 778 (15%) | — |

**Era four's trap holds on its home ground.** The synthesis log states the
absorption for every lane — the operand pair, the product and the sum — and
the weight ROM reaches the array's B port with ZERO logic levels between
them, at +0.122. The address registers before the memory and the data
registers after it are what buy that.

**THE CRITICAL PATH MOVED, AND IT IS NOT THE ARRAY.** Ring 1 read the array
alone at +2.761 register to register; the whole engine reads +0.050, and the
five worst paths are one structure: 19 logic levels, twelve of them CARRY4,
ending on the SET pins of a 16-bit register. That is the epilogue's second
stage — the variable shift, the bias and the first `clamp16` in one cycle —
and the set pins are the saturate-to-32767 arm of the clamp. It is 9.437 ns,
63 percent of it route.

The design does not move for it: it meets, and this project measures before
it optimizes. What the reading buys is the next known weak point and its
answer — the epilogue's stage 2 splits in two, and the cost is one more
cycle of a pipeline whose tag already travels, thus no caller counts it.
Held, like the capture-select reserve, until a full build asks. **The
capture net did not come to the top in this ring**, thus that reserve stands
where it was.

The next ring is the machine around this one: the walk, the sheet, the draw
and the socket, in context and on the real part.

### The whole machine, measured — AND IT DOES NOT MEET

Ring 3: the board top level at rung 1, `l16-h16-100k` at T 128, G 4 and
N 512, through Vivado on the real part IN CONTEXT. The weights are the
checkpoint's; the geometry is `gen_verilog`'s three numbers.

| | measured | against |
|---|---|---|
| WNS | **−6.125 — VIOLATED**, 11 002 failing of 59 482 endpoints | ring 2's +0.050 |
| WHS | +0.051 | MET |
| DSP48E1 | 192 (80%) | exactly one for each lane |
| block RAM | 103.5 (76.7%) | the cost model's ~100 (74%) |
| LUTs | 18 150 (28.6%) | ring 2's engine alone, 13 498 |
| registers | 23 274 (18.4%) | ring 2's 18 778 |

**EVERY RESOURCE LANDS ON THE MODEL AND THE TIMING DOES NOT.** The cost
model's rung-1 row is confirmed at the tile: 103.5 against ~100, and 192 DSPs
against one for each lane. Era four's trap holds where the engine applies it —
all 192 DSPs carry mode `((C:0x0) or P)+(A2*B'')'` with seven elements
absorbed into each. The design is NOT congested: no congestion window stands
above level 5, and the device holds 28 percent of its LUTs.

**THE FAILURE IS BROADCAST, AND IT IS A FAMILY AND NOT A PATH.** Ranked, with
the evidence of `report_timing` and `report_high_fanout_nets` behind each:

| rank | slack | what it is | fanout | route share |
|---|---|---|---|---|
| 1 | −6.125 | the draw's magnitude into the exp2 ROM address | — | 56% |
| 2 | −5.260 | the weight ROM's sign bits into the array's B ports | 528, four nets | 83% |
| 3 | −3.781 | the capture select and the capture enable | 6 019 and 6 144 | — |
| 4 | −2.926 | `band_row` | 3 073 | — |
| 5 | −2.6 | the three window slots | 768 each | — |

- **The draw's magnitude is the one path of LOGIC**, and it is 21 levels with
  ten CARRY4: the logit file register, the 48-way class mux over a 768-bit
  column, the peak subtract, the shift to Q12, the temper multiply and the
  saturate — INTO A MEMORY'S ADDRESS PINS with no register between. That is
  this document's own rule, broken in `Exp2` alone, which era six forked from
  `lib/nn` and which registers its entry and its shift but not its address.
  Registering the address is necessary and NOT sufficient: the 14 ns stands
  BEFORE the address, thus the magnitude wants two stages and not one.
- **The weight broadcast is 12.4 ns of ROUTE on ZERO logic levels.** One
  RAMB36 output bit drives 528 pins — a weight byte's sign, sign-extended into
  the top eleven bits of an 18-bit B port on each of 48 rows — and the placer
  put that BRAM at Y28 and its DSP at Y1. Ring 2 read this very path at
  **+0.122**; the same wire in context reads −5.260. **THE RULE THAT BUYS THE
  ZERO LOGIC LEVELS IS WHAT LEAVES THE NET ONE DRIVER**: the data register
  after the memory is absorbed INTO the BRAM's own output register, thus no
  flop stands in the fabric to replicate.
- **The capture select is here at the fanout ring 1 predicted**, 6 019, and it
  is only the THIRD cause. The reserve is real and it is not enough alone.

**WHAT THE OUT-OF-CONTEXT PROBES COULD NOT SEE.** The rule of this document is
that an OOC number is optimistic by construction, and ring 1 named the reason
it would be: a net at high fanout and 92 percent route is the kind whose length
moves. The mechanism was called correctly and THE COUNT WAS NOT — the engine
holds five broadcast families and the probe had the die to itself for every one
of them. A probe answers for the logic it holds and never for the wire the rest
of the design will take from it.

**The stage stopped here.** Two of the three leading causes stand outside the
reserves the round licensed, thus the board was not programmed and no unit
moved: what the machine needs is a design round over the broadcast, and the
reserves enter it as two answers of several rather than as the answer.

### The broadcast round — ring 3 rebuilt, AND IT MEETS

The design round over the broadcast, settled and built 2026-08-27: WNS
**+0.010**, no failing endpoint of 59 849, on the first roll.

**THE CENSUS FIRST, because the stage's own instruments had misread it.** One
path for each endpoint over the failed build's checkpoint — `-nworst 1`, where
the stage's report took forty thousand paths of the ONE worst endpoint — reads
13 657 endpoints within half a nanosecond and corrects the table above three
ways:

- **Rank 1 is one cone with three faces.** The worst paths into the exp2
  table's address (−6.125), into the FDSE set pins (−5.655) and into an FDRE
  of the walk (−4.982) all start at one bit of the logit column register and
  are the same magnitude cone, 21 to 22 levels — and the cone opens with the
  SEAT mux, before the class mux the table records.
- **The epilogue's stage-2 clamp path was failing on its own, at −2.051** and
  18 to 19 levels — ring 2's known weak point, invisible to both instruments:
  the path report was flooded and the fanout report does not see low-fanout
  logic. One of the two licensed reserves was therefore needed after all.
- **A seventh family stood unnamed: the X and Y store address registers at
  −1.78**, into about 43 RAMB36 address pins each, already replicated five
  times by the tools on their own.

**THE DESIGN IS ONE DISCIPLINE: NO ARRAY-SCALE NET KEEPS A SINGLE DRIVER.**
Every take of a column-wide register bank stands in a bank of replicas, one
copy for each slice of eight rows, `dont_touch` so the tools neither merge
the copies nor absorb them into a primitive. The window slots, the output
band and the logit file are sliced to match; the chain takes one capture
register for each stage — the ring-1 reserve, applied. Two cuts go deeper
than replication:

- **The weight operand register moved from inside the DSP into the fabric
  bank.** The era-four rule put a register after the ROM and one at the
  operand, and the tools absorbed the first into the BRAM's own output
  register and the second into each DSP's B port — leaving the broadcast net
  ONE driver of 528 pins with no flop in the fabric to replicate. The bank IS
  the operand register: the depth of the pipe does not move, no counter
  moves, and the sign now fans 88 pins for each copy. The synthesis log
  states the new mode for all 192 lanes — `((C:0x0) or P)+(A2*B)` — the B
  port direct from the replica bank, the A and accumulator absorptions
  untouched.
- **The draw's magnitude cone is cut in four**: the walk register behind the
  class mux — all three walks share it, thus the one-mux rule stands — the
  temper register, then the table's own address and entry registers, the
  era-four rule applied to the `Exp2` fork. Cycles are the resource the walk
  has and levels are what break: the pipe adds seven cycles to a hidden cell
  (`busy_cycles` 147 to 154) and the service moves from 2.7 to 2.8 percent
  of a pass. The retire pipe carries its walk's state, because the peak
  walk's tail rides into the weigh's first cycles and an untagged pipe would
  take it into the total.

The epilogue split its stage 2 — the shift with the bias, then the ReLU with
the first clamp; latency 3 to 4, and the tag travels, thus no caller moved.

| | ring 3, failed | the round |
|---|---|---|
| WNS | −6.125, 11 002 failing | **+0.010, 0 failing** |
| endpoints within 0.5 ns | 13 657 | 145, scattered |
| LUTs | 18 150 (28.6%) | 21 312 (33.6%) |
| registers | 23 274 | 23 718 |
| block RAM, DSPs | 103.5, 192 | the same |

**What the round left alone, the rebuild judged.** The residual band's load
decodes and the store address registers were left for the rebuild's verdict,
and every one now stands positive — the worst high-fanout net reads +0.893,
and the seventh family cleared with the congestion the five families had
made. The worst path of the build is a ten-level socket-side path at 75
percent route: the natural edge of the design, not a structure. **The slack
is inside the lottery band and the variance reserves are now spent**, thus a
future wobble of THIS netlist is the seed round's 0.1 ns lottery and the
build simply rolls again — the one case where the re-roll is the answer.

### The board rung — Gate B closes

Measured 2026-08-27, the round's bitstream on the board, the panel seed as
it stood: **47872**. The smoke first: the S-1 plays, and the stopwatch reads
about 5.7 seconds from the push to the first note against the model's 574 M
cycles — fallback (c)'s check, confirmed on silicon.

Then the capture, driven whole from the host: the RUN cell cycled over the
console UART, `amidi` on the S-1's thru, and the reference's wire bytes from
`play_diffusion -quantized -seeds 47872 -fade 0 -play -device <file>` — the
fade off, because the fade is the software player's and Phase II owns the
board's. The driver and the bitstream came from one repository state, as the
standing rule demands.

**The two streams are 804 bytes and 268 messages EACH, and they agree BYTE
FOR BYTE IN ORDER.** The gate only demanded an order-tolerant alignment —
the thru reorders locally under dense chord bursts, measured in the chorale
era — and the allowance went unused: every message aligned at displacement
zero. The board plays the twin's sheet exactly, thus instrument 5 closes
and Gate B stands whole at rung 1.

### The ROM round — the padding taken back, and rung 2 fits

The design change of "The weight ROM", built on both rungs 2026-08-27. Rung 1
is the board top level as it stands; rung 2 is the same tree with the one-line
checkpoint swap, in a scratch worktree.

| | rung 1 before | rung 1 banked | rung 2 first build | rung 2 banked |
|---|---|---|---|---|
| WNS | +0.010 MET | **+0.007 MET** | +0.003 MET | **+0.008 MET** |
| endpoints | 59 849 | 62 002 | 63 241 | 62 518 |
| block RAM | 103.5 (76.7%) | **96.0 (71.1%)** | 86 (63.7%) | **124.0 (91.9%)** |
| RAMB36 / RAMB18 | 102 / 3 | 94 / 4 | 86 / **0** | 123 / 2 |
| LUTs | 21 312 (33.6%) | 21 331 (33.7%) | 42 599 (67.2%) | **21 419 (33.8%)** |
| registers | 23 718 | 23 682 | 24 430 | 23 744 |
| DSP48E1 | 192 | 192 | 192 | 192 |

**THE COST MODEL WAS RIGHT AND THE BUILD WAS PAYING PADDING.** Rung 1 drops
7.5 tiles, which is 16 against 8.5 — the weight ROM's full address space
against its banks — and nothing else in the design moves: 19 LUTs for the mux
and its decode, 36 registers fewer, and the same 192 DSPs. Both rungs now land
UNDER the climb table: 96.0 against the model's ~100, and 124.0 against ~127.

**AND THE ROUND IS WHAT MAKES RUNG 2 A BUILD AND NOT A DEMOTION.** The first
rung-2 build met at +0.003 with 86 tiles and 42 599 LUTs, and those numbers
are the failure and not the result. **ITS RAMB18 COUNT IS ZERO**, which is the
whole story in one number: 86 tiles is the two activation stores ALONE, and
every ROM of the design — the weights, the norms, the anneal table and the
exp2 table — stands in fabric, because the padded weight image asked 64 tiles
against the 49 the stores left free. Banked, the same checkpoint holds all of
them in block RAM and the LUTs halve. The tiles then account for the whole
design exactly: 86 for the stores, 32 and 4 for the two weight banks, 1 for
the norms, and the two RAMB18 of the anneal and the exp2 tables.

**THE TIMING IS THE LOTTERY'S AND NOT THE ROUND'S.** Both rungs met on the
first roll, inside the band of about 0.1 ns that three builds of one netlist
have measured. The mux stands between a BRAM's output register and the operand
replicas, thus it adds one level to that path and rung 1 reads 3 ps of it.

**What this round does NOT settle** is the rung-2 election. The tiles and the
timing say the circuit fits; whether `l64-h16` sings is the listening gate's,
and the board and the flash stay behind it, as the climb states.

### The clock-skew mortgage — a build that STA blesses and silicon refuses

The service-cut round closed with a trap that no instrument before the byte
gate can see, measured 2026-08-27 on the first cut build.

**THE FAILURE.** The cut's netlist was sound — Cyclesim exact against the
twin, the phase gate exact write for write — and its build met at WNS +0.004
with every constraint green. On the board it drew a DIFFERENT coherent sheet
at a fixed seed on every run: 762, 594 and 618 bytes across three captures at
one panel seed, a power cycle curing nothing, while the pre-cut bitstream on
the same rig and seed answered byte for byte. The board plays plausibly
throughout — right channel, right density, right shape — thus THE SMOKE TEST
CANNOT CATCH THIS CLASS, and the finished sheet alone convicts nothing. Only
the byte gate against the twin can.

**THE MECHANISM.** The roll placed badly (post-place −0.143, the worst of the
day) and the post-route phys_opt rescued setup by ADJUSTING CLOCK SKEW — the
`Physopt 32-703` move — on six bits of the uniform shift register, a register
replica and a BUFG replica among them. Each adjustment re-taps one flop's
leaf clock so it arrives later: the failing setup path gains, and every short
path into that flop pays. Neighbour bits of the shift register ended on
different clock trees, and the byte-shift path `uniform_reg[7] →
uniform_reg[15]` closed at 0.001 ns of hold at the fast corner — met, in the
report's own words. A corrupted uniform scrambles which cells hide and which
classes draw, which is exactly a coherent wrong sheet.

**THE RULE: STA met by a picosecond is not met.** The refusal instrument, at
every build from this round on:

- **No `Physopt 32-703` on a register family.** Grep the build log. One
  adjustment on an isolated synthesis replica can stand when the margins
  bound it (the fixed build carries one, at 29 ps); a family of them on a
  shift register or a counter is the mortgage, and the build re-rolls.
- **WHS at or above about 0.010.** The three builds that passed the byte
  gate read +0.026, +0.012 and +0.029; the one that failed read +0.001.
- **ONE `BUFGCTRL` IN THE UTILIZATION REPORT.** A `Physopt 32-703` moves a
  register to a second clock tree, thus the COUNT OF TREES is the mortgage
  made visible in one number, and it needs no log grep. Measured over eight
  builds of this design: every sound one carries ONE, every mortgaged one
  carries two or three, and the mux-before-the-data-hold build carried five.
  Read it beside the census.

**THE FIX IS STRUCTURE AND NOT THE LOTTERY.** An Explore re-roll failed the
same family honestly — setup −0.014 and twelve mortgages on the same cone —
thus the fault was a real path: one `u` register feeding the opening
multiply, the mask compare and the draw's threshold at a fanout near sixty,
two thirds of the failing path pure route. The answer is the broadcast
round's own rule applied to the walk: three `dont_touch` replicas of `u`,
fed the same next value, one for each consumer's arithmetic. A replica is
`u` cycle for cycle, thus no gate moves; the placer seats each copy beside
its own carry chain. The fixed build reads WNS +0.018 and WHS +0.029, the
healthiest of the day's five, with the uniform family untouched — and the
byte gate passed twice, byte for byte.

**THE DEBUGGING INSTRUMENT IS THE A/B.** One JTAG program of the last proven
bitstream, the same rig and the same seed, answers the only question that
matters — the build or the method — in three minutes. It is what separated
this trap from a false reference in one step.

### The store round — a store pads as a ROM pads

A measurement build of `l48-h20-100k` at T 64, G 5, N 512 ran 2026-08-27 to
ask one question of the golden candidate — does an array of 240 DSPs place and
route — and answered a second one nobody had asked. Its reports stand in
`board/_build/rung3-unbanked/`.

| | rung 2 (in flash then) | `l48-h20` at T 64, G 5 |
|---|---|---|
| LUTs / registers | 21 455 / 23 818 | 24 127 / 25 699 |
| DSPs | 192 | **240 of 240** |
| block RAM tiles | 124 | 129 of 135 |
| WNS / WHS | +0.018 / +0.029 | **+0.216** / +0.008 |
| `Physopt 32-703` | 1 | 0 |

**THE ARRAY AT 240 DSPS BUILDS.** The build met with the healthiest setup
margin of any build of this design, and the worst path is in the DRAW —
`uniform_open_reg[7]`, 12 levels, six of them carry — and not in the array.
The one number under its line is WHS +0.008, a hair below the hold
instrument's 0.010; the mortgage the instrument really hunts is absent, and no
bitstream came out of this build, thus the build stands as a measurement and
not as a candidate.

**AND THE STORES PAD.** `report_ram_utilization -detail` maps each store of
1 280 columns as `2048x768`, 43 tiles — the same 43 rung 2 pays for 2 048
columns. The 129 account exactly: 86 for the two stores, 40 and 2 for the two
weight banks, and two RAMB18 for the norms and the anneal table. The mapper
rounds the depth of a RAM up to a power of two as it rounds a ROM's, and the
ROM round's whole finding therefore holds one memory class wider than it was
written for.

**THE COST IS AT T 128 AND NOT HERE.** A store of 2 560 columns rounds to
4 096 and costs 86 tiles alone; banked as 2 048 + 512 it costs 43 + 11 = 54.
The design change is `Elaboration.store_banks` and the banked port of "The
weight ROM". Rung 2 banks its 2 048 columns in ONE bank, thus nothing of the
plan reaches it: the port folded the ROM's hand-built mux into itself and moved
the constant table, thus the Verilog is not byte-identical, but rung 2 rebuilt
from it reads 21 455 LUTs, 23 818 registers, 124 tiles, 192 DSPs and
WNS +0.018 / WHS +0.029 — every number of the flashed build, not a band around
them (`board/_build/rung2-port`). Nothing on the board or in the flash moves.

**T WAS NOT THE ANSWER, AND IT WAS ASKED.** 2 040 columns fit one bank of
2 048, thus T at 102 or below would need no banking at all. T is the musical
parameter — the sheet is eight bars and the ear elected the model at T 128 —
and the masked loss of the float model over the 76 valid sheets does not
move with it: 0.1935 at crop 128, 0.1930 at 102, 0.1833 at 96. The draw window
does not move with T either, because the pass scales with T: the N 512 draw
needs 172 ms for each sequencer step at G 5 at every T, against `STEP_MS` 200.
G 5 is the window condition and not a preference.

**THE FIRST PLACEMENT PUT THE MUX BEFORE THE DATA HOLD, AND THE TILES CAME
WHILE THE TIMING WENT.** It built twice at the same geometry, 2026-08-27, in
`board/_build/rung3-mux-before` and `board/_build/rung3-mux-before-explore`:

| | unbanked | mux before the data hold | the same, Explore |
|---|---|---|---|
| block RAM tiles | 129 | **108.5** | 108.5 |
| LUTs / registers | 24 127 / 25 699 | 26 213 / 27 123 | 26 230 / 27 125 |
| WNS | +0.216 MET | **−0.354** | **−0.607** |
| failing endpoints | 0 | 144 | 285 |
| WHS | +0.008 | +0.014 | +0.011 |
| `Physopt 32-703` | 0 | 15 | 11 |

The tiles are the round's whole claim and they arrive: `report_ram_utilization`
maps each store as `1024x768` and `512x768` with no pad, and 108.5 lands on
the 109 the cost model predicted. **THE SETUP FAILURE IS STRUCTURE AND NOT A
ROLL** — the second roll is worse than the first, twice the band a re-roll has
ever moved — and every failing endpoint of both builds is one class: the store
address cone into a bank's READ pins, under the two names the two bank shapes
carry (`ADDRB` of the 1K by 36 bank, `ADDRA` of the 512 by 72 bank). The worst
path ends in a LUT6 at fanout 24 with 2.599 ns of pure route. The
`Physopt 32-703` list is the store address counters with `uniform_open_reg[3]`
and `signal_reg_40` beside them, which is the mortgage the hold instrument
refuses.

**THE MECHANISM IS THE DATA HOLD, AND THE NETLIST NAMES IT.** A store's read is
two registers: the address hold and the data hold. Unbanked, the data hold
rides the block RAM's own latch and the address hold stands in fabric, thus a
memory's address pins are driven by a REGISTER at zero levels. A mux before the
data hold evicts that hold from the latch into fabric — 1 424 registers, about
two columns of 768, and the LUT3 of the mux behind them — and to keep the read
at two cycles Vivado absorbs the ADDRESS hold into the latch instead. The whole
tap-address cone then lands on the pins: the ±1, the multiply, the add and the
`is_join` mux, eight levels of it. This is NOT the broadcast class: the fanout
fell from 86 to 24 and the last hop from 4.1 ns to 2.6 ns, while the logic
doubled. The weight ROM never met it, because its mux has always stood behind
its data registers.

| | unbanked | mux before the data hold |
|---|---|---|
| a bank's read address pins are driven by | a fabric register, 0 levels | an 8-level cone, 3.5 ns logic |
| registers | 25 699 | 27 123 |

**THE ANSWER IS THE ROM'S OWN PLACEMENT, AND IT MEETS.** `block_memory` holds
the data INSIDE each bank, muxes behind those holds, and holds the select
twice; the weight ROM reads through that same port with an image of a bank, and
the hand-built ROM mux is gone. One port, both memory classes. The rebuild is
`board/_build/rung3`:

| | unbanked | mux before the data hold | **mux behind, one port** |
|---|---|---|---|
| block RAM tiles | 129 | 108.5 | **108** |
| LUTs / registers | 24 127 / 25 699 | 26 213 / 27 123 | **24 356 / 25 563** |
| DSPs | 240 | 240 | 240 |
| WNS / WHS | +0.216 / +0.008 | −0.354 / +0.014 | **+0.122 / +0.019** |
| `Physopt 32-703` | 0 | 15 | **0** |

The stores map `1024x768` and `512x768` with no pad, 21 tiles under the
unbanked 129 and one under the cost model's 109. The registers come back BELOW
the unbanked count and the muxes cost 229 LUTs, not the 1 600 two 768-bit muxes
would stand for on their own: they fold into the LUT6s that already mux the
zero column, the stem and Y against X. A bank's read address pins are driven by
a fabric register at ZERO levels again, and the block RAM's output into the
slot registers reads +1.017 at three levels, against +1.44 with no mux at all.
Setup met on the first roll with no clock-skew adjustment anywhere.

### The fused pair — Y stops being a tensor

The design of "The activation budget", built 2026-08-28. The round moves
MEMORY and not work: every value the engine writes is the value
`Quantized.layer_writes` writes, thus the twin is untouched, Gate B's target
stands and the drift lines stand.

**WHAT THE SOFTWARE HALF LEARNED.** A layer is no longer the unit of the
walk. `Elaboration.turns` states the stem, one turn for each pair, and the
head; `blocks_of_turn` lists a turn's blocks in the order the engine runs
them; `Rtl.next_block` is that same rule as a circuit, and a gate walks
every turn of every shape through both. The elaboration prints the ring and
the turns, and `turn_cycles` replaces `layer_cycles` — the tail belongs to
the turn, thus the fused machine spends `pairs * P` cycles fewer than the
unfused one on the same arithmetic.

**THE THREE TRAPS OF THE INTERLEAVE, and none of them moves a frame.**

- **The drain is a frame of its own.** A group's chain empties `P` cycles
  behind the term that captured it and the epilogue answers three behind
  that, thus THE LAST GROUP OF A BLOCK DRAINS UNDER THE NEXT BLOCK — which
  inside a pair is the other layer. A join flag read from the now frame adds
  A's residual to nothing and drops B's, and the ReLU follows the wrong
  layer; the twin sees a column of zeros where it wants the residual. It was
  invisible while a layer was the unit: a layer's last group drained under
  `Turn`, where the layer register still stood at its own value. The phase
  is captured where the array captures the sums.
- **The X port is arbitrated by the cycle and not by the layer.** A join
  layer used to point X at the residual for its whole run, because its taps
  read Y and nothing else wanted the port. Fused, the fetch of the next A
  block goes out under B's LAST INPUT CHANNEL and needs X while the now
  frame is still in B. The residual takes the port only in the cycles the
  load really addresses it.
- **The source of a fetched column travels with the fetch.** Which memory a
  column comes from is a fact of the FETCH's layer: the columns that land
  under B's last channel are A's and come from X, and a slot that read the
  layer register would take Y. The bit rides beside the zero flag.

**THE DWELL REFUSAL GAINS NINE CYCLES.** The load ends by `P + G + 2` and
the next block's fetch opens at `dwell - 9`, thus `9 * Cin` must reach
`P + G + 11`. At P 48 and G 4 that refuses H 6 — dwell 54, which the
unfused floor accepted exactly — and admits H 7 at 63 against 63. No elected
shape is in the band: the stem dwells 72, rung 2 144, rung 3 180. THE TWIN'S
OWN TEST SHAPE IS IN IT, thus the elaboration's gates, the socket test and
the transaction test carry one channel more than `Model.For_test.config`.

**RUNG 2 FUSED, ON THE BOARD.** Built 2026-08-28 from the tree's
`gen_verilog` constants, `board/_build/fused-rung2-explore`:

| | rung 2 unfused (in flash then) | rung 2 FUSED |
|---|---|---|
| block RAM tiles | 124 | **92 of 135** |
| LUTs / registers | 21 455 / 23 818 | 21 449 / 23 766 |
| DSPs | 192 | 192 |
| WNS / WHS | +0.018 / +0.029 | +0.065 / +0.023 |
| `Physopt 32-703` | 1 | 0 |

**92 IS THE NUMBER THE TABLE PREDICTED**, 43 + 11 + 36 + 2, and it costs
nothing anywhere else: the LUTs fall by six and the registers by fifty-two,
because a ring's address is shorter than a tensor's. The first roll met at
WNS +0.000 with seven `Physopt 32-703` on one register family — the
mortgage, refused by the instrument — and the Explore re-roll is the build
above. THE INSTRUMENT EARNED ITS KEEP: nothing but the hold rule separated a
bitstream that STA blessed from one that is actually met.

**AND THE BOARD PLAYS THE TWIN.** Programmed volatile at the panel seed
48877, the flash holding the unfused rung 2. The capture through the S-1's
soft-thru against
`play_diffusion -quantized -seeds 48877 -fade 0 -step-ms 200`: **498 bytes
and 166 messages EACH, byte for byte, in order**, with the order-tolerant
allowance unused. The fused machine is thereby proven against the reference
that exists, and Gate B stands whole at rung 2 fused.

**RUNG 3 FUSED FITS, AND ITS TIMING IS NOT THE PAIR'S.** The golden
candidate `l48-h20` at T 128, G 5, N 512 elaborates at 4 300 464 cycles for
one forward — the cost model's 4.30 M exactly — and builds at **108.5 tiles
of 135 and 240 of 240 DSPs**, which is the table's 109 and the whole point
of the round. The store banks 2 048 + 512 and the ring is 80 columns in one
bank of 512.

**BUT NEITHER ROLL CLEARS THE HOLD INSTRUMENT.** Default directives read
WNS −0.140 with 31 failing endpoints and 20 `Physopt 32-703`; the Explore
re-roll reads +0.003 with 17. Met by three picoseconds with seventeen
clock-skew adjustments is the mortgage, and the mortgage is refused: no
bitstream went to the board.

**AND THE FAMILY NAMES THE OWNER.** Eight of the seventeen adjustments are
`uniform_draw_reg` — the DRAW's shift register, the very family "The
clock-skew mortgage" convicted before and fixed with `dont_touch` replicas
of `u`. The worst path is fifteen levels with NINE carry chains, 46 percent
logic, and it starts and ends outside the array. The fused pair moved
memory and the memory arrived; what stands between rung 3 and the board is
the draw's arithmetic at G 5, and that is a round of its own.

**WHAT THE GATES SAY.** The stream instrument holds every column the engine
writes against the twin, a TURN of writes at a time, over five shapes
including one where the ring wraps twice: 0 part, 0 misplaced at every one.
The cycle bench reads **9 cycles a turn** and it does not grow with the
blocks — the rotation hides the fetch at a phase change as it hides it at a
column change, which is what "the same cycle count" rests on. A waveform of
one pair prints the schedule itself: `lead_column` walks 0, 1, 2, 0, 3 with
`lead_phase` high on that 0.

### The timing round of the fused machine — five families, three cuts

The fused pair fits the golden candidate and rung 3 would not build
honestly: `l48-h20` at T 128, G 5 read WNS −0.140 with 31 failing endpoints
and 20 `Physopt 32-703` on default directives, and +0.003 with 17 on the
Explore re-roll. A third roll was not the answer. **THE CENSUS NAMED FIVE
FAMILIES INSIDE 0.14 ns**, and the lottery band is 0.1:

| family | worst | levels | owner |
|---|---|---|---|
| the draw's threshold: the uniform times the total, one cycle | −0.140 | 15, nine carry | `Draw`, old |
| the fetch cone: `next_block` → the `s - 2` → `column_address` → a bank's address hold | −0.118 | 11–12 | `Forward`, the fused round's own |
| the draw's class into the sheet | −0.107 | 11 | `Source`, old |
| the epilogue's gain multiply, one cycle | −0.090 | 15–17 | `Epilogue`, old |
| the opening: the uniform times the seat width | −0.077 | 11 | `Source`, old |

The unfused rung 3 met at +0.216 on the same 240 DSPs, thus the DSP count
was never the pressure: the walk's new logic and the tighter dwell floor
pushed every long cone at once.

**THREE CUTS, AND EVERY ONE IS A REGISTER.** No arithmetic moved and no
value moved; each cut adds a pipeline stage inside one unit.

1. **The fetch frame is a register.** The fetch of slot `s` goes out at lead
   cycle `3 s + 2` while the nest advances at cycle 0, thus those
   coordinates carried two cycles of slack a combinational `next_block`
   threw away. **Cost: nothing.**
2. **The draw's threshold in two stages**, split by the uniform's halves —
   an exact identity over the integers, thus the theorem that the threshold
   stands STRICTLY under the total is untouched. `busy_cycles` 154 → 155.
3. **The epilogue's multiply in two stages**, split by the gain's bytes: the
   signed high byte times 256 plus the unsigned low. `latency` 4 → 5, and
   every tag pipeline in the lane follows.

**THE CUTS CLEARED THE OTHER TWO FAMILIES ON THEIR OWN.** Cuts 4 and 5 —
the opening's product and the cell port — were designed and not taken: with
the three deeper cones cut, rung 3's census names NOTHING under +0.05.

| | rung 3 fused, before | rung 3 fused, cuts 1-3 |
|---|---|---|
| WNS / WHS | +0.003 / +0.014 | **+0.147 / +0.013** |
| `Physopt 32-703` | 17 | **0** |
| `BUFGCTRL` | 3 | **1** |
| census under +0.05 | 5 families | **0 paths** |
| tiles / DSPs | 108.5 / 240 | 108.5 of 135 / 240 of 240 |

First roll, default directives. **AND BOTH RUNGS PLAY THE TWIN.** Rung 2
rebuilt with the cuts reads +0.035 / +0.015, one tree, no mortgage, 92
tiles, and its capture at panel seed 48877 is 498 bytes and 166 messages
byte for byte. Rung 3, the golden candidate, was then programmed: the draw
stopwatch reads **22.32 seconds** from RUN to the first note against the
cost model's 22.0, and the capture against `play_diffusion` on the golden
checkpoint is **708 bytes and 236 messages, byte for byte, in order**.

Gate B stands whole at the golden candidate.

**THE FLASH HOLDS IT, 2026-08-28.** The tree elects rung 3 in `gen_verilog`
— `l48-h20-100k`, T 128, G 5, N 512 — and its Verilog is the byte of the
one that built `board/_build/cut-rung3` (md5 `4e367cef…`), thus the flashed
bitstream is the tree's own build and no lottery was rolled again. QSPI
erased, programmed, verified, booted; the cell dump answers over the UART
behind the boot. The unfused rung 2 that held the flash since 2026-08-27
stands aside as `board/_build/top-rung2-unfused.bit`.

## The iteration strategy

**THE MACHINE ELABORATES FROM THE CONTRACT FILE, as the twin wrote it.**
The elaboration reads the shape out of the tensor shapes — no flag states a
dimension — and the layer table, the ROM image, the memory
depths and the group count all follow. The weights were never runtime
state in this project; the bitstream initializes them. That standing rule
becomes the iteration loop:

```
elect a checkpoint
  -> infer.py quantize: the contract file
  -> the drift line (infer.py drift, seconds, on the host)
  -> dune build: the netlist; uv run pytest: the Cyclesim gates at a tiny shape
  -> Vivado -> the board -> the capture gate
```

Each rung of the climb is this loop with new constants. Two rules keep it
honest:

- **Trained checkpoints elect music; drawn weights measure builds only.**
  `Model.For_test.drawn` elaborates a shape that has no
  checkpoint, for a timing or utilization reading — never for a drift
  number (the drift of drawn weights reads the format floor, a known trap)
  and never for a
  rung.
- **Every elaboration parameter has a test shape.** P — the 48 pitch rows
  — is a parameter of the CIRCUIT like the rest, pinned to 48 by the board.
  The twin holds P at `Model.rows`, thus Gate B compares at P 48 and the
  tiny shapes come from T, L, H, G and N. **Parameterizing the twin is
  DEFERRED and not refused**: it moves a frozen reference, and nothing moves
  the reference before a circuit works. When one works, the twin takes P as
  well and the test shapes get smaller.

## The two phases

**Phase I — the machine.** The working network on the board: one sheet
from the panel seed on run, and the simplest circuit that hands it to the
sequencer to play. Then measure: the Vivado build, the real generation
cycles, the utilization against the cost model. Phase I holds every
unknown of the round.

**Phase II — the performance.** Many sheets: draw the next while this
one plays, the gap, the fade, the buffer. Phase II reuses semantics the
software side already pinned.

## Phase I: the locked design

- **The climb starts at `l16-h16-100k`** — first make it work, then make
  it good. The engine, the memories and the walk are the machine above;
  `l64-h16-100k` is the same elaboration with a longer table and a larger
  ROM. The golden candidate came through the fused pair, 2026-08-28.
- **One sheet on run.** Reset releases, the machine draws one sheet from
  the SEED cell, the sequencer plays it once, the machine stops. The reset
  button gives the next run. Continuation belongs to phase II.
- **The sheet is the score.** The engine draws in one sheet memory and
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
     waveform tests beside the units, under `dune runtest`;
  2. the sheet agreement — the circuit against the JAX twin, end to end at
     a small shape, and at MORE THAN ONE shape, so that no address region
     field elaborates empty;
  3. the stream gate, WRITE FOR WRITE — every per-layer activation write
     against the twin's, because era five proved that four real datapath
     faults move no frame at a test-sized shape;
  4. the cycle bench — the schedule prints its cycles at the elected
     shape, thus the cost model above and the machine cannot part;
  5. the board rung — the amidi capture of the board's events against
     `infer.py sample --quantized` at the panel seed, the one gate that
     waits for a person and the hardware.

  Instruments 2 and 3 are `jax/tests/test_rtl.py` and instruments 1 and 4
  are `dune runtest`: the two that need an ORACLE moved to the side that
  holds it, and the two that hold the machine against itself stayed.
- **The measurements of phase I**, the numbers the phase must report: the
  build (WNS, LUTs, block RAM, DSPs), the measured cycles of one pass, the
  utilization against the cost model, and the N the board affords inside
  the playback window.

## Phase II, in short

Nothing here is designed; the chapter waits for phase I's numbers. The
items, so the seams stay clean:

- **The buffer.** Gibbs rewrites the sheet in place, thus the playing
  sheet must be its own copy: two sheet memories in ping-pong. Phase I
  keeps the score read port behind its own interface so the doubling stays
  local.
- **The scheduling.** Draw the next sheet while this one plays; the lane
  count and N come from phase I's measurement.
- **The gap and the fade**, as the software states them: `Player`'s
  `velocity_at ~step` is the fade's one point of variation, and velocity
  is a fact of the onset.
- **The seed succession.** The rule that names the seed of sheet k is a
  contract to pin with `infer.py sample --seeds` and the JAX handoff before
  phase II elaborates.
- **The frames as interfaces.** The lead and now frames of `Forward` and of
  `Source` are packed by hand: a `concat_lsb`, two registers on the word,
  and a `field` unpacker that restates each width. An interface record
  with `[@bits]` and `Of_signal.pipeline` is the same registers under their
  own names, and it deletes the packer, the unpacker and the restated
  widths in both files. It waits here and not in the simplify round because
  it changes the netlist — one wide register becomes several narrow ones —
  thus the `top.v` md5 gate cannot cover it, and the timing lottery asks
  for a build. Phase II reopens `Forward`'s frames and owes that build. One
  check at the time: `Of_signal.pipeline` must take the `~enable:run` that
  `Forward`'s `hold` gives its registers.
