# The transformer in RTL

## Scope

The step-frame model of era four on the board: one step of music is one
pass of the network and one 32-bit frame on the socket. The model is
`docs/transformer.md` and the checkpoint is
`d64-frame-do03-96k-s6-l6-nopos-span4` — d 64, 6 layers, 4 heads, context
256, ALiBi span 4, no window-position table.

This document supersedes the circuit of era three. That circuit played on
the board, and git holds it at `dd2264e` with the document that described
it; the source of era three is never ported, because the source of era
four takes its seat. `Pink` holds the model seat until it does.

The design keeps the project rules. The reference of the circuit is exact
integer arithmetic in OCaml — `Quantized` — and the circuit must match it
bit for bit. The float model is not the reference of the circuit:
post-training quantization separates them, and the drift report measures
that distance.

The modules of the era:

| Module | It owns |
|---|---|
| `Quantized` | the quantization of the checkpoint, and the integer model: the exact arithmetic, the chain and the sampler |
| `Source` | the same integers as a circuit: the schedule, the datapath and the socket machine |
| `Mac` | the walk behind the one multiplier: the issue counters, the tags and the accumulator |
| `Divider`, `Isqrt`, `Exp2` | the arithmetic units the walk cannot do: one division, one square root, one table |

`Sounding_state.Rtl` is not in that table, and the reason is the whole
design: no frame is illegal, thus the grammar of the instrument needs no
registers.

![The transformer source: the five layers of the machine, the memories, and
one step drawn as the program that the counter runs](transformer_rtl.svg)

The two halves of the picture are the two halves of the design. The machine
above is small, and it stays small because the mathematics is a program. The
band below is L2, and no box in it is a block: the list is a value that the
config builds at elaboration, and L3 folds it into the cases of the counter.

The schedule is not free for being a list. It reaches the die as the decode of
that counter and as the parallel case on every register that a case writes,
which is the cone the drawing marks, and "The machine" below measures what that
cone costs when the fold goes wrong.

## What the frame changes

Era three carried one note at a time and drew a sentence of tokens for
each step. Four things follow from carrying a frame instead, and each one
takes something out of the circuit.

**The step is one pass, always.** The worst case of era three was nine
tokens and nine passes; the worst case here is one pass. The outer
machine loses its token loop with it.

**The head is a chain of four readouts.** The stream after the last layer
reads four tables of 48 rows in place of one table of 256, and each read
is smaller than the one it replaces.

**No mask stands before a draw.** `Sounding_state.Rtl`, the 128-bit
sounding vector, the last-ON and last-OFF registers and the seat count
all leave the circuit. So do the seat rule and the seat registers: a
frame states which voice holds which pitch, thus nothing has to choose a
seat.

**No piece boundary.** The walk of era four takes no re-anchor, thus the
release scan, the second reset and the bit-slice of the step counter
leave with it. There is one reset now, and it is the rewind.

## The socket

`Source_intf` carries the frame:

```ocaml
module I = struct
  type 'a t = { clock : 'a; clear : 'a; rewind : 'a; step : 'a }
end

module O = struct
  type 'a t =
    { frame : 'a [@bits 8 * voices]  (** the voice codes; seat 0 is the low byte *)
    ; valid : 'a             (** a strobe: [frame] holds the frame of the step *)
    ; idle : 'a              (** 1 when the source is at rest and can take a command *)
    }
end
```

`Note` leaves the interface and `ready` leaves with it. The sequencer
strobes `step` and waits, thus it is always ready and no handshake is
necessary. `valid` answers `step` one time for each step, and every step
gives a frame — a step where nothing sounds gives four zero bytes and not
silence on the socket.

The decode moves into the sequencer, where `Core.Frame` states it: the
sequencer holds the set of pitches that sound, and it sends all releases
and then all strikes. The two passes are the rule, and
`docs/transformer.md` states why a seat walk breaks on the
exchange and the unison.

**The source answers `step` from a frame it has already drawn.** It draws
the frame of the next step while the sequencer sends the messages of this
one, thus `valid` comes at once and the wire is the only thing the tempo
waits for. If a step ever arrives before the draw finishes, `valid` comes
late and the sequencer waits: the socket is latency-insensitive, and the
rule is a rule of correctness and not of timing.

`Pink.Source` moves onto the same socket, which is the last step of the
plan and the only one that touches music the ear elected.

## The integer model

### The weights

Weights are int8 with a per-tensor exponent that is a power of two:
`w ~ q * 2^-e`, with `e` the largest exponent that keeps
`round(max|w| * 2^e)` at 127 or less. A shift dequantizes; no multiplier
carries a scale.

The four seat tables stand in **one** tensor of `4 x 48 x d`, and the bar
phase shares its exponent, because their rows add. Seat `s` begins at
`base + s * 48 * d`, thus the circuit reaches a seat with a shift and an
add, and the ROM base table holds one entry for all four. Four exponents
would track each voice's own range; the drift report is the instrument
that would ask for them, and it does not ask.

### The formats

The formats of era three carry over, and they were metered again and not
assumed. The frame sums five rows into the stream where the token summed
three, and the chain adds up to three more at the head, thus the stream
runs hotter and the question is real.

Metered over a 320-step walk of the elected checkpoint, 1,216 draws:

| Signal | Format | Era three peak | Era four |
|---|---|---|---|
| residual `h` | int32, Q16 | 310 961 (4.7) | **685 783 (10.5)** |
| normed `y`, `q`, `k`, `v`, context, FFN hidden | int16 | see era three | peak 30 466 of 32 767 |
| sampler weights | uint16, Q15 | 2^15 by construction | unchanged |

**No clamp of int16 fired at all.** The stream runs 2.2 times hotter and
it is int32, thus it stands far from its own bound; the int16 signals
reach 93 percent of their range and none of them crosses it. The peak of
the int16 class is the normed vector, which is bounded by construction.

The KV ring keeps the top byte of a `k` or `v` row and restores eight
zero low bits at the read — `Quantized.coarse_to_ring`. The format stays
Q12 at a granularity of 2^-4, and the ring costs half the block RAM. The
reference holds the same rule, thus the coarse row is not a loss of the
circuit alone.

### The operations

Each operation is one definition in `Quantized`, and the circuit computes
the same integers. Every product fits one DSP48 — 25 by 18, signed — and
the timing of the machine rests on that rule.

`rms_norm`, `matvec`, `attention` and `exp2` are the operations of era
three and they do not change; that document's statement of each one still
holds, and `Quantized` is the definition in any case. Three change:

- **embed**: five rows add — the seat row of each of the four seats, at
  `seats + (s * 48 + c_s) * d`, and the bar-phase row at `step mod 16` —
  then one shift to Q16. The rows share one exponent, thus the sum
  happens before the shift and one Embed operation walks all five.
- **head**: the chain, from the soprano down. For seat 3 to seat 0:
  `rms_norm` of the stream, one matvec against that seat's 48 rows, the
  sampler, and then — for every seat but the last — the drawn row adds
  onto the stream. **The chain accumulates in the `h` RAM itself**: the
  stream is dead after the chain, because the forward pass of the drawn
  frame starts from the embedding and not from it. The chain therefore
  needs no memory of its own.
- **sampler**: no mask and no legal peak — the peak is the peak. Temper
  by `log2(e) / T` as one Q14 multiply, subtract the peak, `exp2` to Q15
  weights, and refuse a weight under the min-p share of the peak, which
  is one compare. The draw is `u24`, three PRNG bytes high first, and the
  pick is the first class whose running total passes
  `(u24 * total) >> 24`, class order ascending. **The pick needs no
  fallback**: `u24` is below 2^24, thus the threshold is below the total,
  thus a running total always passes it and the class it names always
  holds weight. The threshold multiply runs as two DSP passes, the high
  twelve bits of the total and then the low twelve.

## The machine

The mathematics is a program, not a state machine. `Source` keeps the
five layers of era three:

| Layer | What it is |
|---|---|
| L0 | the primitives: `Divider`, `Isqrt`, `Exp2`, `Prng.Rtl` |
| L1 | the datapath: the RAMs, the KV rings, the banked weight ROM, and `Mac` behind the one multiplier |
| L2 | the schedule: the step as a list of operations, built from the config |
| L3 | the compiler: the list folds into cases of a program counter |
| L4 | the outer machine: the step strobe, the lead-in and the socket |

**L2 and L4 are where the frame pays.** The schedule of era three held a
forward pass and a sampler, and the outer machine ran them again and
again until a sentence ended. A step is one pass and four draws now, thus
the outer machine keeps no return address, no token count and no sentence
state. What is left of L4 is: take `rewind`, run the lead-in, hold a drawn
frame, answer `step`, and start the next draw.

**The chain is one seat, and the machine runs it four times.** The four
seats were inlined at first, which made the step one straight-line program
and gave the machine no counter at all. That shape was built and measured:
it cost **4 406 slice LUTs against era three's 2 999** at the same six
layers and the same 93 percent of the block RAM, and the six-layer build
**missed the period by 0.4 ns** where era three met it by 0.110. The cost
is not the operations — control is cheap — but the muxes they share: every
case of the program counter that writes a register widens that register's
parallel case, and inlining put four writers where era three had one. The
seat register is the price of the room, and it returns three quarters of
the chain's control for two bits of state.

One operation still holds the facts of one step of the program: the
tensor base and exponent, the address order, the loop bounds, the
landing. Each operation knows its layer at elaboration; the **seat** is the
one thing a register carries, and it reaches exactly two places, which the
schedule states rather than implies — the address of a tensor is `Fixed`
or `Seat_block`. An operation's finish runs the next operation's entry in
the same cycle, thus the loop back to the head of the chain costs no cycle
of its own.

The last seat accumulates a row that nothing reads. The reference does not
add it; the circuit does not test for it, because the test would cost a
case of the program and the row costs `d` cycles of a step that has
hundreds of thousands. The stream is dead there in any case.

`Mac` walks the terms at one term a cycle. A tag travels beside each term
from its address to its retirement, thus the control needs no knowledge
of the depth of the pipe: the first tag of a row loads the accumulator,
and the last raises the row's done. Rows stream back to back.

The timing rules of era three hold, and they were decided against
measured paths: every read is two cycles from address to data, and the
DSP stays a two-register multiply with a fabric adder behind it. The ROM
spends its two cycles as one address register before the bank tree and
one data register in each bank; the address register is load-bearing.
With a combinational address, the tools retime the data register onto
the address pins of every block RAM primitive and rebuild the whole
op-dispatch address cone inside each one — 27 LUTs a primitive, 12
primitives a layer, which was the entire layer scaling of the source
(3 466 -> 2 352 LUTs at six layers, measured out of context).

### The lead-in and the rewind

The walk opens with one bar of silent frames. The circuit plays them as
steps: the frame is four zero bytes, the chain does not run, and **the
PRNG does not move** — exactly as it does not move in the reference. The
forward pass does run, because the silent context is what the model
attends to when it opens the music.

A lead-in pass costs less than a full step: the ring is nearly empty, thus
attention reads a handful of slots and not 256, and the pass is about 3 ms
at six layers. Each of the sixteen has a whole step period to run in, thus
the lead-in never makes the source late.

There is one reset. `rewind` loads the PRNG from SEED, clears the ring
and the step counter, and returns to the head of the lead-in; `idle`
answers it. The second reset of era three — the piece boundary that
carried the PRNG — has nothing to reset.

## The memories

| Memory | Size | Content |
|---|---|---|
| weight ROM | the checkpoint, flat order | 308 224 x 8 at six layers |
| exp2 ROM | 256 x 16 | the table above |
| `h` | 64 x 32 | the residual stream, and the stream of the chain |
| `y` | 64 x 16 | the normed vector; the context of attention reuses it |
| `q` | 64 x 16 | the query of the step |
| K ring, V ring | 6 x 256 x 64 x 8 each | six layers, 256 slots, 64 dims, the top byte |
| shared RAM | 256 x 32 | the scores of one head, then the FFN hidden, then the 48 logits, then the 48 sampler weights |

The shared RAM keeps its depth: the scores of one head over a full ring
are 256 entries, and the readout of a seat is 48. The tables of the model
are smaller than era three's — four seat tables of 48 rows are 12,288
bytes against a token table of 16,384 — thus the frame buys block RAM and
does not spend it.

The ROM image is exact and is not padded to a power of two: synthesis
would implement the pad. The image splits into banks of at most 2^15
rows, because one RAMB36 is 32 K deep at one bit wide and the block RAM
cascade above 2^15 fails the tools' own check REQP-1962. A bank is an
initialized memory with one gated-off write port, which the tools infer
as block RAM at every depth here; a plain write-portless array demotes to
slice logic.

The budget, from `docs/transformer.md`: 126 tiles of 135 at six
layers, which is one tile under the design that played in era three. The
six-layer build measures exactly that.

## What the six-layer build measures

| | era three, six layers | era four |
|---|---|---|
| slice LUTs | 2 999 | 4 069 |
| slice registers | 1 743 | 1 645 |
| block RAM tiles | 127 | 126 |
| DSPs | 2 | 2 |
| worst negative slack | +0.110 ns | **+0.031 ns** |

**The margin is thin and it needs the post-route pass.** `build.tcl` runs
`phys_opt_design` after `route_design`, and the design does not close
without it: the route measures −0.252 ns and that pass recovers it. At 93
percent of the block RAM, routing is three quarters of every long path,
thus placement wins slack that the route gives back. A change to `Source`
or to `Sequencer` that adds a little logic can put the period out of
reach, and the failure is not loud — the bitstream builds and the board
plays at the wrong baud rate.

Registers **fell** against era three while the LUTs rose by a third. Era
four really does hold less state — the grammar of the instrument, the
seats and four states of the outer machine all left — and pays it back in
combinational area. About a thousand LUTs of that rise are **not** yet
accounted for. The chain loop returned 337 of them, which was measured;
the rest wants each block synthesized on its own, and no reader should
trust a number here that a measurement did not make.

## The cost

The step is constant, thus the cost model is a number and not a worst
case. At d 64, context 256 and one term a cycle:

| Part | Multiplies |
|---|---|
| projections `q`, `k`, `v` | 3 d² = 12 288 for each layer |
| attention, two passes | 2 T d = 32 768 for each layer |
| the `wo` join | d² = 4 096 for each layer |
| the feed-forward | 8 d² = 32 768 for each layer |
| **one layer** | **81 920** |
| the chain: four readouts | 4 x 48 d = 12 288 |

Six layers and the chain give **503 808 multiplies for each step**, thus
about 5.0 ms at 100 MHz before the control, the divisions and the fill
latency. Two layers give 176 128, which is the number the model document
states.

The model reproduces the measured bench of era three where the two meet:
its rewind walk — one forward pass of two layers over a short ring —
measured 115 547 cycles, and this model gives about 98 000 multiplies for
that pass, with the difference in the divisions and the control.

Era three measured its worst step at **61 ms** of a 200 ms period at six
layers. The frame takes about **7 ms** with the control, thus compute
stops being the constraint on the tempo and the wire becomes it: the
worst step sends four Note Offs and four Note Ons, which is 24 bytes and
7.68 ms at 31250 baud. **The floor of `step_ms` is 8, and it is a fact of
the wire.**

A cycle bench pins the model to the circuit, as it did in era three, and
that measurement is owed when the circuit exists.

## The board

- `Top.create` takes the model as an argument. `gen_verilog` loads and
  quantizes the checkpoint at elaboration — the bitstream carries the
  weights, per the design rules — and the checkpoint path is a constant
  of `gen_verilog`.
- `test/test_txn.ml` seats a model that reads no file: the control path
  does not read the weights, and a test must not read a file that git
  ignores.
- No new control cells. SEED, STEP_MS, CHANNEL and VELOCITY serve as
  before, and host-control addresses `0A`–`0B` stay reserved.
- The flash carries the bitstream: `flash.tcl` writes it to the QSPI
  device, and the board boots the model at power-on.

## The tests

- `Quantized` against the float model: `Drift.walk` runs the float model
  on the walk of the quantized engine and compares each draw of the
  chain — the logits, and the pick on the same uniform. The integration
  test pins the measured numbers of a fixed sweep of drawn weights, and a
  QCheck property holds calibrated floors over drawn seed pairs; every
  walk is longer than the window, thus the KV ring wraps. On a
  checkpoint, the same walk is `check_transformer drift`.
- **`Source` against `Quantized`: the gate of this document.** A short
  walk must give the same frames, integer for integer, and the frames
  then give the same events. The walk must cross the lead-in, because the
  first drawn step is the one that reads a context of silence.
- The units against exact oracles, each beside its own code: `Mac`
  against a summed walk, `Divider` and `Isqrt` against the reference
  arithmetic, and `Exp2` against the table rule.
- The schedule prints: the state table is data, and an expect test pins
  it. The cycle bench pins the cost model against the measured circuit.
- The sequencer against `Core.Frame`: the eight cases of the decode, and
  the three properties over a stream of frames — no strike of a pitch
  that sounds, no release of a pitch that does not, and four notes at the
  most.
- The board: the amidi thru capture against the reference's events, as
  the pink era and era three both proved their streams.

## What it does not do

- No int4 and no ternary weights; the ladder waits.
- No batch and no KV recompute.
- No new host-control cells and no runtime configuration: one checkpoint,
  one bitstream.
- No care for the loss of the quantization beyond the report: the ear
  judges the model, as it judged every model of the era.
