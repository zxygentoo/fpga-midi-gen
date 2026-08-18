# The transformer in RTL

## Scope

The transformer of era three on the board: the token model as a note
source behind `Source_intf`, with the checkpoint
`d64-mk-do01-48k-s4-prog` — d 64, 2 layers, 4 heads, context 256, ALiBi
span 8, the piece-position table.

**This document describes the circuit that plays today, and era four
supersedes its model.** The design of the token model is the 2026-08-14
entry of `build-log.md`; `docs/transformer_model.md` now holds era four,
where one step is one frame. When the frame lands, the sections "The
socket" and "The piece boundary" below become wrong, and era four takes
an RTL document of its own.

The design keeps the project rules. The reference of the circuit is
exact integer arithmetic in OCaml — `Quantized` — and the circuit must
match it bit for bit. The float model is not the reference of the
circuit: post-training quantization separates them, and the audition
judges that distance, not a test.

The modules of the era:

| Module | It owns |
|---|---|
| `Quantized` | the quantization of the checkpoint, and the integer model: the exact arithmetic, the mask, the sampler, the seat rule |
| `Source` | the same integers as a circuit: the schedule, the datapath and the socket machine |
| `Mac` | the walk behind the one multiplier: the issue counters, the tags and the accumulator |
| `Divider`, `Isqrt`, `Exp2` | the arithmetic units the walk cannot do: one division, one square root, one table |
| `Sounding_state.Rtl` | the grammar of the instrument in registers, beside the software that states it |

Each unit has an interface file that states its contract, and each has a
block test against an exact oracle. `Source` holds the design of the
machine in its own header; this document holds the design of the whole.

## The socket

`Source_intf.Note` carries `on`: 1 is Note On, 0 is Note Off.

The rules of the sequencer:

- `on` at 1: the seat of `voice` takes the note. If the seat holds a
  note, its Note Off goes first — the pink rule.
- `on` at 0: the seat of `voice` releases its note, one Note Off from
  the stored pair. `pitch` repeats the stored pitch; the seat is the key.

The pink source sends every note with `on` at 1. The transformer source
states its own releases, as the model does.

The seat rule of the transformer source: an On takes the highest free
seat, thus the first On of a sentence — the melody — sits high. An Off
names the seat that holds its pitch. The mask guarantees at most four
sounding pitches and no shared pitch, thus a free seat always exists
and the pitch names one seat.

The sequencer has no gate. It sends a Note Off only to keep its state
true — the steal, when an On arrives at an open seat, and the stop
sweep, when the run ends. The sequencer does not shape the music. A gate
shapes it: it closes a note at a musical time, in the model's place. For
a model that states its releases, a gate is a second writer of the same
state — it would close a note that the source still holds in its
sounding set, and the mask and the seats would split. For a model that
does not state releases — the pink model — the highest voice sustains to
its next articulation, as the lower voices do. This is the honest sound
of that model.

Host-control addresses `0A`–`0B` are reserved: they held GATE_MS.

## The integer model

### The weights

Weights are int8 with a per-tensor exponent that is a power of two:
`w ~ q * 2^-e`, with `e` the largest exponent that keeps
`round(max|w| * 2^e)` at 127 or less. A shift dequantizes; no
multiplier carries a scale. The three tables — token, bar phase, piece
position — share one exponent, because their rows add.

### The formats

The activation formats come from a measurement, not from a guess: the
design round metered the peak of each signal class over a sampled walk,
and the formats hold the measured peak with margin. The reference clamps
where the bound is structural. The meter retired with the round — git
history keeps it — and `checkpoint_tool drift` remains the gate for a
new checkpoint: the top-1 agreement and the cosine of the logits at
every draw.

| Signal | Format | Measured peak (value) |
|---|---|---|
| residual `h` | int32, Q16 | 310 961 (4.7) |
| mean-square sum | 48-bit | 6.9e11 (2^39.3) |
| normed `y` | int16, Q12 | 21 892 (5.3); ±8 by construction |
| `q`, `k`, `v` | int16, Q12 | 18 647 (4.6) |
| scores | int32, Q12 | 269 337 (65.8), ALiBi included |
| context | int16, Q12 | 10 996 (2.7) |
| FFN hidden | int16, Q10 | 4 569 (4.5) |
| logits | int32, Q12 | 106 085 (25.9) |
| sampler weights | uint16, Q15 | 2^15 is the peak, by construction |

The KV ring keeps the top byte of a `k` or `v` row and restores eight
zero low bits at the read — `Quantized.coarse_to_ring`. The format stays
Q12 at a granularity of 2^-4, and the ring costs half the block RAM. The
reference holds the same rule, thus the coarse row is not a loss of the
circuit alone.

### The operations

Each operation is one definition in `Quantized`, and the circuit computes
the same integers.

Every product of the circuit fits one DSP48 — 25 by 18, signed — and
the timing of the machine rests on that rule. The three sites that
wanted more get restructured instead: the rms sum squares a Q12 copy
of the stream, the rms scale divides instead of multiplying by a
reciprocal, and the draw threshold multiplies in two passes.

- **rms_norm**: `s = sum((x >> 4)^2)` — the Q12 copy; `m = s >> 6`
  plus the epsilon `round(1e-6 * 2^24)`; `g = isqrt(m)` (bit-by-bit);
  then `y = clamp16((x << 8) / g)` for each element, toward zero — the
  one division rule of the circuit.
- **matvec**: `acc = sum(y_i * w_i)` over the input length, then one
  arithmetic shift to the target format, then an optional ReLU and an
  optional int16 clamp. The shift count folds the input format, the
  tensor exponent and the output format.
- **attention**: the KV ring holds the newest `min(t, 256)` tokens.
  Age `a` reads slot `(cur - a) & 255`, thus the ALiBi distance is
  `a` itself and no slot stores a time. For each head: pass one walks
  the ages, `score = (dot16(q, k) >> shift) - (a << (12 - E_h))` with
  `E_h` in {2, 4, 6, 8}; pass two subtracts the peak, takes
  `exp2` through the table, and accumulates `num_d += e * v_d` and
  `den += e`; then `ctx_d = num_d / den`, one division for each
  dimension. The current token enters the ring first and attends to
  itself at age 0 — the causal wall is the walk itself.
- **exp2**: `e = exp2(u)` for `u <= 0` in Q12: `I = (-u) >> 12`,
  `F = ((-u) >> 4) & 255`, `e = table[F] >> I`, with
  `table[j] = round(2^15 * 2^-(j/256))`. One 256-entry ROM serves the
  softmax and the sampler.
- **head**: `rms_norm`, then one matvec against the token table read
  backward — the tied head. Logits land in the shared RAM.
- **sampler**: temper by `log2(e) / T` as one Q14 multiply — the
  policy the model carries, committed at elaboration like the weights —
  subtract the legal peak, `exp2` to Q15
  weights, refuse a weight under the model's min-p share of the peak —
  one compare. The draw is `u24` — three PRNG bytes, high first, the
  `Prng.uniform` walk — and the pick is the first code whose running
  total passes `(u24 * total) >> 24`, code order ascending, with the
  fallback of the float sampler. The threshold multiply runs as two
  DSP passes — the high twelve bits of the total, then the low twelve
  — the same integer as one wide multiply.
- **legality**: the sounding set is a 128-bit mask with a count, plus
  `last_on` and `last_off` with their valid bits. The rules of
  `Sounding_state`, combinational for each code in turn.

## The machine

The mathematics is a program, not a state machine. `Source` is five
layers:

| Layer | What it is |
|---|---|
| L0 | the primitives: `Divider`, `Isqrt`, `Exp2`, `Sounding_state.Rtl`, `Prng.Rtl` |
| L1 | the datapath: the RAMs, the KV rings, the banked weight ROM, and `Mac` behind the one multiplier |
| L2 | the schedule: the forward pass and the sampler as lists of operations, built from the config |
| L3 | the compiler: the list folds into cases of a program counter, and the operations dispatch as one parallel case |
| L4 | the outer machine: Idle, Run, Decide, Emit, ForwardDone — the token walk, the seats and the handshake |

One operation holds the facts of one step: the tensor base and exponent,
the address order, the loop bounds, the landing. Each operation knows its
layer at elaboration, thus no register carries a sub-step, a layer index
or a return address, and every per-layer mux folds to a constant. An
operation's finish runs the next operation's entry in the same cycle;
this replaces a hand-kept register reset for each step.

`Mac` walks the terms at one term a cycle. A tag travels beside each term
from its address to its retirement, thus the control needs no knowledge
of the depth of the pipe: the first tag of a row loads the accumulator,
and the last raises the row's done. Rows stream back to back.

The timing rules, decided against measured paths:

- Every memory the walk reads stands two registers from the multiplier.
  The second register packs into the output register of the block RAM,
  which is the stage that pays for the route from a far ROM bank into
  the DSP. At one term a cycle it costs only fill latency.
- The DSP stays a two-register multiply, and the accumulator is a fabric
  adder behind it. The build uses 2 DSPs.

The memories:

| Memory | Size | Content |
|---|---|---|
| weight ROM | 116 736 x 8 | every tensor, the checkpoint flat order |
| exp2 ROM | 256 x 16 | the table above |
| `h` | 64 x 32 | the residual stream |
| `y` | 64 x 16 | the normed vector; the context of attention reuses it |
| `q` | 64 x 16 | the query of the token |
| K ring, V ring | 2 x 32 768 x 8 | two layers, 256 slots, 64 dims, the top byte |
| shared RAM | 256 x 32 | the scores of one head, then the FFN hidden, then the logits, then the sampler weights |

The ROM image is exact and is not padded to a power of two: synthesis
would implement the pad. The image splits into banks of at most 2^15
rows — five banks at this shape — because one RAMB36 is 32 K deep at one
bit wide and the block RAM cascade above 2^15 fails the tools' own check
REQP-1962. A bank is an initialized memory with one gated-off write port,
which the tools infer as block RAM at every depth here; a plain
write-portless array demotes to slice logic.

The cost, measured by the cycle bench against the analytic model:

| Phase | Cycles |
|---|---|
| the rewind walk, one forward pass | 115 547 |
| one token: a draw, its forward pass and the control | about 139 000 |
| a step of four events | 690 131 |
| a piece boundary: the release and the forward of START | 115 554 |

The bench measures the two-layer shape over a short ring. The worst step
is a sentence of nine tokens over a full ring, and the analytic model
gives it as 19.5 ms at two layers and 54.6 ms at six, against a step of
200 ms. A boundary step costs less than that, not more: the boundary
empties the ring, thus the tokens after it attend to a few slots. It is
13.6 ms at two layers and 37.0 ms at six. The budget makes speed
worthless here; the cost model beside the schedule states each operation
exactly, and the bench pins the model to the circuit.

The token walk:

1. rewind: load the PRNG from SEED, clear the ring, the counters and
   the sounding state, feed START at phase 0, bucket 0, and rest.
2. a step strobe: draw tokens. Each draw runs the forward pass over the
   last forwarded token, samples a code, and then: an event token goes
   to the sequencer — wait for `ready` — and feeds back; END feeds back
   and ends the step.
3. the phase of a drawn token is `step mod 16`; the bucket is
   `step / 16 mod 16`; the counters are bit-slices of the step count.
4. a piece boundary: a step whose index is a positive multiple of
   `piece_steps` releases and re-anchors before it draws.

### The piece boundary

The walk plays a sequence of pieces and not one endless stream. An
endless walk decays. The texture leaves the corpus within minutes and
settles at a quarter of the onset rate, with no note shorter than a
quarter note. The cause is the walk and not the circuit: the reference
does the same, and the float model does the same more slowly. The
corpus-like opening is the transient of the boot condition, and nothing
holds the content to the corpus after START leaves the window.

At a boundary the source releases every sounding pitch, climbing, and
returns to the boot context: an empty ring and START. START's row reads
the carried step count, as every row does — at the default arc its phase
and bucket are zero, because the arc and the two table periods divide
it. The release states events and not drawn tokens, because the context
they would enter is cleared in the same walk. The step count and the
PRNG stand.

There are two resets, and the difference between them is the design:

| | ring, counters, sounding | `out_code` | step count | PRNG |
|---|---|---|---|---|
| rewind | clear | START | 0 | reload from SEED |
| piece boundary | clear | START | stands | stands |

A boundary that reloaded the PRNG would play one piece for ever. The
boundary carries it, thus each piece is a new draw and the whole walk
stays one function of SEED.

`piece_steps` is one arc of the piece-position table: 256 steps. The
boundary and bucket 0 then fall on the same step, thus each re-anchor
lands where the table taught the model to close. The corpus puts bucket
15 at the final fermata, and the draw winds down there — a third of the
opening onset rate, and notes 2.6 times longer. The policy is a power of
two, thus the circuit takes the boundary as a bit-slice of its step
counter, and `Quantized.Model.check_shape` holds that rule.

The grammar has one rule for the same reason. `Off 0` has no code,
because code 0 is END, thus a sounding pitch 0 could never be released
and the mask refuses `On 0`. `Jsb.escape_reserved` moves the same two
pitches, thus the corpus never states one either.

## The board

The top level seats the transformer:

- `Top.create` takes the parameters as an argument. `gen_verilog`
  loads and quantizes the checkpoint at elaboration — the bitstream
  carries the weights, per the design rules. The checkpoint path is a
  constant of `gen_verilog`.
- `test/test_txn.ml` draws a parameter set from a seed instead: the
  control path does not read the weights, and the test must not read a
  file that git ignores.
- No new control cells. SEED, STEP_MS, CHANNEL and VELOCITY serve as
  before.
- The flash carries the bitstream: `flash.tcl` writes it to the QSPI
  device, and the board boots the model at power-on.

## The tests

- `Quantized` against the float model: `Drift.walk` runs the float
  model on the walk of the quantized engine and compares each draw —
  the logits, and the pick on the same uniform. The integration test
  pins the measured numbers of a fixed sweep of drawn weights, and a
  QCheck property holds calibrated floors over drawn seed pairs; every
  walk is longer than the window, thus the KV ring wraps against the
  float window. A diff there says the integer scheme moved. On a
  checkpoint, the same walk is `checkpoint_tool drift`: a report, not
  a gate — the audition judges it.
- `Source` against `Quantized`: a short stream gives the same events,
  integer for integer. This is the gate that holds the circuit to the
  reference. A second stream runs a short arc over two piece boundaries,
  thus the release and the re-anchor stand under the same gate. The arc
  of the gates is short because the rule under test is the boundary and
  not its period; a boundary costs a whole forward pass in simulation.
- The units against exact oracles, each beside its own code: `Mac`
  against a summed walk, `Divider` and `Isqrt` against the reference
  arithmetic, `Exp2` against the table rule, and `Sounding_state.Rtl`
  against `Sounding_state.legal_mask` over drawn walks.
- The schedule prints: the state table is data, and an expect test pins
  it. The cycle bench pins the cost model against the measured circuit.
- The sequencer: the release path, `on` at 0.
- The board: the amidi thru capture against the reference's events, as
  the pink era proved its stream.

## What it does not do

- No int4 and no ternary weights; the ladder waits.
- No batch and no KV recompute.
- No new host-control cells and no runtime configuration: one
  checkpoint, one bitstream.
- No care for the loss of the quantization beyond the report: the ear
  judges the model, as it judged every model of the era.
