# The transformer prototype in RTL

## Scope

This is the prototype of era three on the board: the model of
`docs/transformer_model.md` as a note source behind `Source_intf`, with
the checkpoint `d64-mk-do01-48k-s4-prog` — d 64, 2 layers, 4 heads,
context 256, ALiBi span 8, the piece-position table. The goal is a feel
of the model on the hardware and a proven block structure. The
production RTL document comes after, with the lessons of this build.

The design keeps the project rules. The reference of the circuit is
exact integer arithmetic in OCaml — the twin — and the circuit must
match it bit for bit. The float model is not the reference of the
circuit: post-training quantization separates them, and the audition
judges that distance, not a test.

Two new modules carry the era, in the pattern of `Pink` and `Voss`:

| Module | It owns |
|---|---|
| `Fixed` | the quantization of the checkpoint, and the integer model: the exact arithmetic, the mask, the sampler, the seat rule |
| `Vaswani` | the same integers as a circuit: the ROMs, the engine, the KV ring, the draw and the socket FSM |

## The socket

`Source_intf.Note` gains one field:

- `kind` — 1 bit: 1 is Note On, 0 is Note Off.

The rules of the sequencer:

- `kind` On: the seat of `voice` takes the note. If the seat holds a
  note, its Note Off goes first — the pink rule, unchanged.
- `kind` Off: the seat of `voice` releases its note, one Note Off from
  the stored pair. `pitch` repeats the stored pitch; the seat is the key.

`Voss` sends every note with `kind` On and does not change behavior.
The transformer source states its own releases, as the model does.

The seat rule of the transformer source: an On takes the highest free
seat, thus the first On of a sentence — the melody — sits high. An Off
names the seat that holds its pitch. The mask guarantees at most four
sounding pitches and no shared pitch, thus a free seat always exists
and the pitch names one seat.

The gate is removed from the sequencer. The rule that replaces it: the
sequencer sends a Note Off only to keep its state true — the steal,
when an On arrives at an open seat, and the stop sweep, when the run
ends. The sequencer does not shape the music. The gate shaped the
music: it closed a note at a musical time, in the model's place. For a
model that states its releases, the gate is a second writer of the same
state — it would close a note that the source still holds in its
sounding set, and the mask and the seats would split. For a model that
does not state releases — the pink model — the highest voice now
sustains to its next articulation, as the lower voices do. This is the
honest sound of that model.

With the gate go all of its parts: the `GateOff` state, the sampled
gate time, and the GATE_MS cell of the host control. Addresses
`0A`–`0B` become reserved.

## The integer model

### The weights

Weights are int8 with a per-tensor exponent that is a power of two:
`w ~ q * 2^-e`, with `e` the largest exponent that keeps
`round(max|w| * 2^e)` at 127 or less. A shift dequantizes; no
multiplier carries a scale. The three tables — token, bar phase, piece
position — share one exponent, because their rows add.

### The formats

The activation formats come from a measurement, not from a guess:
`checkpoint_tool ranges` runs the float model over a sampled walk and
prints the peak of each signal class. The formats hold the measured
peak with margin, and the twin clamps where the bound is structural.

The measurement — `ranges` on the king checkpoint, 96 steps, seed 42 —
confirmed every format, and the drift against the float model is small:
top-1 agreement 91.7 percent, cosine 0.9998 over 276 draws.

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

### The operations

Each operation is one definition in `Fixed`, and the circuit computes
the same integers.

Every product of the circuit fits one DSP48 — 25 by 18, signed — and
the timing of the engine rests on that rule. The three sites that
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
  Age `a` reads slot `(p - 1 - a) & 255`, thus the ALiBi distance is
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
  backward — the tied head. Logits land in the score RAM.
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

## The engine

One multiplier path, one divider, one isqrt, one big FSM. The MAC runs
at two cycles for each term; the budget makes speed worthless here:
about 350 K cycles for each token, 3.5 ms at 100 MHz, and the worst
sentence of nine tokens is under 32 ms against a step of 250 ms.

The memories:

| Memory | Size | Content |
|---|---|---|
| weight ROM | 128 K x 8 | every tensor, the checkpoint flat order |
| exp2 ROM | 256 x 16 | the table above |
| `h` | 64 x 32 | the residual stream |
| `y` | 64 x 16 | the normed vector; the merged context reuses it |
| `q` | 64 x 16 | the query of the token |
| K ring, V ring | 2 x 32 K x 16 | two layers, 256 slots, 64 dims |
| scores | 256 x 32 | one head at a time; the logits reuse it |
| A | 256 x 32 | the FFN hidden |
| B | 256 x 16 | the sampler weights |

About 66 of the 135 block RAMs, with the weight ROM the largest part.

The token walk of the source FSM:

1. rewind: load the PRNG from SEED, clear the ring, the counters and
   the sounding state, feed START at phase 0, bucket 0, and rest.
2. a step strobe: draw tokens. Each draw runs the engine over the last
   fed token, samples a code, and then: an event token goes to the
   sequencer — wait for `ready` — and feeds back into the engine; END
   feeds back and ends the step.
3. the phase of a drawn token is `step mod 16`; the bucket is
   `step / 16 mod 16`; the counters are bit-slices of the step count.

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
  before. GATE_MS is removed with the gate, and addresses `0A`–`0B`
  are reserved.
- The flash keeps the pink era; the prototype programs over JTAG.

## The tests

- `Fixed` against the float model: the plain-float twin structure
  agrees with the Nx reference on logits, then the quantized model's
  drift is a report, not a gate — the audition judges it.
- `Vaswani` against `Fixed`: one fed token gives the same logits, and
  a short stream gives the same events, integer for integer.
- The sequencer: the `kind` Off path. The tests that watched the gate
  change with it: the melody Note Off moves from the gate time to the
  next articulation of its voice.
- The board: the amidi thru capture against the twin's events, as the
  pink era proved its stream.

## What the prototype does not do

- No int4 and no ternary weights; the ladder waits for the real RTL.
- No pipelining beyond the two-cycle MAC; no batch; no KV recompute.
- No new host-control cells and no runtime configuration: one
  checkpoint, one bitstream.
- No care for the loss of the quantization beyond the report: the ear
  judges the prototype, as it judged every model of the era.
