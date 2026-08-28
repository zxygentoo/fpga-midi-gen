# The OCaml cut — work order

Branch: `feat/diffusion-ocaml-cut`, from `feat/diffusion-rtl` at `0ccc16a`.
Status: DONE, 2026-08-28. This document is the order; the report is at its end.

## The goal

Era six holds four statements of one model: the JAX float model, the OCaml
float model, the OCaml integer twin, and the circuit. The two OCaml models
exist to weld the JAX model to the circuit. Cut them. Put the integer twin in
JAX, beside the float model it quantizes, and gate the circuit against it.

The chain today:

```
JAX float --(A: loss, C: walk text)--> OCaml float --(drift)--> OCaml int8 --(B: exact)--> RTL
```

The chain after the cut:

```
JAX float --(drift, in one framework)--> JAX int8 --(B: exact, pytest drives Cyclesim)--> RTL
```

What the cut buys, beside one layer less: the seed sweep runs on the arithmetic
the board plays. Today the ear elects seeds on the float walk, and the board
plays the int8 walk at 95 percent same-draw. After the cut, the sweep elects on
the int8 walk, batched, and the handoff to the board is exact end to end.

If this works, the eras before six get the same cut in their own rounds. They
are frozen for this round. Do not touch `lib/transformer`, `lib/mamba`,
`lib/nn`, or their tools.

## The rules of this order

- Design first. Read the OCaml twin before you write the JAX twin. Every rule
  of the twin is written down with the line that states it; do not derive a
  rule from the float model.
- Check; do not guess. Each step ends in a gate that is exact. Do not go to the
  next step while a gate fails. Do not delete OCaml code before step 3.
- The circuit does not move. `board/_generated/top.v` regenerated from the
  int8 checkpoint must stay md5-identical to the golden's, `4e367cef`. This is
  the gate of the quantizer, and it costs one second.
- Show the diff for review before each commit. Commit only with permission.
- ASD-STE100 in the documents; the style rules of `CLAUDE.md` in the code;
  `ruff` at 90 columns in Python.

## The accounting

| Unit | Today | After the cut |
|---|---|---|
| `Diffusion.Params`, `logits`, `masked_nll`, `gibbs`, `gate_*`, `Config.of_checkpoint` (float) | the OCaml float model | GONE |
| `Diffusion.rows/voices/cell_order/seat_openings/opening_canvas/hidden_cells/anneal_threshold/tensor_column/frames_of_canvas/over_hidden_cells` | the facts of the walk | STAY — `Elaboration`, `Source` and the benches read them |
| `Quantized.Model.of_params/of_checkpoint/fold_layer/gain_scale` | the quantization | MOVES to `jax/diffusion/quantized.py` |
| `Quantized.Model.t`, `check_shape`, `rom_bits/rom_bases`, `widest_inputs`, `activation_q/bits`, `accumulator_bits` | the int8 model as data | STAY — the reader of the int8 checkpoint |
| `Quantized.Model.For_test.init` | a tiny model from drawn FLOAT weights, quantized | REPLACED by `For_test.drawn`: the int8 record drawn directly in the formats (see step 3) |
| `Quantized.layer_forward/forward/layer_writes/draw_cell/Engine/Drift/Clamps` | the integer twin | MOVES to JAX |
| `Quantized.For_test.plane_activations/plane_column` | the stem's decode | STAY, in `canvas.ml` as the software half of `Canvas` |
| `draw_cell` (6 lines over `Nn_quantized`) | the draw of one cell | STAYS, in `draw.ml` as the software half of `Draw` |
| `bin/play_diffusion`, `bin/check_diffusion` | the tools | GONE; `jax/diffusion/infer.py` plays, and `jax/midi.py` already speaks to the S-1 |
| `test/test_diffusion_drift.ml` | the drift sweep | MOVES to `jax/tests/test_drift.py` |
| `test/test_diffusion_socket.ml` | the sequencer gate | STAYS; it takes its expected frames from `Source`, not from `Engine` (see step 3) |
| `Source`'s walk gate, `Forward`'s stream gate | expect tests with the OCaml twin as oracle | MOVE to pytest: Python is the oracle, an OCaml driver runs Cyclesim |
| `Column_array`, `Epilogue`, `Draw`, `Canvas` expect tests | unit gates over `Nn_quantized` primitives | STAY unchanged |
| `jax/tests/test_parity.py`, era-six Gate A and Gate C | JAX float against OCaml float | GONE at step 3; nothing is left to compare |
| `nx`, `nx.io` in `lib/diffusion/dune` | the float checkpoint | `nx.io` stays for the int8 checkpoint; `nx` leaves if nothing reads it |

## The int8 checkpoint: the contract

One safetensors file, written by JAX, read by `Elaboration` through
`Quantized.Model`. It is the only thing that crosses the seam for a build.

Tensors, named `"0"` upward in layer order, five for each layer, and two
named tensors beside them:

| name | dtype | shape | value |
|---|---|---|---|
| `5i + 0` | int32 | `[3; 3; inputs; outputs]` | the kernel, `q`: int8 values held in int32 |
| `5i + 1` | int32 | `[]` | the kernel exponent, `e` |
| `5i + 2` | int32 | `[outputs]` | the gain `q_value` |
| `5i + 3` | int32 | `[outputs]` | the gain `q` |
| `5i + 4` | int32 | `[outputs]` | the bias, Q6 int16 held in int32 |
| `temper` | int32 | `[2]` | the sampling temper: `q_value`, then `q` |
| `activation_q` | int32 | `[]` | the Q of the activation format: 6 |

**AMENDED AT STEP 1, AND BOTH AMENDMENTS ARE FACTS OF THE READER.**
`Nx_io.load_safetensors` holds F32, F64, I32, F16 and BF16 and SKIPS every
other dtype with a warning on stderr — thus an int8 kernel arrives at the
elaboration as a hole, and every tensor is int32; and it gives no access to
`__metadata__` at all — thus the two numbers the elaboration needs travel as
named tensors. Neither is a taste, and both were measured before the file
format was written.

Metadata (`__metadata__`) is written all the same, for a reader with a Python
tool in hand: `temper_q_value`, `temper_q`, `activation_q` (6), `temperature`.
`quantized.load` reads the temperature back from it, thus a round trip is
exact. `Quantized.Model.check_shape` refuses a file that breaks a rule, as it
refuses a bad record today; `Elaboration.norm_word` refuses a gain `q` outside
six bits, as it does today.

Do not put the population statistics or the float scales in this file. The
fold happens in JAX, one time, and the file carries the result.

## Step 1 — the int8 twin in JAX

New file: `jax/diffusion/quantized.py`. The docstring states the formats as
`lib/diffusion/quantized.mli` states them today.

Every rule below mirrors one function of the OCaml twin. Write each with the
OCaml name in a comment, and keep the order of operations the same.

1. **The exponent rule** — `Nn_quantized.max_exponent`, `quantize`
   (`lib/nn/quantized.ml:184-198`). `e` is the largest integer, from 14 down,
   with `round(max|w| * 2^e) <= 127`; 14 for the all-zero tensor. `q =
   clip(round(w * 2^e), -127, 127)`.
2. **The gain** — `Model.gain_scale` (`quantized.ml`). `gain = scale /
   sqrt(variance + 1e-7)` in float64 from the float32 tensors. `e_g` is the
   largest integer from 30 down with `round(|gain| * 2^e_g) <= 32767`; 30 for a
   gain of 0. `q_value = round(gain * 2^e_g)`, `q = e_g + e` of the kernel.
3. **The bias** — `Model.fold_layer`. `bias = clamp16(round((shift - mean *
   gain) * 64))`.
4. **The temper** — `Nn_quantized.policy` (`lib/nn/quantized.ml:210`).
   `log2e = {q_value: round(2^15 / ln 2), q: 15}`; the temper is
   `{q_value: round((1 / ln 2 / T) * 2^14), q: 14}`.
5. **The planes** — `plane_activations`. Q6 int16: a standing cell writes 64 in
   its class row of plane `voice`; a hidden cell writes 64 in every row of
   plane `voices + voice`. The tensor is `[steps, rows, 2 * voices]`.
6. **The layer** — `layer_forward`. A 3 by 3 convolution over (step, row),
   zero at both edges, int8 weights by int16 activations summed in int32; then
   `(acc * q_value) >> q` (an arithmetic shift, toward minus infinity), plus
   the bias; ReLU where the layer takes one; then the counted clamp to int16.
   The kernel reads as `[dy, dx, input, output]` and the tap `(dy, dx)` reads
   the source at `(step + dy - 1, row + dx - 1)`.
7. **The trunk** — `fold_layer_writes`. The stem with ReLU; for each pair, the
   opening layer with ReLU, the closing layer without, then `joined =
   clamp16(max(0, x + second))` through the SAME counted clamp; the head
   without ReLU. The pair-closing write is the joined tensor. Give back the
   list of every layer's tensor as written, for the stream gate, and the last
   of them as the logits.
8. **The draw** — `draw_cell` and `Nn_quantized.draw`, `exp2_of_magnitude`
   (`lib/nn/quantized.ml:150, 235`). `peak = max(logits)`; for each class
   `d = (logit - peak) << 6` (to Q12), `m = -((d * temper.q_value) >> 14)`,
   `w = 0 if m >> 12 >= 16 else table[(m >> 4) & 255] >> (m >> 12)`. The table
   is `round(32768 * 2^(-j / 256))` for `j` in 0..255. The pick: `total = sum
   w`, `u` the 24-bit word of the generator (`prng.uniform_word`, add it to
   `jax/prng.py` beside `uniform`, first byte highest), `threshold = (u *
   total) >> 24`, and the class is the first `c` whose running total `w_0 + ..
   + w_c` exceeds `threshold`, or the last class.
9. **The walk** — `Engine`. The opening is `infer.opening_canvas`; each pass
   draws the mask as `infer.gibbs` draws it (`floor(anneal * 2^24)` against
   `u * 2^24`, one uniform for each cell in the cell order); one int8 forward;
   then one draw for each HIDDEN cell in the cell order, and no uniform for a
   standing cell. The opening state is `prng.create(seed)` and not
   `create_folded`: the engine takes the seed as the SEED cell does, and seed 0
   stands still. Add `create` beside `create_folded` in `jax/prng.py`.
10. **The clamps** — `Clamps`. Count every activation write, the writes that
    clamped, and the peak before the clamp, over the walk.
11. **The drift** — `Drift.walk`. At every pass, teacher-force the JAX float
    model on the engine's canvas and the engine's mask; per redrawn cell count
    the same peak, the same draw on the very uniform the engine took (the float
    draw is `infer.tempered_pick` with `u = word / 2^24`), and the cosine.
    Report `passes, cells, same_peak, same_draw, mean_cosine,
    activations_clamped, activation_peak` as the OCaml `Drift.stats` does.

The batch. Write the walk over a batch of canvases as `infer.gibbs` does, with
one generator state for each canvas and the `active` rule of `prng.uniform`.
The forward is a `jax.jit` over int32 arrays; the draw is numpy over int64 and
the pick is per row. Do not let a finished element consume a uniform.

Int32 wraps in JAX and an OCaml int does not. The twin's claim is that no sum
reaches 2^31 under `widest_inputs`; keep the accumulator int32 and let a wrap
be the finding it would be on the board.

The writer. `quantized.save(path, model, temperature)` writes the contract
file. `quantized.load(path)` reads it back; a round trip is exact.

The tools. `infer.py sample --quantized` draws through the engine and prints
the same step lines. `infer.py drift --ckpt C --seed 42 --crop 128 --walk 32`
prints the drift report. `infer.py quantize --ckpt C --out C.int8` writes the
contract file.

The OCaml half of this step, small: `Quantized.Model.of_int8_checkpoint path`
reads the contract file into `Model.t` (`Nx_io.load_safetensors`,
`Nx_io.to_typed Nx.int8 / Nx.int32`), and `gen_verilog` gains a flag `-int8
PATH` that takes it instead of the float checkpoint. Nothing else changes yet.

### The gates of step 1

- **G1, the quantizer.** `infer.py quantize` on
  `_train/diffusion/coconet/l48-h20-100k.ckpt`, then `gen_verilog -int8` into
  the scratchpad: `top.v` md5 `4e367cef…`, identical to `board/_generated/top.v`.
  A different md5 says one rounding or one exponent is wrong; diff the norm ROM
  first (the gains and biases), then the weight ROM.
- **G2, the walk.** `play_diffusion -quantized -ckpt C -seeds S -steps T -walk
  N` against `infer.py sample --quantized --ckpt C --seeds S --crop T --walk
  N`: the step lines are equal as text at `(3, 128, 32)` and `(7, 32, 8)`, and
  at seed 0 with `(0, 32, 8)`, where both stand still.
- **G3, the drift.** `check_diffusion drift -ckpt C -seed 42 -steps 128 -walk
  32` against `infer.py drift`: `cells`, `same_peak`, `same_draw` and the clamp
  counts equal as integers; `mean_cosine` equal to four decimals;
  `activation_peak` equal.

Write G2 and G3 as pytest gates in `jax/tests/test_parity.py` beside the era's
Gate A and C. They are temporary: they weld the two twins while both exist,
and step 3 removes them with the OCaml twin.

## Step 2 — the RTL gates move to pytest

The oracle of the walk gate and of the stream gate is the JAX twin. Cyclesim
stays in OCaml. A driver executable runs the bench and prints what the circuit
did; Python states what it must have done and compares.

New executable: `bin/gate_diffusion.ml`, `Core.Command` group with two
subcommands. Both take `-int8 PATH` (the contract file of a tiny model),
`-steps`, `-lanes`, `-walk`, and `-rows` where the elaboration takes it.

- `walk -seed S`: `Elaboration.create`, `Source`'s bench harness, `rewind`,
  then print every write of the walk in order, one line each:
  `write MASK|CLASS step seat value`; then `play` the whole canvas and two
  steps past it and print `frame step value` lines. The Python side computes
  the wanted writes as `Source.Bench.wanted` does today — the opening, then for
  each pass its masks in the cell order and its draws in the cell order — from
  its own engine at the same seed, and compares in order; and the frames
  against `Diffusion.frames_of_canvas` of the engine's canvas, which the driver
  prints too (`want_frame step value`) so that Python holds the sequencer's
  format to nothing of its own.
- `stream -seed S`: `Forward`'s bench, one pass over the stem input the driver
  builds with `plane_activations` from an opening and a mask it draws at `-seed`
  exactly as the stream gate does today (`opening_canvas`, `anneal_threshold
  ~step:0`, `hidden_cells`); print `canvas step seat class` and `hidden step
  seat 0|1` so Python can build the same input; then every store write as
  `write L step channel v0 v1 … v(rows-1)`, with the layer index assigned by
  turn as the gate assigns it today (`turn_columns`), the address decoded to
  `(step, channel)` by `Elaboration.column_address`'s inverse; then the head's
  offered logit columns as `logits step seat v0 … v(rows-1)`; and the counts
  `misplaced N`. Python computes `layer_writes` and compares every column.

Exports: the bench modules of `Source` and `Forward` are not in their `.mli`.
Export them as `Source.For_test.Bench` and `Forward.For_test.Bench`, as the
rule of `CLAUDE.md` states for an export only tests read. The driver is a test.

The tiny models. Python draws them: `numpy.random.default_rng(seed)`, normal
weights at the scale `Diffusion.Params.init ~norm_scale:1.0` uses today (read
it; the norm at scale 1.0 keeps a drawn trunk at O(1) activations — the reason
is in `test/test_diffusion_drift.ml`'s header), quantizes them with the twin,
writes the contract file to a `tmp_path`, and runs the driver on it. The shapes
are today's: for the walk gate `H 8, G 2, two pairs, T 6, N 3` at seeds 1, 2, 0
(seed 0 is the standing walk and must print the opening's full draw count) and
`H 7, G 3, one pair, T 5, N 4`; for the stream gate every `case` of the two
stream expect tests in `forward.ml`, shape for shape.

New file: `jax/tests/test_rtl.py`. It skips when `_build/default/bin/
gate_diffusion.exe` is absent, as `test_parity.py` skips, and fails when the
two sides disagree, naming the first write apart with its phase.

The expect tests that these replace stay in `source.ml` and `forward.ml` until
step 3. The cycle-count expect tests (`where a pass spends its cycles`, the
turn counts, the waveforms) do not read the twin and stay for good.

### The gate of step 2

`uv run pytest tests/test_rtl.py` passes on every shape, and every existing
`dune runtest` still passes; the OCaml twin has not moved.

## Step 3 — the cut

Now delete, in this order, building and testing after each item:

1. `test/test_diffusion_drift.ml` goes; `jax/tests/test_drift.py` takes its
   place: the same drawn shapes (`layers 6, width 8`, four weight seeds, four
   walk seeds, T 32), the fixed numbers pinned as numbers and not thresholds,
   the long walk at 8/32/128 passes, and the floors 0.80/0.70/0.985 over seed
   pairs. The numbers will not equal the OCaml ones — the drawn weights come
   from a different generator — and that is not a fault: pin the new ones.
2. `bin/check_diffusion.ml` and `bin/play_diffusion.ml` go, with their `dune`
   stanzas; the era-six Gate A, Gate C, G2 and G3 of `test_parity.py` go.
3. `Source`'s walk gate and `Forward`'s stream gate expect tests go; the
   `Bench.passes/wanted/apart` and the `stem_input` helper go with them where
   nothing else reads them. What the driver reads stays.
4. `test/test_diffusion_socket.ml` takes its expected frames from the source in
   the seat: the frames `Source`'s bench answers at each `play`, and the
   sequencer's bytes must be `Frame.events_of_frames` over those. The gate's
   stated job — "this holds the sequencer to the frames" — does not change.
   Its model is `Quantized.Model.For_test.drawn`, as every surviving OCaml
   test's is. No file enters the repository.
5. `Quantized`: `Engine`, `Drift`, `Clamps`, `layer_forward`, `forward`,
   `fold_layer_writes`, `layer_writes`, `draw_cell`, `of_params`,
   `of_checkpoint`, `fold_layer`, `gain_scale` go;
   `plane_activations/plane_column` move to `canvas.ml`; `draw_cell` moves to
   `draw.ml`'s bench. What stays is the record, `of_int8_checkpoint`,
   `check_shape`, `rom_bits/bases`, the formats. Rename nothing yet; the
   `.mli` doc says what the module now is: the int8 checkpoint as data.

   `For_test.init` becomes `For_test.drawn config ~seed`: a `Model.t` of the
   shape `config` states, its values drawn under `Prng` straight into the
   formats — kernel `q` in -127..127 at one fixed `e`, gain `q_value` in
   int16 and `q` inside the six-bit field of the norm word, bias in int16, the
   temper of T 1.0 — and it passes `check_shape`. Thirteen surviving expect
   tests build their model with it (`elaboration.ml` 7, `source.ml` 3,
   `forward.ml` 3, and the socket test): they need a model of a SHAPE — cycle
   counts, tile counts, the turn order, the images' self-consistency, a
   picture — and none of them needs the twin's arithmetic, because the two
   that did move to pytest in step 2. It reads no file, as `init` read none.

   Expect blocks whose printed VALUES depend on the model — the classes in
   `Source`'s waveform, the frames a play answers — change one time, because
   the drawn integers are not the old quantized floats: re-pin them. Cycle
   counts, turn counts, tile counts and image sizes must not change; if one
   does, it is a bug and not a re-measurement.
6. `Diffusion`: `Params`, `logits`, `masked_nll`, `gibbs`, `gate_canvases`,
   `gate_mask`, `column`, `over_hidden_cells`, `Config.of_checkpoint` and
   `tensor` go. `Config` stays only if `Elaboration` still reads it; check.
   The `nx` library leaves `lib/diffusion/dune` if nothing reads it.
7. `gen_verilog` takes `-int8` alone; the float checkpoint path constant goes;
   `board/nexys-4/dune` loses `nx` if it had it. `docs/diffusion_rtl.md`'s
   "The references, as built" states the new chain, and `CLAUDE.md`'s "Tests"
   section states that the RTL model gates run under `uv run pytest` in `jax/`
   and that `dune runtest` holds the unit gates.
8. The int8 file of the golden candidate goes beside the float checkpoint
   under `_train/` (git ignores both); `gen_verilog` names it.

### The gates of step 3

- `dune fmt` makes no change; `dune build` has no warning; `dune runtest`
  passes.
- `uv run pytest` passes: `test_rtl.py`, `test_drift.py`, the rest.
- `gen_verilog -int8 _train/…/l48-h20-100k.int8` gives `top.v` at md5
  `4e367cef…`. The circuit did not move; the flash stands.
- On the board: `infer.py sample --quantized --seeds 47872 --walk 512 --fade 0
  --save` against the S6 capture of the flash at panel seed 47872 — the
  message stream that `play_diffusion -quantized -fade 0` gave, byte for byte
  in order (236 messages at the golden). This is the end-to-end proof and it
  needs the board; run it last.

## Traps

- **Rounding.** Base's `Float.iround_nearest_exn` is `floor(x + 0.5)`: a tie
  goes toward plus infinity, `-2.5 → -2`, `2.5 → 3`. Python's `round` and
  `numpy.rint` are half-to-even. Write `np.floor(x + 0.5)` and nothing else.
- **`ldexp` and `2^-e`.** Multiply by `np.ldexp(1.0, e)`; it is exact.
- **Float64 from float32.** Convert the checkpoint tensors to float64 first,
  then compute; the OCaml side reads float32 into doubles. Keep the operation
  order of `fold_layer`: `scale / sqrt(variance + eps)`, then
  `shift - mean * gain`.
- **The shift.** `(v * q_value) >> q` in numpy on int64 is an arithmetic shift;
  in Python ints too. Do not divide.
- **The exp2 table.** Generate it in Python as the OCaml rule does and compare
  it entry for entry against the OCaml table one time (print it from an OCaml
  expect test or from `Nn_quantized.Constants.exp2_bits`); a libm difference
  would show in the last bit of a few entries.
- **The uniform word.** Three bytes, the first highest, and the engine draws a
  word and not a float; the float of the drift is `word / 2^24` after the
  draw. The mask compare uses `u * 2^24 < threshold` on the float, which is
  exact on the grid; the word compare `word < threshold` is the same test.
- **Seed 0.** `Prng.create 0` stands still and every uniform is 0; the opening
  is every seat at its lowest class, every cell hidden at every pass, and every
  draw takes class… whatever a threshold of 0 gives under the pick rule. The
  OCaml twin states it; match it, do not reason it.
- **A standing cell takes no uniform.** The redraw leg draws only where the mask
  hid; the opening and the mask draw for every cell.
- **The pair join.** `max(0, held + second)` then the counted clamp; `held` is
  the pair's input and not the opening layer's output.
- **The head.** No ReLU, and its writes are the logits in Q6.
- **`q` of the gain.** `e_g + e`; the norm word holds it in six bits; a
  quantizer that gets `e` wrong by one moves every gain and G1 says so.

## The report

At the end, state under this heading: each gate and its result; the md5; the
line counts removed and added on each side; what the JAX twin costs in seconds
for the golden candidate at T 128, N 512, one canvas and sixteen; and any rule
you found stated nowhere that you had to read out of the code.

---

# The report

Done 2026-08-28. Every gate of the order passed except the last, which needs
the board and the person.

## The gates

| gate | result |
|---|---|
| **G1**, the quantizer | **PASS, first try.** `infer.py quantize` on the golden checkpoint, then `gen_verilog -int8`: `top.v` md5 `4e367cef6e38b2ae1f06ab3cf42a9c42`, identical to `board/_generated/top.v`. Every exponent, gain, bias and weight byte of the JAX quantizer equals the OCaml one. |
| **G2**, the walk | **PASS** at `(3, 128, 32)`, `(7, 32, 8)` and seed 0 at `(0, 32, 8)`: the step lines are equal as text, character for character. |
| **G3**, the drift | **PASS.** The two reports agree line for line: 32 passes over 128 steps redrew **6326** cells, top-1 **97.2%** (6148/6326), cosine **0.9998**, same draw **95.1%** (6013/6326), clamps **0.0000%**, hottest write **180.8** of the format's 512.0. |
| step 2 | **PASS.** `pytest tests/test_rtl.py`, 11 cases — six walk shapes and five stream shapes. `dune runtest` unchanged. |
| step 3 | **PASS.** `dune fmt` makes no change; `dune build` has no warning; `dune runtest` passes; `uv run pytest` passes (136 tests); `gen_verilog` on `_train/diffusion/coconet/l48-h20-100k.int8` gives `top.v` at md5 `4e367cef6e38b2ae1f06ab3cf42a9c42`. **The circuit did not move; the flash stands.** |
| the board | **NOT RUN.** It needs the hardware and a person. |

The gates bite: a bias row of the twin moved by 3 parts 2 writes of the walk
gate and 180 columns of the stream gate at the smallest shape.

## The lines

| side | added | removed | net |
|---|---|---|---|
| OCaml (`lib/`, `bin/`, `board/`, `test/`) | 853 | 2283 | **−1430** |
| JAX (`jax/`) | 1433 | 119 | **+1314** |
| documents | 449 | 39 | +410 |

Two whole tools and one integration test left OCaml; one driver arrived. The
OCaml side of era six is now the circuit, the facts of the walk, the int8
checkpoint as data, and the two software halves that stand beside the units
that must equal them (`canvas.ml`'s decode, `draw.ml`'s draw).

## What the JAX twin costs

The golden candidate, T 128, N 512, on the CPU:

| canvases | seconds |
|---|---|
| 1 | **57.4** |
| 16 | **1000.3** |

**A BATCH BUYS NOTHING ON THE CPU AT THIS SHAPE**, and the number says why:
16 canvases cost 17.4 times one, which is worse than linear. The convolution
batches — it is one `jax.jit` over int32 — but the draw is a Python loop over
`steps * voices` cells for each pass, and that loop runs 262 144 times whatever
the batch holds. The card is where a sweep belongs (`JAX_PLATFORMS=cuda`), and
a batched draw is the next thing to write if the ear wants a wide sweep.

For scale: the OCaml twin took **39 s** for one canvas at `(3, 128, 32)` where
the JAX twin takes **4.4 s**, and the drift report took **55 s** where JAX
takes **8.6 s**. The suite got faster too — `uv run pytest` fell from 267 s to
106 s, because the two era-six gates that drove OCaml binaries went with the
OCaml twin.

## The rules that were stated nowhere

1. **`Nx_io.load_safetensors` reads five dtypes and SKIPS the rest.** F32,
   F64, I32, F16 and BF16 load; every other dtype is dropped with a warning on
   stderr and the tensor simply is not in the archive. An int8 kernel would
   therefore reach the elaboration as a hole and refuse for the wrong reason.
   **Every tensor of the contract file is int32**, and the kernel holds int8
   values inside it, as the bias holds an int16 one.
2. **`Nx_io` gives no access to `__metadata__` at all.** The loader hands back
   the tensors alone. The two numbers the elaboration needs — the temper and
   the Q the file was quantized at — therefore travel as the named tensors
   `"temper"` and `"activation_q"` beside the numbered layers. The metadata is
   written all the same, for a reader with a Python tool in hand, and
   `quantized.load` reads the temperature back from it.
   *(Both amendments are in the contract table above, with their reasons.)*
3. **`Model.layer.kernel.e` is carried and never read below the seam.** The
   weight ROM carries `q` alone and the norm word carries the gain's own
   shift. That is why `For_test.drawn` may give every layer one fixed
   exponent, and why nothing of the circuit notices.
4. **The hidden-cell count of a pass is the machine's own counter and not the
   model's.** `Source`'s cycle bench read it from `Quantized.Engine`; the
   service takes one uniform for each cell it redraws and nothing else takes
   one there, thus `served Uniform / uniform_ticks` IS the count — which the
   sibling rung-1 measurement already did. The engine was never needed there.
5. **The socket test's reference is `Source` played on its own.** A second
   instance of the same block at the same seed and elaboration answers the
   same frames, and what the gate then measures is everything BETWEEN those
   frames and the wire. The frames past T − 1 need not be in the list: the
   message stream saturates, which the test's own header already argued.
6. **The elaboration's expect tests are SELF-CONSISTENCY checks and not value
   pins.** Every one of them holds the circuit's own unpacker against the
   twin's record, or counts, and none prints a weight. That is why replacing
   the drawn model moved no elaboration number, no cycle count, no turn count
   and no image size — the only two blocks that moved were the socket test's
   message counts and one waveform md5 in `forward.ml`. It was designed that
   way; it is what let `For_test.drawn` land without a re-measurement round.
7. **A drawn int8 model needs its gain COMPUTED and not drawn.** A kernel byte
   of spread `s` over the `9 C` taps of a dwell carries an activation of
   magnitude A into an accumulator of about `sqrt(9 C) * s * A`, thus the gain
   must be `1 / (sqrt(9 C) * s)` and the shift is what puts that multiplier in
   the middle of int16. Measured through the circuit itself at H 8, L 8, T 6:
   **peak 126, which is 2.0 of the format's 512.0, 39.9 percent of the writes
   zero from the ReLU, nothing on the clamp.** A gain drawn flat inside int16
   would clamp every write of the trunk or zero it.
8. **The exp2 tables agree entry for entry.** The trap said a libm difference
   would show in the last bit of a few entries; all 256 were compared against
   `Nn_quantized.Constants.exp2_bits` and **none differed**.

## What is left

- **The board gate.** `infer.py sample --quantized --seeds 47872 --walk 512
  --fade 0 --save` against the S6 capture of the flash at panel seed 47872 —
  236 messages, byte for byte in order. The bitstream in the flash is
  unchanged (the netlist md5 proves it), thus this is a gate of the JAX walk
  against the board and not of a new build.
- **The batched draw**, if a wide seed sweep is wanted: the pick is per row
  and the Python loop over cells dominates a batch.
- **The eras before six**, which the goal names: they hold the same two
  layers and would take the same cut in their own rounds.
