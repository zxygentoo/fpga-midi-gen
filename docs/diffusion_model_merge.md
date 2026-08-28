# The model merge — work order

Branch: `feat/diffusion-ocaml-cut`, on top of the cut's commit.
Status: DONE, 2026-08-28. This document is the order; the report is at its end.

## The goal

The cut left two modules whose names lie. `Diffusion` holds no diffusion: it
is the roll, the registers of the seats, the anneal rule and the walk-order
helpers. `Quantized` quantizes nothing: it is the reader of the contract file
and the formats. Together they are one thing — **the model as the circuit
reads it** — and that thing gets one module with one name: `Model`.

This round moves no arithmetic and no number. The netlist md5 must not change.

## What does not move, and why

Two things looked like candidates for the contract file and are not:

- **The anneal table.** `anneal_threshold ~step ~walk` depends on N, and N is
  a parameter of the ELABORATION and not of the model: `gen_verilog` states
  it, and the tests run N 3, 4, 8 and 512 on one drawn model. A table in the
  file would tie a checkpoint to one geometry. The rule stays in OCaml, and the
  walk gate of `test_rtl.py` holds it: a threshold apart from JAX's moves every
  mask of a pass, write for write.
- **The seat registers.** `seat_openings` is `Jsb.voice_ranges` read through
  the class map — the corpus library is the authority and its own test pins
  the ranges. `jax/measure.py`'s `RANGES` is the hardcoded duplicate. A file
  written by JAX would make the copy the master. They stay, and the walk gate
  holds them too: the opening's classes are compared cell for cell.

## The module

New: `lib/diffusion/model.ml` and `model.mli`. Deleted: `diffusion.ml`,
`diffusion.mli`, `quantized.ml`, `quantized.mli`. The `.mli` carries the full
API document the style rules ask for; the `.ml` keeps the comments that state
a WHY and drops the ones that restated the old module names.

The order of the sections, top to bottom, with what each takes:

1. **The roll.** `rows` (`Vocab.classes`), `voices` (`Frame.voices`).
2. **The formats.** `activation_q`, `activation_bits`, `accumulator_bits`,
   `widest_inputs`, with their comments as they stand. `widest_inputs` leaves
   the `Model` submodule and stands beside the widths: it is a format bound.
3. **The model.** `type quantized`, `type layer`, `type t` (flattened — no
   `Model.Model`), `check_shape`, `of_int8_checkpoint`, `rom_bits`,
   `rom_bases`. Every consumer writes `Model.t`, `Model.check_shape`,
   `Model.of_int8_checkpoint`.
4. **The walk.** `type opening`, `seat_openings`, `anneal_threshold`,
   `cell_order`, `opening_canvas`, `hidden_cells`. `grid`, `under` and
   `over_cells` stay private.
5. **The frames.** `frames_of_canvas`.
6. **`For_test`.** `drawn ~layers ~width ~seed` and `rom_tensors`.

Three deletions inside the merge:

- **`Config` goes.** It named a shape for `For_test.drawn` and nothing else;
  `drawn` takes `~layers ~width` and its twenty-two call sites write the two
  numbers. `For_test.config` (the 4-by-6 test shape) goes with it: each of its
  three readers states `~layers:4 ~width:6` and says so.
- **`class_of_cell`'s refusal goes.** Its reader was the corpus loader of Gate
  A, which the cut deleted; what is left maps the two endpoints of each
  register, which are inside the corpus by construction. Keep the class map
  as one expression — `pitch - Vocab.pitch_low + 1` — where `seat_openings`
  builds its `low`, and let the frames test state its two classes the same
  way. `pitch_high` goes with the refusal.
- **`tensor_column` moves to `canvas.ml`**, private, beside `plane_column`, if
  that is its only reader. Check with `grep` first; if a second reader stands,
  it stays in `Model` under the roll.

## The renames

Every reference on both sides of the seam. On the OCaml side:
`lib/diffusion/*.ml` and `*.mli`, `bin/gate_diffusion.ml`,
`board/nexys-4/gen_verilog.ml`, `board/nexys-4/gen_probe.ml`,
`test/test_txn.ml`, `test/test_diffusion_socket.ml`. The other eras' files
that `grep` also finds (`test_mamba_drift`, `play_mamba`, …) name THEIR own
`Quantized` and are not touched.

| was | becomes |
|---|---|
| `Diffusion.rows`, `.voices` | `Model.rows`, `.voices` |
| `Quantized.activation_q`, `.activation_bits`, `.accumulator_bits` | `Model.…` |
| `Quantized.Model.widest_inputs` | `Model.widest_inputs` |
| `Quantized.Model.t`, `.layer`, `.quantized` | `Model.t`, `.layer`, `.quantized` |
| `Quantized.Model.check_shape`, `.of_int8_checkpoint`, `.rom_bits`, `.rom_bases` | `Model.…` |
| `Quantized.Model.For_test.drawn { Diffusion.Config.layers = L; width = H } ~seed` | `Model.For_test.drawn ~layers:L ~width:H ~seed` |
| `Quantized.Model.For_test.rom_tensors` | `Model.For_test.rom_tensors` |
| `Diffusion.opening`, `.seat_openings`, `.low`, `.width` | `Model.…` |
| `Diffusion.anneal_threshold`, `.cell_order`, `.opening_canvas`, `.hidden_cells` | `Model.…` |
| `Diffusion.frames_of_canvas` | `Model.frames_of_canvas` |
| `Diffusion.tensor_column` | `Canvas`'s own, or `Model.tensor_column` (see above) |

The documents: `docs/diffusion_rtl.md` names `lib/diffusion/diffusion.ml`,
`quantized.ml`, `Quantized.Model.of_int8_checkpoint` and `Diffusion.*` in its
CURRENT-STATE sections ("The references, as built", the consumption order, the
iteration strategy, Phase II). Update those. Do not touch the history sections
— a record names the code as it stood when the record was made.

The JAX side: `jax/diffusion/quantized.py`'s docstring and `infer.py quantize`
name `Quantized.Model.of_int8_checkpoint`; they say `Model.of_int8_checkpoint`.
Nothing else on that side changes.

`lib/diffusion/dune`'s comment names the module that shadowed `Mgen_nn`'s
`Quantized`; after the merge nothing does, and the comment says why `Mgen_nn`
is still not opened (its `Quantized` would then be reachable unqualified, and
the file's own `Nn_quantized` alias is the one name the units read it by).

## The gates

- `dune fmt` makes no change; `dune build` has no warning; `dune runtest`
  passes WITH NO EXPECT BLOCK MOVING. A rename moves no number, no cycle count,
  no picture and no message count; if one moves, stop — something other than a
  name changed.
- `uv run pytest` passes: `test_rtl.py` through the rebuilt driver, and G1.
- `gen_verilog` gives `top.v` at md5 `4e367cef6e38b2ae1f06ab3cf42a9c42`.
- `git diff --stat` reads as a move: the lines added to `model.ml` are the
  lines deleted from the two files, less the comments that named them, less
  `Config`, `class_of_cell`'s refusal and `pitch_high`.

## The report

At the end, state under this heading: each gate and its result, the md5, the
line count of `model.ml` against the two it replaced, whether `tensor_column`
moved, and any reader of the two old modules that `grep` found and this order
did not name.

---

# The report

Done 2026-08-28. Every gate passed. The round moved names and nothing else.

## The gates

| gate | result |
|---|---|
| `dune fmt` | **no change** |
| `dune build` | **no error, no warning** |
| `dune runtest` | **passes, AND NO EXPECT BLOCK MOVED** — not one number, cycle count, tile count, picture or message count. A rename moves nothing, and nothing moved. |
| `uv run pytest` | **148 passed**, including `test_rtl.py` through the rebuilt driver and G1 through `gen_verilog`. |
| the netlist | `top.v` md5 **`4e367cef6e38b2ae1f06ab3cf42a9c42`**, the golden's. The circuit did not move; the flash stands. |

## The lines

| file | lines |
|---|---|
| `model.ml` | 472 |
| `model.mli` | 211 |
| **together** | **683** |
| `diffusion.ml` + `.mli` + `quantized.ml` + `.mli` | **744** |

**−61**, and every line of it is the three deletions plus the comments that
named the two old modules to each other. No rule left and none arrived.

## What moved, and what did not

- **`tensor_column` MOVED.** `grep` found exactly one reader — `canvas.ml`,
  where `plane_column` is `tensor_column` at the plane count — thus it went
  there, private, and `Model` states one index rule fewer. `canvas.mli` no
  longer points outside itself for it.
- **`Config` is gone**, and `For_test.config` with it. `drawn ~layers ~width
  ~seed` has 24 call sites and each writes the two numbers. Two of them read
  better for it: `elaboration.ml`'s `tiny_shape` record became the partial
  application `tiny_model = Model.For_test.drawn ~layers:4 ~width:8`, which is
  what its four readers wanted, and the refusal gate that used to reach for
  the twin's 4-by-6 now states `~layers:4 ~width:6` beside the comment that
  says why H 6 is the shape this elaboration REFUSES.
- **`class_of_cell`'s refusal is gone**, with `pitch_high`. The class map is
  one expression where `seat_openings` builds its `low`, and the frames test
  states its two classes with a local of the same expression. (The
  `class_of_cell` that `grep` still finds in `canvas.ml` is a different thing:
  it slices the class bits out of a packed cell in the RTL.)
- **The anneal table and the seat registers did not move**, for the reasons
  the order states. Both are held by the walk gate, and it passes.

## The reader the order did not name

One, and it is a dangling name rather than a compile error:

- **`docs/diffusion_rtl.md:943`** — "The column engine, measured" says the
  weights are drawn "under `Quantized.Model.For_test.drawn`". That name no
  longer exists. The line sits in a MEASUREMENT RECORD and not in a
  current-state section, thus this order's rule says leave it, and it is left.
  It is worth noting that the OCaml cut already renamed this same line once
  (it read `Params.init` before), because a record that names nothing helps no
  reader. **Say the word and it becomes `Model.For_test.drawn`.**

Everything else `grep` found was in the files the order named, or in the
history sections of `docs/diffusion_rtl.md` (the fused-pair round's
`Quantized.layer_writes`), which name the code as it stood and stay.

On the JAX side only the two LIVE cross-references changed, as the order says:
`quantized.py`'s file-layout paragraph and `infer.py quantize`'s help both now
name `Model.of_int8_checkpoint`. The other `Quantized.Model.*` mentions in
`quantized.py` are the "mirrors one function of the OCaml twin this module
replaced" series — `gain_scale`, `fold_layer`, `of_params`, `check_shape`,
`layer` — and they cite the twin AS IT WAS. They are history, and renaming
half of a citation series would be worse than renaming none.
