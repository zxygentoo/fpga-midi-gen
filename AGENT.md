# RULES

1. ALWAYS show the diff for review before you make a commit. Make a commit
   automatically ONLY if the user gives permission.
2. Write ALL technical documents in ASD-STE100 English.
3. Design first, implement later.
4. Check; do not guess.
5. Measure before optimizing.

# Style

- Prefer the functional style: pure functions and clear data abstractions.
- Names:
  - Give each function a name that states its work. A vague name (`clean`,
    `split`, `best`) makes the reader deduce the work from the context. A
    clear name (`escape_zero_pitch`, `chorales`, `vote`) carries the work
    alone.
  - Name the helper of a `map`, a `filter` or a `fold` instead of a dense
    inline literal. The name documents the step. When no good name exists, a
    generic `aux` still reads better than a long closure. A short literal can
    stay inline.
  - Decompose a dense function into named stages; the top function then reads
    as the algorithm (`cadential_holds`, `vote`, `metre`).
- Comments:
  - An interface file has a full API document comment: what the data and each
    field are, what each function takes and gives, what the reader must know
    to use the interface correctly, and the important design choices.
  - Do not overdo the other comments: a comment states only what the code
    cannot say.
  - The *what* is easy to see in the code; the *why* is not. Comment the why:
    the design and the reasoning behind the code.
  - Some *what* comments are necessary — a tie rule, or a part that looks
    unusual. For example, some software here is unconventional because it
    must agree with the circuit.
  - Keep inline comments sparse and terse.
- Datatypes:
  - Do not make a tuple of more than three items. Use a record, and give each
    field a good name.
- Prefer the pipeline (`|>`) where a value passes through steps in sequence:
  the steps stand in the order they happen, and no name holds a value that
  only waits for the next step.
- Mutation is permitted only with a real justification:
  - global state at the outer edge of the program
  - local mutation with a large, measured performance win
  - an idiomatic use of a mutable data structure
  - code that is much more clear in the imperative style
- Software and RTL are in the same `lib/` directory, thus code sharing between
  OCaml and Hardcaml is natural. If a software module and a circuit want the
  same name, put them in one file: the top level holds the OCaml code (the
  hardware can share some of it), and an `Rtl` module holds the Hardcaml
  definitions.
- A module in a file should have a purpose: a group of functions over a data
  abstraction, or a grouping like a namespace that makes the consumer code
  much more clear. Do not use a module as an escape hatch to prevent a simple
  collision of names, or only to make the code compile. Each of these is a
  common signal that the code needs a proper refactor.
- A library module has a documented `.mli` file. A top-level module — an
  executable in `bin/`, the board top level — can omit it.
- Module exports and refactors:
  - If an OCaml export is used only by tests outside the module, export it in
    a `For_test` module. A Hardcaml circuit usually does not have this
    problem; if the circumstances call for it, ask the user for direction.
  - If nothing outside the module uses it, do not export it.
  - If nothing inside the module uses it either (this sometimes happens after
    a refactor), then it is dead code: remove it.
  - Be vigilant for stale tests. An export can stay alive only because a test
    uses it, and that test can itself need a refactor or a removal.
- Use Janestreet's Base/Core instead of Stdlib.
- Base gives `<`, `>`, `=` and the other compares for integers only, thus a
  comparison of floats must name the type. Write the local open,
  `Float.(a > b)`, and not the applied operator, `Float.( > ) a b`. A local
  open also makes `+` and `*` work on floats, thus integer arithmetic stays
  outside the parentheses.
- Format all code with ocamlformat, profile `janestreet`.
- Write each Hardcaml block in the standard idiom: the interface modules `I`
  and `O` with `[@@deriving hardcaml]`. Give the fields clear names, and
  avoid `[@rtlname]`: if a field seems to need it, correct the name instead.

# Basic

This project makes MIDI on an FPGA. An external synthesizer plays it.

## Toolchain

OCaml for the circuit, the elaboration, the drivers and the corpus tools.
Hardcaml for the RTL. Python (JAX) for the models: the trainers, the float
models, the integer twins and the oracle gates, all in `jax/`.

- opam switch: `5.2.0+ox` (OxCaml)
- Hardcaml version: `v0.18~preview`
- Vivado: hardware synthesis
- dune
- `uv` runs everything in `jax/`: `uv run pytest`, `uv run ruff check`,
  `uv run python -m diffusion.infer`. Never bare `python` or `pip`.
- `jax`, `jaxlib` and `jax-cuda12-plugin` move together or not at all. A
  plugin one release behind the runtime is refused, and the trainer falls
  back to the CPU with no message, ten times slower.
- Era six is Flax NNX and optax. The frozen eras keep `nn.adamw`,
  `nn.schedule` and `nn.train`.
- `ruff` at line-length 90, the width of ocamlformat. E501 is not selected,
  thus ruff does not check the width of a docstring.

## Hardware

- FPGA board: Digilent Nexys 4, XC7A100T-1CSG324. This is **not** the Nexys 4
  DDR.
- Synthesizer: Roland S-1.

## Connections

- Pmod JD header pin 1 -> 33 Ω -> Jack L (tip)
- Pmod JD header pin 5 -> Jack G (sleeve)
- Pmod JD header pin 6 -> 33 Ω -> Jack R (ring)
- Jack -> Cable -> S-1 MIDI IN
- Host -> Nexys 4 USB UART (`/dev/ttyUSB1`)
- Host -> S-1 USB

The Jack is a 3.5 mm TRS adapter with screw terminals. The Cable is a 3.5 mm
male-male audio cable.

Header pin 1 is the only FPGA pin here. Header pins 5 and 6 are the ground and
the 3.3 V supply of the Pmod connector, and the FPGA cannot control them. The
ring stays at 3.3 V, and it needs no RTL. This is MIDI TRS Type A.

The MIDI input of the synth is an isolated current loop with a TLP2368
photocoupler. Therefore the two USB connections cannot make a ground loop. The
loop is about 286 Ω and gives about 5.4 mA, against a worst-case threshold of
5 mA. Do not add resistance in series, because the margin is only about 24 Ω.
The 33 Ω value is correct for a 3.3 V driver, and the older 220 Ω value is for
5 V.

## Safety

**WARNING: Remove the power from the board before you connect or disconnect the
Cable.** A TRS plug makes a short circuit between tip, ring and sleeve as it
moves into the Jack. If the FPGA output is low at that moment, the current is
about 50 mA. This is more than the limit of an Artix-7 pin.

**WARNING: Examine Pmod header pins 5 and 6 before the first power-on.** Pin 5
is the ground and pin 6 is the 3.3 V supply. If the two wires are not in the
correct positions, you make a short circuit on the supply. The Nexys 4 has no
fuse for each Pmod connector.

**WARNING: Do not make a short circuit between the tip terminal and the VCC
terminal.** The current is about 100 mA.

# Board reference

The MIDI data pin is header pin 1. In the XDC this pin is `JD[0]`, and the
package pin is H4. The header pin number and the XDC index are not the same.

| Signal | Package pin |
|---|---|
| `clk` | E3 (100 MHz) |
| `btnCpuReset` | C12 |
| `RsRx` (host to FPGA) | C4 |
| `RsTx` (FPGA to host) | D4 |
| `JD[0]` to `JD[7]` | H4, H1, G1, G3, H2, G4, G2, F3 |

These properties are necessary:

- `DRIVE 8` on `JD[0]`. The pin must sink 5.4 mA with a low output voltage,
  because that voltage subtracts from the loop current.
- `DRIVE 4` on the other JD pins. They have no connection. A low drive limits
  the current if the wires are not in the correct positions.
- `PULLTYPE NONE` and `SLEW SLOW` on `JD[*]`.

MIDI is 31250 baud, 8N1. The divisor from 100 MHz is exactly 3200, and the
clock error of this hardware is −279 ppm. The console UART to the host is
115200 baud, 8N1. The maximum MIDI byte rate is 3125 bytes each second.

# The synthesizer

- The default receive channel is **3**, not 1. Roland uses different channels
  so that you can connect more than one AIRA Compact unit.
- If there is no sound but the board shows MIDI output activity, examine the
  channel first.
- The synth receives Program Change on channel **16**.
- Clock, Start and Stop have no channel. Therefore the synth can play its own
  sequencer from your clock and ignore all your notes. Sound does not show that
  the channel is correct. To test the channel, send a note.
- To change the channel: SHIFT + PAD 15 (MENU), dial to `CH`, PAD 2 (ENTER),
  dial to the value, PAD 1 (EXIT).
- CC 74 is the filter cutoff. The synth does not keep the value after a power
  cycle, and the CUTOFF knob has priority over MIDI. Use CC 102 for integrity
  tests, because it has no audible effect.
- A control change is only audible on the next note.
- The S-1 has four voices.

## USB audio

The S-1 is a class-compliant USB audio device. Therefore you can record its
output with no loss of quality. The format is S32_LE, 44100 Hz, 2 channels.
It is not 48000 Hz.

- The firmware must be version 1.02 or later. Push STEP during power-on to see
  the version.
- AIRA LINK must be OFF: SHIFT + pad 15 -> `A.Lnk` -> `OFF`, then power-cycle.
- Do not use a USB hub.

# Layout

- `lib/` — the OCaml libraries, software and RTL together:
  - `lib/core` — the host control constants, MIDI, the frame, the PRNG, the
    Cyclesim harness.
  - `lib/board` — the UART, COBS, the control port and transport, the
    sequencer, the socket, the seed switches.
  - `lib/corpus` — the chorales (`Jsb`) and the vocabulary (`Vocab`).
  - `lib/nn` — what is one thing across the eras: the units, the fixed-point
    rules the circuits read, the placement rules, the bounds of the sampling
    policy.
  - one directory for each era: `pink`, `transformer`, `mamba`, `diffusion`.
- `bin/` — executables: the board driver (`board_tool`), the pink player,
  the corpus tool, and one RTL-gate driver for each era with a circuit
  (`gate_transformer`, `gate_mamba`, `gate_diffusion`).
- `board/` — the top level, the configuration and the scripts of each board,
  for example `board/nexys-4`. `board/_generated/` holds the Verilog and
  `board/_build/` the Vivado work; git ignores both.
- `jax/` — the Python side: `data.py`, `nn.py`, `prng.py`, `midi.py` and
  `measure.py` are common, each era has a directory, and `tests/` holds the
  oracle gates. Git ignores `jax/_data/`; `corpus_tool` rebuilds it.
- `corpus/` — the chorale corpus.
- `_train/` — the training runs: the logs and the checkpoints. Git ignores
  it. Every run pipes to `_train/NAME.log` beside its checkpoint.
- `docs/` — the design documents: `<era>.md` for the model and `<era>_rtl.md`
  for the circuit. A work order is process: write it in `docs/`, never commit
  it, and delete it when its round is done.
- `test/` — the integration tests: the socket simulations.

# Design

```
JAX (train, quantize) -> contract file -> Elaboration -> bitstream
Model (RTL/Hardcaml) -- host control -- Drivers (OCaml)
```

- Model: the FPGA does the inference. The host trains the model in JAX,
  `infer.py quantize` writes the int8 contract file, `Elaboration` reads it,
  and the bitstream carries the weights. The weights are not runtime state.
- Six eras so far: pink noise (era one), a Markov chain (era two — a failed
  experiment, on `feat/markov-model` only), a transformer (eras three and
  four), a Mamba hybrid (era five), and the masked sheet of Coconet (era six,
  in the flash). The records are `docs/<era>.md` and `docs/<era>_rtl.md`.
- Host control: one interface for all drivers — the control registers and a
  read/write wire protocol on the UART. The specification is
  `docs/host_control.md`.
- Drivers: self-check, control and other functions.

Rules:

- `lib/core/control_intf.ml` defines all constants of the host control one
  time. The RTL elaboration and the drivers must use the constants from that
  module, and `lib/board/control_frame.ml` is the wire codec that carries
  them. If `docs/host_control.md` and `lib/core/control_intf.ml` do not
  agree, correct one of them before you continue.
- The host control has no runtime version. The driver and the bitstream must
  come from the same repository state. If the board behavior does not agree
  with the specification, program the board again with the current bitstream.

# Tests

Run all tests with `dune runtest`, and then `uv run pytest` in `jax/`.

- Unit tests are expect tests (`ppx_expect`), in the module that they test.
- A waveform expect test is visual documentation. If a waveform can show the
  behavior of a module clearly, write one.
- `test/` holds the integration tests: the socket simulations with Cyclesim.

- The FLOAT model is an audition tool: train, run it on the host, send the
  MIDI to the S-1 through USB, listen. The FPGA is not in this loop, and the
  float model does not have to equal the circuit.
- The INTEGER twin must equal the circuit bit for bit. That is Gate B: the
  unit gates and the cycle benches under `dune runtest`, the walk gate and
  the stream gate against the twin under `uv run pytest`, and the capture of
  the board's MIDI at the panel seed against the twin's.
- The circuit is tested with Cyclesim, block by block: the PRNG, the
  sampler, the timer. Exact test vectors are easy at the block level.
- **A gate that needs a MODEL AS AN ORACLE runs under `uv run pytest` in
  `jax/`, and `dune runtest` holds the unit gates.** An era whose integer
  twin lives in JAX (era six is the first) keeps Cyclesim in OCaml and puts
  the oracle in Python: a driver executable runs the bench and prints what
  the circuit did, and the test states what it must have done. Neither side
  can then pass a gate by agreeing with itself. The gates that hold the
  machine against ITSELF — the cycle counts, the images, the waveforms —
  stay beside the units, because they need no model.
- Randomness is pseudo-randomness, and the seed is an input. The same seed
  gives the same sequence in the simulation and on the board.
- Diagnostics are on the board: the LEDs and the display. The host control
  has no status or counter cells.

# Traps

- Vivado pads an inferred memory to a power of two and says nothing. A
  memory the budget cannot hold is demoted to LUTs, silently, whatever its
  attribute says. Bank the memories by powers of two, and read the block RAM
  tile census and the LUT count, not the warnings.
- STA met by a picosecond is not met. A build that meets setup only because
  `phys_opt` adjusted the clock skew (`Physopt 32-703`) on a data path, or
  that holds by under about 10 ps, plays a different piece on each run.
  Refuse it and place again.

# Gitflow

- This repository uses git-flow, with the branches `main` and `develop`, and
  the prefix `feat/` for features.
- The pre-commit gates are: `dune fmt` makes no change, `dune build`
  completes with no error and no warning, and — when `jax/` moves —
  `uv run ruff check` finds nothing and `uv run pytest` passes.
- A change to the RTL is netlist-identical (the `top.v` md5 gate of
  `jax/tests/test_parity.py`) or it owes a Vivado build before it merges.
