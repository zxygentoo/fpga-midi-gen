# fpga-midi-gen

Make a few models (transformer, mamba, diffusion) learn to generate Bach chorales, then make circuit versions of them to play on an FPGA.

- For the software side, we use [JAX](https://docs.jax.dev/en/latest/)
- For the RTL implementation, we use [Hardcaml](https://hardcaml.org/)

There are three implementations of each model:
1. A JAX float one: your normal machine learning stuff
2. A JAX quantized (int8) one: weights that are easy to integrate into a circuit
3. An RTL one: the FPGA implementation of the inference engine

- 1 and 2 are close in behavior, measured by drift gates
- 2 and 3 are identical in behavior
- identical PRNG (xorshift32) implementation for both JAX and Hardcaml
- given the same seed, 2 and 3 produce the exact same music

Weights are trained on the host and integrated into the RTL side.

## Why are you doing this?

For fun and learning.

- a microcontroller is probably better for this kind of task than an FPGA in almost every way, at least for the auto-regressive variants (transformer/mamba)
- but playing with hardware is really fun and a good learning experience
- and surprisingly freeing: no operating system, no CPU, no GPU, you just make a circuit, power it on and run

## Quickstart

### Prerequisites

- For Python: [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For OCaml: [opam](https://opam.ocaml.org/doc/Install.html)

### Install project deps

For the Python side:

```sh
uv sync
```

*Note: if the JAX CUDA plugin fails to install, the trainer falls back to the CPU silently.*

We use OxCaml for the OCaml side:

```sh
opam switch create 5.2.0+ox \
    --repos ox=git+https://github.com/oxcaml/opam-repository.git,default
eval $(opam env)
opam install dune core core_unix hardcaml hardcaml_waveterm nx yojson \
    ppx_let ppx_expect ocamlformat
```

Hardcaml version is `v0.18~preview`, which is what the ox repository holds.

### Run tests

```sh
make build    # build stuff
make test     # run tests
```

### (Optional) Only if you want to test it on a board

The project is developed and tested with:

- [Vivado](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html): Building the bitstream and programming the board.
- [Nexys 4](https://digilent.com/reference/_media/reference/programmable-logic/nexys-4/nexys4_rm.pdf): An FPGA dev board from Digilent. The free edition of Vivado covers it, so no licence is needed. This is **NOT** the later Nexys 4 DDR version with DDR RAM.
- [Roland S-1](https://www.roland.com/us/products/s-1/): A mighty little synth from Roland.
- Nexys 4 Pmod JD connects to the S-1 MIDI IN as the pin map states, through an improvised adaptor: two 33 Ω resistors and a standard audio cable and connector.
- Neither the board nor the synth is a requirement of the project; they just happen to be the gear I own. Any MIDI player, hardware or software, and any board with comparable resources should do, though the code would need adapting.

## JAX

### Play pretrained weights on the host

A few pretrained weights ship with the code in `weights/`. To sample them:

```sh
# transformer
uv run python -m transformer.infer sample --ckpt weights/transformer.ckpt --seeds 1 --save out.mid
# mamba
uv run python -m mamba.infer sample --ckpt weights/mamba.ckpt --seeds 1 --save out.mid
# diffusion
uv run python -m diffusion.infer sample --ckpt weights/diffusion.ckpt --seeds 1 --save out.mid
```

- the generated MIDI sequence is saved to `out.mid`
- `--quantized` draws from the quantized model instead of the float one
- switching `--save` to `--play` sends MIDI to `/dev/snd/midiC2D0` on **channel 3**, e.g. a hardware synth (developed and tested on a Roland S-1)
- these weights are meant to be quantized and then integrated into FPGA BRAM, hence the small sizes
- `--help` for help

There is also a pink noise model that needs no weights and no JAX (hardware synth only, and it has no `-save` option):

```sh
dune exec bin/play_pink.exe -- -seed 1
```

### Train your own weights

#### Prepare the corpus

```sh
make corpus
```

This exports `corpus/JSB-Chorales-dataset` into `jax/_data` (git ignored) in the format the JAX side uses.

#### Run trainer

```sh
uv run python -m transformer.train --steps 200 --ckpt _train/probe.ckpt
uv run python -m mamba.train --steps 200 --ckpt _train/probe.ckpt
uv run python -m diffusion.train --steps 200 --ckpt _train/probe.ckpt
```

- without `--ckpt` nothing is saved and the run only prints its losses
- what is written is the best checkpoint by validation loss, not the last one
- `_train/` is git ignored: it holds the runs, where `weights/` holds the product
- pipe a run to a log beside its checkpoint, `... | tee _train/NAME.log`, and watch it live
- `--help` lists the shape flags of each model: `--d`, `--layers`, `--plan`, `--crop` etc.
- drop your own checkpoint at `weights/<model>.ckpt` and the board build picks it up
- the recipe behind each shipped checkpoint is in `weights/README.md`, and why it won is in `docs/<model>.md`

## Hardcaml

> [!Caution]
> **The series resistor belongs to your pair of devices, not to this code.** MIDI IN is an isolated current loop, and the value follows from the driver voltage and the receiver. Here it is 33 Ω: a Nexys 4 driving Pmod JD pin 1 at 3.3 V into a Roland S-1, whose input measures about 286 Ω and needs 5 mA in the worst case. Another synth presents another loop, and a 5 V driver wants the classic 220 Ω. Work out your own value before you connect anything.
>
> **Remove the board power before you connect or disconnect the cable.** A TRS plug shorts tip, ring and sleeve together as it slides into the jack, and about 50 mA then flows, which is more than an Artix-7 pin permits.
>
> **Check Pmod pins 5 and 6 before the first power-on.** Pin 5 is ground and pin 6 is the 3.3 V supply. Swap the two wires and you short the supply, which the Nexys 4 does not fuse per connector.
>
> Proceed at your own risk.

### Generate verilog

```sh
make verilog-pink
make verilog-transformer
make verilog-mamba
make verilog-diffusion
```

Verilog is generated to `board/_generated/top.v`, so a later command overwrites the earlier version.

- each model but pink is quantized to `weights/<model>.int8` first, which the elaboration reads
- this needs no Vivado; only the bitstream and the board do

### Nexys 4

Program the board (transformer as the example). This is the fast path, and it does not survive a power cycle:

```sh
make bitstream-transformer
make program
```

Flash the board so it boots from the onboard QSPI flash memory:

```sh
make bitstream-transformer
make flash
```

- `make bitstream-<model>` pulls the Verilog and the quantized weights behind it, so there is nothing to run first

### Host tool

`board_tool` reads and writes the control registers over the console UART, at `/dev/ttyUSB1`:

```sh
dune exec bin/board_tool.exe -- dump          # every cell, each register named
dune exec bin/board_tool.exe -- read 0x00 4   # the seed cell
dune exec bin/board_tool.exe -- write 0x08 1  # RUN: start the music
```

- the registers are `seed`, `velocity`, `step_ms`, `channel` and `run`
- `-device PATH` for another port
- the board needs no host at all: the center button toggles RUN, and the slide switches set the seed
- there is no runtime version, so the driver and the bitstream must come from the same commit

[doc](docs/host_control.md)

## Layout

```
fpga-midi-gen/
├── docs/             the design documents
├── jax/              the Python side: see below
├── lib/              the OCaml libraries, software and RTL together
│   ├── core/         the host control constants, MIDI, the frame, the PRNG, Cyclesim
│   ├── board/        the UART, COBS, the control port, the sequencer, the socket
│   ├── nn/           common part for sources
│   ├── pink/         pink noise source
│   ├── transformer/  transformer source
│   ├── mamba/        mamba source
│   ├── diffusion/    diffusion source
│   └── corpus/       the chorales (Jsb) and the vocabulary (Vocab)
├── bin/              the executables: the drivers, the gate drivers, the elaborators
├── board/            the top level, the pin map and the Vivado scripts of each board
├── corpus/           the chorale corpus
├── weights/          the elected checkpoints for transformer, mamba and diffusion model
└── test/             the OCaml integration tests: the socket simulations
```

```
jax/
├── transformer/      one directory for each era, mamba/ and diffusion/ beside it
│   ├── model.py      the float model
│   ├── train.py      the trainer
│   ├── infer.py      sample on the host, quantize to a contract file
│   └── quantized/    the integer twin, which the circuit mirrors bit for bit
│       ├── model.py  the weights, the formats and the contract file
│       └── infer.py  the walk the board runs
├── train.py          the rate curve, the update rule, the checkpoint -- not a loop
├── sample.py         the float draw; quantized.pick is its integer twin
├── measure.py        the instruments every era is judged on
├── quantized.py      the integer rules every twin follows, and the contract file
├── ar_*.py           eras four and five only: their model, twin, trainer, measure
├── cli.py            the click options more than one command states
├── prng.py           xorshift32, the batched twin of lib/core/prng.ml
├── midi.py           the wire side: to the synth, to a .mid, or to the terminal
├── corpus.py         the chorales and the vocabulary, as lib/corpus holds them
├── tests/            the oracle gates
└── _data/            the packed corpus, git ignored: make corpus writes it
```

## The board

Source models connect to the board through the same interface, `lib/core/source_intf.ml`.

![The blocks of the board](docs/board_rtl.svg)

[doc](docs/board_rtl.md)

## Models

### Transformer

Transformer with a few hardware-in-mind choices: rms_norm, ALiBi etc.

![The transformer model](docs/transformer.svg)

[doc](docs/transformer.md)

#### RTL

![The transformer source](docs/transformer_rtl.svg)

[doc](docs/transformer_rtl.md)

### Mamba

Mamba takes a page from the [Zamba](https://arxiv.org/abs/2405.16712) book and places an attention block near the end.

![The Mamba model](docs/mamba.svg)

[doc](docs/mamba.md)

#### RTL

![The Mamba source](docs/mamba_rtl.svg)

[doc](docs/mamba_rtl.md)

### Diffusion

A variant of [Coconet](https://magenta.withgoogle.com/coconet).

![The Diffusion model](docs/diffusion.svg)

[doc](docs/diffusion.md)

#### RTL

![The Diffusion source](docs/diffusion_rtl.svg)

[doc](docs/diffusion_rtl.md)

### Pink noise

A cute little pink-noise-to-pentatonic circuit.

[doc](docs/pink.md)

#### RTL

![The Pink source](docs/pink_rtl.svg)

[doc](docs/pink_rtl.md)
