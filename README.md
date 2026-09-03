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

TBA

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

TBA

### Nexys-4

TBA

### Host tool

TBA

## Layout

```
lib/         the OCaml libraries, software and RTL together
  core/        the host control constants, MIDI, the frame, the PRNG, Cyclesim
  board/       the UART, COBS, the control port, the sequencer, the socket
  corpus/      the chorales (Jsb) and the vocabulary (Vocab)
  nn/          common part for sources
  pink/        pink noise source
  transformer/ transformer source
  mamba/       mamba source
  diffusion/   diffusion source
bin/         the executables: the drivers, the gate drivers, the elaborators
board/       the top level, the pin map and the Vivado scripts of each board
jax/         the Python side: see below
corpus/      the chorale corpus
weights/     the elected checkpoints for transformer, mamba and diffusion model
docs/        the design documents
test/        the OCaml integration tests: the socket simulations
```

```
jax/
  corpus.py       the chorales and the vocabulary, as lib/corpus holds them
  prng.py         xorshift32, the batched twin of lib/core/prng.ml
  midi.py         the wire side: to the synth, to a .mid, or to the terminal
  sample.py       the float draw; quantized.pick is its integer twin
  measure.py      the instruments every era is judged on
  train.py        the rate curve, the update rule, the checkpoint -- not a loop
  quantized.py    the integer rules every twin follows, and the contract file
  cli.py          the click options more than one command states
  ar_*.py         eras four and five only: their model, twin, trainer, measure
  transformer/    one directory for each era, mamba/ and diffusion/ beside it
    model.py      the float model
    train.py      the trainer
    infer.py      sample on the host, quantize to a contract file
    quantized/    the integer twin, which the circuit mirrors bit for bit
      model.py    the weights, the formats and the contract file
      infer.py    the walk the board runs
  tests/          the oracle gates
  _data/          the packed corpus, git ignored: make corpus writes it
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
