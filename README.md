# fpga-midi-gen

Make a few models (transformer, mamba, diffusion) learn to generate Bach chorales, then make circuit versions of them to play on FPGA.

- For software side, we use [JAX](https://docs.jax.dev/en/latest/)
- For RTL implementation, we use [Hardcaml](https://hardcaml.org/)

Their are three implementations for each model:
1. A JAX float one: your normal machine learning stuff
2. A JAX quantized (int8) one: a weight that's easy to integrate in circuit
3. A RTL one: FPGA implemention of the inference engine

- 1 and 2 is close in behavior, measured by drift gates
- 2 and 3 is identical in behavior
- identical prng (xorshift32) implemention for both JAX and Hardcaml
- giving the same seed, 2 and 3 produces the exact same music

Weights are trained on the host and integrated to the RTL side.

## Why are you doing this?

For fun and learning.

- a microcontroller is probably better for this kind fo task than FPGA in almost every way, at least for the auto-regressive variants (transformer/mamba)
- but playing with hardware is really fun and good learning experience
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

*Note: If JAX CUDA plugin failed to install, the trainer falls back to CPU silently.*

We use OxCaml for the ocaml side:

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
make build    # build OCaml stuff
make test     # run tests, both Python and OCaml
```

## JAX

### Play pretained weights on the host

A few pretrained weight is shipped with the code in `weights/`, to sample them:

```sh
# transformer
uv run python -m transformer.infer sample --ckpt weights/transformer.ckpt --seeds 1 --save out.mid
# mamba
uv run python -m mamba.infer sample --ckpt weights/mamba.ckpt --seeds 1 --save out.mid
# diffusion
uv run python -m diffusion.infer sample --ckpt weights/diffusion.ckpt --seeds 1 --save out.mid
```

- generated MIDI sequnence is saved to `out.mid`
- `--quantized` will draw from the quantied model install of the float one
- switch `--save` to `--play` will send midi to `/dev/snd/midiC2D0` on **channel 3**, eg. a hardware synth (dev tested on a Roland S-1)
- these weights are meant for directly integrated into FPGA BRAM, hence the samll sizes
- `--help` for help

And, there is also a pink noise model need no weight and JAX (and no `-save` option or hardware synth only):

```sh
dune exec bin/play_pink.exe -- -seed 1
```

### Train your own weight

#### Prepare the corpus

```sh
make corpus
```

This will export `corpus/JSB-Chorales-dataset` into `jax/_data` (git ignored) in the format jax side uses.

#### Run tainer

TBA

## Hardcaml

> [!Caution]
> The code here is using PMod JD on the Nexsy-4 for midi output, and you may have to add resistences between the output and you midi device, otherwise it may demage your board or midi device.
> Proceed with caution and at you own risk.

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
  nn/          what is one thing across the eras: the units, the fixed-point
               rules, the placement, the sampler, the program compiler
  pink/ transformer/ mamba/ diffusion/    one directory for each era
bin/         the executables: the drivers, the gate drivers, the elaborators
board/       the top level, the pin map and the Vivado scripts of each board
jax/         the Python side: the trainers, the float models, the integer
             twins, and the oracle gates in jax/tests
corpus/      the chorale corpus
weights/     the elected checkpoint of each era, committed
docs/        the design documents
test/        the integration tests: the socket simulations
```

## The board

Source models connect to the board using the same interface in `lib/core/source_intf.ml`.

![The blocks of the board](docs/board_rtl.svg)

[doc](docs/board_rtl.md)

## Models

### Transoformer

Transformer with a few hardware-in-mind choices: rms_norm, ALiBi etc.

![The transformer model](docs/transformer.svg)

[doc](docs/transformer.md)

#### RTL

![The transformer source](docs/transformer_rtl.svg)

[doc](docs/transformer_rtl.md)

### Mamba

Mamba with a page from the [Zamba](https://arxiv.org/abs/2405.16712) book and places an attention layer at the end.

![The Mamba model](docs/mamba.svg)

[doc](docs/transformer.md)

#### RTL

![The Mamba source](docs/mamba_rtl.svg)

[doc](docs/transformer.md)

### Diffusion

A variant of the [Coconet](https://magenta.withgoogle.com/coconet).

![The Diffusion model](docs/diffusion.svg)

[doc](docs/diffusion.md)

#### RTL

![The Diffusion source](docs/diffusion_rtl.svg)

[doc](docs/diffusion_rtl.md)

### Pink noise

[doc](docs/pink.md)

#### RTL

![The Pink source](docs/pink_rtl.svg)

[doc](docs/pink_rtl.md)
