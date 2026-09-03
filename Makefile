# The commands of this repository, in one place.
#
# It is a SHORTCUT LAYER and never a build system: dune builds the OCaml, uv runs the
# Python, and Vivado makes a bitstream. What stands here is the ORDER those three go in,
# which is the one thing none of them knows -- a bitstream wants a netlist, a netlist
# wants a contract file, and a contract file wants a checkpoint.
#
# AGENT.md points at these targets rather than restating the commands, because a command
# written twice is a command that drifts.

ERAS := pink transformer mamba diffusion
VIVADO_BATCH := vivado -mode batch -journal board/_build/vivado.jou \
                -log board/_build/vivado.log -source

# the lanes of a probe, G: 1 is the fallback rung, 5 the geometry of the fused pair
G ?= 5

# Vivado steps must run in the order they are written, and a bitstream is not a thing to
# race against itself.
.NOTPARALLEL:

# THE PATTERN TARGETS ARE NOT HERE, and that is not an oversight: make does not look for
# a pattern rule to build a target it has been told is phony, thus `verilog-%` would
# answer "nothing to be done" and write nothing.
.PHONY: help gates test fmt lint build corpus clean require-vivado program flash \
        verilog-pink

help:
	@echo 'make gates              the pre-commit gates: fmt, build, ruff, both suites'
	@echo 'make test               dune runtest, then the Python suite'
	@echo 'make fmt                format the OCaml side'
	@echo 'make lint               ruff over the Python side'
	@echo 'make build              dune build, then everything a clone can derive'
	@echo 'make corpus             the JAX seam: the frames and the pieces of jax/_data'
	@echo
	@echo 'make verilog-ERA        the board top level over one era, into board/_generated'
	@echo 'make bitstream-ERA      that netlist through Vivado, into board/_build'
	@echo 'make probe-UNIT [G=n]   one unit out of context: array, epilogue, draw, forward'
	@echo 'make program            the bitstream into the board, over USB JTAG'
	@echo 'make flash              the bitstream into the QSPI flash'
	@echo
	@echo 'ERA is one of: $(ERAS)'
	@echo 'Era one reads no file; the other three read weights/ERA.int8, which is made'
	@echo 'from the committed weights/ERA.ckpt when it is missing.'

# ---- the gates ------------------------------------------------------------

# AGENT.md states these and this is where they live. `dune build @fmt` REPORTS a
# difference where `dune fmt` would write one, thus a tree that needs formatting fails
# here instead of being formatted behind the author.
#
# THE GATES RUN `test` AND DO NOT RESTATE IT. Both suites gate a commit -- the OCaml side
# holds the unit gates, the cycle benches and the waveforms, and the Python side holds the
# oracle gates -- and one target that names them is one place they can move.
gates:
	dune build @fmt
	dune build
	uv run ruff check
	$(MAKE) --no-print-directory test

test:
	dune runtest
	uv run pytest

fmt:
	dune fmt

lint:
	uv run ruff check

# WHAT A CLONE CAN DERIVE, and this is where a clone starts: the OCaml, the corpus of the
# JAX seam, and the contract file of every era that has weights. `make build && make
# gates` then says whether the tree stands in a good place, and no gate is skipped for a
# file that was never written.
build:
	dune build
	$(MAKE) --no-print-directory corpus $(CONTRACTS)

# ---- the corpus -----------------------------------------------------------

# The two files of the JAX seam, from the chorales in corpus/. Git ignores jax/_data/,
# thus a clone writes them one time, before a trainer runs or era six auditions.
#
# EACH SUBCOMMAND IS NAMED BY THE FILE IT WRITES, and the two files are two corpora and
# not two views of one: a stream has no pieces and a sheet holds nothing else.
#
# THE PACKING RULES ARE PREREQUISITES AND NOT THE CHORALES ALONE, for the reason the
# contract file states below: the seam carries data and never rules, thus a moved shift
# policy, rotation or grid makes a kept file wrong and NOTHING ON THE JAX SIDE COULD SAY
# SO -- it reads arrays, and it never reads what packed them.
CHORALES := corpus/JSB-Chorales-dataset/Jsb16thSeparated.json
PACKERS := bin/corpus_tool.ml lib/corpus/jsb.ml

corpus: jax/_data/frames.safetensors jax/_data/pieces.safetensors

jax/_data/frames.safetensors: $(CHORALES) $(PACKERS)
	dune exec bin/corpus_tool.exe -- frames

jax/_data/pieces.safetensors: $(CHORALES) $(PACKERS)
	dune exec bin/corpus_tool.exe -- pieces

# ---- the weights ----------------------------------------------------------

# The contract file of an era, from the checkpoint committed beside it. It is DERIVED and
# git ignores it; weights/README.md holds why. Era one owns no rule here, because its
# model is a value of `lib/pink` and no file states it.
# .PRECIOUS, because a file make builds in the MIDDLE of a chain is one make deletes at
# the end of it: without this the contract file is written, elaborated and then removed,
# and every netlist costs a JAX run again.
.PRECIOUS: weights/%.int8

# WHAT A CONTRACT FILE IS MADE OF IS NOT THE CHECKPOINT ALONE. The quantizer states the
# int8 image, thus a rule that watched the checkpoint by itself would keep a file that a
# moved rounding, exponent or fold had made wrong -- and no gate would say so, because
# `test_parity.py` quantizes into its own directory and never reads this one. The list is
# deliberately broad: a few seconds of quantizing is the cheaper mistake by far.
QUANTIZERS := $(wildcard jax/*.py) \
              $(wildcard jax/transformer/*.py jax/mamba/*.py jax/diffusion/*.py)

weights/%.int8: weights/%.ckpt $(QUANTIZERS)
	uv run python -m $*.infer quantize --ckpt $< --out $@

# every contract file a clone can write, which `make build` does. THE COMMITTED
# CHECKPOINTS STATE WHICH ERAS HAVE ONE and nothing here lists them: era one is a value of
# `lib/pink`, it owns no `.ckpt`, and it drops out by itself.
CONTRACTS := $(patsubst %.ckpt,%.int8,$(wildcard weights/*.ckpt))

# ---- the netlist ----------------------------------------------------------

# Era one stands apart and FIRST, because make takes an explicit rule over a pattern: it
# reads no contract file, thus it elaborates on a clone with nothing in weights/.
verilog-pink:
	dune exec bin/gen_verilog.exe -- pink

verilog-%: weights/%.int8
	dune exec bin/gen_verilog.exe -- $*

# ---- Vivado ---------------------------------------------------------------

require-vivado:
	@command -v vivado >/dev/null || { \
	  echo 'vivado is not on PATH: a bitstream, a probe and the board need it.'; \
	  echo 'Everything else in this Makefile runs without it.'; exit 1; }

# require-vivado STANDS FIRST, because prerequisites run left to right: with it last, a
# machine without Vivado would quantize and elaborate for minutes before failing.
bitstream-%: require-vivado verilog-%
	@mkdir -p board/_build
	$(VIVADO_BATCH) board/nexys-4/build.tcl

# One unit out of context, at G lanes: the timing and the utilization before the machine
# around it exists. docs/diffusion_rtl.md holds what the readings said.
probe-%: require-vivado
	@mkdir -p board/_build
	dune exec bin/gen_probe.exe -- $* $(G)
	$(VIVADO_BATCH) board/nexys-4/probe.tcl

program: require-vivado
	@mkdir -p board/_build
	$(VIVADO_BATCH) board/nexys-4/program.tcl

flash: require-vivado
	@mkdir -p board/_build
	$(VIVADO_BATCH) board/nexys-4/flash.tcl

# ---- clean ----------------------------------------------------------------

# EVERYTHING DERIVED GOES, the contract files and the corpus with them: each is what one
# program said at one moment, and a half-clean tree that keeps them is how a stale one
# survives. `make corpus` writes jax/_data again in seconds.
#
# _train/ STAYS. It is git-ignored as these are, but it is not derived: a checkpoint costs
# hours and no rule here could write it again.
clean:
	dune clean
	rm -rf board/_generated board/_build jax/_data
	rm -f weights/*.int8
