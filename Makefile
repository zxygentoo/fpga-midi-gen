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
.PHONY: help gates test fmt lint build clean require-vivado program flash verilog-pink

help:
	@echo 'make gates              the pre-commit gates: fmt, build, ruff, pytest'
	@echo 'make test               dune runtest, then the Python suite'
	@echo 'make fmt                format the OCaml side'
	@echo 'make lint               ruff over the Python side'
	@echo 'make build              dune build'
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

# AGENT.md states these four and this is where they live. `dune build @fmt` REPORTS a
# difference where `dune fmt` would write one, thus a tree that needs formatting fails
# here instead of being formatted behind the author.
gates:
	dune build @fmt
	dune build
	uv run ruff check
	uv run pytest

test:
	dune runtest
	uv run pytest

fmt:
	dune fmt

lint:
	uv run ruff check

build:
	dune build

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

# EVERYTHING DERIVED GOES, the contract files with it: they are what the quantizer said
# at one moment, and a half-clean tree that keeps them is how a stale one survives.
clean:
	dune clean
	rm -rf board/_generated board/_build
	rm -f weights/*.int8
