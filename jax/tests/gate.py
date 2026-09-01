"""What a gate mounts on: the built driver, the skips for what git ignores, the run that
carries a failure's stderr into the report, and the readers of what a driver printed.

THE TWO SIDES OF A GATE MUST NOT AGREE WITH THEMSELVES. A driver runs the bench and prints
what the circuit did; the test states what it must have done, against a twin that runs no
circuit. This module holds only the part that is NEITHER, thus the eras share the mounting
and share nothing of the judgement.

Two skips and not three: [need] is for a path `dune build` writes and `needs_corpus` for
one `corpus_tool` writes. Neither absence is a failure.
"""

import re
import subprocess

import numpy as np
import pytest
from click.testing import CliRunner

import corpus

ROOT = corpus.JAX_ROOT.parent
BUILT = ROOT / "_build" / "default" / "bin"


def driver(name):
    """the built gate driver of one era, by its executable name"""
    return BUILT / name


def need(*paths):
    """skip when a path the gate needs was never built -- a tree with no dune build
    behind it has nothing to gate, and that is not a failure"""
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"absent, nothing to gate: {', '.join(missing)}")


needs_corpus = pytest.mark.skipif(
    not corpus.PIECES.exists(), reason="needs corpus_tool pieces"
)


def losses_of(command, argv):
    """The losses one short training run printed, and its whole output beside them. A
    TRAINER THAT CANNOT LEARN STILL PRINTS A CORRECT STEP-1 LOSS, thus only a run of
    several steps tells and the caller states the fall it wants."""
    done = CliRunner().invoke(command, [str(word) for word in argv])
    assert done.exit_code == 0, done.output
    return [float(m) for m in re.findall(r"loss (\d+\.\d+)", done.output)], done.output


def run(*argv, cwd=None):
    """The stdout of one run of [argv]. check=False so the assert carries the stderr into
    the report, where a CalledProcessError would show the command alone; [cwd] must be
    passed, because a subprocess inherits pytest's own directory."""
    done = subprocess.run(
        [str(word) for word in argv],
        capture_output=True,
        text=True,
        check=False,
        cwd=None if cwd is None else str(cwd),
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def classes_of_walk(stdout, steps):
    """The classes the circuit drew at each step, out of a driver's `walk`. THE FORMAT IS
    `bin/gate_common.ml`'s, `step N FRAME c0 c1 c2 c3`, and a line that is not a step is
    FILTERED and never indexed -- splitting a blank line would read as a broken test."""
    lines = [line.split() for line in stdout.splitlines() if line.startswith("step")]
    assert len(lines) == steps, f"the driver stated {len(lines)} steps, wanted {steps}"
    return np.array([[int(word) for word in line[3:]] for line in lines])


def assert_one_walk(circuit, twin):
    """the circuit's walk against the twin's, CLASS FOR CLASS AND STEP FOR STEP: the
    message names the first step they part at and what each of them drew there"""
    parted = np.flatnonzero(~(circuit == twin).all(axis=-1))
    assert not len(parted), (
        f"the walks part at step {parted[0]}: the circuit drew "
        f"{list(circuit[parted[0]])} and the twin wants {list(twin[parted[0]])}"
    )
