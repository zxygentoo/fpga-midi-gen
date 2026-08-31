"""What a gate mounts on: the built driver, the skips for what git ignores, the run that
carries a failure's stderr into the report, and the readers of what a driver printed.

THE TWO SIDES OF A GATE MUST NOT AGREE WITH THEMSELVES. A driver executable runs the
bench and prints what the circuit did; the test states what it must have done, against a
twin that runs no circuit. This module holds only the part that is NEITHER -- the
subprocess, the skip, the parse -- thus the eras share the mounting and share nothing of
the judgement.

It stands here and not in one era's test module: a gate is a gate in every era, and the
era that copied these would be the era whose gate drifted from the one before it.

TWO SKIPS AND NOT THREE: [need] is for a path `dune build` writes and `needs_corpus` for
one `corpus_tool` writes. Both are absences of git-ignored work and neither is a failure.
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
    """the losses one short training run printed, and its whole output beside them.

    Every era's trainer smoke reads its run through here. A TRAINER THAT CANNOT LEARN
    STILL PRINTS A CORRECT STEP-1 LOSS -- the loss at step 1 is measured before the first
    update -- thus only a run of several steps tells, and the caller states the fall it
    wants."""
    done = CliRunner().invoke(command, [str(word) for word in argv])
    assert done.exit_code == 0, done.output
    return [float(m) for m in re.findall(r"loss (\d+\.\d+)", done.output)], done.output


def run(*argv, cwd=None):
    """the stdout of one run of [argv].

    check=False: the assert carries the stderr into the report, where a
    CalledProcessError would show the command alone.

    [cwd] STATES THE DIRECTORY, and a caller that needs one must pass it: a subprocess
    inherits pytest's own, which is wherever pytest was started, thus a gate that runs
    `uv run python -m era.infer` reads a different tree from a shell one directory up.
    `corpus.JAX_ROOT` is what those callers pass."""
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
    """the classes the circuit drew at each step, out of one `walk` run of a driver.

    THE FORMAT IS ONE FORMAT, `bin/gate_common.ml`'s: `step N FRAME c0 c1 c2 c3`, one
    line for each step, and a driver may print other lines around them. A LINE THAT IS
    NOT A STEP IS FILTERED AND NEVER INDEXED -- a gate that split a blank line and read
    `line[0]` raises IndexError, which reads as a broken test and not as a circuit that
    said nothing.

    It parses here and not in each era's gate for the same reason `run` does: one printer
    deserves one reader, and the era that copied this one would be the era whose gate
    drifted from the one before it."""
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
