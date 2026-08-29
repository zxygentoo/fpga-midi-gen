"""What every RTL gate mounts its driver in: the built executable, the skip when it is
absent, and the run that carries a failure's stderr into the report.

THE TWO SIDES OF A GATE MUST NOT AGREE WITH THEMSELVES. A driver executable runs the
bench and prints what the circuit did; the test states what it must have done, against a
twin that runs no circuit. This module holds only the part that is neither -- the
subprocess -- thus the eras share the mounting and share nothing of the judgement.

It stands here and not in one era's test module: a gate is a gate in every era, and the
era that copied these would be the era whose gate drifted from the one before it.
"""

import subprocess

import pytest

import data

ROOT = data.JAX_ROOT.parent
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


def run(*argv):
    """the stdout of one driver run.

    check=False: the assert carries the stderr into the report, where a
    CalledProcessError would show the command alone."""
    done = subprocess.run(
        [str(word) for word in argv], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    return done.stdout
