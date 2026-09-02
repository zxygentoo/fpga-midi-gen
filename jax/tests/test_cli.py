"""The command line: the grammar of the shared flags, and that every module still runs.

`cli.py` holds the options no single era owns, and the only arithmetic among them is
`parse_seeds`. A sweep states its seeds as a range and an audition as a list; a range read
exclusive at the top would drop the last seed of every sweep this project has reported,
and say nothing.

THE SMOKE BELOW IT READS WHAT NO OTHER GATE DOES. A click option's default is evaluated at
IMPORT TIME, thus a module whose defaults name a moved constant dies before any command
runs -- and a module nothing imports dies unseen. `mamba/measure.py` did exactly that
twice: it is the one CLI module with no function a test calls, and ruff cannot see the
fault, because an attribute on a module that exists is not an undefined name. Importing
the module and asking its group for `--help` is the cheapest thing that catches it: it
runs no model and needs no checkpoint.
"""

import importlib

import pytest
from click.testing import CliRunner

import cli


def seeds(value):
    """the seeds of one --seeds string; click hands a callback its context and its
    parameter, and a caller with a string in hand passes None for both"""
    return cli.parse_seeds(None, None, value)


def test_a_list_of_seeds_reads_in_the_order_it_is_written():
    """the default is one seed, and a list is the several an audition plays in turn"""
    assert seeds("1") == [1]
    assert seeds("1,3,7") == [1, 3, 7]


def test_a_range_of_seeds_holds_both_of_its_ends():
    """LOW-HIGH IS INCLUSIVE AT BOTH ENDS: 0-15 is the sixteen seeds the panel switches
    can state, and not fifteen of them."""
    assert seeds("4-7") == [4, 5, 6, 7]
    assert seeds("0-15") == list(range(16))
    assert seeds("3-3") == [3]


# every module in `jax/` that carries a click group. Era four has no `measure`: both its
# halves stand in `ar_measure.py`, which `transformer/infer.py` reads.
COMMANDS = [
    "diffusion.infer",
    "diffusion.measure",
    "diffusion.train",
    "mamba.infer",
    "mamba.measure",
    "mamba.train",
    "transformer.infer",
    "transformer.train",
]


@pytest.mark.parametrize("name", COMMANDS)
def test_a_command_module_imports_and_answers_help(name):
    """the import is half the gate and the `--help` is the other half: the first holds
    the module's own defaults, the second holds every option of every command under it"""
    module = importlib.import_module(name)
    answered = CliRunner().invoke(module.main, ["--help"])
    assert answered.exit_code == 0, f"{name} --help exited {answered.exit_code}"
    assert answered.output.startswith("Usage:"), f"{name} --help stated no usage"
