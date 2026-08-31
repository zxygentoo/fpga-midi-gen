"""The click options more than one command of this tree states.

A `click.option` written twice is a default that can drift, and a default that drifts here
is an audition of one policy reported as another. Everything below is one set because the
thing behind it is one thing -- one checkpoint format, one draw, one wire.

WHAT IS NOT HERE: a flag that names an era's own shape. `--ring` is era five's, `--heads`
and `--context` are era four's, `--crop` and `--walk` are era six's, and each stands in
its own module beside the command that reads it.
"""

import click

import quantized

# the checkpoint of a command that READS one. The two trainers write theirs and take
# `default=None` instead, thus they are not this option.
ckpt_option = click.option(
    "--ckpt", required=True, type=click.Path(exists=True, dir_okay=False)
)


def add_options(options):
    """[options] as one decorator, in the order they are written.

    click reads a stack of decorators FROM THE BOTTOM UP, thus the reverse here is what
    keeps --help in the order the list states. Every option set below and in `midi.py` and
    `ar_train.py` is applied through this, so that the rule stands one time."""

    def decorate(command):
        for option in reversed(options):
            command = option(command)
        return command

    return decorate


def parse_seeds(ctx, param, value):
    """a list, or LOW-HIGH. It is a click callback, and a caller with a string in hand
    passes `None, None, value`."""
    del ctx, param
    if "-" in value:
        low, high = value.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(seed) for seed in value.split(",")]


# The draw of the step-frame eras, and where a batch of walks comes from. The temper and
# the floor were elected by ear 2026-08-17 over a sweep of temperature 0.7 to 1.3 against
# min_p 0.0039 to 0.15, and the numbers agree with the ear at both edges, which is rare
# here. Hotter draws more from the tail: at 1.2 the onset rate passes the corpus and the
# chords go strange. A higher floor smooths the arrivals and costs the music: min_p 0.15
# leaves about one and a half classes standing at a draw, and it reads as dull and MORE
# silent -- silence 5.83 percent against 4.22, gaps 13.4 steps against 9.8, where the
# corpus gives 4.19 and 9.9. Era five carried the pair over unmeasured and has not
# re-elected it, thus the two eras are auditioned on one policy.
#
# `quantized.ELECTED_*` is the one home of the two numbers since the all-era cut took the
# OCaml `Policy` away, thus the audition and the contract file temper alike.
sampler_options = add_options(
    [
        ckpt_option,
        click.option(
            "--seeds", default="1", callback=parse_seeds, help="a list, or LOW-HIGH"
        ),
        click.option(
            "--steps", default=256, help="steps to draw, the silent lead-in inside"
        ),
        click.option("--temperature", default=quantized.ELECTED_TEMPERATURE),
        click.option("--min-p", default=quantized.ELECTED_MIN_P),
        click.option(
            "--quantized",
            "twin",
            is_flag=True,
            help="draw the integer twin of the circuit: the piece the board plays at "
            "this seed",
        ),
    ]
)
