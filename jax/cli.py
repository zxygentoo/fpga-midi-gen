"""The click options more than one command of this tree states.

An option written twice is a default that can drift, and a default that drifts here is an
audition of one policy reported as another. A flag that names an era's own shape is NOT
here: `--ring` is era five's, `--heads` and `--context` era four's, `--crop` and `--walk`
era six's, and each stands beside the command that reads it.
"""

import click

import quantized

# the checkpoint of a command that READS one; the trainers write theirs and take
# `default=None` instead
ckpt_option = click.option(
    "--ckpt", required=True, type=click.Path(exists=True, dir_okay=False)
)


def add_options(options):
    """[options] as one decorator, in the order they are written. click reads a stack of
    decorators FROM THE BOTTOM UP, thus the reverse here is what keeps --help in that
    order."""

    def decorate(command):
        for option in reversed(options):
            command = option(command)
        return command

    return decorate


def parse_seeds(ctx, param, value):
    """a list, or LOW-HIGH; a caller with a string in hand passes `None, None, value`"""
    del ctx, param
    if "-" in value:
        low, high = value.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(seed) for seed in value.split(",")]


# The draw of the step-frame eras. The temper and the floor were elected by ear over
# temperature 0.7 to 1.3 against min_p 0.0039 to 0.15, and the numbers agree with the ear:
# hotter draws from the tail (at 1.2 the onset rate passes the corpus), and a higher floor
# smooths the arrivals and dulls the music (min_p 0.15 leaves about one and a half classes
# standing). Era five carried the pair over unmeasured, thus both eras audition on one
# policy.
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
            is_flag=True,
            help="draw the integer twin of the circuit: the piece the board plays at "
            "this seed",
        ),
    ]
)
