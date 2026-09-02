"""The audition of the masked sheet: independent blocked Gibbs, eight measures at a time.

    uv run python -m diffusion.infer sample --ckpt C --seeds 1-8 --play
    uv run python -m diffusion.infer sample --ckpt C --seeds 7 --walk 32

`sample` is the ear's path: draw, print the battery against the corpus row, and speak the
music to the synthesizer or to a .mid. A batch is several whole pieces and not one piece
in parts, thus --gap puts a silence between two of them. It does not make a crop ARRIVE.

EVERY DRAW COMES FROM THE SHARED GENERATOR under the consumption order of
docs/diffusion_rtl.md: one seed names one SHEET -- its opening, its masks and its redraws
-- alone or in any batch. That is what gives the era the seed handoff.

CPU IS THE DEFAULT PLATFORM, deliberately: a walk is a few hundred forward passes of one
small sheet and the GPU belongs to the trainer. JAX_PLATFORMS=cuda overrides it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
from pathlib import Path

import click
import jax.numpy as jnp
import numpy as np

import cli
import corpus
import midi
import prng
import quantized as q
from diffusion import measure as referee
from diffusion import model
from diffusion import quantized as integer
from diffusion.sample import gibbs_passes, tempered_pick


def gibbs(coconet, given, states, *, walk, temperature):
    """The FLOAT walk of the era: the `gibbs_passes` of `diffusion/sample.py`, in float64
    over the trained model. The loop, the schedule and the order of the draws stand there
    once for both walks; what is here is this walk's arithmetic alone. It gives the sheets
    and the generator behind them, as `integer.gibbs` does."""

    def forward(classes, hidden):
        return np.asarray(
            coconet.logits(jnp.asarray(classes), jnp.asarray(hidden)),
            dtype=np.float64,
        )

    def redraw(states, logits, step, voice, active):
        states, u = prng.uniform(states, active)
        return states, tempered_pick(logits[:, step, :, voice], temperature, u)

    classes = given
    for taken in gibbs_passes(
        states, given, passes=walk, forward=forward, redraw=redraw
    ):
        classes, states = taken.redrawn, taken.states
    return classes, states


def audition_path(path, at, count):
    """the file one sheet writes: the caller's name for one sheet, and that name numbered
    for several -- a batch is a set of whole pieces"""
    if count == 1:
        return path
    name = Path(path)
    return str(name.with_name(f"{name.stem}-{at}{name.suffix}"))


def draw(coconet, *, crop, seeds, walk, temperature, quantized):
    """one batch of sheets, and the seconds the walk cost.

    [quantized] draws the INTEGER twin of the circuit: the piece the board plays at
    this seed. The two walks open on different generators -- the float walk folds its
    seed and the twin takes it as the SEED cell does -- and a seed inside 32 bits names
    itself under both. SEED 0 IS THE EXCEPTION, where the twin stands still."""
    if quantized:
        twin = integer.Coconet.from_float(coconet, temperature)
        states, given = model.opening_sheet(q.engine_states(seeds), crop)

        def walked():
            return integer.gibbs(twin, states, given, walk=walk)[0]
    else:
        states, given = model.opening_sheet(prng.states(seeds), crop)

        def walked():
            return gibbs(coconet, given, states, walk=walk, temperature=temperature)[0]

    started = time.perf_counter()
    return walked(), time.perf_counter() - started


@click.group(help=__doc__)
def main():
    pass


@main.command(help=gibbs.__doc__)
@cli.ckpt_option
@click.option("--corpus", "corpus_path", default=str(corpus.PIECES))
@click.option("--split", default="valid", type=click.Choice(corpus.SPLITS))
@click.option("--crop", default=model.CROP, help="T; the training crop")
@click.option(
    "--seeds",
    default="1",
    callback=cli.parse_seeds,
    help="a list, or LOW-HIGH; each seed is one sheet, one whole piece",
)
# `integer.ELECTED_TEMPERATURE` is the one home of this era's draw and states why
@click.option("--temperature", default=integer.ELECTED_TEMPERATURE)
@click.option(
    "--quantized",
    is_flag=True,
    help="draw the integer twin of the circuit: the piece the board plays at this seed",
)
@click.option("--corpus-seed", default=1, help="the crop draw of the battery row")
@click.option(
    "--walk", default=model.CROP * model.VOICES, help="N, the paper's I times T"
)
@midi.playback_options
@click.option(
    "--gap",
    default=32,
    help="steps of silence between two sheets; 32 is two bars, 0 is none",
)
def sample(
    ckpt,
    walk,
    to_synth,
    to_file,
    device,
    step_ms,
    gap,
    channel,
    velocity,
    corpus_path,
    split,
    corpus_seed,
    crop,
    seeds,
    temperature,
    quantized,
):
    coconet = model.Coconet.load(ckpt)
    reference = referee.corpus_sheets(corpus_path, split, crop, corpus_seed)
    classes, seconds = draw(
        coconet,
        crop=crop,
        seeds=seeds,
        walk=walk,
        temperature=temperature,
        quantized=quantized,
    )
    referee.echo_battery("the corpus", reference)
    referee.echo_battery(f"N {walk}, {len(classes)} sheets", classes)
    click.echo(f"# {seconds:.1f} s, {walk} passes of {len(classes)} sheets")

    # a sheet is a whole piece, thus several of them are several pieces: the synth hears
    # them in turn and the disk takes one file for each
    music = [corpus.decode(drawn) for drawn in classes]
    for at, piece in enumerate(music):
        if len(music) > 1:
            click.echo(f"# sheet {at}")
        if to_file:
            path = audition_path(to_file, at, len(music))
            midi.save(
                piece,
                path,
                step_ms=step_ms,
                channel=channel,
                velocity=velocity,
            )
            click.echo(f"wrote {path}")
        if to_synth:
            if at:
                midi.rest(gap, step_ms=step_ms)
            midi.play(
                piece,
                device=device,
                step_ms=step_ms,
                channel=channel,
                velocity=velocity,
            )
        if not (to_synth or to_file):
            click.echo(
                "\n".join(
                    midi.step_line(step, events) for step, events in enumerate(piece)
                )
            )


@main.command()
@cli.ckpt_option
@click.option("--out", required=True, type=click.Path(dir_okay=False))
@click.option("--temperature", default=integer.ELECTED_TEMPERATURE)
def quantize(ckpt, out, temperature):
    """Write the contract file of one checkpoint: the quantized model, and nothing else.

    It is the only thing that crosses the seam for a build. The population statistics and
    the float scales do not travel: the fold happens here, one time."""
    coconet = model.Coconet.load(ckpt)
    twin = integer.Coconet.from_float(coconet, temperature)
    integer.save(out, twin)
    layers = twin.layers()
    widths = " ".join(f"{layer.inputs}->{layer.outputs}" for layer in layers)
    click.echo(f"wrote {out}: {len(layers)} layers, {widths}")
    click.echo(
        f"temper {twin.temper.q_value} at Q{twin.temper.q}, "
        f"temperature {temperature}"
    )


if __name__ == "__main__":
    main()
