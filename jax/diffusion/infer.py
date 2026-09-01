"""The audition of the masked sheet: independent blocked Gibbs, eight measures at a time.

    uv run python -m diffusion.infer sample --ckpt C --seeds 1-8 --play
    uv run python -m diffusion.infer sample --ckpt C --seeds 7 --walk 32

`sample` is the ear's path: draw, print the battery against the corpus row, and speak the
music to the synthesizer or to a .mid. A batch is several whole pieces and not one piece
in parts, thus --gap puts a silence between two of them and --fade takes the velocity down
over the last bar of each. Neither makes a crop ARRIVE.

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
from diffusion import measure as referee
from diffusion import model, quantized
from diffusion.sample import gibbs_passes, tempered_pick
from quantized import engine_states


def gibbs(coconet, given, states, *, walk, temperature):
    """The FLOAT walk of the era: the `gibbs_passes` of `diffusion/sample.py`, in float64
    over the trained model. The loop, the schedule and the order of the draws stand there
    once for both walks; what is here is this walk's arithmetic alone. It gives the sheets
    and the generator behind them, as `quantized.gibbs` does."""

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


def draw(coconet, *, crop, seeds, walk, temperature, twin):
    """one batch of sheets, and the seconds the walk cost.

    [twin] draws the INTEGER twin of the circuit: the piece the board plays at this seed.
    The two walks open on different generators -- the float walk folds its seed and the
    twin takes it as the SEED cell does -- and a seed inside 32 bits names itself under
    both. SEED 0 IS THE EXCEPTION, where the twin stands still."""
    if twin:
        engine = quantized.Coconet.from_float(coconet, temperature)
        states, given = model.opening_sheet(engine_states(seeds), crop)

        def walked():
            return quantized.gibbs(engine, states, given, walk=walk)[0]
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
# `quantized.ELECTED_TEMPERATURE` is the one home of this era's draw and states why
@click.option("--temperature", default=quantized.ELECTED_TEMPERATURE)
@click.option(
    "--quantized",
    "twin",
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
@click.option(
    "--fade",
    default=16,
    help="steps of diminuendo at the end of a sheet; 16 is one bar, 0 is none",
)
def sample(
    ckpt,
    walk,
    to_synth,
    to_file,
    device,
    step_ms,
    gap,
    fade,
    channel,
    velocity,
    corpus_path,
    split,
    corpus_seed,
    crop,
    seeds,
    temperature,
    twin,
):
    coconet = model.Coconet.load(ckpt)
    reference = referee.corpus_sheets(corpus_path, split, crop, corpus_seed)
    classes, seconds = draw(
        coconet,
        crop=crop,
        seeds=seeds,
        walk=walk,
        temperature=temperature,
        twin=twin,
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
                fade=fade,
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
                fade=fade,
            )
        if not (to_synth or to_file):
            click.echo(
                "\n".join(
                    midi.step_line(step, events) for step, events in enumerate(piece)
                )
            )


@main.command()
@cli.ckpt_option
@click.option("--crop", default=model.CROP, help="T; the steps of the sheet")
@click.option("--seed", default=42, help="N, the seed of the walk")
@click.option("--walk", default=32, help="N, the Gibbs passes to compare")
@click.option("--temperature", default=quantized.ELECTED_TEMPERATURE)
def drift(ckpt, crop, seed, walk, temperature):
    """What the quantization costs, measured on the walk the board takes: at every pass
    the float model is teacher-forced on the ENGINE'S sheet and mask, thus what stands
    between the two is the arithmetic alone."""
    coconet = model.Coconet.load(ckpt)
    states, given = model.opening_sheet(engine_states([seed]), crop)
    said = quantized.drift(coconet, states, given, walk=walk, temperature=temperature)
    seen = said.cells

    def share(count):
        return 100.0 * count / max(1, seen)

    click.echo(f"{said.passes} passes over {crop} steps redrew {seen} cells")
    click.echo(
        f"against the float model: top-1 {share(said.same_peak):.1f}% "
        f"({said.same_peak}/{seen})  cosine {said.mean_cosine:.4f}  "
        f"same draw {share(said.same_draw):.1f}% ({said.same_draw}/{seen})"
    )
    click.echo(
        f"activations on the clamp: {100.0 * said.activations_clamped:.4f}%  "
        f"the hottest write: {said.activation_peak:.1f} of the format's "
        f"{quantized.ACTIVATION_CEILING:.1f}"
    )


@main.command()
@cli.ckpt_option
@click.option("--out", required=True, type=click.Path(dir_okay=False))
@click.option("--temperature", default=quantized.ELECTED_TEMPERATURE)
def quantize(ckpt, out, temperature):
    """Write the contract file of one checkpoint: the quantized model, and nothing else.

    It is the only thing that crosses the seam for a build. The population statistics and
    the float scales do not travel: the fold happens here, one time."""
    coconet = model.Coconet.load(ckpt)
    twin = quantized.Coconet.from_float(coconet, temperature)
    quantized.save(out, twin)
    layers = twin.layers()
    widths = " ".join(f"{layer.inputs}->{layer.outputs}" for layer in layers)
    click.echo(f"wrote {out}: {len(layers)} layers, {widths}")
    click.echo(
        f"temper {twin.temper.q_value} at Q{twin.temper.q}, "
        f"temperature {twin.temper.temperature}"
    )


if __name__ == "__main__":
    main()
