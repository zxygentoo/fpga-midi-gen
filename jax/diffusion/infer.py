"""The audition of the masked canvas: independent blocked Gibbs, eight measures at a time.

    uv run python -m diffusion.infer sample --ckpt C --seeds 1-8 --play
    uv run python -m diffusion.infer sample --ckpt C --seeds 7 --walk 32

`sample` is the ear's path: draw, print the battery against the corpus row, and speak the
music to the synthesizer or to a .mid. A batch is several whole pieces and not one piece
in parts, thus --gap puts a silence between two of them on the wire, as a performer
breathes between two chorales, and --fade takes the velocity down over the last bar of
each. Neither makes a crop ARRIVE.

EVERY DRAW OF THE WALK COMES FROM THE SHARED GENERATOR, jax/prng.py, the batched twin of
the circuit's xorshift32, under the consumption order of docs/diffusion_rtl.md: one seed
names one CANVAS, its opening, its masks and its redraws, alone or in any batch. That is
what gives the era the seed handoff -- a sweep here nominates a seed, and the OCaml
reference and the board play the same piece -- and what Gate C of test_parity.py holds:
this walk and lib/diffusion's print the same step lines. --seeds names the walks, as it
names them in every era. Quality against N is the same seeds at two --walk values: the
openings agree by construction, thus no sweep command exists.

CPU is the default platform here, and deliberately: a walk is a few hundred forward passes
of one small canvas and the GPU belongs to the trainer. Pass JAX_PLATFORMS=cuda to override
it -- at N 512 and sixteen canvases the card is worth having.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
import time
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np

import data
import measure
import midi
import nn
import prng
from diffusion import measure as canvas
from diffusion import model


def opening_canvas(states, steps):
    """A canvas of random notes for each walk of the batch, each voice inside the register
    of its own seat: one uniform for each cell in the cell order -- step-major, seat-minor
    -- and the class [low + floor(u * width)] over [measure.RANGES].

    WHY THE WALK DOES NOT OPEN ON SILENCE, which is the paper's own opening. The paper
    starts on "an empty (zero everywhere) piano roll" and its roll has no silence row, thus
    an empty cell there states nothing. THIS roll holds silence as a class, so an empty cell
    states a REST with the authority of context, and the corpus rests in 0.35 percent of its
    cells; a canvas of notes needs no special first step, and four voices sounding is 99.8
    percent of the corpus. Measured 2026-08-25 over 256 canvases, the two openings are the
    same instrument, and the silent one was removed.

    The draw is over the registers and not the whole roll, because a bass at 81 and a
    soprano at 36 are further from this corpus than a rest is. The product [u * width] is
    exact on the 24-bit grid, thus the OCaml reference states the same class from the same
    seed -- the rule and the consumption are [Diffusion.opening_canvas]'s."""
    canvases = len(states)
    classes = np.zeros((canvases, steps, model.VOICES), np.int32)
    everyone = np.ones(canvases, bool)
    lows = np.array([low - data.PITCH_LOW + 1 for low, _ in measure.RANGES])
    widths = np.array([high - low + 1 for low, high in measure.RANGES])
    for step in range(steps):
        for voice in range(model.VOICES):
            states, u = prng.uniform(states, everyone)
            classes[:, step, voice] = lows[voice] + np.floor(u * widths[voice]).astype(
                np.int32
            )
    return states, classes


@jax.jit
def forward(params, stats, classes, hidden):
    """The logits of one pass over the batch. It takes the weights as ARGUMENTS and stands
    at the module level, thus its compiled form is keyed on the shapes and every walk of a
    run reuses the first compile."""
    said, _ = model.logits(params, stats, model.planes(classes, hidden))
    return said


def tempered_pick(raw, temperature, uniform):
    """The draw of one cell over the batch: Policy.draw_class of the OCaml reference, row
    for row. [raw] is [canvases, ROWS] float64.

    The era draws with no min-p floor, thus the temper is the peak alone. One `pick`
    answers for all three eras, and its docstring holds the argument that no fallback is
    needed here: the peak weighs one, thus the last running total is one or more, and the
    draw is strictly under it, thus a class always passes."""
    return nn.pick(nn.temper(raw, temperature, 0.0), uniform)


def gibbs(params, stats, given, states, *, walk, temperature):
    """Independent blocked Gibbs with the annealed schedule of Yao et al., on the shared
    generator.

    At pass n of [walk] each cell draws one uniform in the cell order and hides exactly
    when [u * 2^24] falls under [floor(anneal(n, walk) * 2^24)] -- the threshold rule of
    docs/diffusion_rtl.md, exact on the grid of the generator. One forward pass runs, and
    each hidden cell draws one uniform in the cell order and redraws through
    [tempered_pick]. The cells are not conditionally independent, which is exactly why the
    schedule anneals: a high masking probability mixes fast and resamples badly, and as it
    falls the block shrinks toward the one-variable-at-a-time chain it approximates.

    EVERY CELL OF THE CANVAS IS FREE. Nothing is given to a walk of this era, thus the
    walk carries no mask over the mask and the machine of the next round carries none
    either; conditioning returns with the whole-piece round.

    [states] holds one generator for each canvas, thus every canvas of a batch is one
    reproducible piece: the walk of seed 7 is the walk of seed 7 in any company, here, in
    the OCaml reference and on the board."""
    canvases, steps, _ = given.shape
    classes = given.copy()
    everyone = np.ones(canvases, dtype=bool)
    for step in range(walk):
        threshold = math.floor(model.anneal(step, walk) * 2**24)
        hidden = np.zeros(given.shape, dtype=bool)
        for at in range(steps):
            for voice in range(model.VOICES):
                states, u = prng.uniform(states, everyone)
                hidden[:, at, voice] = u * 2.0**24 < threshold
        said = np.asarray(
            forward(params, stats, jnp.asarray(classes), jnp.asarray(hidden)),
            dtype=np.float64,
        )
        for at in range(steps):
            for voice in range(model.VOICES):
                active = hidden[:, at, voice]
                states, u = prng.uniform(states, active)
                if active.any():
                    picked = tempered_pick(said[:, at, :, voice], temperature, u)
                    classes[active, at, voice] = picked[active]
    return classes, states


def audition_path(path, at, count):
    """The file one canvas writes: the name the caller gave when there is one canvas, and
    that name numbered when there are several.

    A batch is a set of whole pieces and not one piece in parts, thus each one takes a file
    of its own and none of them is the batch."""
    if count == 1:
        return path
    name = Path(path)
    return str(name.with_name(f"{name.stem}-{at}{name.suffix}"))


def draw(params, stats, *, crop, seeds, walk, temperature):
    """one batch of canvases, and the seconds the walk cost"""
    states, given = opening_canvas(prng.states(seeds), crop)
    started = time.perf_counter()
    classes, _ = gibbs(params, stats, given, states, walk=walk, temperature=temperature)
    return classes, time.perf_counter() - started


@click.group(help=__doc__)
def main():
    pass


@main.command(help=gibbs.__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--corpus", "corpus_path", default=canvas.CORPUS)
@click.option("--split", default="valid", type=click.Choice(data.SPLITS))
@click.option("--crop", default=model.CROP, help="T; the training crop")
@click.option(
    "--seeds",
    default="1",
    callback=midi.parse_seeds,
    help="a list, or LOW-HIGH; each seed is one canvas, one whole piece",
)
# the code release's sampler defaults to 0.99, which is not a measurable difference from
# 1.0; the flag is here because the ear may want one
@click.option("--temperature", default=1.0)
@click.option("--corpus-seed", default=1, help="the crop draw of the battery row")
@click.option(
    "--walk", default=model.CROP * model.VOICES, help="N, the paper's I times T"
)
@click.option(
    "--play", "to_synth", is_flag=True, help=f"send to the synth on {midi.DEVICE}"
)
@click.option("--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid")
@click.option("--device", default=midi.DEVICE)
@click.option("--step-ms", default=200)
@click.option(
    "--gap",
    default=32,
    help="steps of silence between two canvases; 32 is two bars, 0 is none",
)
@click.option(
    "--fade",
    default=16,
    help="steps of diminuendo at the end of a canvas; 16 is one bar, 0 is none",
)
@click.option("--channel", default=2, help="the S-1 factory default, MIDI channel 3")
@click.option("--velocity", default=100)
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
    **flags,
):
    params, stats = model.load_params(ckpt)
    corpus = canvas.corpus_canvases(corpus_path, split, flags["crop"], corpus_seed)
    classes, seconds = draw(params, stats, walk=walk, **flags)
    canvas.echo_structure("the corpus", corpus)
    canvas.echo_structure(f"N {walk}, {len(classes)} canvases", classes)
    click.echo(f"# {seconds:.1f} s, {walk} passes of {len(classes)} canvases")

    # a canvas is a whole piece, thus several of them are several pieces: the synth hears
    # them in turn and the disk takes one file for each
    music = [data.decode(drawn) for drawn in classes]
    for at, piece in enumerate(music):
        if len(music) > 1:
            click.echo(f"# canvas {at}")
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


if __name__ == "__main__":
    main()
