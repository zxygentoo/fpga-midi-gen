"""The audition of the masked canvas: independent blocked Gibbs, eight measures at a time.

    uv run python -m coconet.infer sample --ckpt C --canvases 8 --seed 7 --play
    uv run python -m coconet.infer sample --ckpt C --harmonize --canvases 4
    uv run python -m coconet.infer curve  --ckpt C

THE CURVE IS THE DELIVERABLE OF THE ROUND. `curve` draws the same canvases at N of 32, 64,
128, 256 and 512 and prints the structure battery of each against the corpus row, with the
seconds each one cost. The paper's rule of thumb is N = I times T = 512 evaluations and it
states that a lower N costs a little quality; the board's silence window affords tens of
canvas passes, thus this one curve decides whether the masked era reaches the RTL.

`sample` is the ear's path: draw, print the battery, and speak the music to the synthesizer
or to a .mid. A batch is several whole pieces and not one piece in parts, thus --gap puts a
silence between two of them on the wire, as a performer breathes between two chorales. With --harmonize the walk keeps the soprano of a corpus crop and writes the
three voices under it, which is the completion task the trunk is strongest in and which the
mask planes give for one flag.

CPU is the default platform here, and deliberately: a walk is a few hundred forward passes
of one small canvas and the GPU belongs to the trainer. Pass JAX_PLATFORMS=cuda to override
it -- at N 512 and sixteen canvases the card is worth having.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np

import data
import measure
import midi
from coconet import measure as canvas
from coconet import model

# seat 3, as data.py and the chained head of the earlier eras read the seats
SOPRANO = model.VOICES - 1
CURVE = "32,64,128,256,512"


def free_cells(canvases, steps, harmonize):
    """The region the walk may write: the whole canvas, or everything under a given
    soprano.

    Harmonization needs no second model and no second training. The mask planes already say
    which cells are given, thus the task is one flag on the sampler -- which is the whole
    argument for a masked canvas over a one-way chain."""
    free = np.ones((canvases, steps, model.VOICES), dtype=bool)
    if harmonize:
        free[..., SOPRANO] = False
    return free


def gibbs(params, stats, given, free, *, walk, temperature, seed):
    """Independent blocked Gibbs with the annealed schedule of Yao et al.

    At step n of [walk] each free cell masks with probability model.anneal(n, walk), one
    forward pass runs, and every masked cell resamples INDEPENDENTLY from its own softmax.
    The cells are not conditionally independent, which is exactly why the schedule anneals:
    a high masking probability mixes fast and resamples badly, and as it falls the block
    shrinks toward the one-variable-at-a-time chain it approximates.

    The seed names the whole batch and not a row of it. Every canvas of a batch shares the
    walk and draws its own masks, thus a batch of one is one reproducible piece and a batch
    of sixteen is one reproducible set of sixteen."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def turn(key, classes, hidden, heat):
        said, _ = model.logits(params, stats, model.planes(classes, hidden))
        drawn = jax.random.categorical(key, said / heat, axis=-2)
        return jnp.where(hidden, drawn, classes)

    classes = jnp.asarray(given)
    for step in range(walk):
        # THE WALK OPENS BY MASKING THE WHOLE FREE REGION, where the code release opens
        # with the same Bernoulli as every other step. The reason is the vocabulary and not
        # a taste: this roll holds a silence row where the paper's holds none, thus a cell
        # the opening Bernoulli happens to leave unmasked is not a blank the model reads as
        # "nothing stated here" -- it is a REST, stated with the authority of context, and
        # the corpus rests in 0.35 percent of its cells. At alpha_max 0.9 that would freeze
        # a tenth of the canvas into false rests, and at a low N most of them would never be
        # masked again to be corrected.
        masking = model.anneal(step, walk)
        hidden = free if step == 0 else free & (rng.random(free.shape) < masking)
        key, turn_key = jax.random.split(key)
        classes = turn(turn_key, classes, jnp.asarray(hidden), jnp.float32(temperature))
    return np.asarray(classes)


def audition_path(path, at, count):
    """The file one canvas writes: the name the caller gave when there is one canvas, and
    that name numbered when there are several.

    A batch is a set of whole pieces and not one piece in parts, thus each one takes a file
    of its own and none of them is the batch."""
    if count == 1:
        return path
    name = Path(path)
    return str(name.with_name(f"{name.stem}-{at}{name.suffix}"))


def opening(corpus_path, split, crop, canvases, harmonize, seed):
    """The canvas the walk opens on: silence everywhere, or the first [canvases] corpus
    crops when a soprano is given.

    A silent opening is never read as music. The whole free region is masked at step 0,
    thus the model is handed an all-masked canvas and states the prior of the corpus."""
    if not harmonize:
        return np.zeros((canvases, crop, model.VOICES), dtype=np.int32)
    return canvas.corpus_canvases(corpus_path, split, crop, seed)[:canvases].copy()


def draw(params, stats, *, corpus_path, split, crop, canvases, harmonize, walk, temperature, seed):
    """one batch of canvases, and the seconds the walk cost"""
    given = opening(corpus_path, split, crop, canvases, harmonize, seed)
    free = free_cells(len(given), crop, harmonize)
    started = time.perf_counter()
    classes = gibbs(
        params, stats, given, free, walk=walk, temperature=temperature, seed=seed
    )
    return classes, time.perf_counter() - started


@click.group(help=__doc__)
def main():
    pass


def sampling_options(command):
    """the flags every drawing command takes; the walk is the one they disagree on"""
    for option in reversed(
        [
            click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False)),
            click.option("--corpus", "corpus_path", default=canvas.CORPUS),
            click.option("--split", default="valid", type=click.Choice(data.SPLITS)),
            click.option("--crop", default=model.CROP, help="T; the training crop"),
            click.option(
                "--canvases",
                default=1,
                help="pieces to draw in one walk; --play speaks them in turn and --save "
                "numbers them",
            ),
            click.option(
                "--harmonize",
                is_flag=True,
                help="keep the soprano of a corpus crop and write the three voices under it",
            ),
            # the code release's sampler defaults to 0.99, which is not a measurable
            # difference from 1.0; the flag is here because the ear may want one
            click.option("--temperature", default=1.0),
            click.option("--seed", default=1, help="the batch, not a row of it"),
        ]
    ):
        command = option(command)
    return command


@main.command(help=gibbs.__doc__)
@sampling_options
@click.option("--walk", default=model.CROP * model.VOICES, help="N, the paper's I times T")
@click.option("--play", "to_synth", is_flag=True, help=f"send to the synth on {midi.DEVICE}")
@click.option("--save", "to_file", type=click.Path(dir_okay=False), help="write a .mid")
@click.option("--device", default=midi.DEVICE)
@click.option("--step-ms", default=200)
@click.option(
    "--gap",
    default=16,
    help="steps of silence between two canvases; 16 is one bar, 0 is none",
)
@click.option("--channel", default=2, help="the S-1 factory default, MIDI channel 3")
@click.option("--velocity", default=100)
def sample(
    ckpt, walk, to_synth, to_file, device, step_ms, gap, channel, velocity, **flags
):
    params, stats = model.load_params(ckpt)
    classes, seconds = draw(params, stats, walk=walk, **flags)
    corpus = canvas.corpus_canvases(
        flags["corpus_path"], flags["split"], flags["crop"], flags["seed"]
    )
    label = f"N {walk}, {len(classes)} canvases"
    for line in measure.structure_lines("the corpus", measure.structure(corpus)):
        click.echo(line)
    for line in measure.structure_lines(label, measure.structure(classes)):
        click.echo(line)
    click.echo(f"# {seconds:.1f} s, {walk} passes of {len(classes)} canvases")

    # a canvas is a whole piece, thus several of them are several pieces: the synth hears
    # them in turn and the disk takes one file for each
    music = [data.decode(canvas) for canvas in classes]
    for at, piece in enumerate(music):
        if len(music) > 1:
            click.echo(f"# canvas {at}")
        if to_file:
            path = audition_path(to_file, at, len(music))
            midi.save(piece, path, step_ms=step_ms, channel=channel, velocity=velocity)
            click.echo(f"wrote {path}")
        if to_synth:
            if at:
                midi.rest(gap, step_ms=step_ms)
            midi.play(piece, device=device, step_ms=step_ms, channel=channel, velocity=velocity)
        if not (to_synth or to_file):
            click.echo(
                "\n".join(midi.step_line(step, events) for step, events in enumerate(piece))
            )


@main.command()
@sampling_options
@click.option("--walks", default=CURVE, help="the N of each row, a comma list")
def curve(ckpt, walks, **flags):
    """Quality against N: the same canvases drawn at each step budget, read by the
    structure battery against the corpus row.

    The paper sets N = I times T = 512 as a rule of thumb and says a lower N costs a little
    quality. The board affords tens of passes and not hundreds, thus what this round needs
    to know is the shape of that cost and not its existence."""
    params, stats = model.load_params(ckpt)
    corpus = canvas.corpus_canvases(
        flags["corpus_path"], flags["split"], flags["crop"], flags["seed"]
    )
    for line in measure.structure_lines("the corpus", measure.structure(corpus)):
        click.echo(line)
    for walk in [int(n) for n in walks.split(",")]:
        classes, seconds = draw(params, stats, walk=walk, **flags)
        for line in measure.structure_lines(f"N {walk:4d}", measure.structure(classes)):
            click.echo(line)
        click.echo(f"{'':<22} {seconds:.1f} s for {len(classes)} canvases")


if __name__ == "__main__":
    main()
