"""What the masked canvas of docs/diffusion.md measures with its OWN model: the paper's
Algorithm 1, and the tail of it.

    uv run python -m diffusion.measure nll     --ckpt ../_train/diffusion/NAME.ckpt
    uv run python -m diffusion.measure corpus

The structure battery is NOT here. It is arithmetic over a stack of class indices and it
knows nothing of a canvas or a mask, thus it stands in the common home, jax/measure.py,
where every era reads it. What stands here needs [model.logits] and the mask planes.

THE LIKELIHOOD is the paper's Algorithm 1, and it is the one number of this round that
compares outside the repository: Table 1 of arXiv 1903.07227 reads **0.57 +- 0.01** nats
for each frame on the sixteenth grid of this corpus lineage. The protocol is pinned to the
paper and to its code release, because a referee that computes a different number reads
nothing. What that means in practice is stated on [framewise_lls]: the ordering is over
FRAMES, the model conditions on the ground truth of the frames before it and on its own
predictions inside the frame it is writing, and five orderings are averaged in probability
space and not in log space.

THE CORPUS ROW IS THE REFEREE OF EVERY NUMBER and NOTHING HERE RANKS A MODEL, which are
the standing rules of jax/measure.py and are earned ten times over in this project.

Nothing here draws a canvas: diffusion/infer.py draws and calls in here.
"""

import time

import click
import jax
import jax.numpy as jnp
import numpy as np

import data
import measure
import nn
from diffusion import model

JAX_ROOT = nn.JAX_ROOT
CORPUS = str(JAX_ROOT / "_data" / "pieces.safetensors")


def corpus_canvases(corpus_path, split, crop, seed):
    """one crop of every piece of a split that holds one, at a fixed seed: the rows the
    corpus row and the likelihood referee both read"""
    return data.Crops(data.load_pieces(corpus_path)[split], crop).every_piece(seed)


# ==================================================================== #
# THE LIKELIHOOD — the paper's Algorithm 1                             #
# ==================================================================== #

# Table 1 of the paper on this corpus lineage, sixteenth grid, random orderings. It is the
# anchor of the round and never a target to optimise against.
PAPER_NLL = 0.57
# the paper's M: "averaging likelihoods across an ensemble of M = 5 orderings"
ORDERINGS = 5


def frame_ordering(rng, steps):
    """One ordering of Algorithm 1: a permutation of the frames, and a permutation of the
    voices inside each frame.

    The paper's random ordering is over FRAMES and not over all D variables -- that is the
    difference between its framewise measurement and a notewise one. Its chronological
    variant keeps the frames in time order and shuffles only the voices; this round reports
    the random one, which is the row of Table 1 that reads 0.57."""
    return rng.permutation(steps), np.stack(
        [rng.permutation(model.VOICES) for _ in range(steps)]
    )


def forward_in_chunks(forward, classes, hidden, chunk):
    """the log probabilities of a stack of canvases, [chunk] at a time: one canvas of the
    stack is one frame of the piece, thus the stack is as tall as the crop and a 12 GB card
    wants it cut"""
    return np.concatenate(
        [
            np.asarray(
                forward(jnp.asarray(classes[at : at + chunk]), hidden[at : at + chunk])
            )
            for at in range(0, len(classes), chunk)
        ]
    )


def framewise_lls(forward, classes, ordering, chunk):
    """The log-likelihood of every frame of one canvas under one ordering: Algorithm 1.

    THE FRAMES ARE INDEPENDENT GIVEN THE ORDERING, and that is the whole reason this referee
    is affordable. Algorithm 1 restores the ground truth of a frame the moment it finishes
    writing it, thus frame l conditions on the TRUE frames that stand before it in the
    ordering and on nothing the model wrote outside itself. Therefore the T frames run as
    one stack and the walk costs I forward passes and not I times T. The code release does
    exactly this, and this function is its shape.

    Inside a frame the model does condition on itself: voice k reads what the model put in
    the k - 1 voices before it. That is what makes this framewise and not notewise -- the
    frame is the unit of prediction and an error inside one accumulates, which the paper
    states is the point.

    The model's own value is written as the ARGMAX and not as a draw. The paper's Algorithm
    1 samples there; its code release takes the argmax, and the code release is what
    produced 0.57."""
    frames, voices = ordering
    steps = len(classes)
    # canvas l of the stack reveals the frames that stand before position l in the ordering
    states = np.tile(classes, (steps, 1, 1))
    hidden = np.zeros((steps, steps, model.VOICES), dtype=bool)
    for at in range(steps):
        hidden[at, frames[at:], :] = True
    row = np.arange(steps)
    lls = np.zeros(steps, dtype=np.float64)
    for turn in range(model.VOICES):
        said = forward_in_chunks(forward, states, hidden, chunk)
        # the frame each canvas of the stack is writing, and the voice of it whose turn
        # this is: one cell for each canvas of the stack, thus one distribution over the
        # pitch rows for each
        voice = voices[frames, turn]
        logp = said[row, frames, :, voice]
        lls[frames] += logp[row, classes[frames, voice]]
        states[row, frames, voice] = np.argmax(logp, axis=-1)
        hidden[row, frames, voice] = False
    return lls


def piece_nll(forward, classes, rng, orderings, chunk):
    """Algorithm 1 for one canvas, frame by frame: the nats of every frame of it.

    The caller means these, which is Algorithm 1's return, AND keeps them. A mean cannot
    see a rare bad moment and the ear can, thus the frames are the tail and the tail is a
    measurement of its own.

    The orderings are combined IN PROBABILITY SPACE, one frame at a time -- logsumexp over
    the ensemble, less the log of its size. A mean of log-likelihoods would be an
    unnormalised geometric mean and would waste probability mass, and the paper's line is
    explicit about it."""
    lls = np.stack(
        [
            framewise_lls(forward, classes, frame_ordering(rng, len(classes)), chunk)
            for _ in range(orderings)
        ]
    )
    return -(np.logaddexp.reduce(lls, axis=0) - np.log(len(lls)))


def framewise_nll(params, stats, canvases, *, orderings, chunk, seed, report=None):
    """The referee over a set of canvases: Algorithm 1's mean nats for each frame, its
    standard error, and the frames themselves.

    The standard error is over the PIECES, which is what the paper's Table 1 reports beside
    its means. The frames are kept for [tail_line]."""
    # the log softmax runs on the device beside the trunk: the referee then indexes a
    # probability and never normalises one, and the argmax of the two is the same cell
    forward = jax.jit(
        lambda classes, hidden: jax.nn.log_softmax(
            model.logits(params, stats, model.planes(classes, hidden))[0], axis=-2
        )
    )
    rng = np.random.default_rng(seed)
    frames = []
    for at, canvas in enumerate(canvases):
        frames.append(piece_nll(forward, canvas, rng, orderings, chunk))
        if report is not None:
            report(at, float(frames[-1].mean()))
    pieces = np.asarray([piece.mean() for piece in frames])
    return {
        "mean": float(pieces.mean()),
        "error": float(pieces.std() / np.sqrt(len(pieces))),
        "pieces": pieces,
        # [pieces, frames] and not one flat run of them: the tail resamples PIECES, thus
        # its error stands beside the mean's and reads against the same population
        "frames": np.stack(frames),
    }


# nats for one frame above which the ear would call it a wrong moment. A frame is four
# voices, thus 2 nats is a joint probability of 0.14 for the whole sonority.
LOUD = 2.0
# resamples of the pieces behind each percentile. A percentile carries no standard error of
# its own, and two models an eighth of a nat apart cannot be told from each other without
# one -- the round has already been caught by that once, on the parallels.
RESAMPLES = 1000
MARKS = (50, 90, 99)


def tail_shape(frames, seed=0):
    """The tail of the framewise nats: the percentiles of [MARKS], the share of frames over
    [LOUD], and a bootstrap error for each of them.

    [frames] is [pieces, frames]. THE RESAMPLE IS OVER PIECES and not over frames, for the
    reason the mean's own error is: the frames of one chorale are one draw of a composer
    and not 128 of them, thus resampling frames would state an error several times too
    small and every model would separate from every other."""
    rng = np.random.default_rng(seed)
    draws = frames[rng.integers(len(frames), size=(RESAMPLES, len(frames)))]
    draws = draws.reshape(RESAMPLES, -1)
    marks = np.percentile(frames, MARKS)
    loud = 100.0 * np.mean(frames >= LOUD)
    return {
        "marks": marks,
        "mark errors": np.percentile(draws, MARKS, axis=-1).std(axis=-1),
        "loud": loud,
        "loud error": float((100.0 * np.mean(draws >= LOUD, axis=-1)).std()),
    }


def tail_line(frames, seed=0):
    """THE RARE BAD MOMENT, which the mean of [framewise_nll] cannot see.

    One strange chord in a phrase is heard, and it moves the average of 128 frames by
    nothing at all. This is the instrument the ear asked for on 2026-08-25, after it heard
    the ceiling iron out a weirdness that cost the mean 0.008 nats.

    A model with a shorter tail at the same mean is a model that is wrong less often and
    not less badly, which is the trade the ear elects.

    READ IT AGAINST WHAT IT MEASURES. These are corpus canvases, thus a frame of high nats
    is a frame where BACH surprised the model, and not one where the model wrote something
    strange. The two are not the same question, and the second one is [structure]'s clash,
    which reads the model's own draws."""
    read = tail_shape(frames, seed)
    marks = "   ".join(
        f"{name} {value:5.3f} +- {error:.3f}"
        for name, value, error in zip(
            ("median", "90th", "99th"), read["marks"], read["mark errors"]
        )
    )
    return (
        f"{'the tail':<22} {marks}   above {LOUD:.0f} nats "
        f"{read['loud']:4.1f} +- {read['loud error']:.1f}%"
    )


# ==================================================================== #
# The commands                                                         #
# ==================================================================== #


@click.group(help=__doc__)
def main():
    pass


@main.command(help=measure.structure.__doc__)
@click.option("--corpus", "corpus_path", default=CORPUS)
@click.option("--split", default="train", type=click.Choice(data.SPLITS))
@click.option("--crop", default=model.CROP)
@click.option("--seed", default=0, help="the crop draw; fixed, thus the row is fixed")
def corpus(corpus_path, split, crop, seed):
    canvases = corpus_canvases(corpus_path, split, crop, seed)
    for line in measure.structure_lines(
        f"the corpus, {split}", measure.structure(canvases)
    ):
        click.echo(line)


@main.command(help=framewise_nll.__doc__)
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--corpus", "corpus_path", default=CORPUS)
@click.option("--split", default="test", type=click.Choice(data.SPLITS))
@click.option("--crop", default=model.CROP)
@click.option("--orderings", default=ORDERINGS, help="the paper's M")
@click.option(
    "--pieces", default=0, help="how many pieces of the split; 0 is all of them"
)
@click.option("--chunk", default=16, help="canvases of one forward pass")
@click.option("--seed", default=0)
def nll(ckpt, corpus_path, split, crop, orderings, pieces, chunk, seed):
    canvases = corpus_canvases(corpus_path, split, crop, seed)
    if pieces:
        canvases = canvases[:pieces]
    params, stats = model.load_params(ckpt)
    started = time.perf_counter()

    def report(at, value):
        done = time.perf_counter() - started
        click.echo(
            f"piece {at + 1:3d} of {len(canvases)}  {value:6.4f}  "
            f"{done / (at + 1):5.1f} s each"
        )

    read = framewise_nll(
        params,
        stats,
        canvases,
        orderings=orderings,
        chunk=chunk,
        seed=seed,
        report=report,
    )
    click.echo(
        f"framewise NLL on {split}, {len(canvases)} pieces, {orderings} orderings: "
        f"{read['mean']:.4f} +- {read['error']:.4f} nats for each frame"
    )
    click.echo(tail_line(read["frames"]))
    click.echo(f"the paper's Table 1 on the sixteenth grid: {PAPER_NLL:.2f} +- 0.01")


if __name__ == "__main__":
    main()
