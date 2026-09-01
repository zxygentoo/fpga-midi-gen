"""What the masked sheet of docs/diffusion.md measures with its OWN model: the paper's
Algorithm 1, and the tail of it.

    uv run python -m diffusion.measure nll     --ckpt ../_train/diffusion/NAME.ckpt
    uv run python -m diffusion.measure corpus

The common battery is NOT here -- it knows nothing of a sheet or a mask, thus it stands in
jax/measure.py. What stands here needs the model itself and the mask planes, and nothing
here draws a sheet: diffusion/infer.py draws and calls in.

THE LIKELIHOOD is the paper's Algorithm 1, the one number of this round that compares
outside the repository: Table 1 of arXiv 1903.07227 reads **0.57 +- 0.01** nats for each
frame on the sixteenth grid of this corpus lineage. The protocol is pinned to the paper
and to its code release, because a referee that computes a different number reads nothing.
"""

import time
from functools import partial

import click
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import cli
import corpus
import measure
from diffusion import model


def corpus_sheets(corpus_path, split, crop, seed):
    """one crop of every piece of a split that holds one, at a fixed seed: the rows the
    corpus row and the likelihood referee both read"""
    return corpus.Crops(corpus.load_pieces(corpus_path)[split], crop).pieces(seed)


def echo_battery(label, sheets):
    """one row of the battery of jax/measure.py, printed under [label]"""
    for line in measure.battery_lines(label, measure.battery_row(sheets)):
        click.echo(line)


# THE LIKELIHOOD — the paper's Algorithm 1


# Table 1 of the paper on this corpus lineage, sixteenth grid, random orderings. It is the
# anchor of the round and never a target to optimise against.
PAPER_NLL = 0.57
# the paper's M: "averaging likelihoods across an ensemble of M = 5 orderings"
ORDERINGS = 5


def frame_ordering(rng, steps):
    """One ordering of Algorithm 1: a permutation of the frames, and a permutation of
    the voices inside each frame. The paper's random ordering is over FRAMES and not
    over all D variables, which is what makes its measurement framewise and not
    notewise."""
    return rng.permutation(steps), np.stack(
        [rng.permutation(model.VOICES) for _ in range(steps)]
    )


def forward_in_chunks(forward, classes, hidden, chunk):
    """the log probabilities of a stack of sheets, [chunk] at a time: the stack is as tall
    as the crop and a 12 GB card wants it cut. The chunks cross back to the host ONE TIME,
    because a read of a chunk blocks the dispatch of the next."""
    said = [
        forward(jnp.asarray(classes[at : at + chunk]), hidden[at : at + chunk])
        for at in range(0, len(classes), chunk)
    ]
    return np.asarray(jnp.concatenate(said))


def framewise_lls(forward, classes, ordering, chunk):
    """The log-likelihood of every frame of one sheet under one ordering: Algorithm 1.

    THE FRAMES ARE INDEPENDENT GIVEN THE ORDERING, and that is why this referee is
    affordable: Algorithm 1 restores the ground truth of a frame the moment it finishes
    writing it, thus the T frames run as one stack and the walk costs I forward passes and
    not I times T. Inside a frame the model does condition on itself, which is what makes
    this framewise. Its own value is written as the ARGMAX and not as a draw, because the
    code release takes the argmax and the release produced 0.57."""
    frames, voices = ordering
    steps = len(classes)
    # sheet l of the stack reveals the frames that stand before position l in the ordering
    states = np.tile(classes, (steps, 1, 1))
    hidden = np.zeros((steps, steps, model.VOICES), dtype=bool)
    for at in range(steps):
        hidden[at, frames[at:], :] = True
    row = np.arange(steps)
    lls = np.zeros(steps, dtype=np.float64)
    for turn in range(model.VOICES):
        said = forward_in_chunks(forward, states, hidden, chunk)
        # the frame each sheet of the stack is writing, and the voice of it whose turn
        # this is: one cell for each sheet of the stack, thus one distribution over the
        # pitch rows for each
        voice = voices[frames, turn]
        logp = said[row, frames, :, voice]
        lls[frames] += logp[row, classes[frames, voice]]
        states[row, frames, voice] = np.argmax(logp, axis=-1)
        hidden[row, frames, voice] = False
    return lls


def piece_nll(forward, classes, rng, orderings, chunk):
    """Algorithm 1 for one sheet, frame by frame: the nats of every frame of it. The
    caller means these AND keeps them, because a mean cannot see a rare bad moment. The
    orderings combine IN PROBABILITY SPACE -- logsumexp less the log of the ensemble size
    -- where a mean of log-likelihoods would waste probability mass."""
    lls = np.stack(
        [
            framewise_lls(forward, classes, frame_ordering(rng, len(classes)), chunk)
            for _ in range(orderings)
        ]
    )
    return -(np.logaddexp.reduce(lls, axis=0) - np.log(len(lls)))


@nnx.jit
def log_probabilities(coconet, classes, hidden):
    """the log probability of every pitch row of every cell, over a stack of sheets;
    the log softmax runs on the device beside the trunk, thus the referee indexes a
    probability and never normalises one"""
    said, _ = coconet(model.planes(classes, hidden))
    return jax.nn.log_softmax(said, axis=-2)


def framewise_nll(coconet, sheets, *, orderings, chunk, seed, report=None):
    """The referee over a set of sheets: Algorithm 1's mean nats for each frame, its
    standard error over the PIECES as Table 1 reports it, and the frames themselves."""
    forward = partial(log_probabilities, coconet)
    rng = np.random.default_rng(seed)
    frames = []
    for at, sheet in enumerate(sheets):
        frames.append(piece_nll(forward, sheet, rng, orderings, chunk))
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
# resamples of the pieces behind each percentile. A percentile carries no standard error
# of its own, and two models an eighth of a nat apart cannot be told from each other
# without one -- the round has already been caught by that once, on the parallels.
RESAMPLES = 1000
MARKS = (50, 90, 99)


def tail_row(frames, seed=0):
    """The tail of the framewise nats: the percentiles of [MARKS], the share of frames
    over [LOUD], and a bootstrap error for each. THE RESAMPLE IS OVER PIECES: the
    frames of one chorale are one draw of a composer and not 128, thus resampling
    frames would state an error several times too small."""
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
    """THE RARE BAD MOMENT, which the mean of [framewise_nll] cannot see: one strange
    chord in a phrase is heard and moves the average of 128 frames by nothing.

    READ IT AGAINST WHAT IT MEASURES. These are CORPUS sheets, thus a frame of high
    nats is one where Bach surprised the model and not one where the model wrote
    something strange; that second question is [battery_row]'s clash."""
    read = tail_row(frames, seed)
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


# The commands


@click.group(help=__doc__)
def main():
    pass


@main.command("corpus", help=measure.battery_row.__doc__)
@click.option("--corpus", "corpus_path", default=str(corpus.PIECES))
@click.option("--split", default="train", type=click.Choice(corpus.SPLITS))
@click.option("--crop", default=model.CROP)
@click.option("--seed", default=0, help="the crop draw; fixed, thus the row is fixed")
def corpus_battery(corpus_path, split, crop, seed):
    echo_battery(f"the corpus, {split}", corpus_sheets(corpus_path, split, crop, seed))


@main.command(help=framewise_nll.__doc__)
@cli.ckpt_option
@click.option("--corpus", "corpus_path", default=str(corpus.PIECES))
@click.option("--split", default="test", type=click.Choice(corpus.SPLITS))
@click.option("--crop", default=model.CROP)
@click.option("--orderings", default=ORDERINGS, help="the paper's M")
@click.option(
    "--pieces", default=0, help="how many pieces of the split; 0 is all of them"
)
@click.option("--chunk", default=16, help="sheets of one forward pass")
@click.option("--seed", default=0)
def nll(ckpt, corpus_path, split, crop, orderings, pieces, chunk, seed):
    sheets = corpus_sheets(corpus_path, split, crop, seed)
    if pieces:
        sheets = sheets[:pieces]
    coconet = model.Coconet.load(ckpt)
    started = time.perf_counter()

    def report(at, value):
        done = time.perf_counter() - started
        click.echo(
            f"piece {at + 1:3d} of {len(sheets)}  {value:6.4f}  "
            f"{done / (at + 1):5.1f} s each"
        )

    read = framewise_nll(
        coconet,
        sheets,
        orderings=orderings,
        chunk=chunk,
        seed=seed,
        report=report,
    )
    click.echo(
        f"framewise NLL on {split}, {len(sheets)} pieces, {orderings} orderings: "
        f"{read['mean']:.4f} +- {read['error']:.4f} nats for each frame"
    )
    click.echo(tail_line(read["frames"]))
    click.echo(f"the paper's Table 1 on the sixteenth grid: {PAPER_NLL:.2f} +- 0.01")


if __name__ == "__main__":
    main()
