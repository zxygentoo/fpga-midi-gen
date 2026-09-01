"""The sampler and the player of the state-space model.

One step is one step of the recurrence: the state and the convolution taps carry forward,
the four seats are drawn in a chain from the soprano down, on the host, and the frame goes
back in. There is no window and no context flag -- what the model remembers, it remembers
in a state of fixed size.

CPU only, and deliberately: every step needs the drawn frame back on the host before the
next forward, thus the loop is latency-bound and a GPU would take the device from the
trainer. The wire side is midi.py and the decode is corpus.py's.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import click
import jax.numpy as jnp
import numpy as np
from flax import nnx

import ar_model
import cli
import corpus
import midi
import prng
from mamba import model, quantized


@nnx.jit
def float_step(held, carry, classes, phases):
    """one float step of the walk, jitted.

    IT TAKES THE MODEL AS AN ARGUMENT AT THE MODULE LEVEL, thus its compiled form is keyed
    on the shapes and every walk of the process reuses the first compile; an `nnx.jit`
    built inside `draw` would be a new callable at every call."""
    return held.forward_step(carry, classes, phases)


def draw(held, *, seeds, steps, temperature, min_p, ring=model.ATTN_CONTEXT, twin=False):
    """One batched run: [len(seeds)] independent walks of [steps] steps each.

    [ring] is the depth of the attention layer's keys and values, and it exists only where
    the plan attends at all. Training attends over the WHOLE window, thus a shorter ring
    is a truncation -- at span 4 the slowest head weighs e^-8 at distance 128 and a ring
    of 256 reads the same as one of 512, measured; at a longer span it would not.

    The boot is a lead-in of silence: one bar of silent frames, then the draw. The state
    opens at zero, where a training window opens. The lead-in counts inside [steps].

    [twin] draws the INTEGER twin of the circuit: the piece the board plays at this seed.
    The two walks open on different generators -- the float walk folds its seed and the
    twin takes it as the SEED cell does -- and a seed inside 32 bits names itself under
    both. SEED 0 IS THE EXCEPTION, where the twin stands still."""
    if twin:
        engine = quantized.Mamba.from_float(
            held, ring=ring, temperature=temperature, min_p=min_p
        )
        return quantized.walk(engine, seeds, steps)[0]
    batch = len(seeds)
    rng = prng.states(seeds)
    carry = held.initial_carry(batch, context=ring)
    lead = corpus.BAR_STEPS
    silence = np.zeros((batch, corpus.SEATS), dtype=np.int32)
    played = []
    h = None
    for step in range(steps):
        # through the lead-in nothing is drawn and the generator does not move, as the
        # integer twin and the circuit leave it standing
        if step < lead:
            frame = silence
        else:
            rng, frame = held.head.draw_frame(h, rng, temperature, min_p)
        played.append(frame)
        phases = np.full(batch, step % ar_model.PHASE_BUCKETS, dtype=np.int32)
        carry, stream = float_step(
            held, carry, jnp.asarray(frame), jnp.asarray(phases)
        )
        h = np.asarray(stream).astype(np.float64)
    return np.stack(played, axis=1)


@click.group(help=__doc__)
def main():
    pass


@main.command(help=draw.__doc__)
@cli.sampler_options
@click.option(
    "--ring",
    default=model.ATTN_CONTEXT,
    help="the depth of the attention layer's key and value ring, in steps. It is the "
    "one context this model has, and only where the plan attends at all.",
)
@midi.playback_options
def sample(ckpt, seeds, ring, **flags):
    walks = draw(
        model.Mamba.load(ckpt),
        seeds=seeds,
        ring=ring,
        steps=flags.pop("steps"),
        temperature=flags.pop("temperature"),
        min_p=flags.pop("min_p"),
        twin=flags.pop("twin"),
    )
    midi.audition([corpus.decode(walk) for walk in walks], seeds, **flags)


@main.command()
@cli.ckpt_option
@click.option("--out", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--ring",
    default=quantized.ELECTED_RING,
    help="the depth of the attention layer's key and value ring, in steps. It is a "
    "choice of the INFERENCE and no fact of the training run, thus the file carries it.",
)
@click.option("--temperature", default=quantized.ELECTED_TEMPERATURE)
@click.option("--min-p", default=quantized.ELECTED_MIN_P)
def quantize(ckpt, out, ring, temperature, min_p):
    """Write the contract file of one checkpoint: the quantized model, and nothing else.

    Every width and the plan come out of the checkpoint's own shapes; the ring is the one
    number no training run states."""
    twin = quantized.Mamba.from_float(
        model.Mamba.load(ckpt), ring=ring, temperature=temperature, min_p=min_p
    )
    quantized.save(out, twin)
    plan = "".join(quantized.LETTERS[kind] for kind in twin.plan)
    click.echo(
        f"wrote {out}: d {twin.d}, plan {plan}, {twin.heads} heads, "
        f"span {twin.span}, ring {twin.ring}"
    )
    click.echo(
        f"temper {twin.temper.q_value} at Q{twin.temper.q}, "
        f"temperature {twin.temper.temperature}, min weight {twin.min_weight}"
    )


if __name__ == "__main__":
    main()
