"""The trainer of the state-space model of docs/mamba.md.

Run it as a module, from any directory of the tree:

    uv run python -m mamba.train --steps 200

The loop, the evaluation and the checkpoint policy are `ar_train.train`; this file is the
shape of the model -- the plan, the state, the taps, the dt ladder -- and the flags that
spell it. THE ERA IS FROZEN and this trainer is kept, not run. THE DRAW IS THE MODEL'S,
`Mamba.drawn`, because the gates read the same opening.
"""

import click

import ar_model
import ar_train
from mamba import model

PLAN_LETTERS = {
    "m": model.MAMBA,
    "a": model.ATTN,
    "z": model.ZATTN,
    "f": model.MLP,
}


def parse_plan(ctx, param, value):
    """The plan spelt out, one letter for each layer: M a block, A era four's attention
    sublayer, Z the Zamba one that also reads the embedding, F the feed-forward.

    "MMMMMMZF" is six blocks under a Zamba head. Given, it wins over --layers and
    --attention-at, which spell the plans of blocks and attention alone."""
    del ctx, param
    letters = [c for c in value.lower() if not c.isspace()]
    unknown = {c for c in letters} - set(PLAN_LETTERS)
    if unknown:
        raise click.BadParameter(
            f"{sorted(unknown)} are not plan letters {sorted(PLAN_LETTERS)}"
        )
    return [PLAN_LETTERS[c] for c in letters] or None


def parse_attention_at(ctx, param, value):
    """the layers that are attention and not a block, 0 first; nothing is the trunk"""
    del ctx, param
    return tuple(int(at) for at in value.split(",")) if value else ()


def parse_half_lives(ctx, param, value):
    """LOW-HIGH in steps, or nothing at all for the Mamba draw"""
    del ctx, param
    if not value:
        return None
    low, high = value.split("-")
    return float(low), float(high)


@click.command(help=__doc__)
@click.option("--d", default=64)
@click.option("--layers", default=6)
@click.option("--heads", default=4)
@click.option("--state", default=16, help="N, the state width of one head")
@click.option("--taps", default=model.CONV_TAPS, help="K, the convolution width")
@click.option(
    "--alibi-span",
    "alibi_span",
    default=ar_model.SLOPE_SPAN,
    help="the ALiBi exponent span of the attention layers: the slope of head k is "
    "2^-(span (k+1) / heads), thus a LARGER span reaches further. Era four elected 4 on "
    "a pure transformer; the file records whichever this run used.",
)
@click.option(
    "--plan",
    "spelt",
    default="",
    callback=parse_plan,
    help="the plan spelt out, one letter for each layer: M block, A attention, Z the "
    "Zamba attention that reads the embedding, F feed-forward. Wins over --layers.",
)
@click.option(
    "--attention-at",
    "attention_at",
    default="",
    callback=parse_attention_at,
    help="the layers that take era four's attention sublayer instead of a block, 0 "
    "first; empty is the trunk of six blocks",
)
@click.option("--expand", default=model.EXPAND, help="d_in = expand * d")
@click.option(
    "--dt-half-lives",
    "half_lives",
    default="",
    callback=parse_half_lives,
    help="LOW-HIGH in steps: open dt on a log-spaced half-life ladder instead of the "
    "uniform Mamba draw, one head at each rung",
)
@click.option(
    "--conv-scale",
    type=float,
    default=ar_model.DRAW_SCALE,
    help="the draw of the convolution kernel; measured against 1/sqrt(K), see "
    "Mamba.drawn",
)
@ar_train.recipe_options(dropout=0.2)
def main(
    d,
    layers,
    heads,
    state,
    taps,
    alibi_span,
    attention_at,
    spelt,
    expand,
    conv_scale,
    seed,
    half_lives,
    **flags,
):
    ar_train.train(
        model.Mamba.drawn(
            seed,
            d=d,
            layers=layers,
            heads=heads,
            state=state,
            taps=taps,
            expand=expand,
            conv_scale=conv_scale,
            half_lives=half_lives,
            attention_at=attention_at,
            spelt=spelt,
            span=alibi_span,
        ),
        seed=seed,
        note=f", dt half-lives {half_lives or 'the Mamba draw'}",
        **flags,
    )


if __name__ == "__main__":
    main()
