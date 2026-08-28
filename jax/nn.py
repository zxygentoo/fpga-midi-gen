"""The common parts of the model families.

Both eras -- the step-frame transformer and the state-space model -- share one head, one
position rule, one sampling chain and one trainer skeleton. What is one thing across them
stands here one time: a rule changed here changes both models at once, which is the
point. The model modules keep what is theirs alone -- the trunks, the parameter layouts
and the checkpoint walks.

Matmul precision is pinned to true float32 here, no TF32; every model imports this
module, thus the pin holds everywhere.
"""

import time
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

import data
import prng

jax.config.update("jax_default_matmul_precision", "float32")

# The slope of head k is 2^-(SLOPE_SPAN (k+1) / heads). Elected 2026-08-18 over spans 4,
# 8, 16, 24 and 64: the means of 4 and 8 are a dead heat, and the VARIANCE is the finding
# -- 5 to 7 times tighter over six seeds, replicated at two step budgets. Every head is
# then local, and seeds stop latching onto whatever distant structure their init favours.
SLOPE_SPAN = 4
# the phase table IS the bar -- one row for each step of it. Two names for one number let
# the corpus phase and the table part, and a phase outside the table gathers a clamped row
# in silence.
PHASE_BUCKETS = data.BAR_STEPS
TABLES = ("seats", "phase")
JAX_ROOT = Path(__file__).resolve().parent


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def attention_bias(heads, length, span=SLOPE_SPAN):
    """ALiBi plus the causal wall, [1, heads, length, length]."""
    pos = jnp.arange(length, dtype=jnp.float32)
    distance = pos[:, None] - pos[None, :]
    slopes = -(2.0 ** (-span * (jnp.arange(heads, dtype=jnp.float32) + 1.0) / heads))
    alibi = slopes[None, :, None, None] * distance[None, None, :, :]
    wall = jnp.triu(jnp.ones((length, length), dtype=jnp.float32), k=1) * -1e9
    return alibi + wall[None, None, :, :]


def dropout_masks(key, rate, shape):
    """the multiplier form of inverted dropout: 0 or 1/keep, one for each element"""
    keep = 1.0 - rate
    return jax.random.bernoulli(key, keep, shape) / keep


def embed(params, classes):
    """The input of one step: the four seat rows sum.

    A shared table with a voice tag cannot work here, and the reason is arithmetic and not
    capacity. Every step carries all four seats, thus the sum of the four tags is the same
    vector at every position -- a bias, which carries nothing -- and what remains is
    symmetric in the four codes. A soprano on 72 over a bass on 48 would give the vector of
    a soprano on 48 under a bass on 72, and the voices would be thrown away on the way in.
    Four tables break the symmetry, and no voice tag is then necessary anywhere."""
    return sum(params["seats"][seat][classes[..., seat]] for seat in range(data.SEATS))


def seat_logits(params, h, drawn):
    """The chained head: [batch, length, d] -> [batch, length, SEATS, CLASSES].

    Each seat reads the stream that the seats above it have already written:

        h3 = h                   logits(seat 3) = E[3] . rms(h3)
        h2 = h3 + E[3][c3]       logits(seat 2) = E[2] . rms(h2)
        h1 = h2 + E[2][c2]       logits(seat 1) = E[1] . rms(h1)
        h0 = h1 + E[1][c1]       logits(seat 0) = E[0] . rms(h0)

    [drawn] holds the classes the chain conditions on -- the true frame in training, where
    all four heads then run in one pass with no sampling, and the drawn seats at the draw.
    Only seats 3, 2 and 1 are read.

    The chain runs from the soprano down, which keeps the one decision the ear accepted:
    the top voice is chosen first and conditions on no voice under it, as the music is
    written. Four heads that drew in parallel would make the voices conditionally
    independent, and a chord is a joint choice: measured on era four, that costs 0.3157
    nats for each step -- 0.456 bits, sixteen times the seed spread. The chain removes the
    cost for no parameters at all -- parallel heads need the same four tables -- and three
    adds of a vector.

    The table that reads a voice is the table that writes it, which is the tied embedding
    of both eras, one time for each seat. What the chain adds is also what the next step
    reads: the input embedding of step t+1 is a3 + a2 + a1 + a0, and the chain assembles
    it one voice at a time."""
    seats = params["seats"]
    stream = h
    logits = [None] * data.SEATS
    for seat in reversed(range(data.SEATS)):
        logits[seat] = rms_norm(stream) @ seats[seat].T
        if seat:
            stream = stream + seats[seat][drawn[..., seat]]
    return jnp.stack(logits, axis=-2)


def seat_nll_of_hidden(params, h, labels):
    """the negative log likelihood of every voice of every step, over a residual stream
    the era's own [hidden] computed; the caller reduces"""
    logp = jax.nn.log_softmax(seat_logits(params, h, labels), axis=-1)
    return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]


# ---------------------------------------------------------------------
# the host-side draw: numpy, float64, and the PRNG of the circuit
# ---------------------------------------------------------------------


def _host_rms_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def temper(raw, temperature, min_p):
    """the tempered weight of each class against the peak, then the min-p floor; the peak
    weighs one, thus min_p is a share of the peak"""
    weights = np.exp((raw - raw.max(axis=1, keepdims=True)) / temperature)
    if min_p > 0.0:
        weights = np.where(weights >= min_p, weights, 0.0)
    return weights


def pick(weights, uniform):
    """The class whose running total passes the draw.

    It takes the uniform and not a draw, thus one function owns both sums and the total is
    the last running total -- never a second sum of the same weights. numpy adds pairwise in
    sum() and left to right in cumsum(), thus two sums of one array differ in the last bits,
    and a draw made against the other sum can land above every running total, where no class
    passes at all.

    Against this total the draw is strictly below it, because the uniform falls under 1 by
    2**-24 at the least. Therefore the walk always ends on a class, and that class always
    holds weight the floor left standing: to reach the last index is to know that no earlier
    total passed, thus the weight there is the difference of two totals across the draw. No
    fallback is necessary, and none is written."""
    running = np.cumsum(weights, axis=1)
    return (running > (uniform * running[:, -1])[:, None]).argmax(axis=1)


def draw_frame(params, h, state, temperature, min_p):
    """One step of the chained head, on the host: the soprano first, and each seat under it
    reading the stream the seats above have written.

    The chain is the reason a frame is a joint choice and not four independent ones. Seat 0
    is the bass and seat 3 the soprano, thus the loop runs down.

    Every walk of the batch draws. A step is one frame and never a sentence of its own
    length, thus no walk of the batch finishes before another and none has to sit out a
    draw while the rest go on."""
    seats = np.asarray(params["seats"])
    stream = h
    frame = np.zeros((len(h), data.SEATS), dtype=np.int32)
    for seat in reversed(range(data.SEATS)):
        raw = (_host_rms_norm(stream) @ seats[seat].T).astype(np.float64)
        weights = temper(raw, temperature, min_p)
        state, uniform = prng.uniform(state, True)
        frame[:, seat] = pick(weights, uniform)
        if seat:
            stream = stream + seats[seat][frame[:, seat]]
    return state, frame


# ---------------------------------------------------------------------
# the trainer skeleton
# ---------------------------------------------------------------------


def save_checkpoint(path, tensors, span=None):
    """The naming rule of the seam: the tensors named "0" upward, in construction order,
    then the ALiBi span last and alone where the model carries one -- an older file that
    does not still reads, because a reader takes whole layer groups and then one scalar
    if one is there. The era's trainer builds the flat list, because the layer layouts
    are its own."""
    if span is not None:
        tensors = list(tensors) + [np.asarray([span], dtype=np.float32)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({str(i): np.asarray(t) for i, t in enumerate(tensors)}, path)


def optimizer_init(params):
    """the optimizer state at step zero: the two moment trees, zeros in params' shape"""
    moments = jax.tree.map(jnp.zeros_like, params)
    return moments, moments


def schedule(step, peak, warmup, total):
    """linear warmup to the peak, cosine decay to zero; a warmup of zero is a constant"""
    if warmup == 0:
        return peak
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))


def adamw(state, params, grads, t, lr, *, clip, weight_decay):
    """One update: the global-norm clip, then Adam with a decoupled weight decay. [t] is
    the step count, which the bias correction reads.

    It stands apart from [make_step] because the update rule is one thing and the shape of
    a step is another: the sheet era carries the batch-norm population statistics beside
    the two moments, and its step is its own while the rule here stays shared."""
    m, v = state
    if clip > 0.0:
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
        scale = clip / jnp.maximum(norm, clip)
        grads = jax.tree.map(lambda g: g * scale, grads)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    m_hat = jax.tree.map(lambda m_: m_ / (1 - b1**t), m)
    v_hat = jax.tree.map(lambda v_: v_ / (1 - b2**t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + weight_decay * p),
        params,
        m_hat,
        v_hat,
    )
    return params, (m, v)


def make_step(loss, clip, weight_decay):
    """The jitted training step around an era's loss: the gradient and one [adamw] update.
    [loss params classes phases key] is the era's own -- the trunks differ, the update
    does not."""

    def step_fn(params, state, t, classes, phases, lr, key):
        value, grads = jax.value_and_grad(lambda p: loss(p, classes, phases, key))(params)
        params, state = adamw(
            state, params, grads, t, lr, clip=clip, weight_decay=weight_decay
        )
        return value, params, state

    return jax.jit(step_fn)


def make_eval(nll):
    """the jitted evaluation sums around an era's nll: the per-step loss and the step
    count -- the moving-steps instrument lives in measure.py, where elections read it"""

    def eval_fn(params, classes, phases):
        steps = jnp.sum(nll(params, classes, phases), axis=-1)
        return jnp.sum(steps), jnp.size(steps)

    return jax.jit(eval_fn)


def on_device(batches):
    """the evaluation windows are fixed for the whole run, thus they cross to the device
    one time and not at every evaluation"""
    return [(jnp.asarray(classes), jnp.asarray(phases)) for classes, phases in batches]


def eval_loss(eval_fn, params, batches):
    """nats for each step, the mean over the evaluation windows"""
    total = 0.0
    steps = 0
    for classes, phases in batches:
        sums = eval_fn(params, classes, phases)
        total += float(sums[0])
        steps += int(sums[1])
    return total / max(steps, 1)


def train(
    *,
    corpus_path,
    train_on,
    context,
    batch,
    steps,
    lr,
    seed,
    warmup,
    log_every,
    eval_every,
    eval_limit,
    ckpt,
    average_top,
    draw_params,
    step_fn,
    eval_fn,
    save_checkpoint,
    describe=None,
):
    """The training loop both eras run: the batch draw, the schedule, the step, the
    evaluations, the best-by-valid checkpoint and the top-K average. The era passes what
    is its own -- [draw_params key] for the initial tree, the jitted [step_fn] and
    [eval_fn], [save_checkpoint path params] closing over whatever its seam wants, and an
    optional [describe params] line for the head of the log."""
    corpus = data.load_corpus(corpus_path)
    pool = data.train_pool(corpus, train_on)
    train_eval = on_device(data.eval_batches(corpus["train"], context, eval_limit, batch))
    valid_eval = on_device(data.eval_batches(corpus["valid"], context, eval_limit, batch))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, draw_key = jax.random.split(key)
    params = draw_params(draw_key)
    state = optimizer_init(params)
    count = sum(int(np.prod(t.shape)) for t in jax.tree.leaves(params))
    corpus_steps = sum(int(split.index[row, 1]) for split, row in pool)
    click.echo(
        f"corpus: {len(pool)} pool streams, {corpus_steps} steps; eval rows: "
        f"{sum(len(b[0]) for b in train_eval)} train, "
        f"{sum(len(b[0]) for b in valid_eval)} valid; parameters: {count}"
    )
    if describe is not None:
        click.echo(describe(params))

    best = float("inf")
    top = []  # (valid, step, host params) -- the K best snapshots for averaging
    losses = []
    started = time.perf_counter()

    def evaluate(step, params):
        nonlocal best
        train_all = eval_loss(eval_fn, params, train_eval)
        valid_all = eval_loss(eval_fn, params, valid_eval)
        mark = ""
        if valid_all < best:
            best = valid_all
            mark = "  *"
            if ckpt and train_on != "all":
                save_checkpoint(ckpt, params)
        # the snapshot crosses to the host only when it can stay: the sort would drop it
        # again, and the copy is the whole parameter tree
        if average_top > 0 and (len(top) < average_top or valid_all < top[-1][0]):
            top.append((valid_all, step, jax.tree.map(np.asarray, params)))
            top.sort(key=lambda entry: entry[0])
            del top[average_top:]
        click.echo(
            f"step {step:5d}  eval  train {train_all:.4f}  valid {valid_all:.4f}{mark}"
        )

    for step in range(1, steps + 1):
        classes, phases = data.train_batch(rng, pool, batch, context)
        # a name of its own: [lr] is the peak the schedule reads, and a loop that writes
        # its own peak decays the rate geometrically to zero and trains nothing
        rate = schedule(step, lr, warmup, steps)
        key, step_key = jax.random.split(key)
        value, params, state = step_fn(
            params,
            state,
            jnp.float32(step),
            jnp.asarray(classes),
            jnp.asarray(phases),
            jnp.float32(rate),
            step_key,
        )
        # the device array, NOT float(value): a read blocks until the step finishes, and
        # the loop then cannot overlap the next batch draw and its transfer with the
        # compute of this one. The log below reads, thus the run-ahead stays inside one
        # log window.
        losses.append(value)
        if step % log_every == 0 or step == 1:
            # the training number is nats for each step too: the mean over the predictions
            # times the four seats
            mean = float(jnp.mean(jnp.stack(losses)))
            click.echo(f"step {step:5d}  loss {data.SEATS * mean:.4f}")
            losses = []
        if step % eval_every == 0 or step == steps:
            evaluate(step, params)

    seconds = time.perf_counter() - started
    click.echo(
        f"time: {seconds:.0f} s, {seconds / steps * 1000:.0f} ms each step, "
        f"the evaluations inside"
    )
    click.echo(f"best valid {best:.4f}")
    if ckpt:
        if train_on == "all":
            save_checkpoint(ckpt, params)
            click.echo(f"checkpoint of the last step: {ckpt}")
        else:
            click.echo(f"checkpoint of the best: {ckpt}")
        if average_top > 0 and top:
            averaged = jax.tree.map(
                lambda *tensors: np.mean(np.stack(tensors), axis=0),
                *[entry[2] for entry in top],
            )
            path = ckpt.replace(".ckpt", "-avg.ckpt")
            save_checkpoint(path, averaged)
            click.echo(
                f"average of {len(top)} best snapshots "
                f"(steps {[entry[1] for entry in top]}): {path}"
            )
