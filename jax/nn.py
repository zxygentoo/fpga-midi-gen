"""The common parts of the eras, above the seam and below it.

ABOVE THE SEAM stand the float models: the step-frame transformer and the state-space
model share one head, one position rule, one sampling chain and one trainer skeleton, and
each model module keeps what is its own -- the trunk, the parameter layout and the
checkpoint walk. BELOW IT stand the integer rules every TWIN is built on: the fixed-point
rails, the exponent rule of a checkpoint, the sampling policy, the shared table and the
integer draw, which are the twin of `lib/nn/quantized.ml`.

What is one thing across the eras stands here one time: a rule changed here changes every
model, or every twin, at once -- which is the point.

Matmul precision is pinned to true float32 here, no TF32; every model imports this
module, thus the pin holds everywhere.
"""

import math
import time
from pathlib import Path
from typing import NamedTuple

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


def pick_share(weights, share):
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
    return (running > (share * running[:, -1])[:, None]).argmax(axis=1)


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
        frame[:, seat] = pick_share(weights, uniform)
        if seat:
            stream = stream + seats[seat][frame[:, seat]]
    return state, frame


# ---------------------------------------------------------------------
# the integer rules of the twins
# ---------------------------------------------------------------------

# WHAT STANDS HERE IS THE PART OF THE INTEGER ARITHMETIC THAT IS ONE THING ACROSS THE
# ERAS: the fixed-point rails, the exponent rule of a checkpoint, the sampling policy, the
# shared table and the integer draw. Each era's twin keeps what is its own -- the
# parameter structures, the state formats of a recurrence, and the engines themselves. A
# rule written here is read by every twin and, through them, by every circuit.
#
# `lib/nn/quantized.ml` is the same module in OCaml and the elaborations read it. The two
# are TWO STATEMENTS OF ONE RULE and nothing in the types welds them; what holds them
# together is `tests/test_quantized.py`, which states the numbers both must give.

# the rails of int16: a value that passes them saturates and never wraps. Every clamp of
# every twin reads them here, thus none can write a rail of its own and part from its
# circuit in silence.
INT16_BITS = 16
INT16_LOW = -(1 << (INT16_BITS - 1))
INT16_HIGH = (1 << (INT16_BITS - 1)) - 1

# the Q of log2(e), and the Q the temper takes: one below it. The extra bit is headroom for
# the temperature -- the circuits carry this constant on an 18-bit signed port, thus the Q
# of log2(e) would overflow that port under a temperature of about 0.36.
LOG2E_Q = 15
TEMPER_Q = LOG2E_Q - 1

# the Q the exp2 unit reads its magnitudes at, and the Q of its answer
EXP2_IN_Q = 12
EXP2_OUT_Q = 15


def round_half_up(x):
    """Base's `Float.iround_nearest_exn`: floor(x + 0.5).

    A TIE GOES TOWARD PLUS INFINITY, thus -2.5 is -2 and 2.5 is 3, where Python's `round`
    and `numpy.rint` are half-to-even. Every rounding of every twin goes through here."""
    return np.floor(np.asarray(x, np.float64) + 0.5)


def largest_exponent(magnitude, *, opening, cap):
    """The largest exponent, from [opening] down, that keeps round(magnitude * 2^e) at
    [cap] or less.

    [opening] caps the all-zero value, where every exponent fits. The predicate falls
    monotonically in e, thus the first e that fits is the largest. Its readings differ
    only in where they open and what they must fit."""
    if magnitude <= 0.0:
        return opening
    e = opening
    while round_half_up(np.ldexp(magnitude, e)) > cap:
        e -= 1
    return e


def max_exponent(peak):
    """`Nn_quantized.max_exponent`: the exponent of one int8 tensor -- from 14 down, the
    largest that keeps round(peak * 2^e) inside the byte."""
    return largest_exponent(peak, opening=14, cap=127)


def quantize(weights, e=None):
    """`Nn_quantized.quantize`: the int8 form of one tensor, and the exponent that reads it.

    The byte is two's complement and the negative end is not used: the clamp is -127 and
    not -128, thus the image is symmetric and a negated weight is a negated byte. [e]
    overrides the exponent of the tensor's own peak, where tensors whose rows add share
    one."""
    weights = np.asarray(weights, np.float64)
    if e is None:
        e = max_exponent(float(np.abs(weights).max(initial=0.0)))
    return np.clip(round_half_up(np.ldexp(weights, e)), -127, 127).astype(np.int32), e


def fixed_q12(values, bound):
    """a per-head number in Q12, clamped to the PORT that carries it.

    Era five's `dt_bias` joins an int16 sum and its `d_skip` rides an 18-bit operand port,
    thus the bound is a fact of the circuit and the caller states it."""
    values = np.ldexp(np.asarray(values, np.float64), 12)
    return np.clip(round_half_up(values), -bound, bound).astype(np.int32)


def temper_of(temperature):
    """`Nn_quantized.policy`: the sampling temper, log2(e) / T, as (q_value, q)."""
    if temperature <= 0.0:
        raise ValueError("the temperature is positive")
    return int(round_half_up(np.ldexp(1.0 / math.log(2.0) / temperature, TEMPER_Q))), (
        TEMPER_Q
    )


def min_weight_of(min_p):
    """`Nn_quantized.policy`: the min-p floor as a share of the peak weight.

    The peak weighs 2^`EXP2_OUT_Q` after the temper, thus the floor is a plain share of it
    and the circuit compares two integers."""
    if not 0.0 <= min_p < 1.0:
        raise ValueError("min_p is 0 up to 1")
    return int(round_half_up(min_p * float(1 << EXP2_OUT_Q)))


class Temper(NamedTuple):
    """The sampling temper as the bitstream carries it: log2(e) / T at [q].

    The temperature is PROVENANCE and not arithmetic -- the temper is already folded -- thus
    it travels in the metadata of a contract file alone, and a file written by an older
    tool can read back with no temperature at all."""

    q_value: int
    q: int
    temperature: float

    @classmethod
    def of(cls, temperature):
        q_value, q = temper_of(temperature)
        return cls(q_value, q, temperature)


def write_tally():
    """a running tally: the activation writes, the writes that rode the clamp, and the
    hottest write BEFORE it.

    A clamp that fires is the finding that says which format is wrong, thus it is counted and
    never assumed away. The peak reads before the clamp, thus it answers the format question
    directly."""
    return {"seen": 0, "clamped": 0, "peak": 0}


def tallied_write(tally, value):
    """every activation write goes through here: the clamp is counted and the peak kept.

    A peak inside the format proves that nothing clamped, thus the clip is skipped — the walk
    writes millions of these and the short circuit is the whole of the difference."""
    high, low = int(value.max()), int(value.min())
    tally["seen"] += value.size
    tally["peak"] = max(tally["peak"], high, -low)
    if high <= INT16_HIGH and low >= INT16_LOW:
        return value.astype(np.int32)
    tally["clamped"] += int(np.count_nonzero(value > INT16_HIGH))
    tally["clamped"] += int(np.count_nonzero(value < INT16_LOW))
    return np.clip(value, INT16_LOW, INT16_HIGH).astype(np.int32)


# the quantized exponential: exp2 of -j/256 in Q15 -- `Nn_quantized.Constants.exp2_table`,
# the one table the samplers of every era read
EXP2_TABLE = np.array(
    [
        int(round_half_up(float(1 << EXP2_OUT_Q) * 2.0 ** (-j / 256.0)))
        for j in range(256)
    ],
    np.int64,
)


def exp2_of_magnitude(magnitude):
    """`Nn_quantized.exp2_of_magnitude`: 2^-m in Q15 over a nonnegative Q12 magnitude.

    The integer part shifts and the top eight bits of the fraction index the table; a
    magnitude of 16 or more is 0. The shift is held under the width of the host word where
    the answer is 0 anyway, because a shift past the width states nothing in either
    language."""
    whole = magnitude >> EXP2_IN_Q
    entry = EXP2_TABLE[(magnitude >> (EXP2_IN_Q - 8)) & 255]
    return np.where(whole >= 16, 0, entry >> np.minimum(whole, 62))


def pick(weights, word):
    """`Nn_quantized.draw`: the class a 24-bit uniform word lands, over the batch.

    The total is the last running total and never a second sum of the same weights. THE PICK
    ALWAYS LANDS: the peak weighs 2^15, thus the total is 2^15 or more, and the word falls
    under 2^24, thus the threshold stands strictly under it. No fallback is written."""
    running = np.cumsum(weights, axis=-1)
    threshold = (np.asarray(word, np.int64) * running[..., -1]) >> prng.UNIFORM_BITS
    return (running > threshold[..., None]).argmax(axis=-1)


def engine_states(seeds):
    """the generator of each walk: THE SEED AS IT STANDS, which is the board's SEED cell rule,
    thus seed 0 is the walk that stands still as the circuit stands still on it.

    `prng.states` folds instead, which is the float walk's rule: a seed inside 32 bits names
    itself under both, and 0 is the one seed where the two walks are not one walk."""
    return np.array([prng.create(int(seed)) for seed in seeds], dtype=np.uint32)


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
