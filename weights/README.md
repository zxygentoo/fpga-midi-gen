# The elected model of each era

One model for each era that has weights, committed so that a clone can audition
and build with no training run behind it. `_train/` holds the RUNS and git
ignores it; this directory holds the PRODUCT, and the two do not mix.

**The era is the name.** A run name states a sweep — `do03`, `96k`, `s6`,
`span4` — because a run must be unique among the runs beside it. There is one
elected model for each era, thus the era names it, and the same key reads
`lib/<era>`, `jax/<era>`, `bin/gate_<era>`, `docs/<era>.md` and
`make verilog-<era>`.

| file | era | the run it came from | elected |
|---|---|---|---|
| `transformer.ckpt` | four, the step-frame transformer | `_train/transformer/d64-frame-do03-96k-s6-l6-nopos-span4` | 2026-08-14, by the ear |
| `mamba.ckpt` | five, the Mamba hybrid | `_train/mamba/d64-mamba-k4-n16-zamba-ff-do03-48k-s7` | 2026-08-23, by the ear |
| `diffusion.ckpt` | six, the masked sheet of Coconet | `_train/diffusion/coconet/l48-h20-100k` | 2026-08-25, by the ear |

What each election measured is in `docs/<era>.md`, and the shape of each
circuit in `docs/<era>_rtl.md`.

**ERA ONE HAS NO FILE.** Pink noise is parameters and not weights:
`Pink.default` is a value of `lib/pink`, thus era one elaborates with nothing
from this directory and `make verilog-pink` runs on a bare clone.

## The contract file is derived and git ignores it

`weights/<era>.int8` is what crosses the seam to the elaboration, and
`make verilog-<era>` writes it from the checkpoint beside it:

    uv run python -m <era>.infer quantize --ckpt weights/<era>.ckpt \
        --out weights/<era>.int8

IT IS NOT COMMITTED, for two measured reasons. It is no smaller than the
checkpoint — an int8 image travels as int32, because `Nx_io` skips a dtype it
does not hold — and it is not reproducible byte for byte, because
`safetensors` writes its metadata out of a hash map that each process orders
differently. What IS reproducible is the NETLIST the file states, and
`jax/tests/test_parity.py` pins the md5 of that for every era. A contract file
rebuilt here must state the pinned netlist, thus nothing is lost by rebuilding
it and a stale one could not hide.
