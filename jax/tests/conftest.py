"""Tests run on the CPU: a sweep or a training run owns the GPU for half an hour at a
time, and a test that reached for it would fail with CUDA_ERROR_OUT_OF_MEMORY rather than
with anything about the code. This runs before any test module imports jax.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
