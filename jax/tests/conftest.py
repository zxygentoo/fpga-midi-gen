"""Tests run on the CPU.

A sweep or a training run owns the GPU for half an hour at a time, and a test that
reaches for it fails with CUDA_ERROR_OUT_OF_MEMORY rather than with anything about the
code. This runs before any test module imports jax, so the choice is made once.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
