"""Numerical backend configuration.

JAX is the canonical geometry backend for this project. We enable 64-bit
floating point because active-set and projection calculations are sensitive to
small boundary tolerances.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

__all__ = ["jax", "jnp"]

