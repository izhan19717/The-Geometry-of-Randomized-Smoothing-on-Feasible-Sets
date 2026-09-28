"""Dirichlet smoothing utilities for full-budget simplex experiments.

These helpers are empirical infrastructure.  They use the standard closed-form
Dirichlet KL identity numerically; the repository's Lean artifact currently
verifies only the positive-parameter algebra for this law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, gammaln

FloatArray = np.ndarray


@dataclass(frozen=True)
class DirichletScaleMatch:
    """Concentration parameters matched to affine Gaussian tangent variance."""

    alpha_min: float
    kappa: float
    total_concentration: float
    matched_tangent_variance: float


def dirichlet_parameters(
    action: FloatArray,
    *,
    alpha_min: float,
    kappa: float,
    budget: float = 1.0,
) -> FloatArray:
    """Boundary-safe affine Dirichlet parameterization.

    For a full-budget simplex action ``a``, the parameter is
    ``alpha_i(a) = alpha_min + kappa * a_i / budget``.
    """

    point = np.asarray(action, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError("action must be one-dimensional")
    if budget <= 0.0:
        raise ValueError("budget must be positive")
    if alpha_min <= 0.0:
        raise ValueError("alpha_min must be positive")
    if kappa < 0.0:
        raise ValueError("kappa must be nonnegative")
    return alpha_min + (kappa / budget) * point


def match_dirichlet_concentration_to_sigma(
    dimension: int,
    sigma: float,
    *,
    alpha_min: float = 0.05,
) -> DirichletScaleMatch:
    """Match Dirichlet tangent variance at the uniform simplex point.

    At the uniform point with total concentration ``S``, every unit tangent
    direction has variance ``1 / (d * (S + 1))``.  Matching that to ``sigma^2``
    gives ``S = 1 / (d sigma^2) - 1``.
    """

    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if alpha_min <= 0.0:
        raise ValueError("alpha_min must be positive")
    total_concentration = 1.0 / (dimension * sigma * sigma) - 1.0
    kappa = total_concentration - dimension * alpha_min
    if kappa < 0.0:
        raise ValueError(
            "sigma is too large for the requested alpha_min scale match; "
            f"need total concentration >= {dimension * alpha_min:.6g}"
        )
    return DirichletScaleMatch(
        alpha_min=alpha_min,
        kappa=float(kappa),
        total_concentration=float(total_concentration),
        matched_tangent_variance=float(sigma * sigma),
    )


def sample_dirichlet_actions(
    action: FloatArray,
    *,
    alpha_min: float,
    kappa: float,
    count: int,
    budget: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """Sample feasible full-budget simplex actions from the Dirichlet smoother."""

    if count <= 0:
        raise ValueError("count must be positive")
    alpha = dirichlet_parameters(action, alpha_min=alpha_min, kappa=kappa, budget=budget)
    rng = np.random.default_rng(seed)
    return budget * rng.dirichlet(alpha, size=count)


def dirichlet_kl(alpha: FloatArray, beta: FloatArray) -> float:
    """KL divergence ``KL(Dir(alpha) || Dir(beta))``."""

    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("alpha and beta must be one-dimensional arrays with the same shape")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("Dirichlet parameters must be strictly positive")
    sum_a = float(np.sum(a))
    sum_b = float(np.sum(b))
    value = gammaln(sum_a) - np.sum(gammaln(a))
    value -= gammaln(sum_b) - np.sum(gammaln(b))
    value += float(np.sum((a - b) * (digamma(a) - digamma(sum_a))))
    return max(0.0, float(value))


def dirichlet_pinsker_tv_bound(alpha: FloatArray, beta: FloatArray) -> float:
    """Pinsker TV bound from the closed-form Dirichlet KL value."""

    return float(np.sqrt(0.5 * dirichlet_kl(alpha, beta)))
