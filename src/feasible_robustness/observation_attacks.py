"""Observation-space attacks for trained resource controllers.

The formal FACETS certificate is action-space.  These helpers are empirical
attack utilities for trained controllers whose observations are bounded feature
vectors.  They keep the perturbation set explicit so an experiment can report
which features were allowed to move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Objective = Callable[[FloatArray], float]


@dataclass(frozen=True)
class AttackResult:
    """Result of an observation-space attack."""

    adversarial: FloatArray
    objective_value: float
    perturbation_linf: float
    perturbation_l2: float
    steps: int


def project_linf_box(
    candidate: ArrayLike,
    center: ArrayLike,
    radius: float,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    mutable: ArrayLike | None = None,
) -> FloatArray:
    """Project ``candidate`` to an ``L_inf`` ball around ``center`` and box bounds."""

    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    point = np.asarray(candidate, dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if point.shape != origin.shape or point.shape != lo.shape or point.shape != hi.shape:
        raise ValueError("candidate, center, lower, and upper must have the same shape")
    clipped = np.clip(point, origin - radius, origin + radius)
    clipped = np.clip(clipped, lo, hi)
    if mutable is not None:
        mask = np.asarray(mutable, dtype=bool)
        if mask.shape != point.shape:
            raise ValueError("mutable mask must have the same shape as candidate")
        clipped = np.where(mask, clipped, origin)
    return clipped


def spsa_gradient(
    objective: Objective,
    point: ArrayLike,
    *,
    delta: float,
    samples: int,
    seed: int,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    mutable: ArrayLike | None = None,
) -> FloatArray:
    """Estimate a black-box gradient by symmetric SPSA perturbations."""

    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    x = np.asarray(point, dtype=np.float64)
    gradient = np.zeros_like(x)
    rng = np.random.default_rng(seed)
    if mutable is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        mask = np.asarray(mutable, dtype=bool)
        if mask.shape != x.shape:
            raise ValueError("mutable mask must match point")
    lo = np.full_like(x, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64)
    hi = np.full_like(x, np.inf) if upper is None else np.asarray(upper, dtype=np.float64)
    for _ in range(samples):
        direction = rng.choice(np.array([-1.0, 1.0]), size=x.shape)
        direction = np.where(mask, direction, 0.0)
        plus = np.clip(x + delta * direction, lo, hi)
        minus = np.clip(x - delta * direction, lo, hi)
        gradient += ((objective(plus) - objective(minus)) / (2.0 * delta)) * direction
    return gradient / float(samples)


def nes_gradient(
    objective: Objective,
    point: ArrayLike,
    *,
    sigma: float,
    samples: int,
    seed: int,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    mutable: ArrayLike | None = None,
) -> FloatArray:
    """Estimate a black-box gradient using antithetic Gaussian NES directions."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    x = np.asarray(point, dtype=np.float64)
    gradient = np.zeros_like(x)
    rng = np.random.default_rng(seed)
    if mutable is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        mask = np.asarray(mutable, dtype=bool)
        if mask.shape != x.shape:
            raise ValueError("mutable mask must match point")
    lo = np.full_like(x, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64)
    hi = np.full_like(x, np.inf) if upper is None else np.asarray(upper, dtype=np.float64)
    for _ in range(samples):
        direction = rng.normal(size=x.shape)
        direction = np.where(mask, direction, 0.0)
        plus = np.clip(x + sigma * direction, lo, hi)
        minus = np.clip(x - sigma * direction, lo, hi)
        gradient += ((objective(plus) - objective(minus)) / (2.0 * sigma)) * direction
    return gradient / float(samples)


def iterative_linf_attack(
    objective: Objective,
    point: ArrayLike,
    *,
    radius: float,
    step_size: float,
    steps: int,
    lower: ArrayLike,
    upper: ArrayLike,
    mutable: ArrayLike | None = None,
    gradient_estimator: Callable[[Objective, FloatArray, int], FloatArray],
) -> AttackResult:
    """Maximize ``objective`` by projected sign-gradient ascent."""

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    origin = np.asarray(point, dtype=np.float64)
    current = origin.copy()
    for step in range(steps):
        gradient = gradient_estimator(objective, current, step)
        current = current + step_size * np.sign(gradient)
        current = project_linf_box(
            current,
            origin,
            radius,
            lower,
            upper,
            mutable=mutable,
        )
    delta = current - origin
    return AttackResult(
        adversarial=current,
        objective_value=float(objective(current)),
        perturbation_linf=float(np.max(np.abs(delta))),
        perturbation_l2=float(np.linalg.norm(delta)),
        steps=steps,
    )


def spsa_linf_attack(
    objective: Objective,
    point: ArrayLike,
    *,
    radius: float,
    step_size: float,
    steps: int,
    delta: float,
    samples_per_step: int,
    lower: ArrayLike,
    upper: ArrayLike,
    mutable: ArrayLike | None = None,
    seed: int = 0,
) -> AttackResult:
    """Projected SPSA ``L_inf`` attack."""

    return iterative_linf_attack(
        objective,
        point,
        radius=radius,
        step_size=step_size,
        steps=steps,
        lower=lower,
        upper=upper,
        mutable=mutable,
        gradient_estimator=lambda obj, x, step: spsa_gradient(
            obj,
            x,
            delta=delta,
            samples=samples_per_step,
            seed=seed + step,
            lower=lower,
            upper=upper,
            mutable=mutable,
        ),
    )


def nes_linf_attack(
    objective: Objective,
    point: ArrayLike,
    *,
    radius: float,
    step_size: float,
    steps: int,
    sigma: float,
    samples_per_step: int,
    lower: ArrayLike,
    upper: ArrayLike,
    mutable: ArrayLike | None = None,
    seed: int = 0,
) -> AttackResult:
    """Projected NES ``L_inf`` attack."""

    return iterative_linf_attack(
        objective,
        point,
        radius=radius,
        step_size=step_size,
        steps=steps,
        lower=lower,
        upper=upper,
        mutable=mutable,
        gradient_estimator=lambda obj, x, step: nes_gradient(
            obj,
            x,
            sigma=sigma,
            samples=samples_per_step,
            seed=seed + step,
            lower=lower,
            upper=upper,
            mutable=mutable,
        ),
    )

