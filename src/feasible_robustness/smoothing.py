"""Sampling and certification helpers for feasible randomized smoothing."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable, Hashable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta, chi2, norm, truncnorm

from .geometry import (
    capped_simplex_linear_maximize,
    capped_simplex_linear_minimize,
    capped_simplex_project_batch,
)


FloatArray = NDArray[np.float64]
Label = Hashable


@dataclass(frozen=True)
class RadiusEstimate:
    """Monte Carlo radius estimate with confidence-adjusted probabilities."""

    top_label: Label
    runner_up: Label | None
    top_lower: float
    runner_upper: float
    radius: float


@dataclass(frozen=True)
class LogRatioBounds:
    """Lower and upper bounds for a log normalizer ratio."""

    lower: float
    upper: float


@dataclass(frozen=True)
class TruncatedGaussianSamples:
    """Accepted samples and rejection-sampling diagnostics."""

    samples: FloatArray
    attempts: int
    acceptance_rate: float


@dataclass(frozen=True)
class GaussianHitAndRunSamples:
    """MCMC samples and diagnostics for Gaussian hit-and-run."""

    samples: FloatArray
    steps: int
    burn_in: int
    thinning: int
    mean_abs_step: float
    max_box_violation: float
    max_budget_violation: float

    @property
    def attempts(self) -> int:
        return self.steps

    @property
    def acceptance_rate(self) -> float:
        return 1.0


@dataclass(frozen=True)
class ReflectiveHMCSamples:
    """MCMC samples and diagnostics for exact reflective Gaussian HMC."""

    samples: FloatArray
    steps: int
    burn_in: int
    thinning: int
    trajectory_length: float
    mean_reflections: float
    max_reflections: int
    max_box_violation: float
    max_budget_violation: float

    @property
    def attempts(self) -> int:
        return self.steps

    @property
    def acceptance_rate(self) -> float:
        return 1.0


def projected_gaussian_samples(
    center: ArrayLike,
    sigma: float,
    count: int,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    seed: int | None = None,
) -> FloatArray:
    """Sample ``P_K(center + sigma * noise)`` for a capped simplex ``K``."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    center_vector = np.asarray(center, dtype=np.float64)
    if center_vector.ndim != 1:
        raise ValueError("center must be a one-dimensional vector")
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=(count, center_vector.size))
    if equality:
        noise -= noise.mean(axis=1, keepdims=True)
    raw = center_vector + noise
    return capped_simplex_project_batch(raw, budget, caps, equality=equality)


def truncated_gaussian_capped_simplex_samples(
    center: ArrayLike,
    sigma: float,
    count: int,
    budget: float,
    caps: ArrayLike,
    *,
    equality: bool = True,
    seed: int | None = None,
    batch_size: int | None = None,
    max_attempts: int = 1_000_000,
    tol: float = 1e-10,
) -> TruncatedGaussianSamples:
    """Sample a Gaussian conditioned on a capped simplex by rejection.

    For ``equality=True`` the proposal is the isotropic Gaussian in the affine
    hull ``sum x = budget`` centered at ``center``.  Accepted samples are exact
    draws from that proposal conditioned on the capped simplex.  The function
    reports the acceptance rate because near-boundary regimes may be expensive.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if max_attempts < count:
        raise ValueError("max_attempts must be at least count")
    center_vector = np.asarray(center, dtype=np.float64)
    cap_vector = np.asarray(caps, dtype=np.float64)
    if center_vector.ndim != 1:
        raise ValueError("center must be a one-dimensional vector")
    if cap_vector.shape != center_vector.shape:
        raise ValueError("caps must match center shape")
    dimension = center_vector.size
    if np.any(cap_vector < 0.0) or not np.all(np.isfinite(cap_vector)):
        raise ValueError("caps must be finite and non-negative")
    if np.any(center_vector < -tol) or np.any(center_vector > cap_vector + tol):
        raise ValueError("center must satisfy box constraints")
    if equality and abs(float(center_vector.sum()) - budget) > tol:
        raise ValueError("center must satisfy the equality budget")
    if not equality and float(center_vector.sum()) - budget > tol:
        raise ValueError("center must satisfy the inequality budget")

    rng = np.random.default_rng(seed)
    accepted: list[FloatArray] = []
    accepted_total = 0
    attempts = 0
    batch = batch_size or max(1024, min(16384, 4 * count))
    while accepted_total < count and attempts < max_attempts:
        draw_count = min(batch, max_attempts - attempts)
        noise = rng.normal(loc=0.0, scale=sigma, size=(draw_count, center_vector.size))
        if equality:
            noise -= noise.mean(axis=1, keepdims=True)
        proposals = center_vector + noise
        feasible = np.all(proposals >= -tol, axis=1) & np.all(proposals <= cap_vector + tol, axis=1)
        if equality:
            feasible &= np.abs(proposals.sum(axis=1) - budget) <= 10.0 * tol
        else:
            feasible &= proposals.sum(axis=1) <= budget + tol
        block = proposals[feasible]
        if block.size:
            accepted.append(block)
            accepted_total += int(block.shape[0])
        attempts += draw_count

    if not accepted:
        raise RuntimeError("rejection sampler accepted no samples")
    samples = np.vstack(accepted)[:count]
    if samples.shape[0] < count:
        raise RuntimeError(
            f"accepted {samples.shape[0]} samples after {attempts} attempts; "
            "increase max_attempts or reduce sigma"
        )
    return TruncatedGaussianSamples(
        samples=samples,
        attempts=attempts,
        acceptance_rate=float(accepted_total / attempts),
    )


def minimax_tilting_capped_simplex_samples(
    center: ArrayLike,
    sigma: float,
    count: int,
    budget: float,
    caps: ArrayLike,
    *,
    equality: bool = True,
    seed: int | None = None,
    batch_size: int | None = None,
    max_attempts: int = 1_000_000,
    dropped_coordinate: int | None = None,
    sampler_path: str | None = None,
    tol: float = 1e-10,
) -> TruncatedGaussianSamples:
    """Exact iid capped-simplex truncated-Gaussian samples via minimax tilting.

    The equality coordinate is eliminated.  The remaining ``n-1`` coordinates
    have a nonsingular Gaussian law and box constraints; those are sampled by
    Botev's minimax-tilting truncated-normal algorithm.  A final independent
    rejection step enforces the single sum-band constraint induced by the
    dropped coordinate.  Since the proposal is exact iid, accepted samples are
    exact iid draws from the capped-simplex truncation.
    """

    if not equality:
        raise NotImplementedError("minimax tilting sampler currently supports equality budgets only")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if max_attempts < count:
        raise ValueError("max_attempts must be at least count")
    center_vector = np.asarray(center, dtype=np.float64)
    cap_vector = np.asarray(caps, dtype=np.float64)
    if center_vector.ndim != 1:
        raise ValueError("center must be a one-dimensional vector")
    if cap_vector.shape != center_vector.shape:
        raise ValueError("caps must match center shape")
    n = center_vector.size
    if n < 2:
        raise ValueError("dimension must be at least two")
    if np.any(cap_vector < 0.0) or not np.all(np.isfinite(cap_vector)):
        raise ValueError("caps must be finite and non-negative")
    if np.any(center_vector < -tol) or np.any(center_vector > cap_vector + tol):
        raise ValueError("center must satisfy box constraints")
    if abs(float(center_vector.sum()) - budget) > tol:
        raise ValueError("center must satisfy the equality budget")

    drop = n - 1 if dropped_coordinate is None else int(dropped_coordinate)
    if drop < 0 or drop >= n:
        raise ValueError("dropped_coordinate is out of range")
    keep = np.array([i for i in range(n) if i != drop], dtype=int)
    dropped_cap = float(cap_vector[drop])

    mu = center_vector[keep]
    covariance = sigma**2 * (np.eye(n - 1) - np.ones((n - 1, n - 1)) / n)
    lower = np.zeros(n - 1, dtype=np.float64)
    upper = cap_vector[keep].astype(np.float64, copy=True)
    sum_lower = budget - dropped_cap
    sum_upper = budget

    TruncatedMVN = _load_botev_truncated_mvn(sampler_path)
    tmvn = TruncatedMVN(mu, covariance, lower, upper, seed=seed)

    accepted: list[FloatArray] = []
    accepted_total = 0
    attempts = 0
    batch = batch_size or max(256, min(8192, 2 * count))
    while accepted_total < count and attempts < max_attempts:
        draw_count = min(batch, max_attempts - attempts)
        proposal_keep = np.asarray(tmvn.sample(draw_count), dtype=np.float64).T
        sums = proposal_keep.sum(axis=1)
        feasible = (sums >= sum_lower - tol) & (sums <= sum_upper + tol)
        kept = proposal_keep[feasible]
        if kept.size:
            block = np.empty((kept.shape[0], n), dtype=np.float64)
            block[:, keep] = kept
            block[:, drop] = budget - kept.sum(axis=1)
            block = np.vstack([
                _repair_capped_simplex_roundoff(row, budget, cap_vector, locked=None, tol=tol)
                for row in block
            ])
            accepted.append(block)
            accepted_total += int(block.shape[0])
        attempts += draw_count

    if not accepted:
        raise RuntimeError("minimax tilting box proposal accepted no capped-simplex samples")
    samples = np.vstack(accepted)[:count]
    if samples.shape[0] < count:
        raise RuntimeError(
            f"accepted {samples.shape[0]} samples after {attempts} minimax proposals; "
            "increase max_attempts"
        )
    max_violation = max(
        float(np.max(-samples)),
        float(np.max(samples - cap_vector)),
        abs(float(np.max(samples.sum(axis=1) - budget))),
        abs(float(np.min(samples.sum(axis=1) - budget))),
    )
    if max_violation > 1e-7:
        raise RuntimeError(f"minimax tilting sampler produced infeasible samples: {max_violation}")
    return TruncatedGaussianSamples(
        samples=samples,
        attempts=attempts,
        acceptance_rate=float(accepted_total / attempts),
    )


def _load_botev_truncated_mvn(sampler_path: str | None):
    candidate = sampler_path or os.environ.get("FACETS_TMVN_SAMPLER_PATH")
    if candidate:
        candidate_path = os.path.abspath(candidate)
        if candidate_path not in sys.path:
            sys.path.insert(0, candidate_path)
    try:
        from minimax_tilting_sampler import TruncatedMVN
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "minimax_tilting_sampler is required for exact minimax-tilting "
            "capped-simplex sampling. Clone "
            "https://github.com/brunzema/truncated-mvn-sampler and set "
            "FACETS_TMVN_SAMPLER_PATH to that directory."
        ) from error
    return TruncatedMVN


def hitandrun_truncated_gaussian_capped_simplex_samples(
    center: ArrayLike,
    sigma: float,
    count: int,
    budget: float,
    caps: ArrayLike,
    *,
    equality: bool = True,
    seed: int | None = None,
    burn_in: int = 512,
    thinning: int = 8,
    start: ArrayLike | None = None,
    tol: float = 1e-10,
) -> GaussianHitAndRunSamples:
    """Sample a capped-simplex truncated Gaussian by Gaussian hit-and-run.

    The transition chooses a random direction in the affine hull and samples
    exactly from the one-dimensional Gaussian conditional on the feasible chord.
    The truncated Gaussian is the stationary distribution of this Markov chain.
    Returned samples are therefore MCMC samples, not independent rejection
    samples; use burn-in/thinning diagnostics before citing high-dimensional
    results.
    """

    if not equality:
        raise NotImplementedError("hit-and-run sampler currently supports equality budgets only")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning <= 0:
        raise ValueError("thinning must be positive")

    center_vector = np.asarray(center, dtype=np.float64)
    cap_vector = np.asarray(caps, dtype=np.float64)
    if center_vector.ndim != 1:
        raise ValueError("center must be a one-dimensional vector")
    if cap_vector.shape != center_vector.shape:
        raise ValueError("caps must match center shape")
    dimension = center_vector.size
    if np.any(cap_vector < 0.0) or not np.all(np.isfinite(cap_vector)):
        raise ValueError("caps must be finite and non-negative")
    if np.any(center_vector < -tol) or np.any(center_vector > cap_vector + tol):
        raise ValueError("center must satisfy box constraints")
    if abs(float(center_vector.sum()) - budget) > tol:
        raise ValueError("center must satisfy the equality budget")
    cap_sum_total = float(np.sum(cap_vector))
    if cap_sum_total + tol < budget:
        raise ValueError("budget exceeds the sum of caps")

    if budget <= tol or cap_sum_total <= budget + tol:
        samples = np.repeat(center_vector.reshape(1, dimension), count, axis=0)
        return GaussianHitAndRunSamples(
            samples=samples,
            steps=0,
            burn_in=burn_in,
            thinning=thinning,
            mean_abs_step=0.0,
            max_box_violation=0.0,
            max_budget_violation=0.0,
        )

    if start is None:
        if cap_sum_total <= 0.0:
            raise ValueError("sum of caps must be positive")
        current = budget * cap_vector / cap_sum_total
        if abs(float(current.sum()) - budget) > tol:
            current = center_vector.copy()
    else:
        current = np.asarray(start, dtype=np.float64).copy()
        if current.shape != center_vector.shape:
            raise ValueError("start must match center shape")
        if np.any(current < -tol) or np.any(current > cap_vector + tol):
            raise ValueError("start must satisfy box constraints")
        if abs(float(current.sum()) - budget) > tol:
            raise ValueError("start must satisfy the equality budget")

    if dimension == 1:
        samples = np.repeat(center_vector.reshape(1, 1), count, axis=0)
        return GaussianHitAndRunSamples(
            samples=samples,
            steps=0,
            burn_in=burn_in,
            thinning=thinning,
            mean_abs_step=0.0,
            max_box_violation=0.0,
            max_budget_violation=0.0,
        )

    rng = np.random.default_rng(seed)
    total_steps = burn_in + count * thinning
    samples: list[FloatArray] = []
    abs_steps: list[float] = []

    for step in range(total_steps):
        for _ in range(256):
            direction = rng.normal(loc=0.0, scale=1.0, size=dimension)
            direction -= direction.mean()
            norm_value = float(np.linalg.norm(direction))
            if norm_value <= 1e-14:
                continue
            direction /= norm_value

            lower = -np.inf
            upper = np.inf
            for value, slope, cap in zip(current, direction, cap_vector):
                if slope > tol:
                    lower = max(lower, -value / slope)
                    upper = min(upper, (cap - value) / slope)
                elif slope < -tol:
                    lower = max(lower, (cap - value) / slope)
                    upper = min(upper, -value / slope)

            if upper - lower > 1e-14:
                break
        else:
            raise RuntimeError("could not find a positive-length hit-and-run chord")

        conditional_mean = float(np.dot(center_vector - current, direction))
        standardized_lower = (lower - conditional_mean) / sigma
        standardized_upper = (upper - conditional_mean) / sigma
        draw = float(
            truncnorm.rvs(
                standardized_lower,
                standardized_upper,
                loc=conditional_mean,
                scale=sigma,
                random_state=rng,
            )
        )
        current = current + draw * direction
        current = np.clip(current, 0.0, cap_vector)
        residual = budget - float(current.sum())
        if abs(residual) > tol:
            free = np.flatnonzero((current > tol) & (current < cap_vector - tol))
            if free.size:
                current[int(free[0])] += residual
        abs_steps.append(abs(draw))

        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples.append(current.copy())

    sample_array = np.vstack(samples)
    if sample_array.shape[0] != count:
        raise RuntimeError("internal hit-and-run sample count mismatch")
    lower_violation = max(0.0, float(-np.min(sample_array)))
    upper_violation = max(0.0, float(np.max(sample_array - cap_vector[None, :])))
    return GaussianHitAndRunSamples(
        samples=sample_array,
        steps=total_steps,
        burn_in=burn_in,
        thinning=thinning,
        mean_abs_step=float(np.mean(abs_steps)) if abs_steps else 0.0,
        max_box_violation=max(lower_violation, upper_violation),
        max_budget_violation=float(np.max(np.abs(sample_array.sum(axis=1) - budget))),
    )


def reflective_hmc_truncated_gaussian_capped_simplex_samples(
    center: ArrayLike,
    sigma: float,
    count: int,
    budget: float,
    caps: ArrayLike,
    *,
    equality: bool = True,
    seed: int | None = None,
    burn_in: int = 256,
    thinning: int = 4,
    trajectory_length: float = np.pi / 2.0,
    min_trajectory_length: float = 0.25,
    max_reflections: int = 512,
    start: ArrayLike | None = None,
    tol: float = 1e-10,
) -> ReflectiveHMCSamples:
    """Sample a capped-simplex truncated Gaussian by exact reflective HMC.

    The sampler evolves the Gaussian Hamiltonian exactly in the equality
    affine hull and reflects momentum at active box constraints.  It targets
    the same truncated Gaussian as rejection sampling, but returned draws are
    MCMC samples and must be diagnosed before paper-grade use.
    """

    if not equality:
        raise NotImplementedError("reflective HMC currently supports equality budgets only")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning <= 0:
        raise ValueError("thinning must be positive")
    if trajectory_length <= 0.0:
        raise ValueError("trajectory_length must be positive")
    if min_trajectory_length < 0.0 or min_trajectory_length > trajectory_length:
        raise ValueError("min_trajectory_length must lie in [0, trajectory_length]")
    if max_reflections <= 0:
        raise ValueError("max_reflections must be positive")

    center_vector = np.asarray(center, dtype=np.float64)
    cap_vector = np.asarray(caps, dtype=np.float64)
    if center_vector.ndim != 1:
        raise ValueError("center must be a one-dimensional vector")
    if cap_vector.shape != center_vector.shape:
        raise ValueError("caps must match center shape")
    dimension = center_vector.size
    if np.any(cap_vector < 0.0) or not np.all(np.isfinite(cap_vector)):
        raise ValueError("caps must be finite and non-negative")
    if np.any(center_vector < -tol) or np.any(center_vector > cap_vector + tol):
        raise ValueError("center must satisfy box constraints")
    if abs(float(center_vector.sum()) - budget) > tol:
        raise ValueError("center must satisfy the equality budget")
    cap_sum_total = float(np.sum(cap_vector))
    if cap_sum_total + tol < budget:
        raise ValueError("budget exceeds the sum of caps")

    if budget <= tol or cap_sum_total <= budget + tol or dimension == 1:
        samples = np.repeat(center_vector.reshape(1, dimension), count, axis=0)
        return ReflectiveHMCSamples(
            samples=samples,
            steps=0,
            burn_in=burn_in,
            thinning=thinning,
            trajectory_length=trajectory_length,
            mean_reflections=0.0,
            max_reflections=0,
            max_box_violation=0.0,
            max_budget_violation=0.0,
        )

    if start is None:
        current = _capped_simplex_interior_start(budget, cap_vector, center_vector, tol=tol)
    else:
        current = np.asarray(start, dtype=np.float64).copy()
        _validate_capped_simplex_point(current, budget, cap_vector, tol=tol, name="start")

    rng = np.random.default_rng(seed)
    total_steps = burn_in + count * thinning
    samples: list[FloatArray] = []
    reflection_counts: list[int] = []

    for step in range(total_steps):
        momentum = rng.normal(loc=0.0, scale=1.0, size=dimension)
        momentum -= momentum.mean()
        duration = float(rng.uniform(min_trajectory_length, trajectory_length))
        current, momentum, reflections = _reflective_hmc_trajectory(
            current,
            momentum,
            center_vector,
            sigma,
            cap_vector,
            budget,
            duration,
            max_reflections=max_reflections,
            tol=tol,
        )
        reflection_counts.append(reflections)
        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples.append(current.copy())

    sample_array = np.vstack(samples)
    if sample_array.shape[0] != count:
        raise RuntimeError("internal reflective-HMC sample count mismatch")
    lower_violation = max(0.0, float(-np.min(sample_array)))
    upper_violation = max(0.0, float(np.max(sample_array - cap_vector[None, :])))
    return ReflectiveHMCSamples(
        samples=sample_array,
        steps=total_steps,
        burn_in=burn_in,
        thinning=thinning,
        trajectory_length=trajectory_length,
        mean_reflections=float(np.mean(reflection_counts)) if reflection_counts else 0.0,
        max_reflections=max(reflection_counts) if reflection_counts else 0,
        max_box_violation=max(lower_violation, upper_violation),
        max_budget_violation=float(np.max(np.abs(sample_array.sum(axis=1) - budget))),
    )


def _validate_capped_simplex_point(
    point: FloatArray,
    budget: float,
    caps: FloatArray,
    *,
    tol: float,
    name: str,
) -> None:
    if point.shape != caps.shape:
        raise ValueError(f"{name} must match caps shape")
    if np.any(point < -tol) or np.any(point > caps + tol):
        raise ValueError(f"{name} must satisfy box constraints")
    if abs(float(point.sum()) - budget) > tol:
        raise ValueError(f"{name} must satisfy the equality budget")


def _capped_simplex_interior_start(
    budget: float,
    caps: FloatArray,
    fallback: FloatArray,
    *,
    tol: float,
) -> FloatArray:
    cap_sum = float(np.sum(caps))
    if cap_sum <= 0.0:
        raise ValueError("sum of caps must be positive")
    start = budget * caps / cap_sum
    if np.all(start >= -tol) and np.all(start <= caps + tol) and abs(float(start.sum()) - budget) <= tol:
        return start.copy()
    return fallback.copy()


def _reflective_hmc_trajectory(
    position: FloatArray,
    momentum: FloatArray,
    center: FloatArray,
    sigma: float,
    caps: FloatArray,
    budget: float,
    duration: float,
    *,
    max_reflections: int,
    tol: float,
) -> tuple[FloatArray, FloatArray, int]:
    q = position.copy()
    p = momentum.copy()
    remaining = duration
    reflections = 0
    while remaining > 1e-12:
        collision = _next_reflective_hmc_collision(q, p, center, sigma, caps, remaining, tol=tol)
        if collision is None:
            q, p = _gaussian_hamiltonian_advance(q, p, center, sigma, remaining)
            remaining = 0.0
            break

        angle, coordinate, bound = collision
        q, p = _gaussian_hamiltonian_advance(q, p, center, sigma, angle)
        q[coordinate] = bound
        q = _repair_capped_simplex_roundoff(q, budget, caps, locked=coordinate, tol=tol)
        p = _reflect_affine_box_momentum(p, coordinate)
        remaining -= angle
        reflections += 1
        if reflections > max_reflections:
            raise RuntimeError("reflective HMC exceeded max_reflections")

    q = _repair_capped_simplex_roundoff(q, budget, caps, locked=None, tol=tol)
    p -= p.mean()
    return q, p, reflections


def _gaussian_hamiltonian_advance(
    position: FloatArray,
    momentum: FloatArray,
    center: FloatArray,
    sigma: float,
    angle: float,
) -> tuple[FloatArray, FloatArray]:
    cos_value = float(np.cos(angle))
    sin_value = float(np.sin(angle))
    displacement = position - center
    new_position = center + displacement * cos_value + sigma * momentum * sin_value
    new_momentum = momentum * cos_value - displacement * (sin_value / sigma)
    new_momentum -= new_momentum.mean()
    return new_position, new_momentum


def _next_reflective_hmc_collision(
    position: FloatArray,
    momentum: FloatArray,
    center: FloatArray,
    sigma: float,
    caps: FloatArray,
    max_angle: float,
    *,
    tol: float,
) -> tuple[float, int, float] | None:
    best_angle = np.inf
    best_coordinate = -1
    best_bound = 0.0
    for index, cap in enumerate(caps):
        for bound in (0.0, float(cap)):
            angle = _first_coordinate_boundary_angle(
                float(position[index]),
                float(momentum[index]),
                float(center[index]),
                sigma,
                bound,
                max_angle,
                tol=tol,
            )
            if angle is not None and angle < best_angle:
                best_angle = angle
                best_coordinate = index
                best_bound = bound
    if not np.isfinite(best_angle):
        return None
    return float(best_angle), best_coordinate, best_bound


def _first_coordinate_boundary_angle(
    position: float,
    momentum: float,
    center: float,
    sigma: float,
    bound: float,
    max_angle: float,
    *,
    tol: float,
) -> float | None:
    amplitude_cos = position - center
    amplitude_sin = sigma * momentum
    radius = float(np.hypot(amplitude_cos, amplitude_sin))
    if radius <= tol:
        return None
    target = (bound - center) / radius
    if target < -1.0 - 1e-12 or target > 1.0 + 1e-12:
        return None
    target = float(np.clip(target, -1.0, 1.0))
    phase = float(np.atan2(amplitude_sin, amplitude_cos))
    alpha = float(np.arccos(target))
    period = 2.0 * np.pi
    best = np.inf
    for base in (phase + alpha, phase - alpha):
        k = np.ceil((1e-10 - base) / period)
        candidate = base + period * k
        if candidate <= 1e-10:
            candidate += period
        if candidate <= max_angle + 1e-10 and candidate < best:
            best = candidate
    return float(best) if np.isfinite(best) else None


def _reflect_affine_box_momentum(momentum: FloatArray, coordinate: int) -> FloatArray:
    dimension = momentum.size
    normal = np.full(dimension, -1.0 / dimension, dtype=np.float64)
    normal[coordinate] += 1.0
    denom = float(np.dot(normal, normal))
    if denom <= 0.0:
        return momentum
    reflected = momentum - 2.0 * float(np.dot(momentum, normal)) / denom * normal
    reflected -= reflected.mean()
    return reflected


def _repair_capped_simplex_roundoff(
    point: FloatArray,
    budget: float,
    caps: FloatArray,
    *,
    locked: int | None,
    tol: float,
) -> FloatArray:
    repaired = np.clip(point, 0.0, caps)
    residual = budget - float(repaired.sum())
    if abs(residual) <= 10.0 * tol:
        return repaired
    free = np.flatnonzero((repaired > tol) & (repaired < caps - tol))
    if locked is not None:
        free = free[free != locked]
    for index in free:
        candidate = repaired[index] + residual
        if -tol <= candidate <= caps[index] + tol:
            repaired[index] = np.clip(candidate, 0.0, caps[index])
            return repaired
    if free.size:
        repaired[int(free[0])] = np.clip(repaired[int(free[0])] + residual, 0.0, caps[int(free[0])])
    return repaired


def estimate_label_probabilities(
    samples: FloatArray,
    classifier: Callable[[FloatArray], Label],
) -> dict[Label, int]:
    """Count labels produced by ``classifier`` over sampled feasible actions."""

    counts: dict[Label, int] = {}
    for sample in samples:
        label = classifier(sample)
        counts[label] = counts.get(label, 0) + 1
    return counts


def clopper_pearson(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval for a binomial proportion."""

    if not isinstance(trials, (int, np.integer)) or isinstance(trials, (bool, np.bool_)):
        raise TypeError("trials must be an integer")
    if not isinstance(successes, (int, np.integer)) or isinstance(successes, (bool, np.bool_)):
        raise TypeError("successes must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def simultaneous_clopper_pearson(
    counts: Mapping[Label, int],
    label_universe: Iterable[Label],
    alpha: float,
) -> dict[Label, tuple[float, float]]:
    """Simultaneous two-sided label-probability bounds for IID counts.

    The complete finite label universe is required so unseen labels are
    zero-filled.  Each label receives a two-sided Clopper-Pearson interval at
    level ``alpha / |Y|``; a union bound over both tails and every label gives
    simultaneous coverage at least ``1-alpha``.
    """

    labels = tuple(label_universe)
    if len(labels) < 2:
        raise ValueError("label_universe must contain at least two labels")
    if len(set(labels)) != len(labels):
        raise ValueError("label_universe must not contain duplicates")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    unknown = set(counts) - set(labels)
    if unknown:
        raise ValueError(f"counts contain labels outside label_universe: {unknown!r}")
    complete: dict[Label, int] = {}
    for label in labels:
        count = counts.get(label, 0)
        if not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)):
            raise TypeError("every label count must be an integer")
        if count < 0:
            raise ValueError("label counts must be nonnegative")
        complete[label] = int(count)
    total = sum(complete.values())
    if total <= 0:
        raise ValueError("counts must have positive total")
    per_label_alpha = alpha / len(labels)
    return {
        label: clopper_pearson(count, total, per_label_alpha)
        for label, count in complete.items()
    }


def _simultaneous_top_runner(
    counts: Mapping[Label, int],
    label_universe: Iterable[Label],
    alpha: float,
) -> tuple[Label, Label, float, float]:
    labels = tuple(label_universe)
    bounds = simultaneous_clopper_pearson(counts, labels, alpha)
    complete = {label: int(counts.get(label, 0)) for label in labels}
    top_label = max(labels, key=complete.__getitem__)
    competitors = tuple(label for label in labels if label != top_label)
    runner_up = max(competitors, key=lambda label: bounds[label][1])
    return top_label, runner_up, bounds[top_label][0], bounds[runner_up][1]


def certify_radius_cohen(
    counts: dict[Label, int],
    sigma: float,
    *,
    label_universe: Iterable[Label],
    alpha: float = 0.001,
) -> RadiusEstimate:
    """Compute a Cohen-style radius from IID counts with simultaneous bounds."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    top_label, runner_up, top_lower, runner_upper = _simultaneous_top_runner(
        counts, label_universe, alpha
    )
    if top_lower <= runner_upper:
        radius = 0.0
    else:
        radius = float(0.5 * sigma * (norm.ppf(top_lower) - norm.ppf(runner_upper)))
    return RadiusEstimate(
        top_label=top_label,
        runner_up=runner_up,
        top_lower=top_lower,
        runner_upper=runner_upper,
        radius=radius,
    )


def certify_radius_truncated_gaussian_diameter(
    counts: dict[Label, int],
    sigma: float,
    diameter_upper: float,
    *,
    label_universe: Iterable[Label],
    alpha: float = 0.001,
) -> RadiusEstimate:
    """Conservative TV radius from the truncated-Gaussian diameter KL bound.

    For any ``b`` with ``||a-b|| <= r``, the support-function normalizer bound
    gives ``KL(q_a || q_b) <= diameter(K) * r / sigma^2``.  Combining this with
    Pinsker and the TV margin certificate gives
    ``r < margin^2 sigma^2 / (2 diameter(K))``.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if diameter_upper <= 0:
        raise ValueError("diameter_upper must be positive")
    top_label, runner_up, top_lower, runner_upper = _simultaneous_top_runner(
        counts, label_universe, alpha
    )
    margin = top_lower - runner_upper
    radius = 0.0 if margin <= 0.0 else float((margin**2 * sigma**2) / (2.0 * diameter_upper))
    return RadiusEstimate(
        top_label=top_label,
        runner_up=runner_up,
        top_lower=top_lower,
        runner_upper=runner_upper,
        radius=radius,
    )


def certify_radius_truncated_gaussian_convex(
    counts: dict[Label, int],
    sigma: float,
    *,
    label_universe: Iterable[Label],
    alpha: float = 0.001,
) -> RadiusEstimate:
    """Candidate convex-set truncated-Gaussian certificate.

    For convex ``K``, the standard strong-log-concavity/Brascamp-Lieb route
    gives the manuscript-level bound ``KL(q_a || q_b) <= ||a-b||^2/(2 sigma^2)``.
    Pinsker then gives ``TV(q_a, q_b) <= ||a-b||/(2 sigma)``, so the TV-margin
    certificate radius is ``sigma * margin``.  This helper is intentionally
    separated from the diameter bound because the convex-contraction theorem is
    not yet Lean-verified in this project.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    top_label, runner_up, top_lower, runner_upper = _simultaneous_top_runner(
        counts, label_universe, alpha
    )
    margin = top_lower - runner_upper
    radius = 0.0 if margin <= 0.0 else float(sigma * margin)
    return RadiusEstimate(
        top_label=top_label,
        runner_up=runner_up,
        top_lower=top_lower,
        runner_upper=runner_upper,
        radius=radius,
    )


def finite_support_normalizer_log_ratio_bounds(
    support: ArrayLike,
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
) -> LogRatioBounds:
    """Bound ``log Z(b) - log Z(a)`` over a finite support.

    For truncated Gaussian kernels on a common finite support ``S``,
    ``Z(c)=sum_{z in S} exp(-||z-c||^2/(2 sigma^2))``.  If
    ``Delta(z)=(||z-a||^2-||z-b||^2)/(2 sigma^2)``, then
    ``min_S Delta <= log Z(b)-log Z(a) <= max_S Delta``.

    When ``support`` lists vertices of a convex polytope, the same extrema are
    exact for the support-function bound because ``Delta`` is affine in ``z``.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    points = np.asarray(support, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("support must be a non-empty 2D array or 1D support")
    a = np.asarray(center_a, dtype=np.float64)
    b = np.asarray(center_b, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1)
    if b.ndim == 0:
        b = b.reshape(1)
    if a.shape != b.shape or a.shape != (points.shape[1],):
        raise ValueError("centers must match the support dimension")

    delta = (
        np.sum((points - a) ** 2, axis=1)
        - np.sum((points - b) ** 2, axis=1)
    ) / (2.0 * sigma**2)
    return LogRatioBounds(lower=float(np.min(delta)), upper=float(np.max(delta)))


def capped_simplex_delta_bounds(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    budget: float,
    caps: ArrayLike,
    *,
    equality: bool = True,
) -> LogRatioBounds:
    """Compute exact support-function bounds for ``Delta_{a,b}``.

    ``Delta_{a,b}(z)=(||z-a||^2-||z-b||^2)/(2 sigma^2)`` is affine in ``z``.
    This routine optimizes it exactly over the capped simplex using greedy
    linear optimization.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    a = np.asarray(center_a, dtype=np.float64)
    b = np.asarray(center_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("centers must be one-dimensional vectors with equal shape")
    direction = b - a
    constant = float((np.dot(a, a) - np.dot(b, b)) / (2.0 * sigma**2))
    max_result = capped_simplex_linear_maximize(direction, budget, caps, equality=equality)
    min_result = capped_simplex_linear_minimize(direction, budget, caps, equality=equality)
    lower = float(min_result.value / sigma**2 + constant)
    upper = float(max_result.value / sigma**2 + constant)
    return LogRatioBounds(lower=lower, upper=upper)


def truncated_gaussian_pairwise_kl_tv_bounds(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    budget: float,
    caps: ArrayLike,
    *,
    equality: bool = True,
) -> tuple[float, float, LogRatioBounds]:
    """Return conservative pairwise KL and TV bounds for truncated Gaussians."""

    delta = capped_simplex_delta_bounds(
        center_a,
        center_b,
        sigma,
        budget,
        caps,
        equality=equality,
    )
    kl_upper = max(0.0, delta.upper - delta.lower)
    tv_upper = float(np.sqrt(0.5 * kl_upper))
    return kl_upper, tv_upper, delta


def truncated_gaussian_convex_kl_tv_bounds(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
) -> tuple[float, float]:
    """Candidate convex-set KL/TV bounds for same-variance truncated Gaussians."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    a = np.asarray(center_a, dtype=np.float64)
    b = np.asarray(center_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("centers must be one-dimensional vectors with equal shape")
    distance = float(np.linalg.norm(a - b))
    kl_upper = float(distance**2 / (2.0 * sigma**2))
    tv_upper = float(distance / (2.0 * sigma))
    return kl_upper, tv_upper


def gaussian_ball_probability_lower_bound(
    dimension: int,
    radius: float,
    sigma: float,
) -> float:
    """Gaussian probability of an ``m``-dimensional ball of radius ``radius``.

    If the feasible set contains the ball ``B(a, radius)`` in its affine hull,
    then the truncated Gaussian normalizer at ``a`` is at least this value.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if np.isinf(radius):
        return 1.0
    return float(chi2.cdf((radius / sigma) ** 2, df=dimension))


def boundary_depth_normalizer_log_ratio_bounds(
    dimension: int,
    depth_a: float,
    depth_b: float,
    sigma: float,
) -> LogRatioBounds:
    """Boundary-depth bound for ``log Z(b) - log Z(a)``.

    If ``B(a, depth_a)`` and ``B(b, depth_b)`` are contained in the feasible set
    in its affine hull and ``Z(c)`` uses the normalized Gaussian density, then
    ``Z(c) <= 1`` and
    ``Z(c) >= P(||G|| <= depth_c / sigma)``.  Therefore
    ``log lower_Z(b) <= log Z(b)-log Z(a) <= -log lower_Z(a)``.
    """

    lower_a = gaussian_ball_probability_lower_bound(dimension, depth_a, sigma)
    lower_b = gaussian_ball_probability_lower_bound(dimension, depth_b, sigma)
    lower = -np.inf if lower_b <= 0.0 else float(np.log(lower_b))
    upper = np.inf if lower_a <= 0.0 else float(-np.log(lower_a))
    return LogRatioBounds(lower=lower, upper=upper)
