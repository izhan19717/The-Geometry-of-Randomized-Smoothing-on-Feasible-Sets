"""Geometry primitives for capped-simplex feasible sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backend import jnp


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ActiveSet:
    """Active constraints at a capped-simplex point."""

    lower: tuple[int, ...]
    upper: tuple[int, ...]
    budget: bool


@dataclass(frozen=True)
class LinearOptimizationResult:
    """Result of a linear optimization over a capped simplex."""

    value: float
    point: FloatArray


def _as_float_vector(values: ArrayLike, name: str) -> FloatArray:
    vector = jnp.asarray(values, dtype=jnp.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not bool(jnp.all(jnp.isfinite(vector))):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _caps_or_inf(size: int, caps: ArrayLike | None) -> FloatArray:
    if caps is None:
        return jnp.full(size, jnp.inf, dtype=jnp.float64)
    cap_vector = _as_float_vector(caps, "caps")
    if cap_vector.size != size:
        raise ValueError("caps must have the same length as the input vector")
    if bool(jnp.any(cap_vector < 0)):
        raise ValueError("caps must be non-negative")
    return cap_vector


def _validate_budget_and_caps(
    size: int,
    budget: float,
    caps: FloatArray,
    *,
    equality: bool,
    tol: float,
) -> None:
    if budget < -tol:
        raise ValueError("budget must be non-negative")
    if bool(jnp.any(jnp.isinf(caps))):
        cap_sum = np.inf
    else:
        cap_sum = float(jnp.sum(caps))
    if equality and budget - cap_sum > tol:
        raise ValueError("budget exceeds the sum of caps")
    if size == 0:
        raise ValueError("dimension must be positive")


def capped_simplex_project(
    values: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> FloatArray:
    """Project a vector onto a capped simplex.

    The feasible set is ``0 <= x_i <= caps_i`` and either
    ``sum_i x_i == budget`` or ``sum_i x_i <= budget``.
    """

    vector = _as_float_vector(values, "values")
    cap_vector = _caps_or_inf(vector.size, caps)
    if budget < -tol:
        raise ValueError("budget must be non-negative")
    finite_cap_sum = float(jnp.sum(cap_vector[jnp.isfinite(cap_vector)]))
    if bool(jnp.any(jnp.isinf(cap_vector))):
        cap_sum = np.inf
    else:
        cap_sum = finite_cap_sum
    if budget - cap_sum > tol:
        raise ValueError("budget exceeds the sum of caps")

    clipped = jnp.clip(vector, 0.0, cap_vector)
    if not equality and float(clipped.sum()) <= budget + tol:
        return np.asarray(clipped)
    if budget <= tol:
        return np.asarray(jnp.zeros_like(vector))

    def projected_sum(theta: float) -> float:
        return float(jnp.clip(vector - theta, 0.0, cap_vector).sum())

    finite_caps = jnp.where(jnp.isfinite(cap_vector), cap_vector, 0.0)
    lower = float(jnp.min(vector - finite_caps) - abs(budget) - jnp.max(jnp.abs(vector)) - 1.0)
    upper = float(jnp.max(vector))

    while projected_sum(lower) < budget:
        lower *= 2.0
    while projected_sum(upper) > budget:
        upper *= 2.0 if upper > 0 else 0.5
        if abs(upper) < tol:
            upper = 1.0

    for _ in range(max_iter):
        theta = 0.5 * (lower + upper)
        total = projected_sum(theta)
        if abs(total - budget) <= tol:
            break
        if total > budget:
            lower = theta
        else:
            upper = theta

    theta = 0.5 * (lower + upper)
    projected = jnp.clip(vector - theta, 0.0, cap_vector)
    if equality:
        residual = budget - float(projected.sum())
        free = np.flatnonzero(np.asarray((projected > tol) & (projected < cap_vector - tol)))
        if free.size and abs(residual) > tol:
            projected = projected.at[int(free[0])].add(residual)
    return np.asarray(projected)


def capped_simplex_project_batch(
    values: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> FloatArray:
    """Project many vectors onto the same capped simplex.

    This is the batch analogue of :func:`capped_simplex_project`.  The equality
    case solves for one threshold per row by vectorized bisection.  It is used
    by larger experiments to avoid thousands of scalar projection calls.
    """

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must have nonzero batch and feature dimensions")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must contain only finite entries")
    if budget < -tol:
        raise ValueError("budget must be non-negative")

    if caps is None:
        cap_vector = np.full(matrix.shape[1], np.inf, dtype=np.float64)
    else:
        cap_vector = np.asarray(caps, dtype=np.float64)
        if cap_vector.ndim != 1:
            raise ValueError("caps must be a one-dimensional vector")
        if cap_vector.size != matrix.shape[1]:
            raise ValueError("caps must have the same length as each row")
        if np.any(cap_vector < 0.0) or not np.all(np.isfinite(cap_vector)):
            raise ValueError("caps must be finite and non-negative")

    cap_sum = np.inf if np.any(np.isinf(cap_vector)) else float(np.sum(cap_vector))
    if equality and budget - cap_sum > tol:
        raise ValueError("budget exceeds the sum of caps")

    clipped = np.clip(matrix, 0.0, cap_vector)
    if not equality:
        output = clipped.copy()
        needs_projection = output.sum(axis=1) > budget + tol
        if not np.any(needs_projection):
            return output
        output[needs_projection] = capped_simplex_project_batch(
            matrix[needs_projection],
            budget,
            cap_vector,
            equality=True,
            tol=tol,
            max_iter=max_iter,
        )
        return output

    if budget <= tol:
        return np.zeros_like(matrix)

    finite_caps = np.where(np.isfinite(cap_vector), cap_vector, 0.0)
    lower = (
        np.min(matrix - finite_caps[None, :], axis=1)
        - abs(float(budget))
        - np.max(np.abs(matrix), axis=1)
        - 1.0
    )
    upper = np.max(matrix, axis=1)
    theta = 0.5 * (lower + upper)

    for _ in range(max_iter):
        theta = 0.5 * (lower + upper)
        totals = np.clip(matrix - theta[:, None], 0.0, cap_vector).sum(axis=1)
        if np.max(np.abs(totals - budget)) <= tol:
            break
        too_large = totals > budget
        lower = np.where(too_large, theta, lower)
        upper = np.where(too_large, upper, theta)

    projected = np.clip(matrix - theta[:, None], 0.0, cap_vector)
    residuals = budget - projected.sum(axis=1)
    if np.max(np.abs(residuals)) > tol:
        free = (projected > tol) & (projected < cap_vector[None, :] - tol)
        for row_index, residual in enumerate(residuals):
            if abs(float(residual)) <= tol:
                continue
            free_indices = np.flatnonzero(free[row_index])
            if free_indices.size:
                projected[row_index, int(free_indices[0])] += residual
    return projected


def capped_simplex_linear_maximize(
    objective: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    tol: float = 1e-12,
) -> LinearOptimizationResult:
    """Maximize a linear objective over a capped simplex.

    The problem is ``max objective @ x`` subject to ``0 <= x <= caps`` and
    either ``sum x == budget`` or ``sum x <= budget``.  The greedy fill is exact
    because this is a bounded fractional knapsack linear program.
    """

    coeffs = _as_float_vector(objective, "objective")
    cap_vector = _caps_or_inf(coeffs.size, caps)
    _validate_budget_and_caps(coeffs.size, budget, cap_vector, equality=equality, tol=tol)
    if bool(jnp.any(jnp.isinf(cap_vector))):
        raise ValueError("finite caps are required for linear optimization")

    remaining = float(budget)
    point = np.zeros(coeffs.size, dtype=np.float64)
    order = np.argsort(-np.asarray(coeffs))
    for index in order:
        coefficient = float(coeffs[int(index)])
        if not equality and coefficient <= 0.0:
            break
        if remaining <= tol:
            break
        amount = min(float(cap_vector[int(index)]), remaining)
        point[int(index)] = amount
        remaining -= amount

    if equality and remaining > tol:
        raise ValueError("budget exceeds the sum of caps")
    value = float(np.dot(np.asarray(coeffs), point))
    return LinearOptimizationResult(value=value, point=point)


def capped_simplex_linear_minimize(
    objective: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    tol: float = 1e-12,
) -> LinearOptimizationResult:
    """Minimize a linear objective over a capped simplex."""

    result = capped_simplex_linear_maximize(
        -np.asarray(objective, dtype=np.float64),
        budget,
        caps,
        equality=equality,
        tol=tol,
    )
    return LinearOptimizationResult(value=-result.value, point=result.point)


def capped_simplex_boundary_depth(
    point: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    tol: float = 1e-12,
) -> float:
    """Distance to the nearest capped-simplex boundary in the affine hull.

    For equality-budget simplexes this measures distance inside the hyperplane
    ``sum x = budget``.  For inequality-budget simplexes it uses the ambient
    Euclidean distance to nonnegativity, caps, and the budget hyperplane.
    """

    vector = _as_float_vector(point, "point")
    cap_vector = _caps_or_inf(vector.size, caps)
    if bool(jnp.any(jnp.isinf(cap_vector))):
        raise ValueError("finite caps are required for boundary depth")
    _validate_budget_and_caps(vector.size, budget, cap_vector, equality=equality, tol=tol)
    if bool(jnp.any(vector < -tol)) or bool(jnp.any(vector > cap_vector + tol)):
        raise ValueError("point must satisfy box constraints")
    if equality and abs(float(jnp.sum(vector)) - budget) > tol:
        raise ValueError("point must satisfy the equality budget")
    if not equality and float(jnp.sum(vector)) - budget > tol:
        raise ValueError("point must satisfy the inequality budget")

    if vector.size == 1 and equality:
        return np.inf

    scale = np.sqrt(1.0 - 1.0 / float(vector.size)) if equality else 1.0
    lower_depth = float(jnp.min(vector)) / scale
    upper_depth = float(jnp.min(cap_vector - vector)) / scale
    depth = min(lower_depth, upper_depth)
    if not equality:
        budget_depth = (budget - float(jnp.sum(vector))) / np.sqrt(float(vector.size))
        depth = min(depth, budget_depth)
    return max(0.0, depth)


def active_set(
    point: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    tol: float = 1e-9,
) -> ActiveSet:
    """Return active lower, upper, and budget constraints."""

    vector = _as_float_vector(point, "point")
    cap_vector = _caps_or_inf(vector.size, caps)
    lower = tuple(np.flatnonzero(np.asarray(vector <= tol)).tolist())
    upper = tuple(
        np.flatnonzero(np.asarray(jnp.isfinite(cap_vector) & (vector >= cap_vector - tol))).tolist()
    )
    budget_active = equality or float(vector.sum()) >= budget - tol
    return ActiveSet(lower=lower, upper=upper, budget=budget_active)


def face_dimension(
    point: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    tol: float = 1e-9,
) -> int:
    """Estimate the dimension of the active face containing ``point``."""

    vector = _as_float_vector(point, "point")
    active = active_set(vector, budget, caps, equality=equality, tol=tol)
    rows: list[FloatArray] = []
    if active.budget:
        rows.append(jnp.ones(vector.size, dtype=jnp.float64))
    for index in active.lower + active.upper:
        row = jnp.zeros(vector.size, dtype=jnp.float64).at[index].set(1.0)
        rows.append(row)
    if not rows:
        return vector.size
    matrix = jnp.vstack(rows)
    rank = int(jnp.linalg.matrix_rank(matrix, tol=tol))
    return max(0, vector.size - rank)


def _sample_affine_gaussian(
    rng: np.random.Generator,
    count: int,
    dimension: int,
    *,
    equality: bool,
) -> FloatArray:
    samples = rng.normal(loc=0.0, scale=1.0, size=(count, dimension)).astype(np.float64)
    if equality:
        samples -= samples.mean(axis=1, keepdims=True)
    return samples


def solid_angle_estimate(
    point: ArrayLike,
    budget: float,
    caps: ArrayLike | None = None,
    *,
    equality: bool = True,
    samples: int = 20_000,
    seed: int | None = 0,
    tol: float = 1e-9,
) -> float:
    """Monte Carlo estimate of the tangent cone's Gaussian solid angle."""

    vector = _as_float_vector(point, "point")
    cap_vector = _caps_or_inf(vector.size, caps)
    active = active_set(vector, budget, cap_vector, equality=equality, tol=tol)
    rng = np.random.default_rng(seed)
    directions = _sample_affine_gaussian(rng, samples, vector.size, equality=equality)
    feasible = jnp.ones(samples, dtype=bool)

    if active.lower:
        feasible = feasible & jnp.all(directions[:, active.lower] >= -tol, axis=1)
    if active.upper:
        feasible = feasible & jnp.all(directions[:, active.upper] <= tol, axis=1)
    if active.budget and not equality:
        feasible = feasible & (directions.sum(axis=1) <= tol)

    return float(feasible.mean())
