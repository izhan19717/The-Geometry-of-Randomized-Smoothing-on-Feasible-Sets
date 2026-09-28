"""Correlated capped feasible smoothing primitives.

The routines in this module implement a finite, center-independent proposal
bundle followed by a first-feasible selector and an explicit fallback.  They
do not implement unbounded rejection and do not infer observation-space
certificates from action-center certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CCFSSelection:
    """Result of inspecting one finite proposal bundle."""

    action: Array
    selected_index: int | None
    inspected_count: int
    used_fallback: bool


def _validate_bundle_parameters(cap: int, correlation: float) -> None:
    if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap < 1:
        raise ValueError("cap must be a positive integer")
    if not math.isfinite(correlation) or not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must lie in [0,1)")


def comparison_multiplier(cap: int, correlation: float) -> float:
    """Return the exact common-mean Gaussian comparison multiplier.

    For an equicorrelated bundle of ``cap`` proposals with marginal variance
    one and pairwise correlation ``correlation``, this is
    ``1^T C^{-1} 1``.  Divergence is multiplied by this value and a binary
    Gaussian radius is divided by its square root.
    """

    _validate_bundle_parameters(cap, correlation)
    return float(cap / (1.0 + (cap - 1) * correlation))


def correlation_for_multiplier(cap: int, target_multiplier: float) -> float:
    """Choose equicorrelation for an exact comparison-cost budget.

    For ``cap>1``, a nonsingular bundle can realize precisely the interval
    ``1 < target_multiplier <= cap``.  The lower endpoint is attained only as
    the correlation tends to one.  A one-proposal bundle has multiplier one
    and correlation zero by convention.
    """

    _validate_bundle_parameters(cap, 0.0)
    if not math.isfinite(target_multiplier):
        raise ValueError("target_multiplier must be finite")
    if cap == 1:
        if not math.isclose(target_multiplier, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("a one-proposal bundle has multiplier one")
        return 0.0
    if not 1.0 < target_multiplier <= float(cap):
        raise ValueError("for cap>1, target_multiplier must lie in (1,cap]")
    correlation = (cap / target_multiplier - 1.0) / (cap - 1.0)
    _validate_bundle_parameters(cap, correlation)
    return float(correlation)


def equicorrelation_matrix(cap: int, correlation: float) -> Array:
    """Construct the positive-definite equicorrelation matrix."""

    _validate_bundle_parameters(cap, correlation)
    identity = np.eye(cap, dtype=np.float64)
    ones = np.ones((cap, cap), dtype=np.float64)
    return (1.0 - correlation) * identity + correlation * ones


def sample_equicorrelated_proposals(
    center: Array | Sequence[float],
    sigma: float,
    cap: int,
    correlation: float,
    rng: np.random.Generator,
) -> Array:
    """Draw an equicorrelated Gaussian proposal bundle.

    Every row has law ``N(center, sigma^2 I)`` and distinct rows have
    cross-covariance ``correlation * sigma^2 I``.
    """

    _validate_bundle_parameters(cap, correlation)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    center_array = np.asarray(center, dtype=np.float64)
    if center_array.ndim != 1 or center_array.size == 0:
        raise ValueError("center must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(center_array)):
        raise ValueError("center must contain only finite values")

    shared = rng.standard_normal(center_array.shape)
    individual = rng.standard_normal((cap, center_array.size))
    noise = math.sqrt(correlation) * shared[None, :]
    noise = noise + math.sqrt(1.0 - correlation) * individual
    return center_array[None, :] + sigma * noise


def select_first_feasible(
    proposals: Array,
    is_feasible: Callable[[Array], bool],
    fallback: Callable[[Array], Array | Sequence[float]],
) -> CCFSSelection:
    """Return the first feasible proposal or an explicit common fallback.

    ``fallback`` receives the complete finite bundle and is called only after
    every proposal has failed.  Its output is checked with the same feasibility
    predicate; an invalid fallback raises rather than silently weakening an
    exact-feasibility claim.

    Soundness is a caller contract: for a certified comparison,
    ``is_feasible`` and ``fallback`` must be the same measurable maps under
    both compared centers.  A closure that reads the raw center, changes the
    feasible set, or changes the fallback across the comparison creates an
    uncharged side channel.
    """

    proposal_array = np.asarray(proposals, dtype=np.float64)
    if proposal_array.ndim != 2 or proposal_array.shape[0] < 1:
        raise ValueError("proposals must have shape (cap, action_dimension)")
    if proposal_array.shape[1] < 1 or not np.all(np.isfinite(proposal_array)):
        raise ValueError("proposals must be finite and nonempty")

    for index, proposal in enumerate(proposal_array):
        candidate = np.asarray(proposal, dtype=np.float64)
        if bool(is_feasible(candidate)):
            return CCFSSelection(
                action=candidate.copy(),
                selected_index=index,
                inspected_count=index + 1,
                used_fallback=False,
            )

    repaired = np.asarray(fallback(proposal_array.copy()), dtype=np.float64)
    if repaired.shape != proposal_array.shape[1:]:
        raise ValueError("fallback output shape must equal one action shape")
    if not np.all(np.isfinite(repaired)):
        raise ValueError("fallback output must be finite")
    if not bool(is_feasible(repaired)):
        raise ValueError("fallback returned an infeasible action")
    return CCFSSelection(
        action=repaired.copy(),
        selected_index=None,
        inspected_count=proposal_array.shape[0],
        used_fallback=True,
    )


def ccfs_step(
    center: Array | Sequence[float],
    sigma: float,
    cap: int,
    correlation: float,
    rng: np.random.Generator,
    is_feasible: Callable[[Array], bool],
    fallback: Callable[[Array], Array | Sequence[float]],
) -> CCFSSelection:
    """Sample and execute one CCFS proposal bundle."""

    proposals = sample_equicorrelated_proposals(
        center=center,
        sigma=sigma,
        cap=cap,
        correlation=correlation,
        rng=rng,
    )
    return select_first_feasible(proposals, is_feasible, fallback)


def trajectory_comparison_distance(
    first_centers: Array,
    second_centers: Array,
    sigmas: Array | Sequence[float],
    caps: Sequence[int],
    correlations: Array | Sequence[float],
) -> float:
    """Return the exact joint-Gaussian trajectory comparison distance.

    Time blocks are assumed independent and their center sequences are
    predeclared.  The returned value is ``R_traj`` in the Gaussian event shift,
    so finite-order Renyi divergence is bounded by ``alpha*R_traj**2/2``.
    """

    first = np.asarray(first_centers, dtype=np.float64)
    second = np.asarray(second_centers, dtype=np.float64)
    scale = np.asarray(sigmas, dtype=np.float64)
    rho = np.asarray(correlations, dtype=np.float64)
    cap_values = tuple(caps)
    if (
        first.ndim != 2
        or first.shape != second.shape
        or first.shape[0] < 1
        or first.shape[1] < 1
    ):
        raise ValueError("center arrays must share nonempty shape (horizon, dimension)")
    horizon = first.shape[0]
    if scale.shape != (horizon,) or rho.shape != (horizon,):
        raise ValueError("sigmas and correlations must have one value per time block")
    if len(cap_values) != horizon:
        raise ValueError("caps must have one value per time block")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("centers must be finite")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("sigmas must be finite and positive")

    squared = 0.0
    for index, cap in enumerate(cap_values):
        multiplier = comparison_multiplier(cap, float(rho[index]))
        difference = first[index] - second[index]
        squared += multiplier * float(np.dot(difference, difference)) / scale[index] ** 2
    return float(math.sqrt(squared))


def independent_exhaustion_probability(occupancy: float, cap: int) -> float:
    """Return ``(1-occupancy)**cap`` for an IID proposal bundle."""

    _validate_bundle_parameters(cap, 0.0)
    if not math.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
        raise ValueError("occupancy must lie in [0,1]")
    return float((1.0 - occupancy) ** cap)


def independent_expected_inspections(occupancy: float, cap: int) -> float:
    """Expected lazy feasibility checks for an IID capped bundle."""

    _validate_bundle_parameters(cap, 0.0)
    if not math.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
        raise ValueError("occupancy must lie in [0,1]")
    if occupancy == 0.0:
        return float(cap)
    if occupancy == 1.0:
        return 1.0
    return float(-math.expm1(cap * math.log1p(-occupancy)) / occupancy)
