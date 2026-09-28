"""Exact laws for IID capped first-feasible selection.

The clean identities here assume a fixed feasible set, a common fixed
fallback, and an accepted conditional law with zero mass at the fallback.
The algebra itself does not depend on the fallback's location.  Interpreting
the result as an exact-feasible selector additionally requires that fallback
to belong to the feasible set; callers and verification protocols must check
that condition explicitly.
They are pointwise divergence identities, not automatically uniform
attack-ball certificates.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


Array = np.ndarray


def _validate_occupancy(occupancy: float, *, interior: bool = False) -> float:
    value = float(occupancy)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("occupancy must lie in [0,1]")
    if interior and not 0.0 < value < 1.0:
        raise ValueError("this identity requires occupancy strictly between zero and one")
    return value


def _validate_cap(cap: int) -> int:
    if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap < 1:
        raise ValueError("cap must be a positive integer")
    return int(cap)


def capped_failure_probability(occupancy: float, cap: int) -> float:
    occupancy = _validate_occupancy(occupancy)
    cap = _validate_cap(cap)
    return float((1.0 - occupancy) ** cap)


def capped_success_probability(occupancy: float, cap: int) -> float:
    occupancy = _validate_occupancy(occupancy)
    cap = _validate_cap(cap)
    if occupancy == 0.0:
        return 0.0
    if occupancy == 1.0:
        return 1.0
    return float(-math.expm1(cap * math.log1p(-occupancy)))


def capped_geometric_factor(occupancy: float, cap: int) -> float:
    """Return the continuous-density multiplier ``s/a``."""

    occupancy = _validate_occupancy(occupancy)
    cap = _validate_cap(cap)
    if occupancy == 0.0:
        return float(cap)
    return float(capped_success_probability(occupancy, cap) / occupancy)


def _xlog_ratio(value: float, reference: float) -> float:
    if value == 0.0:
        return 0.0
    if reference == 0.0:
        return math.inf
    return float(value * math.log(value / reference))


def bernoulli_kl(first: float, second: float) -> float:
    first = _validate_occupancy(first)
    second = _validate_occupancy(second)
    return float(
        _xlog_ratio(first, second)
        + _xlog_ratio(1.0 - first, 1.0 - second)
    )


def capped_success_indicator_kl(
    first_occupancy: float, second_occupancy: float, cap: int
) -> float:
    """Stable KL between the two capped success/fallback indicators."""

    first_occupancy = _validate_occupancy(first_occupancy, interior=True)
    second_occupancy = _validate_occupancy(second_occupancy, interior=True)
    cap = _validate_cap(cap)
    first_log_failure = _log_failure(first_occupancy, cap)
    second_log_failure = _log_failure(second_occupancy, cap)
    first_log_success = _log_success(first_occupancy, cap)
    second_log_success = _log_success(second_occupancy, cap)
    first_failure = math.exp(first_log_failure)
    first_success = -math.expm1(first_log_failure)
    return float(
        first_failure * (first_log_failure - second_log_failure)
        + first_success * (first_log_success - second_log_success)
    )


def selected_output_kl(
    first_occupancy: float,
    second_occupancy: float,
    cap: int,
    conditional_kl: float,
) -> float:
    """Exact KL of two atom-free accepted laws plus a common fallback atom."""

    first_occupancy = _validate_occupancy(first_occupancy)
    second_occupancy = _validate_occupancy(second_occupancy)
    cap = _validate_cap(cap)
    if conditional_kl < 0.0 or math.isnan(conditional_kl):
        raise ValueError("conditional_kl must be nonnegative")
    first_success = capped_success_probability(first_occupancy, cap)
    if 0.0 < first_occupancy < 1.0 and 0.0 < second_occupancy < 1.0:
        binary = capped_success_indicator_kl(
            first_occupancy, second_occupancy, cap
        )
    else:
        second_success = capped_success_probability(second_occupancy, cap)
        binary = bernoulli_kl(first_success, second_success)
    if math.isinf(binary):
        return math.inf
    if first_success == 0.0:
        return binary
    return float(binary + first_success * conditional_kl)


def _logsumexp_pair(first: float, second: float) -> float:
    maximum = max(first, second)
    if math.isinf(maximum):
        return maximum
    return float(maximum + math.log(math.exp(first - maximum) + math.exp(second - maximum)))


def _log_failure(occupancy: float, cap: int) -> float:
    if occupancy == 1.0:
        return -math.inf
    return float(cap * math.log1p(-occupancy))


def _log_success(occupancy: float, cap: int) -> float:
    if occupancy == 0.0:
        return -math.inf
    if occupancy == 1.0:
        return 0.0
    return float(math.log(-math.expm1(_log_failure(occupancy, cap))))


def selected_output_renyi(
    first_occupancy: float,
    second_occupancy: float,
    cap: int,
    alpha: float,
    conditional_renyi: float,
) -> float:
    """Exact finite-order Renyi divergence for the disjoint mixture law.

    This implementation uses interior occupancies.  Endpoint support cases are
    best handled explicitly because negative powers at zero depend on the
    divergence direction.
    """

    first_occupancy = _validate_occupancy(first_occupancy, interior=True)
    second_occupancy = _validate_occupancy(second_occupancy, interior=True)
    cap = _validate_cap(cap)
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha == 1.0:
        raise ValueError("alpha must be positive, finite, and different from one")
    if conditional_renyi < 0.0 or math.isnan(conditional_renyi):
        raise ValueError("conditional_renyi must be nonnegative")
    atom = (
        alpha * _log_failure(first_occupancy, cap)
        + (1.0 - alpha) * _log_failure(second_occupancy, cap)
    )
    accepted = (
        alpha * _log_success(first_occupancy, cap)
        + (1.0 - alpha) * _log_success(second_occupancy, cap)
        + (alpha - 1.0) * conditional_renyi
    )
    return float(_logsumexp_pair(atom, accepted) / (alpha - 1.0))


def gaussian_conditional_kl(
    first_center: Array | Sequence[float],
    second_center: Array | Sequence[float],
    sigma: float,
    first_occupancy: float,
    second_occupancy: float,
    first_conditional_mean: Array | Sequence[float],
) -> float:
    """Exact KL between equal-covariance Gaussians conditioned on one fixed set."""

    first_occupancy = _validate_occupancy(first_occupancy, interior=True)
    second_occupancy = _validate_occupancy(second_occupancy, interior=True)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    first = np.asarray(first_center, dtype=np.float64)
    second = np.asarray(second_center, dtype=np.float64)
    mean = np.asarray(first_conditional_mean, dtype=np.float64)
    if first.ndim != 1 or first.size < 1 or second.shape != first.shape or mean.shape != first.shape:
        raise ValueError("centers and conditional mean must share a nonempty vector shape")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)) or not np.all(np.isfinite(mean)):
        raise ValueError("centers and conditional mean must be finite")
    shift = second - first
    value = (
        float(np.dot(shift, shift)) / (2.0 * sigma**2)
        - float(np.dot(shift, mean - first)) / sigma**2
        + math.log(second_occupancy / first_occupancy)
    )
    if value < -5e-12:
        raise FloatingPointError("computed conditional KL is negative")
    return float(max(0.0, value))


def gaussian_selected_output_renyi(
    first_center: Array | Sequence[float],
    second_center: Array | Sequence[float],
    sigma: float,
    cap: int,
    alpha: float,
    first_occupancy: float,
    second_occupancy: float,
    tilted_occupancy: float,
) -> float:
    """Closed Renyi formula using occupancy at the alpha-tilted center."""

    first_occupancy = _validate_occupancy(first_occupancy, interior=True)
    second_occupancy = _validate_occupancy(second_occupancy, interior=True)
    tilted_occupancy = _validate_occupancy(tilted_occupancy, interior=True)
    cap = _validate_cap(cap)
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha == 1.0:
        raise ValueError("alpha must be positive, finite, and different from one")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    first = np.asarray(first_center, dtype=np.float64)
    second = np.asarray(second_center, dtype=np.float64)
    if first.ndim != 1 or first.size < 1 or second.shape != first.shape:
        raise ValueError("centers must share a nonempty vector shape")
    shift_squared = float(np.dot(first - second, first - second))
    atom = (
        alpha * _log_failure(first_occupancy, cap)
        + (1.0 - alpha) * _log_failure(second_occupancy, cap)
    )
    log_first_g = _log_success(first_occupancy, cap) - math.log(first_occupancy)
    log_second_g = _log_success(second_occupancy, cap) - math.log(second_occupancy)
    accepted = (
        alpha * log_first_g
        + (1.0 - alpha) * log_second_g
        + alpha * (alpha - 1.0) * shift_squared / (2.0 * sigma**2)
        + math.log(tilted_occupancy)
    )
    return float(_logsumexp_pair(atom, accepted) / (alpha - 1.0))


def selected_output_tv_from_overlap(
    first_occupancy: float,
    second_occupancy: float,
    cap: int,
    accepted_scaled_overlap: float,
) -> float:
    """Exact TV given ``integral min(s_u r_u, s_v r_v)``."""

    first_occupancy = _validate_occupancy(first_occupancy)
    second_occupancy = _validate_occupancy(second_occupancy)
    cap = _validate_cap(cap)
    if not math.isfinite(accepted_scaled_overlap) or accepted_scaled_overlap < 0.0:
        raise ValueError("accepted_scaled_overlap must be finite and nonnegative")
    atom_overlap = min(
        capped_failure_probability(first_occupancy, cap),
        capped_failure_probability(second_occupancy, cap),
    )
    total_overlap = atom_overlap + accepted_scaled_overlap
    if total_overlap > 1.0 + 1e-12:
        raise ValueError("total overlap cannot exceed one")
    return float(max(0.0, 1.0 - min(1.0, total_overlap)))


def selected_output_tv_bounds(
    first_occupancy: float,
    second_occupancy: float,
    cap: int,
    conditional_tv: float,
) -> tuple[float, float]:
    first_success = capped_success_probability(first_occupancy, cap)
    second_success = capped_success_probability(second_occupancy, cap)
    conditional_tv = _validate_occupancy(conditional_tv)
    lower = abs(first_success - second_success)
    upper = lower + min(first_success, second_success) * conditional_tv
    return float(lower), float(min(1.0, upper))


def selected_output_fisher(
    occupancy: float,
    cap: int,
    sigma: float,
    occupancy_gradient: Array | Sequence[float],
    conditional_covariance: Array,
) -> Array:
    """Exact local Fisher matrix of the selected output center family."""

    occupancy = _validate_occupancy(occupancy, interior=True)
    cap = _validate_cap(cap)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    gradient = np.asarray(occupancy_gradient, dtype=np.float64)
    covariance = np.asarray(conditional_covariance, dtype=np.float64)
    if gradient.ndim != 1 or gradient.size < 1 or covariance.shape != (gradient.size, gradient.size):
        raise ValueError("gradient and covariance shapes are incompatible")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(covariance)):
        raise ValueError("gradient and covariance must be finite")
    success = capped_success_probability(occupancy, cap)
    accepted_information = success * covariance / sigma**4
    bit_coefficient = cap**2 * (1.0 - occupancy) ** (cap - 2) / success
    return accepted_information + bit_coefficient * np.outer(gradient, gradient)


def renyi_fallback_log_base(
    first_occupancy: float, second_occupancy: float, alpha: float
) -> float:
    """Log base of the order-alpha fallback atom contribution per proposal."""

    first_occupancy = _validate_occupancy(first_occupancy, interior=True)
    second_occupancy = _validate_occupancy(second_occupancy, interior=True)
    if not math.isfinite(alpha) or alpha <= 1.0:
        raise ValueError("rare-fallback slope requires finite alpha greater than one")
    return float(
        alpha * math.log1p(-first_occupancy)
        + (1.0 - alpha) * math.log1p(-second_occupancy)
    )


def renyi_fallback_asymptotic_slope(
    first_occupancy: float, second_occupancy: float, alpha: float
) -> float:
    return float(
        max(0.0, renyi_fallback_log_base(first_occupancy, second_occupancy, alpha))
        / (alpha - 1.0)
    )
