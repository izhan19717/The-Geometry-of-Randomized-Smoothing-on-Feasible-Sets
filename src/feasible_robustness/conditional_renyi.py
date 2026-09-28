"""Conditional certificates under a uniform bound on Gaussian covariance.

The categorical reversal cost is from Li et al., NeurIPS 2019, Lemma 1.
For order alpha > 1, the covariance premise must hold out to alpha times
the certified displacement. Ordinary floating-point evaluation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable

DEFAULT_ORDERS = (
    (1.0,)
    + tuple(1.0 + k / 20 for k in range(1, 81))
    + (
        6.0,
        8.0,
        12.0,
        16.0,
        32.0,
        64.0,
    )
)


@dataclass(frozen=True)
class ConditionalRenyiCertificate:
    radius: float
    order: float
    reverse_kl_radius: float
    certified_region_radius: float
    covariance_factor_upper: float


def _probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probabilities must be finite and in [0, 1]")
    return value


def categorical_reversal_cost(p: float, q: float, order: float = 1.0) -> float:
    """Least D_order(new labels || nominal labels) permitting a competitor tie.

    p and q are the selected and competitor probabilities, or conservative
    lower and upper bounds with p + q <= 1. The limit at order one is reverse
    KL. A nonpositive margin returns zero. Zero cells use continuous limits.
    """
    p, q = _probability(p), _probability(q)
    order = float(order)
    if not math.isfinite(order) or order < 1.0:
        raise ValueError("order must be finite and at least one")
    if p + q > 1.0 + 1e-14:
        raise ValueError(
            "selected and competitor probabilities must sum to at most one"
        )
    if p <= q:
        return 0.0
    if q == 0.0:
        return math.inf if p == 1.0 else -math.log1p(-p)

    root_p, root_q = math.sqrt(p), math.sqrt(q)
    difference = (p - q) / (root_p + root_q)
    removed_mass = difference * difference
    if order > 1.0:
        # M_t = sqrt(pq) exp(log(cosh(t log(p/q)/2))/t), t < 0.
        # This form avoids subtracting nearly equal power means near p == q.
        t = 1.0 - order
        ratio = (p - q) / q
        half_log_ratio = 0.5 * (
            math.log1p(ratio) if math.isfinite(ratio) else math.log(p) - math.log(q)
        )
        x = abs(t * half_log_ratio)
        log_cosh = (
            math.log1p(2.0 * math.sinh(x / 2.0) ** 2)
            if x < 20.0
            else x + math.log1p(math.exp(-2.0 * x)) - math.log(2.0)
        )
        log_mean_adjustment = log_cosh / t if math.isfinite(x) else -half_log_ratio
        removed_mass += 2.0 * root_p * root_q * (-math.expm1(log_mean_adjustment))
    if removed_mass < 0.9:
        return -math.log1p(-removed_mass)
    # Retain the small remainder when p is very close to one.
    if order == 1.0:
        mean = root_p * root_q
    else:
        mean = root_p * root_q * math.exp(log_mean_adjustment)
    remainder = max(0.0, 1.0 - p - q) + 2.0 * mean
    return math.inf if remainder == 0.0 else -math.log(remainder)


def conditioned_covariance_renyi_radius(
    selected_probability_lower: float,
    runner_probability_upper: float,
    sigma: float,
    covariance_factor_upper: float,
    *,
    certified_region_radius: float = math.inf,
    orders: Iterable[float] = DEFAULT_ORDERS,
) -> ConditionalRenyiCertificate:
    """Maximize valid conditional radii over a finite set of Renyi orders.

    The caller must certify Cov(Q_c) <= factor * sigma**2 * I throughout
    the open center ball of the supplied region radius. A sample covariance
    at the nominal center is insufficient. At alpha > 1, the region cap is
    R / alpha because the tilted center lies beyond the queried endpoint.
    Selection of the best order reuses the same probability confidence event.
    """
    p = _probability(selected_probability_lower)
    q = min(_probability(runner_probability_upper), 1.0 - p)
    sigma, factor, region = map(
        float, (sigma, covariance_factor_upper, certified_region_radius)
    )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("covariance_factor_upper must be finite and positive")
    if math.isnan(region) or region <= 0.0:
        raise ValueError("certified_region_radius must be positive")
    candidates = tuple(sorted(set((1.0, *(float(a) for a in orders)))))
    if any(not math.isfinite(a) or a < 1.0 for a in candidates):
        raise ValueError("every order must be finite and at least one")
    best_radius, best_order, reverse = 0.0, 1.0, 0.0
    for order in candidates:
        cost = categorical_reversal_cost(p, q, order)
        radius = min(region / order, sigma * math.sqrt(2.0 * cost / (order * factor)))
        if order == 1.0:
            reverse = radius
        if radius > best_radius:
            best_radius, best_order = radius, order
    return ConditionalRenyiCertificate(best_radius, best_order, reverse, region, factor)
