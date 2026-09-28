"""Outward numerical evaluation of binomial and smoothing certificates.

Requires python-flint. Arb encloses every arithmetic and transcendental step.
This validates inference on integer counts, not Gaussian sampling or model
execution in floating-point arithmetic. Each process owns its Arb context.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Literal

from scipy.stats import beta

from feasible_robustness.conditional_renyi import DEFAULT_ORDERS

try:
    from flint import arb, ctx
except ImportError as error:
    raise ImportError("Validated numerics requires python-flint==0.9.0") from error


def exact_rational(value: Fraction | str | int | float) -> arb:
    """Represent the stated rational value without decimal reinterpretation."""
    rational = Fraction(value)
    return arb(rational.numerator) / rational.denominator


def outward_float(value: arb, side: Literal["lower", "upper"]) -> float:
    if side not in ("lower", "upper"):
        raise ValueError("side must be lower or upper")
    endpoint = value.lower() if side == "lower" else value.upper()
    result = float(endpoint)
    if not math.isfinite(result):
        raise ArithmeticError("nonfinite ball endpoint")
    if side == "lower":
        while not arb(result) <= endpoint:
            result = math.nextafter(result, -math.inf)
    else:
        while not arb(result) >= endpoint:
            result = math.nextafter(result, math.inf)
    return result


def binomial_upper_tail(n: int, k: int, p: arb, *, precision: int = 128) -> arb:
    """Enclose P[Binomial(n,p) >= k] by a positive finite sum and remainder.

    Past the mode, successive term ratios decrease. The remaining finite
    sum is bounded above by an infinite geometric tail. Unlike a numerical
    incomplete-beta estimate, this stopping bound is itself an Arb enclosure.
    """
    if not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError("require integer counts with 0 <= k <= n")
    if not 0 <= p <= 1:
        raise ValueError("probability enclosure must be contained in [0,1]")
    with ctx.workprec(precision):
        if k == 0 or p == 1:
            return arb(1)
        if p == 0:
            return arb(0)
        if (n + 1) * p > k:
            return 1 - binomial_upper_tail(n, n - k + 1, 1 - p, precision=precision)
        log_coefficient = (
            arb(n + 1).lgamma() - arb(k + 1).lgamma() - arb(n - k + 1).lgamma()
        )
        term = (log_coefficient + k * p.log() + (n - k) * (1 - p).log()).exp()
        total = term
        tolerance = arb(2) ** (-(precision // 2))
        for j in range(k, n):
            ratio = arb(n - j) * p / ((j + 1) * (1 - p))
            if ratio < 1:
                remainder = term * ratio / (1 - ratio)
                if remainder < total * tolerance:
                    return total + arb(0, remainder.upper())
            term *= ratio
            total += term
        return total


def binomial_endpoint(
    successes: int,
    trials: int,
    error: Fraction | str,
    side: Literal["lower", "upper"],
    *,
    precision: int = 128,
) -> float:
    """Return a float proved outward of the corresponding CP endpoint.

    SciPy supplies an initial candidate only. A candidate is returned only
    after Arb proves the binomial-tail inequality against the exact error.
    For the upper endpoint the initial inverse survival function avoids
    subtracting a small error probability from one.
    """
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("counts must be integers")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    eta_fraction = Fraction(error)
    if not 0 < eta_fraction < 1:
        raise ValueError("error must lie in (0,1)")
    if side not in ("lower", "upper"):
        raise ValueError("side must be lower or upper")
    if side == "lower" and successes == 0:
        return 0.0
    if side == "upper" and successes == trials:
        return 1.0
    if side == "lower":
        candidate = float(
            beta.ppf(float(eta_fraction), successes, trials - successes + 1)
        )
    else:
        candidate = float(
            beta.isf(float(eta_fraction), successes + 1, trials - successes)
        )
    if not math.isfinite(candidate) or not 0 <= candidate <= 1:
        raise ArithmeticError("invalid initial binomial endpoint")
    step = math.ulp(candidate)
    with ctx.workprec(precision):
        eta = exact_rational(eta_fraction)
        for _ in range(128):
            probability = arb(candidate)
            tail = (
                binomial_upper_tail(trials, successes, probability, precision=precision)
                if side == "lower"
                else binomial_upper_tail(
                    trials, trials - successes, 1 - probability, precision=precision
                )
            )
            if tail <= eta:
                return candidate
            # Inverse-beta approximations can miss by thousands of ulps.
            # Widen outwards geometrically, retaining the exact tail test.
            if side == "lower":
                candidate = max(0.0, math.nextafter(candidate - step, 0.0))
            else:
                candidate = min(1.0, math.nextafter(candidate + step, 1.0))
            step *= 2
    raise ArithmeticError("could not prove an outward binomial endpoint")


def gaussian_quantile(p: arb) -> arb:
    if not 0 < p < 1:
        raise ValueError("finite Gaussian quantiles require 0 < p < 1")
    return -arb(2).sqrt() * (2 * p).erfcinv()


def gaussian_radius(
    selected: float,
    competitor: float,
    sigma: float,
    *,
    side: Literal["lower", "upper"] = "lower",
    precision: int = 128,
) -> float:
    """Round the Gaussian two-mass formula outward in the requested direction."""
    if side not in ("lower", "upper"):
        raise ValueError("side must be lower or upper")
    if (
        not 0 <= selected <= 1
        or not 0 <= competitor <= 1
        or not (math.isfinite(sigma) and sigma > 0)
    ):
        raise ValueError("invalid probabilities or noise scale")
    if selected <= competitor:
        return 0.0
    if selected == 1 or competitor == 0:
        return math.inf
    with ctx.workprec(precision):
        radius = (
            arb(sigma)
            / 2
            * (gaussian_quantile(arb(selected)) - gaussian_quantile(arb(competitor)))
        )
        return max(0.0, outward_float(radius, side))


@dataclass(frozen=True)
class ValidatedRenyiCertificate:
    radius_lower: float
    reverse_kl_radius_lower: float
    selected_order: float
    precision_bits: int


def conditional_renyi_radius(
    selected_lower: float,
    competitor_upper: float,
    sigma: float,
    factor: float,
    *,
    region: float = math.inf,
    orders=DEFAULT_ORDERS,
    precision: int = 128,
) -> ValidatedRenyiCertificate:
    """Enclose the finite-order conditional certificate using valid CI inputs."""
    if not (0 <= selected_lower <= 1 and 0 <= competitor_upper <= 1):
        raise ValueError("probabilities must lie in [0,1]")
    if not (
        math.isfinite(sigma) and sigma > 0 and math.isfinite(factor) and factor > 0
    ):
        raise ValueError("sigma and covariance factor must be positive and finite")
    if math.isnan(region) or region <= 0:
        raise ValueError("region radius must be positive")
    candidates = sorted(set([1.0, *orders]))
    if any(not math.isfinite(a) or a < 1 for a in candidates):
        raise ValueError("orders must be finite and at least one")
    best, reverse, best_order = 0.0, 0.0, 1.0
    with ctx.workprec(precision):
        p = arb(selected_lower)
        q = arb(competitor_upper)
        if q > 1 - p:
            q = 1 - p
        if p <= q:
            return ValidatedRenyiCertificate(0.0, 0.0, 1.0, precision)
        if q == 0 and p == 1:
            return ValidatedRenyiCertificate(region, region, 1.0, precision)
        for order in candidates:
            alpha = arb(order)
            if q == 0:
                cost = -(1 - p).log()
            elif order == 1:
                cost = -(1 - p - q + 2 * (p * q).sqrt()).log()
            else:
                t = 1 - alpha
                mean = ((p**t + q**t) / 2) ** (1 / t)
                cost = -(1 - p - q + 2 * mean).log()
            if not cost > 0:
                lower = 0.0
            else:
                radius = arb(sigma) * (2 * cost / (alpha * arb(factor))).sqrt()
                lower = max(0.0, outward_float(radius, "lower"))
                if math.isfinite(region):
                    lower = min(lower, outward_float(arb(region) / alpha, "lower"))
            if order == 1:
                reverse = lower
            if lower > best:
                best, best_order = lower, float(order)
    return ValidatedRenyiCertificate(best, reverse, best_order, precision)
