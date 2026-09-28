"""Certify the nonconvex two-band covariance-radius example.

The numerical proof uses exact ``Fraction`` arithmetic and outward rational
intervals. It does not use floating-point transcendental functions.

Run from the repository root with

``PYTHONPATH=src python3 experiments/certify_two_band_covariance_radius.py``.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from feasible_robustness.rational_interval import (
    RationalInterval,
    decimal_string,
    gaussian_integral_from_zero,
    gaussian_kernel_mass,
    log_interval,
    outward_decimal_interval,
    pi_interval,
    sqrt_interval,
)


Q = Fraction
TOLERANCE = Q(1, 10**55)
DISPLAY_DIGITS = 20


def standard_normalizer() -> RationalInterval:
    return sqrt_interval(2 * pi_interval(TOLERANCE), 50)


def normal_cdf(x: Fraction, normalizer: RationalInterval) -> RationalInterval:
    return (
        RationalInterval.point(Q(1, 2))
        + gaussian_integral_from_zero(x, TOLERANCE) / normalizer
    )


def normal_quantile(
    probability: RationalInterval,
    lower: Fraction,
    upper: Fraction,
    normalizer: RationalInterval,
) -> RationalInterval:
    """Enclose every Gaussian quantile for the probability interval."""

    if normal_cdf(lower, normalizer).hi >= probability.lo:
        raise AssertionError("lower Gaussian quantile bracket is not strict")
    if normal_cdf(upper, normalizer).lo <= probability.hi:
        raise AssertionError("upper Gaussian quantile bracket is not strict")
    for _ in range(220):
        midpoint = (lower + upper) / 2
        midpoint_cdf = normal_cdf(midpoint, normalizer)
        if midpoint_cdf.hi < probability.lo:
            lower = midpoint
        elif midpoint_cdf.lo > probability.hi:
            upper = midpoint
        else:
            break
    return RationalInterval(lower, upper)


def certify() -> dict[str, Any]:
    """Return outward intervals and strict checks for the published example."""

    inner_edge = Q(9, 10)
    nominal_distance = Q(1, 10)
    normalizer = standard_normalizer()

    # Reflection maps the lower band under center -delta to [rho-delta,1-delta].
    selected_kernel = gaussian_kernel_mass(
        inner_edge - nominal_distance,
        1 - nominal_distance,
        TOLERANCE,
    )
    runner_kernel = gaussian_kernel_mass(
        inner_edge + nominal_distance,
        1 + nominal_distance,
        TOLERANCE,
    )
    selected_mass = selected_kernel / normalizer
    runner_mass = runner_kernel / normalizer
    occupancy = selected_mass + runner_mass
    selected_probability = selected_kernel / (selected_kernel + runner_kernel)
    runner_probability = RationalInterval.point(1) - selected_probability

    substituted_radius = normal_quantile(
        selected_probability, Q(0), Q(3, 10), normalizer
    )
    selected_quantile = normal_quantile(
        selected_mass, Q(-3), Q(-1), normalizer
    )
    runner_quantile = normal_quantile(
        runner_mass, Q(-3), Q(-1), normalizer
    )
    joint_mass_radius = (selected_quantile - runner_quantile) / 2

    categorical_kl = selected_probability * log_interval(
        2 * selected_probability, TOLERANCE
    ) + runner_probability * log_interval(2 * runner_probability, TOLERANCE)
    covariance_radius = sqrt_interval(2 * categorical_kl, 45)

    gaussian_boundary_probability = normal_cdf(nominal_distance, normalizer)
    checks = {
        "selected_band_is_top": selected_mass.lo > runner_mass.hi,
        "substituted_radius_crosses_exact_boundary": (
            selected_probability.lo > gaussian_boundary_probability.hi
        ),
        "covariance_radius_beats_joint_mass": (
            covariance_radius.lo > joint_mass_radius.hi
        ),
        "covariance_radius_is_more_than_twice_joint_mass": (
            covariance_radius.lo > 2 * joint_mass_radius.hi
        ),
        "covariance_radius_below_exact_boundary": (
            covariance_radius.hi < nominal_distance
        ),
        "covariance_radius_recovers_more_than_94_percent": (
            covariance_radius.lo > Q(94, 100) * nominal_distance
        ),
        "substituted_radius_exceeds_118_percent": (
            substituted_radius.lo > Q(118, 100) * nominal_distance
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    return {
        "family": {
            "affine_space": "R^2",
            "sigma": "1",
            "retained_set": "R x ([-1,-9/10] union [9/10,1])",
            "nominal_center": ["0", "-1/10"],
            "exact_decision_boundary": "second center coordinate equals 0",
            "global_covariance_factor": "Lambda=1",
        },
        "intervals": {
            "occupancy": occupancy,
            "selected_joint_mass": selected_mass,
            "runner_joint_mass": runner_mass,
            "selected_conditional_probability": selected_probability,
            "runner_conditional_probability": runner_probability,
            "substituted_conditional_radius": substituted_radius,
            "joint_mass_radius": joint_mass_radius,
            "categorical_kl_to_tie": categorical_kl,
            "covariance_controlled_radius": covariance_radius,
            "exact_boundary_distance": RationalInterval.point(nominal_distance),
        },
        "certified": checks,
    }


def interval_payload(value: RationalInterval) -> dict[str, str]:
    lower, upper = outward_decimal_interval(value, DISPLAY_DIGITS)
    return {
        "lower": decimal_string(lower, DISPLAY_DIGITS),
        "upper": decimal_string(upper, DISPLAY_DIGITS),
        "lower_rational": f"{lower.numerator}/{lower.denominator}",
        "upper_rational": f"{upper.numerator}/{upper.denominator}",
    }


def public_payload(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": certificate["family"],
        "method": (
            "Exact Fraction arithmetic with alternating Gaussian integral "
            "bounds, an atanh logarithm bound, and outward square roots"
        ),
        "analytic_guarantees": {
            "factorization": (
                "The first coordinate remains N(c_1,1) and is independent of "
                "the conditioned second coordinate"
            ),
            "uniform_covariance": (
                "The retained second coordinate lies in [-1,1], so Popoviciu "
                "gives variance at most 1 at every center; the free coordinate "
                "has variance 1"
            ),
            "nonconvex_unbounded_support": (
                "The two horizontal bands form a nonconvex set of infinite diameter"
            ),
            "exact_boundary": (
                "Reflection gives equal band masses at second-coordinate center 0; "
                "pointwise Gaussian likelihood ordering gives opposite strict order "
                "on either side"
            ),
        },
        "internal_tolerance": "1/10^55",
        "published_decimal_digits": DISPLAY_DIGITS,
        "intervals": {
            key: interval_payload(value)
            for key, value in certificate["intervals"].items()
        },
        "certified": certificate["certified"],
        "trust_assumptions": [
            "Correctness of Python arbitrary-precision integers and fractions.Fraction",
            "Correct implementation of the documented rational interval primitives",
            "The factorization, Popoviciu variance bound, and radius formulas proved in the paper",
        ],
        "outside_trusted_computing_base": [
            "NumPy, SciPy, mpmath, and platform floating-point transcendental functions",
            "Monte Carlo and numerical quadrature",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/two_band_covariance_radius_certificate.json"),
    )
    args = parser.parse_args()
    payload = public_payload(certify())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "certified": payload["certified"]}, indent=2))


if __name__ == "__main__":
    main()
