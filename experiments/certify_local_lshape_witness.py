"""Rigorously certify the rational local L-shape KL and exact-TV witness.

The proof computation uses exact ``Fraction`` arithmetic and the outward
enclosures in ``feasible_robustness.rational_interval``.  It intentionally
does not call NumPy, SciPy, mpmath, or platform transcendental functions.

Run from the repository root with::

    PYTHONPATH=src python3 experiments/certify_local_lshape_witness.py
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from feasible_robustness.rational_interval import (
    RationalInterval,
    decimal_string,
    gaussian_kernel_mass,
    gaussian_kernel_value,
    log_interval,
    outward_decimal_interval,
    pi_interval,
    sqrt_interval,
)

Q = Fraction
TOLERANCE = Q(1, 10**40)
DISPLAY_DIGITS = 24


def mass(lower: Fraction, upper: Fraction) -> RationalInterval:
    return gaussian_kernel_mass(lower, upper, TOLERANCE)


def variable_upper_mass(lower: Fraction, upper: RationalInterval) -> RationalInterval:
    """Use positivity of the kernel to enclose an interval-valued upper limit."""

    lower_mass = mass(lower, upper.lo)
    upper_mass = mass(lower, upper.hi)
    return RationalInterval(lower_mass.lo, upper_mass.hi)


def first_moment_factor(
    center: Fraction,
    sigma: Fraction,
    alpha: Fraction,
    beta: Fraction,
    interval_mass: RationalInterval,
) -> RationalInterval:
    """Unnormalized 1D factor ``c J + sigma(k(alpha)-k(beta))``."""

    return center * interval_mass + sigma * (
        gaussian_kernel_value(alpha, TOLERANCE)
        - gaussian_kernel_value(beta, TOLERANCE)
    )


def second_moment_factor(
    center: Fraction,
    sigma: Fraction,
    alpha: Fraction,
    beta: Fraction,
    interval_mass: RationalInterval,
) -> RationalInterval:
    """Unnormalized 1D second-moment factor over fixed standardized bounds."""

    kernel_alpha = gaussian_kernel_value(alpha, TOLERANCE)
    kernel_beta = gaussian_kernel_value(beta, TOLERANCE)
    first_standardized = kernel_alpha - kernel_beta
    second_standardized = (
        interval_mass + alpha * kernel_alpha - beta * kernel_beta
    )
    return (
        center * center * interval_mass
        + 2 * center * sigma * first_standardized
        + sigma * sigma * second_standardized
    )


def variable_upper_first_moment_factor(
    center: Fraction,
    sigma: Fraction,
    alpha: Fraction,
    beta: RationalInterval,
) -> RationalInterval:
    """Enclose the first moment when a positive physical upper limit varies.

    In this witness the physical integral is over ``[0, upper]`` with
    ``upper in (1/2, 1)``.  Its integrand ``x exp(-(x-center)^2/(2 sigma^2))``
    is non-negative, so the integral is monotone in the upper endpoint.
    """

    lower_mass = mass(alpha, beta.lo)
    upper_mass = mass(alpha, beta.hi)
    lower_value = first_moment_factor(center, sigma, alpha, beta.lo, lower_mass)
    upper_value = first_moment_factor(center, sigma, alpha, beta.hi, upper_mass)
    return RationalInterval(lower_value.lo, upper_value.hi)


def certify() -> dict[str, Any]:
    sigma = Q(3, 20)
    ax = Q(13, 20)
    bx = Q(66, 100)
    ay = Q(49, 100)
    displacement = bx - ax
    if sigma <= 0 or displacement <= 0:
        raise AssertionError("witness requires sigma>0 and a positive horizontal displacement")

    # Exact standardized endpoints for K=([0,1]x[0,1/2]) U
    # ([0,1/2]x[1/2,1]).
    ay_bottom_lo, ay_bottom_hi = Q(-49, 15), Q(1, 15)
    ay_upper_lo, ay_upper_hi = Q(1, 15), Q(17, 5)
    ax_full_lo, ax_full_hi = Q(-13, 3), Q(7, 3)
    ax_left_hi = Q(-1)
    bx_full_lo, bx_full_hi = Q(-22, 5), Q(34, 15)
    bx_left_hi = Q(-16, 15)

    y_bottom = mass(ay_bottom_lo, ay_bottom_hi)
    y_upper = mass(ay_upper_lo, ay_upper_hi)
    a_full_x = mass(ax_full_lo, ax_full_hi)
    a_left_x = mass(ax_full_lo, ax_left_hi)
    b_full_x = mass(bx_full_lo, bx_full_hi)
    b_left_x = mass(bx_full_lo, bx_left_hi)

    weight_a = a_full_x * y_bottom + a_left_x * y_upper
    weight_b = b_full_x * y_bottom + b_left_x * y_upper
    if weight_a.lo <= 0 or weight_b.lo <= 0:
        raise AssertionError("rectangle-union weights were not certified positive")
    log_normalizer_ratio = log_interval(weight_b / weight_a, TOLERANCE)

    first_a = (
        first_moment_factor(ax, sigma, ax_full_lo, ax_full_hi, a_full_x) * y_bottom
        + first_moment_factor(ax, sigma, ax_full_lo, ax_left_hi, a_left_x) * y_upper
    )
    mean_a_x = first_a / weight_a

    second_a = (
        second_moment_factor(ax, sigma, ax_full_lo, ax_full_hi, a_full_x) * y_bottom
        + second_moment_factor(ax, sigma, ax_full_lo, ax_left_hi, a_left_x) * y_upper
    )
    second_moment_a_x = second_a / weight_a
    variance_a_x = second_moment_a_x - mean_a_x * mean_a_x
    variance_ratio = variance_a_x / (sigma * sigma)

    # For the mean absolute deviation, certify the split point first.  Since
    # E[X-mu]=0, E|X-mu|=2 E[(mu-X)_+].  The rounded interval still encloses
    # the exact mean but keeps variable-endpoint series denominators compact.
    mean_split_lower, mean_split_upper = outward_decimal_interval(mean_a_x, 30)
    mean_split = RationalInterval(mean_split_lower, mean_split_upper)
    if mean_split.lo <= Q(1, 2) or mean_split.hi >= 1:
        raise AssertionError("MAD split location was not certified inside (1/2, 1)")
    mean_standardized = (mean_split - ax) / sigma
    bottom_left_mass_at_mean = variable_upper_mass(ax_full_lo, mean_standardized)
    bottom_left_first_at_mean = variable_upper_first_moment_factor(
        ax, sigma, ax_full_lo, mean_standardized
    )
    upper_left_first = first_moment_factor(
        ax, sigma, ax_full_lo, ax_left_hi, a_left_x
    )
    left_weight_at_mean = bottom_left_mass_at_mean * y_bottom + a_left_x * y_upper
    left_first_at_mean = (
        bottom_left_first_at_mean * y_bottom + upper_left_first * y_upper
    )
    left_deviation = mean_split * left_weight_at_mean - left_first_at_mean
    conditional_mad_a_x = 2 * left_deviation / weight_a

    gaussian_kl = displacement * displacement / (2 * sigma * sigma)
    if gaussian_kl != Q(1, 450):
        raise AssertionError("unexpected Gaussian KL comparator")
    mean_drift_excess = (mean_a_x - ax) * (ax - bx) / (sigma * sigma)
    kl_excess = log_normalizer_ratio + mean_drift_excess
    kl = gaussian_kl + kl_excess

    crossing_x_exact = (ax + bx) / 2 + (
        sigma * sigma / displacement
    ) * log_normalizer_ratio
    # The log-series endpoints have very large exact denominators.  Rounding
    # this already-certified interval outwards to a 30-digit rational grid
    # keeps the subsequent power-series arithmetic compact without sacrificing
    # any relevant precision.
    crossing_lower, crossing_upper = outward_decimal_interval(crossing_x_exact, 30)
    crossing_x = RationalInterval(crossing_lower, crossing_upper)
    if crossing_x.lo <= Q(1, 2) or crossing_x.hi >= 1:
        raise AssertionError("likelihood crossing was not certified inside (1/2, 1)")

    a_cross_standardized = (crossing_x - ax) / sigma
    b_cross_standardized = (crossing_x - bx) / sigma
    a_bottom_left = variable_upper_mass(ax_full_lo, a_cross_standardized)
    b_bottom_left = variable_upper_mass(bx_full_lo, b_cross_standardized)
    left_weight_a = a_bottom_left * y_bottom + a_left_x * y_upper
    left_weight_b = b_bottom_left * y_bottom + b_left_x * y_upper
    cdf_a = left_weight_a / weight_a
    cdf_b = left_weight_b / weight_b
    exact_tv = cdf_a - cdf_b

    # A second, simpler TV proof uses the fixed rational event
    # H={z_1 <= 623/1000}.  Since TV is the supremum over events, its exact
    # probability gap is a lower bound independent of the likelihood crossing.
    fixed_cut = Q(623, 1000)
    if not Q(1, 2) < fixed_cut < 1:
        raise AssertionError("fixed TV event must contain the whole upper-left rectangle")
    fixed_a_bottom = mass(ax_full_lo, (fixed_cut - ax) / sigma)  # upper = -9/50
    fixed_b_bottom = mass(bx_full_lo, (fixed_cut - bx) / sigma)  # upper = -37/150
    fixed_cdf_a = (fixed_a_bottom * y_bottom + a_left_x * y_upper) / weight_a
    fixed_cdf_b = (fixed_b_bottom * y_bottom + b_left_x * y_upper) / weight_b
    fixed_event_gap = fixed_cdf_a - fixed_cdf_b

    # Operational binary classifier f(z)=1{z_1 <= 633/1000}.  This cut lies in
    # the bottom rectangle's x range and includes the entire upper-left box.
    decision_cut = Q(633, 1000)
    if not Q(1, 2) < decision_cut < 1:
        raise AssertionError("decision event must contain the whole upper-left rectangle")
    decision_a_bottom = mass(ax_full_lo, (decision_cut - ax) / sigma)  # -17/150
    decision_b_bottom = mass(bx_full_lo, (decision_cut - bx) / sigma)  # -9/50
    decision_probability_a = (
        decision_a_bottom * y_bottom + a_left_x * y_upper
    ) / weight_a
    decision_probability_b = (
        decision_b_bottom * y_bottom + b_left_x * y_upper
    ) / weight_b
    decision_margin_a = 2 * decision_probability_a - 1
    decision_half_gap_b = Q(1, 2) - decision_probability_b

    pi = pi_interval(TOLERANCE)
    standard_normalizer = sqrt_interval(2 * pi, 38)
    gaussian_tv = mass(Q(-1, 30), Q(1, 30)) / standard_normalizer
    gaussian_mad = 2 * sigma / standard_normalizer
    mad_ratio = conditional_mad_a_x / gaussian_mad
    tv_excess = exact_tv - gaussian_tv
    fixed_event_excess = fixed_event_gap - gaussian_tv
    gaussian_rate_decision_slack = decision_margin_a - 2 * gaussian_tv
    cohen_probability_threshold = Q(1, 2) + mass(Q(0), Q(1, 15)) / standard_normalizer
    cohen_probability_slack = decision_probability_a - cohen_probability_threshold

    if (
        kl_excess.lo <= 0
        or variance_ratio.lo <= 1
        or mad_ratio.lo <= 1
        or tv_excess.lo <= 0
        or fixed_event_excess.lo <= 0
        or decision_probability_a.lo <= Q(1, 2)
        or decision_probability_b.hi >= Q(1, 2)
        or gaussian_rate_decision_slack.lo <= 0
        or cohen_probability_slack.lo <= 0
    ):
        raise AssertionError("strict witness excess was not certified")

    intervals = {
        "weight_a_unnormalized": weight_a,
        "weight_b_unnormalized": weight_b,
        "log_normalizer_ratio": log_normalizer_ratio,
        "mean_a_x": mean_a_x,
        "second_moment_a_x": second_moment_a_x,
        "variance_a_x": variance_a_x,
        "variance_over_sigma_squared": variance_ratio,
        "mad_split_location": mean_split,
        "conditional_mad_a_x": conditional_mad_a_x,
        "gaussian_mad": gaussian_mad,
        "conditional_mad_over_gaussian_mad": mad_ratio,
        "mean_drift_excess": mean_drift_excess,
        "gaussian_kl": RationalInterval.point(gaussian_kl),
        "kl": kl,
        "kl_excess": kl_excess,
        "crossing_x": crossing_x,
        "cdf_a_at_crossing": cdf_a,
        "cdf_b_at_crossing": cdf_b,
        "exact_tv": exact_tv,
        "fixed_event_cut": RationalInterval.point(fixed_cut),
        "fixed_event_probability_gap": fixed_event_gap,
        "gaussian_tv": gaussian_tv,
        "tv_excess": tv_excess,
        "fixed_event_excess_over_gaussian_tv": fixed_event_excess,
        "decision_event_cut": RationalInterval.point(decision_cut),
        "decision_event_probability_a": decision_probability_a,
        "decision_event_probability_b": decision_probability_b,
        "decision_event_margin_a": decision_margin_a,
        "decision_event_half_gap_b": decision_half_gap_b,
        "twice_gaussian_tv": 2 * gaussian_tv,
        "gaussian_rate_false_certificate_slack": gaussian_rate_decision_slack,
        "cohen_probability_threshold_phi_1_over_15": cohen_probability_threshold,
        "cohen_probability_slack": cohen_probability_slack,
        "pi": pi,
        "sqrt_2pi": standard_normalizer,
    }
    return {
        "witness": {
            "region": "([0,1]x[0,1/2]) union ([0,1/2]x[1/2,1])",
            "sigma": "3/20",
            "a": ["13/20", "49/100"],
            "b": ["66/100", "49/100"],
        },
        "internal_tolerance": "1/10^40",
        "intervals": intervals,
        "certified": {
            "crossing_in_half_to_one": True,
            "exact_tv_value_enclosed": True,
            "kl_strictly_exceeds_gaussian": True,
            "tv_strictly_exceeds_gaussian": True,
            "fixed_rational_event_alone_proves_tv_excess": True,
            "binary_smoothed_label_flips_between_a_and_b": True,
            "gaussian_rate_tv_test_would_include_b": True,
            "local_kl_variance_rate_exceeds_gaussian": True,
            "local_tv_mad_rate_exceeds_gaussian": True,
            "ordinary_binary_cohen_certificate_would_include_b": True,
        },
        "rational_margin_guarantees": {
            "kl_excess_strictly_greater_than": "1/5000",
            "exact_tv_excess_strictly_greater_than": "1/500",
            "fixed_event_excess_strictly_greater_than": "1/500",
            "decision_probability_a_strictly_greater_than": "527/1000",
            "decision_probability_b_strictly_less_than": "499/1000",
            "gaussian_rate_false_certificate_slack_strictly_greater_than": "1/1000",
            "variance_over_sigma_squared_strictly_greater_than": "109/100",
            "conditional_mad_over_gaussian_mad_strictly_greater_than": "107/100",
            "cohen_probability_slack_strictly_greater_than": "9/10000",
        },
    }


def interval_payload(value: RationalInterval, digits: int = DISPLAY_DIGITS) -> dict[str, str]:
    lower, upper = outward_decimal_interval(value, digits)
    return {
        "lower": decimal_string(lower, digits),
        "upper": decimal_string(upper, digits),
        "lower_rational": f"{lower.numerator}/{lower.denominator}",
        "upper_rational": f"{upper.numerator}/{upper.denominator}",
    }


def public_payload(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "witness": certificate["witness"],
        "method": (
            "Exact Fraction arithmetic; alternating Gaussian-kernel/integral and "
            "Machin-arctangent brackets; atanh log bound; decimal-grid outward rounding"
        ),
        "series_preconditions_and_remainders": {
            "exp_neg": (
                "For y>=0, A_n=y^n/n!. Once n+1>=y the terms decrease; "
                "S_(2m+1)<=exp(-y)<=S_(2m), with bracket width A_(2m+1)."
            ),
            "gaussian_integral": (
                "For t>=0, A_n=t^(2n+1)/(2^n n! (2n+1)); "
                "A_(n+1)/A_n < (t^2/2)/(n+1). After n+1>=t^2/2, "
                "adjacent odd/even partial sums bracket the integral. Negative t uses oddness."
            ),
            "log": (
                "For x>0, z=(x-1)/(x+1) has |z|<1 and "
                "log(x)=2 sum z^(2k+1)/(2k+1). For z>=0 the omitted tail is at most "
                "2*z^(2N+3)/((2N+3)*(1-z^2)); x<1 is handled by log(x)=-log(1/x)."
            ),
            "pi": (
                "Machin identity pi=16 atan(1/5)-4 atan(1/239); each atan series has "
                "decreasing alternating terms for 0<=z<=1, so adjacent partial sums bracket it."
            ),
            "sqrt_and_decimal_rounding": (
                "Inputs are nonnegative rational intervals; integer square-root comparisons and "
                "integer floor/ceiling division round both endpoints outwards."
            ),
        },
        "interval_propagation_preconditions": {
            "sigma_positive": "sigma=3/20>0",
            "horizontal_displacement_positive": "b_1-a_1=1/100>0",
            "normalizers_positive": (
                "Both unnormalized rectangle-union weight intervals have positive lower bounds"
            ),
            "division": "Every divisor interval is checked not to contain zero",
            "log": "The normalizer-ratio interval has a strictly positive lower endpoint",
            "exact_tv_event_shape": "The certified crossing interval is contained in (1/2,1)",
            "fixed_tv_event_shape": "623/1000 is contained in (1/2,1)",
            "decision_event_shape": "633/1000 is contained in (1/2,1)",
            "mad_split_shape": "The outward mean interval is contained in (1/2,1)",
            "cohen_equivalence": (
                "Phi is strictly increasing and ||a-b||/sigma=1/15"
            ),
        },
        "analytic_moment_and_certificate_identities": {
            "second_moment": (
                "On standardized [alpha,beta], the unnormalized second moment is "
                "c^2 J + 2 c sigma (k(alpha)-k(beta)) + sigma^2 "
                "(J + alpha k(alpha) - beta k(beta))."
            ),
            "mean_absolute_deviation": (
                "For mu=E[X], E|X-mu|=2 E[(mu-X)_+]; the certified mu interval "
                "lies in (1/2,1), fixing which L-shape rectangles are split."
            ),
            "binary_cohen": (
                "For a binary Gaussian smoother, Cohen radius is sigma*Phi^-1(p_A); "
                "strict monotonicity makes radius>||a-b|| equivalent to "
                "p_A>Phi(||a-b||/sigma)=Phi(1/15)."
            ),
        },
        "internal_tolerance": certificate["internal_tolerance"],
        "published_decimal_digits": DISPLAY_DIGITS,
        "intervals": {
            name: interval_payload(value) for name, value in certificate["intervals"].items()
        },
        "certified": certificate["certified"],
        "rational_margin_guarantees": certificate["rational_margin_guarantees"],
        "trust_assumptions": [
            "Correctness of Python arbitrary-precision integers and fractions.Fraction",
            (
                "Correct implementation of the elementary series remainder arguments "
                "documented in rational_interval.py"
            ),
            (
                "The analytic rectangle-union KL/first- and second-moment identities, "
                "split-at-mean MAD identity, one-crossing total-variation identity, "
                "and binary Cohen formula proved in Appendix F of the manuscript"
            ),
        ],
        "outside_trusted_computing_base": [
            "python-flint/Arb 0.9.0 (optional independent cross-check; not imported)",
            "NumPy, SciPy, mpmath, and platform floating-point transcendental functions",
            "Monte Carlo and numerical quadrature",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/local_lshape_interval_certificate.json"),
    )
    args = parser.parse_args()
    payload = public_payload(certify())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "certified": payload["certified"]}, indent=2))


if __name__ == "__main__":
    main()
