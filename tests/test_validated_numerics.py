"""Independent exact-arithmetic checks for numerical probability enclosures."""

from fractions import Fraction
from math import comb

import pytest

pytest.importorskip("flint")
from flint import arb, ctx
from feasible_robustness.validated_numerics import (
    binomial_upper_tail,
    binomial_endpoint,
    conditional_renyi_radius,
    exact_rational,
    gaussian_radius,
    outward_float,
)


@pytest.mark.parametrize(
    "n,k,p",
    [
        (8, 3, Fraction(1, 4)),
        (20, 13, Fraction(2, 3)),
        (40, 2, Fraction(1, 50)),
        (8, 2, Fraction(3, 4)),
    ],
)
def test_binomial_tail_contains_exact_rational_sum(n, k, p):
    exact = sum(
        Fraction(comb(n, j)) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1)
    )
    with ctx.workprec(256):
        ball = binomial_upper_tail(n, k, exact_rational(p), precision=128)
        assert ball.contains(exact_rational(exact))


@pytest.mark.parametrize("k,n", [(0, 20), (1, 20), (10, 20), (19, 20), (20, 20)])
def test_cp_endpoints_satisfy_exact_rational_tail_test(k, n):
    eta = Fraction(1, 1000)
    lower = Fraction(binomial_endpoint(k, n, eta, "lower"))
    upper = Fraction(binomial_endpoint(k, n, eta, "upper"))
    if k:
        tail = sum(
            Fraction(comb(n, j)) * lower**j * (1 - lower) ** (n - j)
            for j in range(k, n + 1)
        )
        assert tail <= eta
    if k < n:
        tail = sum(
            Fraction(comb(n, j)) * upper**j * (1 - upper) ** (n - j)
            for j in range(k + 1)
        )
        assert tail <= eta


def test_outward_float_is_not_nearest_rounding():
    with ctx.workprec(128):
        value = arb(1) / 10
        assert Fraction(outward_float(value, "lower")) <= Fraction(1, 10)
        assert Fraction(outward_float(value, "upper")) >= Fraction(1, 10)


def test_probability_extremes_and_local_region():
    cert = conditional_renyi_radius(0.99, 0.01, 0.25, 1.0, region=0.01)
    assert cert.radius_lower <= 0.01
    assert conditional_renyi_radius(0.4, 0.5, 0.25, 1.0).radius_lower == 0
    assert gaussian_radius(0.4, 0.5, 0.25) == 0


def test_gaussian_radius_encloses_high_precision_formula():
    with ctx.workprec(256):
        p, q = arb(0.9), arb(0.02)
        exact = (
            arb(0.25)
            / 2
            * arb(2).sqrt()
            * ((2 * p - 1).erfinv() - (2 * q - 1).erfinv())
        )
        assert arb(gaussian_radius(0.9, 0.02, 0.25, side="lower")) <= exact
        assert arb(gaussian_radius(0.9, 0.02, 0.25, side="upper")) >= exact


def test_renyi_radius_is_no_larger_than_high_precision_maximum():
    low = conditional_renyi_radius(0.999, 0.0001, 0.25, 1.0, precision=128)
    high = conditional_renyi_radius(0.999, 0.0001, 0.25, 1.0, precision=256)
    assert low.radius_lower <= high.radius_lower
    assert low.radius_lower >= 0.99 * high.radius_lower


def test_large_count_candidate_is_proved_even_when_more_than_1024_ulps_away():
    eta = Fraction(1, 64_000_000)
    bound = binomial_endpoint(5722, 100000, eta, "lower")
    with ctx.workprec(256):
        assert binomial_upper_tail(
            100000, 5722, arb(bound), precision=256
        ) <= exact_rational(eta)


@pytest.mark.parametrize("side", ["nearest", "invalid"])
def test_invalid_rounding_direction_rejected(side):
    with pytest.raises(ValueError):
        gaussian_radius(0.9, 0.1, 0.25, side=side)
    with pytest.raises(ValueError):
        outward_float(arb(1), side)


def test_nonfinite_noise_scale_rejected():
    with pytest.raises(ValueError):
        gaussian_radius(0.9, 0.1, float("inf"))
