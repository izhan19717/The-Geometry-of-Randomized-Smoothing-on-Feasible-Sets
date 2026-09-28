"""Checks of the categorical projection and conditional Renyi certificate."""

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from feasible_robustness.conditional_renyi import (
    categorical_reversal_cost,
    conditioned_covariance_renyi_radius,
)
from feasible_robustness.filtered_certificate import conditioned_covariance_kl_radius


@pytest.mark.parametrize("order", [1.0, 1.001, 1.2, 2.0, 8.0])
@pytest.mark.parametrize("pair", [(0.6, 0.2), (0.95, 0.01), (0.34, 0.33)])
def test_cost_matches_independent_categorical_minimization(order, pair):
    p, q = pair
    nominal = np.array([p, q, 1.0 - p - q])

    def objective(t):
        new = np.array([t, t, 1.0 - 2 * t])
        if order == 1:
            return float(np.sum(new * np.log(new / nominal)))
        return float(np.log(np.sum(new**order * nominal ** (1 - order))) / (order - 1))

    optimum = minimize_scalar(
        objective,
        bounds=(1e-10, 0.5 - 1e-10),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert categorical_reversal_cost(p, q, order) == pytest.approx(
        optimum.fun, abs=2e-11
    )


def test_reverse_kl_removes_forward_kl_ceiling_and_dominates():
    for p in np.linspace(0.5001, 0.9999, 100):
        old = conditioned_covariance_kl_radius(p, 1 - p, 0.25, 1.0)
        new = conditioned_covariance_renyi_radius(p, 1 - p, 0.25, 1.0)
        assert new.radius >= new.reverse_kl_radius >= old - 1e-13
    assert conditioned_covariance_renyi_radius(
        0.9999, 0.0001, 0.25, 1.0
    ).radius > 0.25 * math.sqrt(2 * math.log(2))


def test_finite_region_requires_extrapolated_center():
    cert = conditioned_covariance_renyi_radius(
        0.99, 0.01, 1.0, 1.0, certified_region_radius=0.6, orders=[2.0]
    )
    assert cert.radius == 0.6  # order one, not the incorrectly uncapped order two
    assert cert.order == 1.0


def test_gaussian_binary_boundary_is_not_overcertified():
    for distance in [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]:
        p = norm.cdf(distance)
        cert = conditioned_covariance_renyi_radius(p, 1 - p, 1.0, 1.0)
        assert 0 < cert.radius <= distance + 1e-10


def test_near_tie_and_order_one_limit():
    p, q = 0.4000000001, 0.4
    reference = -math.log1p(-((math.sqrt(p) - math.sqrt(q)) ** 2))
    assert categorical_reversal_cost(p, q) == pytest.approx(reference, rel=1e-5)
    assert categorical_reversal_cost(0.9, 0.03, 1 + 1e-10) == pytest.approx(
        categorical_reversal_cost(0.9, 0.03), rel=1e-9
    )


@pytest.mark.parametrize(
    "args", [(-0.1, 0.1, 1), (0.6, 0.5, 1), (0.6, 0.1, 0.5), (math.nan, 0.1, 1)]
)
def test_invalid_cost_inputs(args):
    with pytest.raises(ValueError):
        categorical_reversal_cost(*args)


def test_probability_bounds_and_zero_cells():
    assert conditioned_covariance_renyi_radius(0.3, 0.4, 1.0, 1.0).radius == 0
    assert categorical_reversal_cost(1.0, 0.0) == math.inf
    assert categorical_reversal_cost(0.8, 0.0, 2.0) == pytest.approx(-math.log(0.2))
    # The simplex upper bound is a valid tightening of an imprecise competitor bound.
    assert conditioned_covariance_renyi_radius(
        0.8, 0.5, 1.0, 1.0
    ) == conditioned_covariance_renyi_radius(0.8, 0.2, 1.0, 1.0)


@pytest.mark.parametrize("order", [1.0, 1.1, 2.0, 64.0])
def test_cost_is_monotone_in_probability_bounds(order):
    assert categorical_reversal_cost(0.8, 0.1, order) > categorical_reversal_cost(
        0.7, 0.1, order
    )
    assert categorical_reversal_cost(0.8, 0.1, order) > categorical_reversal_cost(
        0.8, 0.2, order
    )


def test_subnormal_probability_and_large_order_remain_finite():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    assert categorical_reversal_cost(0.8, tiny, 2.0) == pytest.approx(-math.log(0.2))
    assert categorical_reversal_cost(0.8, 0.1, 1e308) == pytest.approx(-math.log(0.3))


def test_analytic_two_band_boundary_is_not_overcertified():
    for distance in np.geomspace(0.0001, 3.0, 60):
        selected = norm.cdf(1 - distance) - norm.cdf(0.9 - distance)
        runner = norm.cdf(1 + distance) - norm.cdf(0.9 + distance)
        p, q = selected / (selected + runner), runner / (selected + runner)
        cert = conditioned_covariance_renyi_radius(p, q, 1.0, 1.0)
        assert 0 < cert.radius <= distance + 2e-11
