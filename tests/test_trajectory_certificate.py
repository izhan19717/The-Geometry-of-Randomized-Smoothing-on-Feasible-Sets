import math

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import beta

from feasible_robustness.ccfs import comparison_multiplier
from feasible_robustness.trajectory_certificate import (
    certify_trajectory_event_from_counts,
    limit_shift_to_remaining_energy,
    one_sided_binomial_lower_bound,
    trajectory_shift_distance,
)


def test_one_sided_lower_bound_matches_exact_beta_quantile() -> None:
    actual = one_sided_binomial_lower_bound(91, 100, 0.01)
    expected = float(beta.ppf(0.01, 91, 10))
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
    assert one_sided_binomial_lower_bound(0, 100, 0.01) == 0.0


def test_rounded_unit_lower_bound_remains_finite_and_conservative() -> None:
    alpha = float(np.nextafter(1.0, 0.0))
    lower = one_sided_binomial_lower_bound(100, 100, alpha)
    assert lower == np.nextafter(1.0, 0.0)
    result = certify_trajectory_event_from_counts(100, 100, alpha, 0.5)
    assert result.certified
    assert math.isfinite(result.normalized_energy_radius)


def test_count_certificate_returns_normalized_and_constant_schedule_radii() -> None:
    result = certify_trajectory_event_from_counts(
        99,
        100,
        0.001,
        0.8,
        sigma=0.2,
        cap=4,
        correlation=0.75,
    )
    assert result.certified
    expected_normalized = ndtri(result.lower_probability) - ndtri(0.8)
    expected_multiplier = comparison_multiplier(4, 0.75)
    assert math.isclose(
        result.normalized_energy_radius,
        expected_normalized,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(result.comparison_multiplier, expected_multiplier)
    assert math.isclose(
        result.center_energy_radius,
        0.2 * expected_normalized / math.sqrt(expected_multiplier),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_count_certificate_fails_closed_below_target() -> None:
    result = certify_trajectory_event_from_counts(60, 100, 0.05, 0.8)
    assert not result.certified
    assert result.normalized_energy_radius is None
    assert result.center_energy_radius is None


def test_schedule_arguments_must_be_complete() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        certify_trajectory_event_from_counts(
            90,
            100,
            0.05,
            0.7,
            sigma=0.2,
        )


def test_trajectory_shift_distance_matches_block_energy() -> None:
    shifts = np.array([[0.3, 0.4], [-0.1, 0.2]])
    sigmas = np.array([0.5, 0.25])
    caps = (4, 8)
    correlations = np.array([0.5, 0.75])
    expected_squared = sum(
        comparison_multiplier(cap, rho) * float(np.dot(shift, shift)) / sigma**2
        for shift, sigma, cap, rho in zip(
            shifts, sigmas, caps, correlations, strict=True
        )
    )
    assert math.isclose(
        trajectory_shift_distance(shifts, sigmas, caps, correlations),
        math.sqrt(expected_squared),
        rel_tol=0.0,
        abs_tol=2e-15,
    )


def test_trajectory_shift_distance_preserves_representable_extremes() -> None:
    tiny = trajectory_shift_distance([[1e-200]], [1.0], [1], [0.0])
    large = trajectory_shift_distance([[1e200]], [1.0], [1], [0.0])
    assert tiny == 1e-200
    assert large == 1e200


def test_trajectory_distance_handles_smallest_scale_without_nan() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    zero = trajectory_shift_distance([[0.0]], [tiny], [1], [0.0])
    matched = trajectory_shift_distance([[tiny]], [tiny], [1], [0.0])
    assert zero == 0.0
    assert matched == 1.0


def test_energy_limiter_preserves_or_scales_pre_block_shift() -> None:
    unchanged = limit_shift_to_remaining_energy([0.1, 0.0], 1.0, 4, 0.75, 1.0)
    assert not unchanged.was_scaled
    np.testing.assert_array_equal(unchanged.shift, np.array([0.1, 0.0]))

    limited = limit_shift_to_remaining_energy([3.0, 4.0], 1.0, 4, 0.75, 0.25)
    assert limited.was_scaled
    assert math.isclose(limited.charged_energy_squared, 0.25)
    kappa = comparison_multiplier(4, 0.75)
    charge = kappa * float(np.dot(limited.shift, limited.shift))
    assert charge <= 0.25
    assert math.isclose(charge, 0.25, rel_tol=2e-12, abs_tol=0.0)


def test_energy_limiter_is_conservative_at_zero_and_after_rounding() -> None:
    zero = limit_shift_to_remaining_energy([1e-200], 1.0, 1, 0.0, 0.0)
    assert zero.was_scaled
    np.testing.assert_array_equal(zero.shift, np.array([0.0]))
    assert zero.charged_energy_squared == 0.0

    rng = np.random.default_rng(20260905)
    for _ in range(2_000):
        shift = rng.normal(size=3) * 10.0 ** rng.uniform(-150.0, 150.0)
        sigma = 10.0 ** rng.uniform(-150.0, 150.0)
        remaining = 10.0 ** rng.uniform(-300.0, 300.0)
        cap = int(rng.integers(1, 33))
        correlation = float(rng.uniform(0.0, 0.999))
        result = limit_shift_to_remaining_energy(
            shift,
            sigma,
            cap,
            correlation,
            remaining,
        )
        kappa = comparison_multiplier(cap, correlation)
        norm = math.hypot(*(abs(float(value)) for value in result.shift))
        actual_charge = (math.sqrt(kappa) * norm / sigma) ** 2
        assert actual_charge <= remaining
        assert result.charged_energy_squared <= remaining
        assert result.charged_energy_squared >= actual_charge


def test_energy_limiter_handles_subnormal_remaining_budget() -> None:
    remaining = 2.2223e-320
    result = limit_shift_to_remaining_energy(
        [1.79810557e-191],
        3.553125112547226e-51,
        40,
        0.42146190247549326,
        remaining,
    )
    actual_norm = math.hypot(*(abs(float(value)) for value in result.shift))
    kappa = comparison_multiplier(40, 0.42146190247549326)
    actual_charge = (math.sqrt(kappa) * actual_norm / 3.553125112547226e-51) ** 2
    assert actual_charge <= remaining
    assert result.charged_energy_squared <= remaining


@pytest.mark.parametrize(
    "args",
    [
        (True, 10, 0.05),
        (1, 0, 0.05),
        (11, 10, 0.05),
        (1, 10, 0.0),
    ],
)
def test_invalid_count_inputs_fail_closed(args: tuple[object, int, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_sided_binomial_lower_bound(*args)
