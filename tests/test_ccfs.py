import math

import numpy as np
import pytest

from feasible_robustness.ccfs import (
    ccfs_step,
    comparison_multiplier,
    correlation_for_multiplier,
    equicorrelation_matrix,
    independent_exhaustion_probability,
    independent_expected_inspections,
    sample_equicorrelated_proposals,
    select_first_feasible,
    trajectory_comparison_distance,
)


def test_comparison_multiplier_matches_direct_mahalanobis_calculation():
    for cap in (1, 2, 4, 8):
        for rho in (0.0, 0.5, 0.9):
            matrix = equicorrelation_matrix(cap, rho)
            ones = np.ones(cap)
            direct = float(ones @ np.linalg.solve(matrix, ones))
            assert np.isclose(
                comparison_multiplier(cap, rho), direct, atol=2e-14, rtol=2e-14
            )
    assert comparison_multiplier(1, 0.9) == 1.0
    assert comparison_multiplier(8, 0.0) == 8.0


def test_correlation_for_multiplier_holds_cost_fixed_as_cap_grows():
    target = 1.25
    correlations = []
    for cap in (2, 4, 8, 32, 128):
        rho = correlation_for_multiplier(cap, target)
        correlations.append(rho)
        assert np.isclose(
            comparison_multiplier(cap, rho), target, atol=3e-15, rtol=3e-15
        )
    assert all(left < right for left, right in zip(correlations, correlations[1:]))
    assert np.isclose(correlations[-1], 1.0 / target, atol=0.01)
    assert correlation_for_multiplier(1, 1.0) == 0.0
    with pytest.raises(ValueError):
        correlation_for_multiplier(4, 1.0)
    with pytest.raises(ValueError):
        correlation_for_multiplier(4, 4.1)


def test_invalid_bundle_parameters_are_rejected():
    for cap in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            comparison_multiplier(cap, 0.0)
    for rho in (-0.1, 1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            comparison_multiplier(2, rho)


def test_sampler_is_deterministic_and_has_requested_shape():
    first = sample_equicorrelated_proposals(
        [0.2, -0.1], 0.4, 4, 0.75, np.random.default_rng(17)
    )
    second = sample_equicorrelated_proposals(
        [0.2, -0.1], 0.4, 4, 0.75, np.random.default_rng(17)
    )
    assert first.shape == (4, 2)
    assert np.array_equal(first, second)


def test_sampler_empirical_marginal_and_cross_covariance():
    rng = np.random.default_rng(20270831)
    cap = 4
    rho = 0.6
    draws = np.stack(
        [sample_equicorrelated_proposals([0.0], 1.0, cap, rho, rng)[:, 0] for _ in range(30000)]
    )
    covariance = np.cov(draws, rowvar=False, ddof=0)
    assert np.allclose(np.diag(covariance), 1.0, atol=0.025, rtol=0.0)
    off_diagonal = covariance[~np.eye(cap, dtype=bool)]
    assert np.allclose(off_diagonal, rho, atol=0.025, rtol=0.0)


def test_selector_uses_first_feasible_candidate_without_calling_fallback():
    proposals = np.asarray([[-2.0], [0.2], [0.7]])

    def forbidden_fallback(_):
        raise AssertionError("fallback should not be called")

    result = select_first_feasible(
        proposals,
        is_feasible=lambda action: 0.0 <= action[0] <= 1.0,
        fallback=forbidden_fallback,
    )
    assert result.selected_index == 1
    assert result.inspected_count == 2
    assert not result.used_fallback
    assert np.array_equal(result.action, [0.2])


def test_selector_checks_fallback_feasibility():
    proposals = np.asarray([[-2.0], [-1.0]])
    predicate = lambda action: 0.0 <= action[0] <= 1.0
    result = select_first_feasible(
        proposals, predicate, fallback=lambda _: np.asarray([0.5])
    )
    assert result.selected_index is None
    assert result.inspected_count == 2
    assert result.used_fallback
    assert np.array_equal(result.action, [0.5])
    with pytest.raises(ValueError, match="infeasible"):
        select_first_feasible(
            proposals, predicate, fallback=lambda _: np.asarray([-0.5])
        )


def test_ccfs_step_always_returns_a_verified_action_with_exact_fallback():
    result = ccfs_step(
        center=[-10.0],
        sigma=0.01,
        cap=4,
        correlation=0.5,
        rng=np.random.default_rng(31),
        is_feasible=lambda action: action[0] >= 0.0,
        fallback=lambda _: np.asarray([0.0]),
    )
    assert result.used_fallback
    assert result.action[0] == 0.0


def test_trajectory_distance_matches_manual_block_sum():
    first = np.asarray([[0.0, 0.0], [1.0, -1.0]])
    second = np.asarray([[0.3, 0.4], [0.8, -0.9]])
    sigmas = np.asarray([0.5, 0.25])
    caps = (4, 8)
    correlations = np.asarray([0.5, 0.75])
    expected_squared = sum(
        comparison_multiplier(cap, rho)
        * float(np.dot(a - b, a - b))
        / sigma**2
        for a, b, sigma, cap, rho in zip(
            first, second, sigmas, caps, correlations, strict=True
        )
    )
    assert np.isclose(
        trajectory_comparison_distance(
            first, second, sigmas, caps, correlations
        ),
        math.sqrt(expected_squared),
        atol=2e-15,
        rtol=2e-15,
    )


def test_trajectory_distance_rejects_zero_action_dimension():
    with pytest.raises(ValueError, match="nonempty shape"):
        trajectory_comparison_distance(
            np.empty((2, 0)),
            np.empty((2, 0)),
            np.ones(2),
            (2, 2),
            np.zeros(2),
        )


def test_independent_bundle_cost_formulas_include_edge_cases():
    assert independent_exhaustion_probability(0.0, 4) == 1.0
    assert independent_exhaustion_probability(1.0, 4) == 0.0
    assert independent_expected_inspections(0.0, 4) == 4.0
    assert independent_expected_inspections(1.0, 4) == 1.0
    occupancy = 0.3
    expected = sum((1.0 - occupancy) ** index for index in range(6))
    assert np.isclose(
        independent_expected_inspections(occupancy, 6), expected, atol=2e-15
    )
