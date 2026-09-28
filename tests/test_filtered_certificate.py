import math

import pytest
from scipy.stats import norm

from feasible_robustness import (
    categorical_order_reversal_kl,
    certify_filtered_hybrid_from_counts,
    certify_filtered_joint_mass_from_counts,
    conditioned_diameter_log_odds_radius,
    conditioned_diameter_pinsker_radius,
    conditioned_covariance_kl_radius,
    gaussian_joint_mass_radius,
)


def test_confidence_filter_counterexample_has_sharp_joint_mass_radius():
    center = -0.1
    left_mass = norm.cdf(-1.0 - center)
    right_mass = 1.0 - norm.cdf(1.0 - center)
    conditional_left = left_mass / (left_mass + right_mass)

    imported = norm.ppf(conditional_left)
    corrected = gaussian_joint_mass_radius(left_mass, right_mass, 1.0)

    assert imported == pytest.approx(0.19085520776319986)
    assert corrected == pytest.approx(0.1)
    assert corrected < imported

    attacked_center = 0.01
    attacked_left = norm.cdf(-1.0 - attacked_center)
    attacked_right = 1.0 - norm.cdf(1.0 - attacked_center)
    assert attacked_right > attacked_left
    assert attacked_center - center < imported
    assert attacked_center - center > corrected


def test_separated_half_lines_attain_the_joint_mass_radius():
    selected_mass = 0.3
    runner_mass = 0.1
    sigma = 0.7
    radius = gaussian_joint_mass_radius(selected_mass, runner_mass, sigma)
    shifted_selected = norm.cdf(norm.ppf(selected_mass) - radius / sigma)
    shifted_runner = norm.cdf(norm.ppf(runner_mass) + radius / sigma)
    assert shifted_selected == pytest.approx(shifted_runner)


def test_filtered_proposals_remain_in_joint_mass_denominator():
    certificate = certify_filtered_joint_mass_from_counts(
        {"top": 600, "runner": 100},
        1000,
        0.25,
        label_universe=("top", "runner"),
        delta=0.001,
    )
    assert certificate.selected_label == "top"
    assert certificate.selected_count == 600
    assert certificate.proposal_count == 1000
    assert certificate.selected_lower < 0.6
    assert certificate.runner_upper > 0.1
    assert certificate.radius > 0.0


def test_hybrid_certificate_uses_tighter_simultaneous_runner_bound():
    certificate = certify_filtered_hybrid_from_counts(
        {0: 450, 1: 45, 2: 5},
        1000,
        0.25,
        label_universe=(0, 1, 2),
        selected_label=0,
        delta=0.001,
        selection_independent=True,
    )
    assert certificate.selected_label == 0
    assert certificate.runner_up == 1
    assert certificate.runner_upper == min(
        certificate.explicit_runner_upper,
        certificate.complement_runner_upper,
    )
    assert certificate.radius > 0.0


def test_hybrid_certificate_counts_rejection_in_selected_union():
    certificate = certify_filtered_hybrid_from_counts(
        {0: 400, 1: 100},
        1000,
        0.2,
        label_universe=(0, 1),
        selected_label=0,
        selection_independent=True,
    )
    assert 0.10 < certificate.complement_runner_upper < 0.14
    assert certificate.radius > 0.0

    with pytest.raises(ValueError, match="independent selection"):
        certify_filtered_hybrid_from_counts(
            {0: 400, 1: 100},
            1000,
            0.2,
            label_universe=(0, 1),
            selected_label=0,
        )


def test_independent_selection_uses_the_declared_label():
    certificate = certify_filtered_joint_mass_from_counts(
        {0: 350, 1: 100, 2: 50},
        1000,
        0.5,
        label_universe=(0, 1, 2),
        selected_label=0,
        selection_independent=True,
    )
    assert certificate.selected_label == 0
    assert certificate.runner_up == 1
    assert certificate.selection_independent
    with pytest.raises(ValueError, match="selection_independent"):
        certify_filtered_joint_mass_from_counts(
            {0: 350, 1: 100},
            1000,
            0.5,
            label_universe=(0, 1),
            selected_label=0,
        )


def test_rejected_samples_are_not_a_competing_task_label():
    with_rejections = certify_filtered_joint_mass_from_counts(
        {0: 450, 1: 50},
        1000,
        0.2,
        label_universe=(0, 1),
    )
    treating_rejection_as_a_label = certify_filtered_joint_mass_from_counts(
        {0: 450, 1: 50, "reject": 500},
        1000,
        0.2,
        label_universe=(0, 1, "reject"),
    )
    assert with_rejections.radius > 0.0
    assert treating_rejection_as_a_label.radius == 0.0


def test_bounded_support_radii_match_closed_forms():
    p_top = 0.6
    p_runner = 0.4
    sigma = 0.15
    diameter = math.sqrt(2.0)
    assert conditioned_diameter_pinsker_radius(
        p_top, p_runner, sigma, diameter
    ) == pytest.approx(2.0 * sigma**2 * 0.2 / diameter)
    assert conditioned_diameter_log_odds_radius(
        p_top, p_runner, sigma, diameter
    ) == pytest.approx(sigma**2 * math.log(1.5) / diameter)
    assert math.isinf(
        conditioned_diameter_log_odds_radius(
            p_top, 0.0, sigma, diameter
        )
    )


def test_covariance_radius_uses_exact_order_reversal_cost():
    selected = 0.6
    runner = 0.3
    sigma = 0.25
    factor = 1.2
    midpoint = 0.5 * (selected + runner)
    expected_kl = selected * math.log(selected / midpoint) + runner * math.log(
        runner / midpoint
    )
    assert categorical_order_reversal_kl(selected, runner) == pytest.approx(
        expected_kl
    )
    assert conditioned_covariance_kl_radius(
        selected,
        runner,
        sigma,
        factor,
    ) == pytest.approx(sigma * math.sqrt(2.0 * expected_kl / factor))


def test_covariance_radius_respects_certified_region_and_zero_margin():
    assert conditioned_covariance_kl_radius(
        0.8,
        0.1,
        0.5,
        0.25,
        certified_region_radius=0.2,
    ) == pytest.approx(0.2)
    assert conditioned_covariance_kl_radius(0.4, 0.4, 0.5, 1.0) == 0.0
    assert categorical_order_reversal_kl(0.4, 0.0) == pytest.approx(
        0.4 * math.log(2.0)
    )
    with pytest.raises(ValueError, match="covariance_factor_upper"):
        conditioned_covariance_kl_radius(0.6, 0.2, 0.5, 0.0)


def test_covariance_radius_is_stable_for_nearly_tied_probabilities():
    selected = 0.500000001
    runner = 0.499999999
    reversal_kl = categorical_order_reversal_kl(selected, runner)
    assert reversal_kl > 0.0
    assert reversal_kl == pytest.approx(2.0e-18, rel=1.0e-7)
    assert conditioned_covariance_kl_radius(
        selected, runner, 0.25, 1.0
    ) == pytest.approx(0.25 * math.sqrt(2.0 * reversal_kl))


def test_order_reversal_cost_tightens_runner_by_remaining_mass():
    assert categorical_order_reversal_kl(0.7, 0.6) == pytest.approx(
        categorical_order_reversal_kl(0.7, 0.3)
    )


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        certify_filtered_joint_mass_from_counts(
            {0: 600, 1: 500}, 1000, 0.2, label_universe=(0, 1)
        )
    with pytest.raises(ValueError, match="outside"):
        certify_filtered_joint_mass_from_counts(
            {0: 10, 1: 10, 2: 1}, 100, 0.2, label_universe=(0, 1)
        )
