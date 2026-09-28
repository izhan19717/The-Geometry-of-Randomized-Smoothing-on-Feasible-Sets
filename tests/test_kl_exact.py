import numpy as np
import pytest
from scipy.stats import norm

from feasible_robustness import (
    RectangleUnion,
    canonical_l_shape,
    gaussian_box_conditional_mean,
    gaussian_box_first_moment,
    gaussian_box_probability,
    gaussian_box_second_moment,
    rectangle_region,
    rectangle_union_centered_absolute_projection_moment,
    rectangle_union_first_moment,
    rectangle_union_geodesic_distance,
    rectangle_union_normalizer,
    rectangle_union_projected_probability,
    rectangle_union_covariance,
    rectangle_union_path_integrated_kl,
    rectangle_union_segment_is_visible,
    rectangle_union_total_variation,
    truncated_gaussian_kl_rectangle_union,
)


def test_interval_normalizer_matches_gaussian_cdf_difference():
    region = rectangle_region([-1.0], [2.0])
    center = np.array([0.25])
    sigma = 0.7
    expected = norm.cdf((2.0 - center[0]) / sigma) - norm.cdf((-1.0 - center[0]) / sigma)
    assert np.isclose(rectangle_union_normalizer(region, center, sigma), expected)


def test_symmetric_interval_first_moment_is_zero():
    region = rectangle_region([-1.0], [1.0])
    moment = rectangle_union_first_moment(region, [0.0], sigma=0.3)
    assert abs(float(moment[0])) < 1e-14


def test_gaussian_box_conditional_mean_matches_first_moment_ratio():
    lower = [-0.2, -0.4]
    upper = [0.8, 0.7]
    center = [0.1, 0.2]
    sigma = 0.3
    moment = gaussian_box_first_moment(lower, upper, center, sigma)
    probability = gaussian_box_probability(lower, upper, center, sigma)
    mean = gaussian_box_conditional_mean(lower, upper, center, sigma)
    np.testing.assert_allclose(mean, moment / probability)


def test_large_box_second_moment_recovers_gaussian_covariance():
    center = np.array([0.1, -0.2])
    sigma = 0.4
    probability = gaussian_box_probability([-8.0, -8.0], [8.0, 8.0], center, sigma)
    raw_second = gaussian_box_second_moment([-8.0, -8.0], [8.0, 8.0], center, sigma)
    expected = sigma**2 * np.eye(2) + np.outer(center, center)
    np.testing.assert_allclose(raw_second / probability, expected, atol=1e-12)


def test_positive_tail_box_probability_uses_stable_survival_difference():
    observed = gaussian_box_probability([9.0], [10.0], [0.0], sigma=1.0)
    expected = norm.sf(9.0) - norm.sf(10.0)
    assert observed > 0.0
    assert np.isclose(observed, expected, rtol=2e-14, atol=0.0)


def test_narrow_interval_second_moment_remains_positive():
    width = 1e-12
    observed = gaussian_box_second_moment([0.0], [width], [0.0], sigma=1.0)[0, 0]
    leading_term = norm.pdf(0.0) * width**3 / 3.0
    assert observed > 0.0
    assert np.isclose(observed, leading_term, rtol=1e-12, atol=0.0)


def test_extremely_asymmetric_interval_moments_do_not_overflow():
    lower = [-100.0]
    upper = [1.0]
    probability = gaussian_box_probability(lower, upper, [0.0], sigma=1.0)
    first = gaussian_box_first_moment(lower, upper, [0.0], sigma=1.0)[0]
    second = gaussian_box_second_moment(lower, upper, [0.0], sigma=1.0)[0, 0]
    assert np.isclose(probability, norm.cdf(1.0), atol=1e-15)
    assert np.isclose(first, -norm.pdf(1.0), atol=1e-15)
    assert np.isclose(second, norm.cdf(1.0) - norm.pdf(1.0), atol=1e-15)


def test_rectangle_convex_kl_does_not_exceed_convex_bound():
    region = rectangle_region([0.0, 0.0], [1.0, 1.0])
    result = truncated_gaussian_kl_rectangle_union(
        [0.35, 0.45],
        [0.55, 0.50],
        sigma=0.2,
        region=region,
    )
    assert result.kl >= 0.0
    assert result.excess <= 1e-10
    assert np.isclose(
        result.excess,
        result.normalizer_log_ratio + result.mean_drift_excess,
        atol=1e-12,
    )


def test_large_rectangle_approaches_untruncated_gaussian_kl():
    region = rectangle_region([-8.0, -8.0], [8.0, 8.0])
    result = truncated_gaussian_kl_rectangle_union(
        [0.1, -0.2],
        [0.4, 0.0],
        sigma=0.5,
        region=region,
    )
    assert np.isclose(result.kl, result.convex_bound, atol=1e-10)


def test_projected_probability_matches_axis_aligned_box_cdf():
    region = rectangle_region([0.0, -1.0], [1.0, 2.0])
    center = np.array([0.25, 0.4])
    sigma = 0.35
    threshold = 0.6
    expected = (
        gaussian_box_probability([0.0, -1.0], [threshold, 2.0], center, sigma)
        / rectangle_union_normalizer(region, center, sigma)
    )
    observed = rectangle_union_projected_probability(
        region,
        center,
        sigma,
        direction=[1.0, 0.0],
        threshold=threshold,
    )
    assert np.isclose(observed, expected, atol=1e-12)


def test_large_rectangle_total_variation_matches_gaussian_formula():
    region = rectangle_region([-10.0, -10.0], [10.0, 10.0])
    a = np.array([0.1, -0.2])
    b = np.array([0.5, 0.1])
    sigma = 0.7
    distance = float(np.linalg.norm(a - b))
    expected = 2.0 * norm.cdf(distance / (2.0 * sigma)) - 1.0
    observed = rectangle_union_total_variation(a, b, sigma, region, quadrature_order=80)
    assert np.isclose(observed, expected, atol=1e-8)


def test_rectangle_union_total_variation_is_symmetric_and_zero_on_diagonal():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    a = np.array([0.67, 0.49])
    b = np.array([0.33, 0.96])
    sigma = 0.08
    forward = rectangle_union_total_variation(a, b, sigma, region, quadrature_order=80)
    backward = rectangle_union_total_variation(b, a, sigma, region, quadrature_order=80)
    assert 0.0 <= forward <= 1.0
    assert np.isclose(forward, backward, atol=1e-10)
    assert rectangle_union_total_variation(a, a, sigma, region) == 0.0


def test_rectangle_union_segment_visibility_detects_notch_obstruction():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    assert rectangle_union_segment_is_visible(region, [0.2, 0.2], [0.8, 0.2])
    assert not rectangle_union_segment_is_visible(region, [0.67, 0.49], [0.33, 0.96])


def test_l_shape_flagship_geodesic_distance_matches_reference_value():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    observed = rectangle_union_geodesic_distance(region, [0.67, 0.49], [0.33, 0.96])
    expected = np.linalg.norm(np.array([0.67, 0.49]) - np.array([0.5, 0.5])) + np.linalg.norm(
        np.array([0.5, 0.5]) - np.array([0.33, 0.96])
    )
    assert np.isclose(observed, expected, atol=1e-12)
    assert f"{observed:.5f}" == "0.66070"


def test_l_shape_is_nonconvex_but_computable():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    assert region.contains([0.75, 0.25])
    assert region.contains([0.25, 0.75])
    assert not region.contains([0.75, 0.75])
    result = truncated_gaussian_kl_rectangle_union(
        [0.45, 0.25],
        [0.25, 0.45],
        sigma=0.18,
        region=region,
    )
    assert np.isfinite(result.kl)
    assert result.normalizer_a > 0.0
    assert result.normalizer_b > 0.0
    assert np.isfinite(result.normalizer_log_ratio)
    assert np.isfinite(result.mean_drift_excess)


def test_rounded_l_shape_counterexample_has_large_excess():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    result = truncated_gaussian_kl_rectangle_union(
        [0.67, 0.49],
        [0.33, 0.96],
        sigma=0.04,
        region=region,
    )
    assert result.excess > 7.0
    assert result.mean_drift_excess > result.normalizer_log_ratio
    assert np.isclose(
        result.excess,
        result.normalizer_log_ratio + result.mean_drift_excess,
        atol=1e-12,
    )


def test_l_shape_has_local_rate_loss_inside_one_convex_bar():
    """The full nonconvex support matters even for a locally visible pair.

    Both centers and their connecting segment lie strictly inside the bottom
    rectangle of the canonical L-shape.  Nevertheless, conditioning on the
    *whole* L-shape produces both KL and exact-TV rate loss relative to the
    untruncated Gaussian.  This guards against treating a convex subregion
    containing the centers as though it were the support in the convex
    contraction theorem.
    """

    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    a = np.array([0.55, 0.4])
    b = np.array([0.56, 0.396])
    sigma = 0.15

    assert rectangle_union_segment_is_visible(region, a, b)
    result = truncated_gaussian_kl_rectangle_union(a, b, sigma, region)
    distance = float(np.linalg.norm(a - b))
    gaussian_tv = 2.0 * norm.cdf(distance / (2.0 * sigma)) - 1.0
    truncated_tv = rectangle_union_total_variation(
        a,
        b,
        sigma,
        region,
        quadrature_order=96,
    )

    assert result.excess > 2.3e-4
    assert truncated_tv < 0.031
    assert truncated_tv > 1.06 * gaussian_tv

    direction = (b - a) / distance
    covariance = rectangle_union_covariance(region, a, sigma)
    local_kl_ratio = float(direction @ covariance @ direction) / sigma**2
    assert local_kl_ratio > 1.09


def test_rational_l_shape_manuscript_witness_matches_reported_values():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    a = np.array([13.0 / 20.0, 49.0 / 100.0])
    b = np.array([66.0 / 100.0, 49.0 / 100.0])
    sigma = 3.0 / 20.0

    assert rectangle_union_segment_is_visible(region, a, b)
    covariance = rectangle_union_covariance(region, a, sigma)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert float(np.linalg.eigvalsh(covariance).min()) > 0.0
    assert np.isclose(covariance[0, 0] / sigma**2, 1.09153866764843, atol=2e-13)

    result = truncated_gaussian_kl_rectangle_union(a, b, sigma, region)
    tv = rectangle_union_total_variation(a, b, sigma, region, quadrature_order=96)
    assert np.isclose(result.kl, 0.002423524483776904, atol=3e-15)
    assert result.kl > result.convex_bound
    assert np.isclose(tv, 0.028641422872326627, atol=2e-13)
    assert tv > 2.0 * norm.cdf(np.linalg.norm(b - a) / (2.0 * sigma)) - 1.0


def test_covariance_path_integral_recovers_exact_conditioned_kl():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    a = np.array([13.0 / 20.0, 49.0 / 100.0])
    b = np.array([66.0 / 100.0, 49.0 / 100.0])
    sigma = 3.0 / 20.0
    direct = truncated_gaussian_kl_rectangle_union(a, b, sigma, region).kl
    integrated = rectangle_union_path_integrated_kl(
        a,
        b,
        sigma,
        region,
        quadrature_order=48,
    )
    assert integrated == pytest.approx(direct, abs=2e-14)


def test_centered_absolute_projection_matches_local_l_shape_prediction():
    region = canonical_l_shape(notch_x=0.5, notch_y=0.5)
    center = np.array([13.0 / 20.0, 49.0 / 100.0])
    sigma = 3.0 / 20.0
    observed = rectangle_union_centered_absolute_projection_moment(
        region,
        center,
        sigma,
        [1.0, 0.0],
    )
    assert np.isclose(observed, 0.1290724364219014, atol=3e-15)
    doubled = rectangle_union_centered_absolute_projection_moment(
        region,
        center,
        sigma,
        [2.0, 0.0],
    )
    assert np.isclose(doubled, 2.0 * observed, atol=3e-15)


def test_centered_absolute_projection_recovers_gaussian_mad_on_large_box():
    region = rectangle_region([-10.0, -10.0], [10.0, 10.0])
    center = np.array([0.2, -0.1])
    direction = np.array([0.6, -0.8])
    sigma = 0.7
    observed = rectangle_union_centered_absolute_projection_moment(
        region,
        center,
        sigma,
        direction,
    )
    expected = sigma * np.sqrt(2.0 / np.pi) * np.linalg.norm(direction)
    assert np.isclose(observed, expected, atol=2e-12)


def test_overlapping_rectangle_union_is_rejected():
    with pytest.raises(ValueError, match="interiors must be disjoint"):
        RectangleUnion.from_bounds(
            (
                ([0.0, 0.0], [1.0, 1.0]),
                ([0.5, 0.5], [1.5, 1.5]),
            )
        )
