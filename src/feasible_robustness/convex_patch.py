"""Analytic KL and Fisher decompositions for convex-patch Gaussian mixtures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .kl_exact import (
    AxisAlignedBox,
    RectangleUnion,
    gaussian_box_first_moment,
    gaussian_box_probability,
    gaussian_box_second_moment,
    rectangle_union_covariance,
    truncated_gaussian_kl_rectangle_union,
)


FloatArray = NDArray[np.float64]
_RENYI_KL_LIMIT_TOLERANCE = 1e-8


@dataclass(frozen=True)
class BoxConditionalStatistics:
    """Normalizer, mean, and covariance of one box-conditioned Gaussian."""

    normalizer: float
    mean: FloatArray
    covariance: FloatArray


def _validate_probability_vector(probability: FloatArray, name: str) -> FloatArray:
    """Validate a strictly positive finite categorical probability vector."""

    vector = np.asarray(probability, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if np.any(~np.isfinite(vector)) or np.any(vector <= 0.0):
        raise ValueError(f"{name} entries must be finite and strictly positive")
    if not math.isclose(float(np.sum(vector)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return vector


def box_conditional_statistics(
    box: AxisAlignedBox,
    center: FloatArray,
    sigma: float,
) -> BoxConditionalStatistics:
    """Return analytic moments of ``N(center,sigma^2 I)`` restricted to ``box``."""

    normalizer = gaussian_box_probability(box.lower, box.upper, center, sigma)
    if normalizer <= 0.0:
        raise FloatingPointError("box normalizer underflowed to zero")
    mean = gaussian_box_first_moment(box.lower, box.upper, center, sigma) / normalizer
    raw_second = gaussian_box_second_moment(
        box.lower,
        box.upper,
        center,
        sigma,
    ) / normalizer
    covariance = raw_second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return BoxConditionalStatistics(
        normalizer=float(normalizer),
        mean=np.asarray(mean, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
    )


def box_conditioned_kl(
    box: AxisAlignedBox,
    center_a: FloatArray,
    center_b: FloatArray,
    sigma: float,
    stats_a: BoxConditionalStatistics | None = None,
    stats_b: BoxConditionalStatistics | None = None,
) -> float:
    """Exact identity for KL between two Gaussians conditioned on one box."""

    if stats_a is None:
        stats_a = box_conditional_statistics(box, center_a, sigma)
    if stats_b is None:
        stats_b = box_conditional_statistics(box, center_b, sigma)
    quadratic = (
        2.0 * float(np.dot(stats_a.mean, center_a - center_b))
        + float(np.dot(center_b, center_b) - np.dot(center_a, center_a))
    ) / (2.0 * sigma**2)
    return float(quadratic + math.log(stats_b.normalizer / stats_a.normalizer))


def box_conditioned_renyi(
    box: AxisAlignedBox,
    center_a: FloatArray,
    center_b: FloatArray,
    sigma: float,
    alpha: float,
    stats_a: BoxConditionalStatistics | None = None,
    stats_b: BoxConditionalStatistics | None = None,
) -> float:
    """Finite-order Renyi divergence, using the KL limit near order one."""

    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if abs(alpha - 1.0) <= _RENYI_KL_LIMIT_TOLERANCE:
        return box_conditioned_kl(box, center_a, center_b, sigma, stats_a, stats_b)
    if stats_a is None:
        stats_a = box_conditional_statistics(box, center_a, sigma)
    if stats_b is None:
        stats_b = box_conditional_statistics(box, center_b, sigma)
    mixed_center = alpha * center_a + (1.0 - alpha) * center_b
    mixed_normalizer = gaussian_box_probability(
        box.lower,
        box.upper,
        mixed_center,
        sigma,
    )
    if mixed_normalizer <= 0.0 or not math.isfinite(mixed_normalizer):
        raise FloatingPointError(
            "mixed box normalizer is not numerically representable at this Renyi order"
        )
    gaussian = alpha * float(np.dot(center_a - center_b, center_a - center_b)) / (
        2.0 * sigma**2
    )
    correction = (
        math.log(mixed_normalizer)
        - alpha * math.log(stats_a.normalizer)
        - (1.0 - alpha) * math.log(stats_b.normalizer)
    ) / (alpha - 1.0)
    return float(gaussian + correction)


def categorical_kl(probability_a: FloatArray, probability_b: FloatArray) -> float:
    """Return KL between strictly positive finite categorical laws."""

    probability_a = _validate_probability_vector(probability_a, "probability_a")
    probability_b = _validate_probability_vector(probability_b, "probability_b")
    if probability_a.shape != probability_b.shape:
        raise ValueError("categorical probabilities must have the same shape")
    value = float(np.dot(probability_a, np.log(probability_a / probability_b)))
    if value < 0.0 and abs(value) <= 1e-14:
        return 0.0
    return value


def disjoint_patch_mixture_renyi(
    component_divergences: FloatArray,
    probability_a: FloatArray,
    probability_b: FloatArray,
    alpha: float,
) -> float:
    """Tagged-mixture Renyi divergence, using the KL limit near order one."""

    component_divergences = np.asarray(component_divergences, dtype=np.float64)
    probability_a = _validate_probability_vector(probability_a, "probability_a")
    probability_b = _validate_probability_vector(probability_b, "probability_b")
    if component_divergences.ndim != 1 or component_divergences.size == 0:
        raise ValueError("component divergences must be a nonempty one-dimensional vector")
    if component_divergences.shape != probability_a.shape or probability_a.shape != probability_b.shape:
        raise ValueError("component divergences and probabilities must have the same shape")
    if np.any(~np.isfinite(component_divergences)) or np.any(component_divergences < 0.0):
        raise ValueError("component divergences must be finite and nonnegative")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if abs(alpha - 1.0) <= _RENYI_KL_LIMIT_TOLERANCE:
        return categorical_kl(probability_a, probability_b) + float(
            np.dot(probability_a, component_divergences)
        )
    log_terms = (
        alpha * np.log(probability_a)
        + (1.0 - alpha) * np.log(probability_b)
        + (alpha - 1.0) * component_divergences
    )
    maximum = float(np.max(log_terms))
    return float((maximum + math.log(float(np.sum(np.exp(log_terms - maximum))))) / (alpha - 1.0))


def component_statistics(
    region: RectangleUnion,
    center: FloatArray,
    sigma: float,
) -> tuple[BoxConditionalStatistics, ...]:
    """Return conditioned statistics for every box in a rectangle partition."""

    return tuple(box_conditional_statistics(box, center, sigma) for box in region.boxes)


def mixture_weights(statistics: tuple[BoxConditionalStatistics, ...]) -> FloatArray:
    """Normalize component Gaussian occupancies into a categorical gate."""

    normalizers = np.asarray([item.normalizer for item in statistics], dtype=np.float64)
    return normalizers / float(np.sum(normalizers))


def tempered_mixture_weights(
    statistics: tuple[BoxConditionalStatistics, ...],
    tau: float,
    prior: FloatArray | None = None,
) -> FloatArray:
    """Return ``pi_s proportional to prior_s * Z_s**tau`` stably."""

    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    component_count = len(statistics)
    if prior is None:
        prior_array = np.full(component_count, 1.0 / component_count, dtype=np.float64)
    else:
        prior_array = np.asarray(prior, dtype=np.float64)
        if prior_array.shape != (component_count,):
            raise ValueError("prior must have one entry per component")
        if np.any(~np.isfinite(prior_array)) or np.any(prior_array <= 0.0):
            raise ValueError("prior entries must be finite and strictly positive")
        prior_array = prior_array / float(np.sum(prior_array))
    log_weights = np.log(prior_array) + tau * np.log(
        np.asarray([item.normalizer for item in statistics], dtype=np.float64)
    )
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / float(np.sum(weights))


def anchored_tempered_weights(
    anchor_weights: FloatArray,
    current_weights: FloatArray,
    tau: float,
) -> FloatArray:
    """Geometrically interpolate an anchored gate and a current occupancy gate.

    The returned vector is proportional to
    ``anchor_weights**(1-tau) * current_weights**tau``.  It equals the anchor
    gate at ``tau=0`` and the current occupancy gate at ``tau=1``.
    """

    anchor = _validate_probability_vector(anchor_weights, "anchor_weights")
    current = _validate_probability_vector(current_weights, "current_weights")
    if anchor.shape != current.shape:
        raise ValueError("anchor and current weights must have the same shape")
    if not math.isfinite(tau) or tau < 0.0 or tau > 1.0:
        raise ValueError("tau must be finite and lie in [0, 1]")
    log_weights = (1.0 - tau) * np.log(anchor) + tau * np.log(current)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / float(np.sum(weights))


def evaluate_convex_patch_pair(
    region: RectangleUnion,
    center_a: FloatArray,
    center_b: FloatArray,
    sigma: float,
) -> dict[str, Any]:
    """Evaluate the exact KL chain rule for one rectangle-partition pair."""

    whole = truncated_gaussian_kl_rectangle_union(center_a, center_b, sigma, region)
    stats_a = component_statistics(region, center_a, sigma)
    stats_b = component_statistics(region, center_b, sigma)
    weights_a = mixture_weights(stats_a)
    weights_b = mixture_weights(stats_b)
    component_kls = np.asarray(
        [
            box_conditioned_kl(box, center_a, center_b, sigma, left, right)
            for box, left, right in zip(region.boxes, stats_a, stats_b, strict=True)
        ],
        dtype=np.float64,
    )
    gate_kl = categorical_kl(weights_a, weights_b)
    within_kl = float(np.dot(weights_a, component_kls))
    reconstructed = gate_kl + within_kl
    gaussian_kl = float(np.dot(center_a - center_b, center_a - center_b) / (2.0 * sigma**2))
    if gaussian_kl <= 0.0:
        raise ValueError("the two centers must be distinct")

    return {
        "a_x": float(center_a[0]),
        "a_y": float(center_a[1]),
        "b_x": float(center_b[0]),
        "b_y": float(center_b[1]),
        "sigma": float(sigma),
        "distance": float(np.linalg.norm(center_b - center_a)),
        "gaussian_kl": gaussian_kl,
        "whole_conditioning_kl": float(whole.kl),
        "within_patch_kl": within_kl,
        "gate_kl": gate_kl,
        "reconstructed_whole_kl": reconstructed,
        "decomposition_abs_error": abs(float(whole.kl) - reconstructed),
        "whole_ratio": float(whole.kl / gaussian_kl),
        "within_patch_ratio": within_kl / gaussian_kl,
        "gate_ratio": gate_kl / gaussian_kl,
        "whole_excess": float(whole.kl - gaussian_kl),
        "within_patch_slack": gaussian_kl - within_kl,
        "gate_minus_slack": gate_kl - (gaussian_kl - within_kl),
        "weights_a": weights_a.tolist(),
        "weights_b": weights_b.tolist(),
        "component_kls": component_kls.tolist(),
        "component_kl_ratios": (component_kls / gaussian_kl).tolist(),
    }


def evaluate_convex_patch_renyi(
    region: RectangleUnion,
    center_a: FloatArray,
    center_b: FloatArray,
    sigma: float,
    alpha: float,
) -> dict[str, Any]:
    """Compare whole, frozen, and uniform patch mixtures at one Renyi order."""

    stats_a = component_statistics(region, center_a, sigma)
    stats_b = component_statistics(region, center_b, sigma)
    occupancy_a = mixture_weights(stats_a)
    occupancy_b = mixture_weights(stats_b)
    component_divergences = np.asarray(
        [
            box_conditioned_renyi(
                box,
                center_a,
                center_b,
                sigma,
                alpha,
                left,
                right,
            )
            for box, left, right in zip(region.boxes, stats_a, stats_b, strict=True)
        ],
        dtype=np.float64,
    )
    uniform = np.full(len(region.boxes), 1.0 / len(region.boxes), dtype=np.float64)
    gaussian = alpha * float(np.dot(center_a - center_b, center_a - center_b)) / (
        2.0 * sigma**2
    )
    whole = disjoint_patch_mixture_renyi(
        component_divergences,
        occupancy_a,
        occupancy_b,
        alpha,
    )
    frozen = disjoint_patch_mixture_renyi(
        component_divergences,
        occupancy_a,
        occupancy_a,
        alpha,
    )
    uniform_value = disjoint_patch_mixture_renyi(
        component_divergences,
        uniform,
        uniform,
        alpha,
    )
    return {
        "alpha": float(alpha),
        "gaussian_renyi": gaussian,
        "component_divergences": component_divergences.tolist(),
        "whole_conditioning_renyi": whole,
        "frozen_reference_gate_renyi": frozen,
        "uniform_fixed_gate_renyi": uniform_value,
        "whole_ratio": whole / gaussian,
        "frozen_reference_gate_ratio": frozen / gaussian,
        "uniform_fixed_gate_ratio": uniform_value / gaussian,
    }


def _mean_squared_displacement(
    weights: FloatArray,
    statistics: tuple[BoxConditionalStatistics, ...],
    center: FloatArray,
) -> float:
    return float(
        sum(
            weight
            * (float(np.trace(item.covariance)) + float(np.dot(item.mean - center, item.mean - center)))
            for weight, item in zip(weights, statistics, strict=True)
        )
    )


def evaluate_tempered_patch_pair(
    region: RectangleUnion,
    center_a: FloatArray,
    center_b: FloatArray,
    sigma: float,
    tau: float,
    prior: FloatArray | None = None,
) -> dict[str, Any]:
    """Evaluate a tempered occupancy gate on an a.e.-disjoint box partition."""

    stats_a = component_statistics(region, center_a, sigma)
    stats_b = component_statistics(region, center_b, sigma)
    weights_a = tempered_mixture_weights(stats_a, tau, prior)
    weights_b = tempered_mixture_weights(stats_b, tau, prior)
    occupancy_a = mixture_weights(stats_a)
    occupancy_b = mixture_weights(stats_b)
    component_kls = np.asarray(
        [
            box_conditioned_kl(box, center_a, center_b, sigma, left, right)
            for box, left, right in zip(region.boxes, stats_a, stats_b, strict=True)
        ],
        dtype=np.float64,
    )
    gate_kl = categorical_kl(weights_a, weights_b)
    within_kl = float(np.dot(weights_a, component_kls))
    total_kl = gate_kl + within_kl
    gaussian_kl = float(np.dot(center_a - center_b, center_a - center_b) / (2.0 * sigma**2))
    if gaussian_kl <= 0.0:
        raise ValueError("the two centers must be distinct")

    mse_a = _mean_squared_displacement(weights_a, stats_a, center_a)
    whole_mse_a = _mean_squared_displacement(occupancy_a, stats_a, center_a)
    component_rejection_cost_a = float(
        sum(weight / item.normalizer for weight, item in zip(weights_a, stats_a, strict=True))
    )
    whole_rejection_cost_a = 1.0 / float(sum(item.normalizer for item in stats_a))
    return {
        "tau": float(tau),
        "weights_a": weights_a.tolist(),
        "weights_b": weights_b.tolist(),
        "occupancy_weights_a": occupancy_a.tolist(),
        "occupancy_weights_b": occupancy_b.tolist(),
        "gaussian_kl": gaussian_kl,
        "within_patch_kl": within_kl,
        "gate_kl": gate_kl,
        "total_kl": total_kl,
        "within_patch_ratio": within_kl / gaussian_kl,
        "gate_ratio": gate_kl / gaussian_kl,
        "total_ratio": total_kl / gaussian_kl,
        "nominal_tv_from_whole_conditioning": 0.5 * float(np.sum(np.abs(weights_a - occupancy_a))),
        "perturbed_tv_from_whole_conditioning": 0.5 * float(np.sum(np.abs(weights_b - occupancy_b))),
        "nominal_mean_squared_displacement": mse_a,
        "whole_conditioning_mean_squared_displacement": whole_mse_a,
        "nominal_mse_ratio_to_whole": mse_a / whole_mse_a,
        "componentwise_expected_rejection_proposals": component_rejection_cost_a,
        "whole_union_expected_rejection_proposals": whole_rejection_cost_a,
    }


def evaluate_anchored_tempered_patch_pair(
    region: RectangleUnion,
    anchor: FloatArray,
    center_b: FloatArray,
    sigma: float,
    tau: float,
    diameter: float,
) -> dict[str, Any]:
    """Evaluate the anchored tempered family from one anchor to one center.

    The anchor-side gate is the whole-union occupancy gate.  At the second
    center it is geometrically interpolated with the second occupancy gate.
    Because the boxes form an a.e.-disjoint partition, the tagged KL and total
    variation identities are exact.  ``diameter`` must be a certified upper
    bound on the Euclidean diameter of ``region``; this routine cannot verify
    that geometric premise from the scalar supplied by the caller.
    """

    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("certified diameter upper bound must be finite and positive")
    stats_anchor = component_statistics(region, anchor, sigma)
    stats_b = component_statistics(region, center_b, sigma)
    occupancy_anchor = mixture_weights(stats_anchor)
    occupancy_b = mixture_weights(stats_b)
    gate_anchor = occupancy_anchor
    gate_b = anchored_tempered_weights(occupancy_anchor, occupancy_b, tau)
    component_kls = np.asarray(
        [
            box_conditioned_kl(box, anchor, center_b, sigma, left, right)
            for box, left, right in zip(
                region.boxes,
                stats_anchor,
                stats_b,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    gate_kl = categorical_kl(gate_anchor, gate_b)
    within_kl = float(np.dot(gate_anchor, component_kls))
    total_kl = gate_kl + within_kl
    displacement = float(np.linalg.norm(center_b - anchor))
    gaussian_kl = displacement**2 / (2.0 * sigma**2)
    if gaussian_kl <= 0.0:
        raise ValueError("the anchor and second center must be distinct")
    exact_tv_from_whole = 0.5 * float(np.sum(np.abs(gate_b - occupancy_b)))
    tv_bound = math.tanh(
        (1.0 - tau) * diameter * displacement / (4.0 * sigma**2)
    )
    global_kl_bound = (
        1.0 / (2.0 * sigma**2)
        + tau**2 * diameter**2 / (8.0 * sigma**4)
    ) * displacement**2
    return {
        "tau": float(tau),
        "anchor": np.asarray(anchor, dtype=np.float64).tolist(),
        "center_b": np.asarray(center_b, dtype=np.float64).tolist(),
        "sigma": float(sigma),
        "diameter": float(diameter),
        "distance": displacement,
        "occupancy_anchor": occupancy_anchor.tolist(),
        "occupancy_b": occupancy_b.tolist(),
        "gate_anchor": gate_anchor.tolist(),
        "gate_b": gate_b.tolist(),
        "component_kls": component_kls.tolist(),
        "within_kl": within_kl,
        "gate_kl": gate_kl,
        "total_kl": total_kl,
        "gaussian_kl": gaussian_kl,
        "within_ratio": within_kl / gaussian_kl,
        "gate_ratio": gate_kl / gaussian_kl,
        "total_ratio": total_kl / gaussian_kl,
        "global_kl_bound": global_kl_bound,
        "global_kl_bound_ratio": global_kl_bound / gaussian_kl,
        "nominal_tv_from_whole": 0.0,
        "tv_from_whole_at_b": exact_tv_from_whole,
        "tv_from_whole_bound": tv_bound,
    }


def tempered_local_sensitivity(
    region: RectangleUnion,
    center: FloatArray,
    sigma: float,
    direction: FloatArray,
    tau: float,
    prior: FloatArray | None = None,
) -> dict[str, Any]:
    """Exact local KL decomposition for a tempered occupancy gate."""

    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    unit = direction / norm
    stats = component_statistics(region, center, sigma)
    weights = tempered_mixture_weights(stats, tau, prior)
    means = np.vstack([item.mean for item in stats])
    covariances = np.stack([item.covariance for item in stats])
    mixture_mean = np.sum(weights[:, None] * means, axis=0)
    within = np.sum(weights[:, None, None] * covariances, axis=0)
    centered_means = means - mixture_mean
    between = np.einsum("s,si,sj->ij", weights, centered_means, centered_means)
    within_ratio = float(unit @ within @ unit) / sigma**2
    unscaled_between_ratio = float(unit @ between @ unit) / sigma**2
    gate_ratio = tau**2 * unscaled_between_ratio
    return {
        "tau": float(tau),
        "weights": weights.tolist(),
        "within_local_kl_ratio": within_ratio,
        "unscaled_between_mean_ratio": unscaled_between_ratio,
        "gate_local_kl_ratio": gate_ratio,
        "total_local_kl_ratio": within_ratio + gate_ratio,
    }


def local_covariance_decomposition(
    region: RectangleUnion,
    center: FloatArray,
    sigma: float,
    direction: FloatArray,
) -> dict[str, Any]:
    """Decompose local Fisher sensitivity into within- and between-patch parts."""

    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    unit = direction / norm
    stats = component_statistics(region, center, sigma)
    weights = mixture_weights(stats)
    means = np.vstack([item.mean for item in stats])
    covariances = np.stack([item.covariance for item in stats])
    mixture_mean = np.sum(weights[:, None] * means, axis=0)
    within = np.sum(weights[:, None, None] * covariances, axis=0)
    centered_means = means - mixture_mean
    between = np.einsum("s,si,sj->ij", weights, centered_means, centered_means)
    total = rectangle_union_covariance(region, center, sigma)

    within_directional = float(unit @ within @ unit)
    between_directional = float(unit @ between @ unit)
    total_directional = float(unit @ total @ unit)
    return {
        "direction": unit.tolist(),
        "weights": weights.tolist(),
        "component_means": means.tolist(),
        "mixture_mean": mixture_mean.tolist(),
        "within_covariance": within.tolist(),
        "between_covariance": between.tolist(),
        "total_covariance": total.tolist(),
        "covariance_decomposition_max_abs_error": float(np.max(np.abs(total - within - between))),
        "within_directional_variance": within_directional,
        "between_directional_variance": between_directional,
        "total_directional_variance": total_directional,
        "within_local_kl_ratio": within_directional / sigma**2,
        "gate_local_kl_ratio": between_directional / sigma**2,
        "whole_local_kl_ratio": total_directional / sigma**2,
        "ratio_decomposition_abs_error": abs(
            total_directional - within_directional - between_directional
        )
        / sigma**2,
    }
