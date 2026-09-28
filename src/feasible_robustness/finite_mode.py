"""Exact finite-mode RBF gates for native discrete feasible actions.

These utilities deliberately describe a categorical law with respect to
counting measure on a finite support.  They do *not* condition an ambient
continuous Gaussian on a probability-zero set.  The Gaussian radial basis
function is only the score used to define the categorical gate.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validated_support(actions: FloatArray) -> FloatArray:
    support = np.asarray(actions, dtype=np.float64)
    if support.ndim != 2 or support.shape[0] == 0 or support.shape[1] == 0:
        raise ValueError("actions must be a nonempty two-dimensional array")
    if np.any(~np.isfinite(support)):
        raise ValueError("actions must be finite")
    if np.unique(support, axis=0).shape[0] != support.shape[0]:
        raise ValueError("actions must contain distinct point modes")
    return support


def _validated_center(center: FloatArray, dimension: int, name: str) -> FloatArray:
    vector = np.asarray(center, dtype=np.float64)
    if vector.shape != (dimension,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with one entry per action coordinate")
    return vector


def _validated_sigma(sigma: float) -> float:
    value = float(sigma)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return value


def _normalize_log_weights(log_scores: FloatArray) -> FloatArray:
    maximum = float(np.max(log_scores))
    log_normalizer = maximum + math.log(float(np.sum(np.exp(log_scores - maximum))))
    return np.asarray(log_scores - log_normalizer, dtype=np.float64)


def cartesian_product_levels(levels: list[FloatArray] | tuple[FloatArray, ...]) -> FloatArray:
    """Enumerate a finite Cartesian product with one level vector per coordinate."""

    if len(levels) == 0:
        raise ValueError("levels must contain at least one coordinate")
    validated: list[FloatArray] = []
    for coordinate, raw in enumerate(levels):
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"levels[{coordinate}] must be a nonempty vector")
        if np.any(~np.isfinite(values)):
            raise ValueError(f"levels[{coordinate}] must be finite")
        if np.unique(values).size != values.size:
            raise ValueError(f"levels[{coordinate}] must not contain duplicates")
        validated.append(values)
    return np.asarray(list(itertools.product(*validated)), dtype=np.float64)


def finite_mode_rbf_log_weights(
    actions: FloatArray,
    center: FloatArray,
    sigma: float,
) -> FloatArray:
    """Return normalized log weights of a counting-measure RBF gate.

    For point mode ``u`` the unnormalized score is
    ``exp(-||u-center||^2 / (2 sigma^2))``.
    """

    support = _validated_support(actions)
    center_array = _validated_center(center, support.shape[1], "center")
    sigma_value = _validated_sigma(sigma)
    scores = -np.sum((support - center_array[None, :]) ** 2, axis=1) / (
        2.0 * sigma_value**2
    )
    return _normalize_log_weights(scores)


def finite_mode_rbf_weights(
    actions: FloatArray,
    center: FloatArray,
    sigma: float,
) -> FloatArray:
    """Return categorical probabilities of a counting-measure RBF gate."""

    return np.exp(finite_mode_rbf_log_weights(actions, center, sigma))


def anchored_finite_mode_log_weights(
    actions: FloatArray,
    anchor: FloatArray,
    current_center: FloatArray,
    sigma: float,
    tau: float,
) -> FloatArray:
    """Geometrically interpolate anchor and current finite-mode RBF gates.

    The categorical probabilities are proportional to
    ``w_anchor(u)**(1-tau) * w_current(u)**tau``.  Thus ``tau=0`` is the
    center-independent anchored gate and ``tau=1`` is the moving RBF gate.
    """

    tau_value = float(tau)
    if not math.isfinite(tau_value) or not 0.0 <= tau_value <= 1.0:
        raise ValueError("tau must be finite and lie in [0, 1]")
    log_anchor = finite_mode_rbf_log_weights(actions, anchor, sigma)
    log_current = finite_mode_rbf_log_weights(actions, current_center, sigma)
    return _normalize_log_weights(
        (1.0 - tau_value) * log_anchor + tau_value * log_current
    )


def categorical_kl_from_log_weights(
    log_probability_a: FloatArray,
    log_probability_b: FloatArray,
) -> float:
    """Return categorical KL while retaining very small probabilities in log space."""

    log_a = np.asarray(log_probability_a, dtype=np.float64)
    log_b = np.asarray(log_probability_b, dtype=np.float64)
    if log_a.ndim != 1 or log_a.size == 0 or log_b.shape != log_a.shape:
        raise ValueError("log-probability vectors must be nonempty and have the same shape")
    if np.any(~np.isfinite(log_a)) or np.any(~np.isfinite(log_b)):
        raise ValueError("log-probabilities must be finite")
    probability_a = np.exp(log_a)
    value = float(np.dot(probability_a, log_a - log_b))
    return 0.0 if value < 0.0 and abs(value) <= 1e-14 else value


def categorical_total_variation_from_log_weights(
    log_probability_a: FloatArray,
    log_probability_b: FloatArray,
) -> float:
    """Return exact total variation between two finite categorical laws."""

    log_a = np.asarray(log_probability_a, dtype=np.float64)
    log_b = np.asarray(log_probability_b, dtype=np.float64)
    if log_a.ndim != 1 or log_a.size == 0 or log_b.shape != log_a.shape:
        raise ValueError("log-probability vectors must be nonempty and have the same shape")
    if np.any(~np.isfinite(log_a)) or np.any(~np.isfinite(log_b)):
        raise ValueError("log-probabilities must be finite")
    return 0.5 * float(np.sum(np.abs(np.exp(log_a) - np.exp(log_b))))


def finite_mode_moments(
    actions: FloatArray,
    log_weights: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return the exact mean and covariance of a finite categorical action law."""

    support = _validated_support(actions)
    log_probability = np.asarray(log_weights, dtype=np.float64)
    if log_probability.shape != (support.shape[0],) or np.any(~np.isfinite(log_probability)):
        raise ValueError("log_weights must be finite with one entry per point mode")
    probability = np.exp(log_probability)
    total = float(np.sum(probability))
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("log_weights must be normalized")
    mean = probability @ support
    centered = support - mean[None, :]
    covariance = (centered * probability[:, None]).T @ centered
    covariance = 0.5 * (covariance + covariance.T)
    return np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64)


def finite_mode_local_fisher(
    actions: FloatArray,
    center: FloatArray,
    sigma: float,
) -> dict[str, FloatArray | float]:
    """Return the exact center Fisher matrix and worst Gaussian-rate ratio.

    For the RBF categorical family the Fisher matrix is ``Cov(U)/sigma^4``.
    The equal-covariance ambient Gaussian has Fisher matrix ``I/sigma^2``;
    therefore the worst local divergence ratio is
    ``lambda_max(Cov(U))/sigma^2``.
    """

    support = _validated_support(actions)
    sigma_value = _validated_sigma(sigma)
    log_weights = finite_mode_rbf_log_weights(support, center, sigma_value)
    mean, covariance = finite_mode_moments(support, log_weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    return {
        "mean": mean,
        "covariance": covariance,
        "fisher_information": covariance / sigma_value**4,
        "worst_direction": np.asarray(direction, dtype=np.float64),
        "worst_local_kl_ratio": float(eigenvalues[-1] / sigma_value**2),
    }


def evaluate_anchored_finite_mode_pair(
    actions: FloatArray,
    anchor: FloatArray,
    current_center: FloatArray,
    sigma: float,
    tau: float,
) -> dict[str, FloatArray | float]:
    """Evaluate exact divergence and locality metrics for one anchored gate pair."""

    support = _validated_support(actions)
    anchor_array = _validated_center(anchor, support.shape[1], "anchor")
    current_array = _validated_center(current_center, support.shape[1], "current_center")
    sigma_value = _validated_sigma(sigma)
    log_anchor = finite_mode_rbf_log_weights(support, anchor_array, sigma_value)
    log_tempered = anchored_finite_mode_log_weights(
        support,
        anchor_array,
        current_array,
        sigma_value,
        tau,
    )
    weights = np.exp(log_tempered)
    mean, covariance = finite_mode_moments(support, log_tempered)
    gaussian_kl = float(np.dot(current_array - anchor_array, current_array - anchor_array)) / (
        2.0 * sigma_value**2
    )
    gate_kl = categorical_kl_from_log_weights(log_anchor, log_tempered)
    tracking_current = float(
        np.dot(weights, np.sum((support - current_array[None, :]) ** 2, axis=1))
    )
    tracking_anchor = float(
        np.dot(weights, np.sum((support - anchor_array[None, :]) ** 2, axis=1))
    )
    all_off = np.all(support == 0.0, axis=1)
    return {
        "weights": weights,
        "mean": mean,
        "covariance": covariance,
        "gaussian_kl": gaussian_kl,
        "gate_kl": gate_kl,
        "gate_kl_ratio": gate_kl / gaussian_kl if gaussian_kl > 0.0 else 0.0,
        "total_variation": categorical_total_variation_from_log_weights(
            log_anchor,
            log_tempered,
        ),
        "expected_squared_distance_to_current": tracking_current,
        "expected_squared_distance_to_anchor": tracking_anchor,
        "probability_all_off": float(np.sum(weights[all_off])),
    }

