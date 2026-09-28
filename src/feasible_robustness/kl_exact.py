"""Deterministic low-dimensional KL helpers for rectangle-union feasible sets.

The dichotomy experiments need a deterministic way to test whether the convex
truncated-Gaussian KL contraction survives or fails on simple nonconvex sets.
For disjoint unions of axis-aligned rectangles, the normalizer and first
moments of a truncated isotropic Gaussian are products of one-dimensional
Gaussian CDF/PDF terms, so the KL identity can be evaluated without Monte
Carlo.

The identities are analytic.  This implementation evaluates them in IEEE
double precision (with stable tail/narrow-interval fallbacks), so its outputs
are not outward-rounded interval certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.stats import norm


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AxisAlignedBox:
    """Closed axis-aligned box with nonempty interior."""

    lower: FloatArray
    upper: FloatArray

    @classmethod
    def from_bounds(cls, lower: ArrayLike, upper: ArrayLike) -> "AxisAlignedBox":
        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        if lower_arr.ndim != 1 or upper_arr.ndim != 1:
            raise ValueError("box bounds must be one-dimensional")
        if lower_arr.shape != upper_arr.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if lower_arr.size == 0:
            raise ValueError("box dimension must be positive")
        if not np.all(np.isfinite(lower_arr)) or not np.all(np.isfinite(upper_arr)):
            raise ValueError("box bounds must be finite")
        if np.any(lower_arr >= upper_arr):
            raise ValueError("every lower bound must be strictly below its upper bound")
        return cls(lower=lower_arr, upper=upper_arr)

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    def contains(self, point: ArrayLike, *, tol: float = 1e-12) -> bool:
        vector = np.asarray(point, dtype=np.float64)
        if vector.shape != self.lower.shape:
            return False
        return bool(np.all(vector >= self.lower - tol) and np.all(vector <= self.upper + tol))


@dataclass(frozen=True)
class RectangleUnion:
    """Finite disjoint union of axis-aligned boxes."""

    boxes: tuple[AxisAlignedBox, ...]

    @classmethod
    def from_bounds(
        cls,
        boxes: list[tuple[ArrayLike, ArrayLike]] | tuple[tuple[ArrayLike, ArrayLike], ...],
    ) -> "RectangleUnion":
        parsed = tuple(AxisAlignedBox.from_bounds(lower, upper) for lower, upper in boxes)
        return cls(parsed)

    def __post_init__(self) -> None:
        if not self.boxes:
            raise ValueError("rectangle union must contain at least one box")
        dimension = self.boxes[0].dimension
        if any(box.dimension != dimension for box in self.boxes):
            raise ValueError("all boxes must have the same dimension")
        for i, left in enumerate(self.boxes):
            for right in self.boxes[i + 1 :]:
                if _box_interiors_overlap(left, right):
                    raise ValueError("box interiors must be disjoint")

    @property
    def dimension(self) -> int:
        return self.boxes[0].dimension

    def contains(self, point: ArrayLike, *, tol: float = 1e-12) -> bool:
        return any(box.contains(point, tol=tol) for box in self.boxes)

    def bounding_box(self) -> AxisAlignedBox:
        lower = np.min(np.vstack([box.lower for box in self.boxes]), axis=0)
        upper = np.max(np.vstack([box.upper for box in self.boxes]), axis=0)
        return AxisAlignedBox.from_bounds(lower, upper)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "boxes": [
                {"lower": box.lower.tolist(), "upper": box.upper.tolist()}
                for box in self.boxes
            ]
        }


@dataclass(frozen=True)
class TruncatedGaussianKL:
    """Deterministic analytic KL diagnostic for two rectangle-union centers."""

    kl: float
    convex_bound: float
    excess: float
    quadratic_expectation: float
    normalizer_log_ratio: float
    mean_drift_excess: float
    normalizer_a: float
    normalizer_b: float
    mean_a: FloatArray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kl": self.kl,
            "convex_bound": self.convex_bound,
            "excess": self.excess,
            "quadratic_expectation": self.quadratic_expectation,
            "normalizer_log_ratio": self.normalizer_log_ratio,
            "mean_drift_excess": self.mean_drift_excess,
            "normalizer_a": self.normalizer_a,
            "normalizer_b": self.normalizer_b,
            "mean_a": self.mean_a.tolist(),
        }


def canonical_l_shape(
    *,
    notch_x: float = 0.5,
    notch_y: float = 0.5,
    width: float = 1.0,
    height: float = 1.0,
) -> RectangleUnion:
    """Return `[0,width] x [0,height]` with its upper-right notch removed.

    The representation uses two disjoint rectangles:

    - bottom bar: `[0,width] x [0,notch_y]`;
    - upper-left bar: `[0,notch_x] x [notch_y,height]`.
    """

    if not (0.0 < notch_x < width and 0.0 < notch_y < height):
        raise ValueError("notch coordinates must lie strictly inside the outer rectangle")
    return RectangleUnion.from_bounds(
        (
            ((0.0, 0.0), (width, notch_y)),
            ((0.0, notch_y), (notch_x, height)),
        )
    )


def rectangle_region(
    lower: ArrayLike,
    upper: ArrayLike,
) -> RectangleUnion:
    """Convenience wrapper for a single convex rectangle."""

    return RectangleUnion.from_bounds(((lower, upper),))


def rectangle_union_normalizer(
    region: RectangleUnion,
    center: ArrayLike,
    sigma: float,
) -> float:
    """Compute `int_K phi_sigma(z-center) dz` for a rectangle union."""

    _validate_sigma(sigma)
    center_arr = _validate_center(center, region.dimension)
    return float(sum(_box_probability(box, center_arr, sigma) for box in region.boxes))


def gaussian_box_probability(
    lower: ArrayLike,
    upper: ArrayLike,
    center: ArrayLike,
    sigma: float,
) -> float:
    """Probability that `N(center, sigma^2 I)` lies in an axis-aligned box."""

    _validate_sigma(sigma)
    box = AxisAlignedBox.from_bounds(lower, upper)
    center_arr = _validate_center(center, box.dimension)
    return _box_probability(box, center_arr, sigma)


def gaussian_box_first_moment(
    lower: ArrayLike,
    upper: ArrayLike,
    center: ArrayLike,
    sigma: float,
) -> FloatArray:
    """Unnormalized first moment over one axis-aligned Gaussian box."""

    _validate_sigma(sigma)
    box = AxisAlignedBox.from_bounds(lower, upper)
    center_arr = _validate_center(center, box.dimension)
    moment = np.zeros(box.dimension, dtype=np.float64)
    probability_terms = _interval_probabilities(box, center_arr, sigma)
    for axis in range(box.dimension):
        axis_moment = _interval_first_moment(
            box.lower[axis],
            box.upper[axis],
            center_arr[axis],
            sigma,
        )
        other_probability = float(np.prod(np.delete(probability_terms, axis)))
        moment[axis] += axis_moment * other_probability
    return moment


def gaussian_box_second_moment(
    lower: ArrayLike,
    upper: ArrayLike,
    center: ArrayLike,
    sigma: float,
) -> FloatArray:
    """Unnormalized matrix ``int_box z z^T phi_sigma(z-center) dz``."""

    _validate_sigma(sigma)
    box = AxisAlignedBox.from_bounds(lower, upper)
    center_arr = _validate_center(center, box.dimension)
    return _box_second_moment(box, center_arr, sigma)


def gaussian_box_conditional_mean(
    lower: ArrayLike,
    upper: ArrayLike,
    center: ArrayLike,
    sigma: float,
) -> FloatArray:
    """Conditional mean of `N(center,sigma^2 I)` given an axis-aligned box."""

    probability = gaussian_box_probability(lower, upper, center, sigma)
    if probability <= 0.0:
        raise FloatingPointError("Gaussian box probability underflowed to zero")
    return gaussian_box_first_moment(lower, upper, center, sigma) / probability


def rectangle_union_first_moment(
    region: RectangleUnion,
    center: ArrayLike,
    sigma: float,
) -> FloatArray:
    """Compute `int_K z phi_sigma(z-center) dz` for a rectangle union."""

    _validate_sigma(sigma)
    center_arr = _validate_center(center, region.dimension)
    moment = np.zeros(region.dimension, dtype=np.float64)
    for box in region.boxes:
        probability_terms = _interval_probabilities(box, center_arr, sigma)
        for axis in range(region.dimension):
            axis_moment = _interval_first_moment(
                box.lower[axis],
                box.upper[axis],
                center_arr[axis],
                sigma,
            )
            other_probability = float(np.prod(np.delete(probability_terms, axis)))
            moment[axis] += axis_moment * other_probability
    return moment


def rectangle_union_second_moment(
    region: RectangleUnion,
    center: ArrayLike,
    sigma: float,
) -> FloatArray:
    """Compute ``int_K z z^T phi_sigma(z-center) dz`` for a box union."""

    _validate_sigma(sigma)
    center_arr = _validate_center(center, region.dimension)
    second_moment = np.zeros((region.dimension, region.dimension), dtype=np.float64)
    for box in region.boxes:
        second_moment += _box_second_moment(box, center_arr, sigma)
    return second_moment


def rectangle_union_covariance(
    region: RectangleUnion,
    center: ArrayLike,
    sigma: float,
) -> FloatArray:
    """Deterministic analytic covariance for a Gaussian restricted to a box union."""

    _validate_sigma(sigma)
    center_arr = _validate_center(center, region.dimension)
    normalizer = rectangle_union_normalizer(region, center_arr, sigma)
    if normalizer <= 0.0:
        raise FloatingPointError("Gaussian normalizer underflowed to zero")
    mean = rectangle_union_first_moment(region, center_arr, sigma) / normalizer
    raw_second = rectangle_union_second_moment(region, center_arr, sigma) / normalizer
    covariance = raw_second - np.outer(mean, mean)
    return 0.5 * (covariance + covariance.T)


def rectangle_union_centered_absolute_projection_moment(
    region: RectangleUnion,
    center: ArrayLike,
    sigma: float,
    direction: ArrayLike,
    *,
    epsabs: float = 1e-12,
    epsrel: float = 1e-11,
) -> float:
    """Compute ``E|<direction,Z-EZ>|`` for a conditioned 2D Gaussian.

    The support may be a disjoint rectangle union.  One coordinate is
    integrated analytically using Gaussian interval masses and first moments;
    deterministic adaptive quadrature is used only for the remaining scalar
    coordinate.  The direction is not normalized, so this function is the
    support function of the centered ``L_1`` centroid body.
    """

    _validate_sigma(sigma)
    center_arr = _validate_center(center, region.dimension)
    direction_arr = _validate_center(direction, region.dimension)
    if region.dimension != 2:
        raise ValueError("centered absolute projections currently support 2D")
    if not np.all(np.isfinite(direction_arr)) or float(np.linalg.norm(direction_arr)) <= 0.0:
        raise ValueError("direction must be finite and nonzero")
    if epsabs < 0.0 or epsrel <= 0.0:
        raise ValueError("quadrature tolerances must be positive")

    normalizer = rectangle_union_normalizer(region, center_arr, sigma)
    if normalizer <= 0.0:
        raise FloatingPointError("Gaussian normalizer underflowed to zero")
    mean = rectangle_union_first_moment(region, center_arr, sigma) / normalizer

    # Integrate analytically along the coordinate carrying the larger
    # directional coefficient.  This avoids unstable division by a nearly zero
    # coefficient when locating the absolute-value kink.
    inner_axis = int(np.argmax(np.abs(direction_arr)))
    outer_axis = 1 - inner_axis
    inner_coefficient = float(direction_arr[inner_axis])
    outer_coefficient = float(direction_arr[outer_axis])
    total = 0.0
    for box in region.boxes:
        inner_lower = float(box.lower[inner_axis])
        inner_upper = float(box.upper[inner_axis])
        outer_lower = float(box.lower[outer_axis])
        outer_upper = float(box.upper[outer_axis])
        inner_center = float(center_arr[inner_axis])
        outer_center = float(center_arr[outer_axis])
        inner_mean = float(mean[inner_axis])
        outer_mean = float(mean[outer_axis])

        if abs(outer_coefficient) <= 1e-15 * abs(inner_coefficient):
            inner_absolute = _interval_absolute_deviation(
                inner_lower,
                inner_upper,
                inner_center,
                sigma,
                inner_mean,
            )
            outer_probability = _interval_probability(
                outer_lower,
                outer_upper,
                outer_center,
                sigma,
            )
            total += abs(inner_coefficient) * inner_absolute * outer_probability
            continue

        ratio = outer_coefficient / inner_coefficient

        def integrand(outer_value: float) -> float:
            location = inner_mean - ratio * (outer_value - outer_mean)
            inner_absolute = _interval_absolute_deviation(
                inner_lower,
                inner_upper,
                inner_center,
                sigma,
                location,
            )
            outer_density = float(norm.pdf((outer_value - outer_center) / sigma) / sigma)
            return abs(inner_coefficient) * inner_absolute * outer_density

        # The analytic inner expression changes branch only when its zero
        # crossing reaches an interval endpoint.  Supplying those locations to
        # QUADPACK makes the scalar integral reproducible at tight tolerances.
        points: list[float] = []
        for endpoint in (inner_lower, inner_upper):
            outer_break = outer_mean + (inner_mean - endpoint) / ratio
            if outer_lower < outer_break < outer_upper:
                points.append(float(outer_break))
        value, _error = quad(
            integrand,
            outer_lower,
            outer_upper,
            points=sorted(set(points)) or None,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=200,
        )
        total += float(value)
    return float(max(0.0, total / normalizer))


def truncated_gaussian_kl_rectangle_union(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    region: RectangleUnion,
) -> TruncatedGaussianKL:
    """Compute `KL(q_a || q_b)` for a rectangle-union truncated Gaussian."""

    _validate_sigma(sigma)
    a = _validate_center(center_a, region.dimension)
    b = _validate_center(center_b, region.dimension)
    if not region.contains(a):
        raise ValueError("center_a must lie in the rectangle union")
    if not region.contains(b):
        raise ValueError("center_b must lie in the rectangle union")

    normalizer_a = rectangle_union_normalizer(region, a, sigma)
    normalizer_b = rectangle_union_normalizer(region, b, sigma)
    if normalizer_a <= 0.0 or normalizer_b <= 0.0:
        raise FloatingPointError("Gaussian normalizer underflowed to zero")

    first_moment = rectangle_union_first_moment(region, a, sigma)
    mean_a = first_moment / normalizer_a
    quadratic_expectation = (
        2.0 * float(np.dot(mean_a, a - b))
        + float(np.dot(b, b) - np.dot(a, a))
    ) / (2.0 * sigma**2)
    normalizer_log_ratio = float(np.log(normalizer_b / normalizer_a))
    kl = float(quadratic_expectation + normalizer_log_ratio)
    bound = convex_gaussian_kl_bound(a, b, sigma)
    mean_drift_excess = float(np.dot(mean_a - a, a - b) / sigma**2)
    return TruncatedGaussianKL(
        kl=kl,
        convex_bound=bound,
        excess=kl - bound,
        quadratic_expectation=quadratic_expectation,
        normalizer_log_ratio=normalizer_log_ratio,
        mean_drift_excess=mean_drift_excess,
        normalizer_a=normalizer_a,
        normalizer_b=normalizer_b,
        mean_a=mean_a,
    )


def rectangle_union_path_integrated_kl(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    region: RectangleUnion,
    *,
    quadrature_order: int = 32,
) -> float:
    """Evaluate the exact KL path integral through conditional covariance.

    For ``d = center_b - center_a``, the identity is

    ``KL(Q_a || Q_b) = sigma**-4 integral_0^1
    (1-t) d.T Cov(Q_{a+t d}) d dt``.

    Only the scalar integral is numerical.  Conditional covariance at every
    quadrature node is analytic for a finite rectangle union.
    """

    _validate_sigma(sigma)
    a = _validate_center(center_a, region.dimension)
    b = _validate_center(center_b, region.dimension)
    if not region.contains(a):
        raise ValueError("center_a must lie in the rectangle union")
    if not region.contains(b):
        raise ValueError("center_b must lie in the rectangle union")
    if not isinstance(quadrature_order, int) or quadrature_order < 1:
        raise ValueError("quadrature_order must be a positive integer")
    displacement = b - a
    if not np.any(displacement):
        return 0.0
    nodes, weights = leggauss(quadrature_order)
    parameters = 0.5 * (nodes + 1.0)
    integral = 0.0
    for parameter, weight in zip(parameters, weights, strict=True):
        center = a + float(parameter) * displacement
        covariance = rectangle_union_covariance(region, center, sigma)
        directional_variance = float(displacement @ covariance @ displacement)
        integral += 0.5 * float(weight) * (1.0 - float(parameter)) * directional_variance
    return float(integral / sigma**4)


def rectangle_union_projected_probability(
    region: RectangleUnion,
    center: ArrayLike,
    sigma: float,
    direction: ArrayLike,
    threshold: float,
    *,
    quadrature_order: int = 64,
) -> float:
    """Return `P(<u,Z> <= threshold | Z in K)` for a rectangle union.

    Here `Z ~ N(center, sigma^2 I)`, `K` is the rectangle union, and `u` is the
    normalized version of `direction`.  In two dimensions the numerator is a
    deterministic one-dimensional quadrature over the projected coordinate.
    """

    _validate_sigma(sigma)
    center_arr = _validate_center(center, region.dimension)
    direction_arr = _validate_center(direction, region.dimension)
    if region.dimension != 2:
        raise ValueError("projected rectangle-union probabilities currently support 2D")
    direction_norm = float(np.linalg.norm(direction_arr))
    if direction_norm <= 0.0 or not np.isfinite(direction_norm):
        raise ValueError("direction must be finite and nonzero")
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")

    normalizer = rectangle_union_normalizer(region, center_arr, sigma)
    if normalizer <= 0.0:
        raise FloatingPointError("Gaussian normalizer underflowed to zero")
    unit = direction_arr / direction_norm
    numerator = sum(
        _box_projected_probability(
            box,
            center_arr,
            sigma,
            unit,
            threshold,
            quadrature_order=quadrature_order,
        )
        for box in region.boxes
    )
    return float(np.clip(numerator / normalizer, 0.0, 1.0))


def rectangle_union_total_variation(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    region: RectangleUnion,
    *,
    quadrature_order: int = 64,
) -> float:
    """Compute total variation between two truncated Gaussians on a 2D union.

    For equal isotropic covariance and common support, the likelihood-ratio
    crossing set is a halfspace.  Therefore TV is the difference between two
    projected CDFs at the crossing threshold.
    """

    _validate_sigma(sigma)
    a = _validate_center(center_a, region.dimension)
    b = _validate_center(center_b, region.dimension)
    if region.dimension != 2:
        raise ValueError("rectangle_union_total_variation currently supports 2D")
    if not region.contains(a):
        raise ValueError("center_a must lie in the rectangle union")
    if not region.contains(b):
        raise ValueError("center_b must lie in the rectangle union")
    direction = b - a
    distance = float(np.linalg.norm(direction))
    if distance <= 0.0:
        return 0.0

    normalizer_a = rectangle_union_normalizer(region, a, sigma)
    normalizer_b = rectangle_union_normalizer(region, b, sigma)
    if normalizer_a <= 0.0 or normalizer_b <= 0.0:
        raise FloatingPointError("Gaussian normalizer underflowed to zero")
    threshold_dot = (
        0.5 * (float(np.dot(b, b)) - float(np.dot(a, a)))
        + sigma**2 * float(np.log(normalizer_b / normalizer_a))
    )
    threshold = threshold_dot / distance
    unit = direction / distance
    probability_a = rectangle_union_projected_probability(
        region,
        a,
        sigma,
        unit,
        threshold,
        quadrature_order=quadrature_order,
    )
    probability_b = rectangle_union_projected_probability(
        region,
        b,
        sigma,
        unit,
        threshold,
        quadrature_order=quadrature_order,
    )
    return float(np.clip(abs(probability_a - probability_b), 0.0, 1.0))


def rectangle_union_segment_is_visible(
    region: RectangleUnion,
    start: ArrayLike,
    end: ArrayLike,
    *,
    tol: float = 1e-12,
) -> bool:
    """Return whether the closed segment from `start` to `end` stays in `region`.

    The check is exact for finite unions of closed axis-aligned boxes up to the
    supplied floating-point tolerance: each box contributes an interval of
    segment parameters, and the segment is visible iff the union of those
    intervals covers `[0,1]`.
    """

    start_arr = _validate_center(start, region.dimension)
    end_arr = _validate_center(end, region.dimension)
    if region.dimension != 2:
        raise ValueError("segment visibility currently supports 2D rectangle unions")
    if not region.contains(start_arr, tol=tol):
        raise ValueError("start must lie in the rectangle union")
    if not region.contains(end_arr, tol=tol):
        raise ValueError("end must lie in the rectangle union")

    intervals = []
    for box in region.boxes:
        interval = _box_segment_parameter_interval(box, start_arr, end_arr, tol=tol)
        if interval is not None:
            intervals.append(interval)
    if not intervals:
        return False

    intervals.sort(key=lambda item: item[0])
    covered_until = 0.0
    for lower, upper in intervals:
        if lower > covered_until + tol:
            return False
        covered_until = max(covered_until, upper)
        if covered_until >= 1.0 - tol:
            return True
    return covered_until >= 1.0 - tol


def rectangle_union_boundary_vertices(region: RectangleUnion) -> FloatArray:
    """Return unique box vertices used as visibility-graph candidates."""

    if region.dimension != 2:
        raise ValueError("boundary-vertex extraction currently supports 2D")
    vertices: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for box in region.boxes:
        for x in (box.lower[0], box.upper[0]):
            for y in (box.lower[1], box.upper[1]):
                key = (float(x), float(y))
                if key not in seen:
                    seen.add(key)
                    vertices.append(key)
    return np.array(vertices, dtype=np.float64)


def rectangle_union_geodesic_distance(
    region: RectangleUnion,
    start: ArrayLike,
    end: ArrayLike,
    *,
    tol: float = 1e-12,
) -> float:
    """Shortest-path distance inside a connected 2D rectangle union.

    The visibility graph contains the endpoints and all rectangle vertices.
    For the L-shaped and rectangle-union geometries used in the dichotomy
    experiments, shortest Euclidean paths bend only at such vertices.
    """

    start_arr = _validate_center(start, region.dimension)
    end_arr = _validate_center(end, region.dimension)
    if region.dimension != 2:
        raise ValueError("rectangle-union geodesics currently support 2D")
    if not region.contains(start_arr, tol=tol):
        raise ValueError("start must lie in the rectangle union")
    if not region.contains(end_arr, tol=tol):
        raise ValueError("end must lie in the rectangle union")
    if np.linalg.norm(start_arr - end_arr) <= tol:
        return 0.0
    if rectangle_union_segment_is_visible(region, start_arr, end_arr, tol=tol):
        return float(np.linalg.norm(start_arr - end_arr))

    candidates = [start_arr, end_arr]
    for vertex in rectangle_union_boundary_vertices(region):
        if region.contains(vertex, tol=tol):
            candidates.append(vertex)
    nodes = _unique_points(candidates, tol=tol)
    start_index = 0
    end_index = 1

    node_count = len(nodes)
    distances = [math.inf] * node_count
    visited = [False] * node_count
    distances[start_index] = 0.0
    for _ in range(node_count):
        current = min(
            (index for index in range(node_count) if not visited[index]),
            key=lambda index: distances[index],
            default=None,
        )
        if current is None or math.isinf(distances[current]):
            break
        if current == end_index:
            return float(distances[current])
        visited[current] = True
        for neighbor in range(node_count):
            if visited[neighbor] or neighbor == current:
                continue
            if not rectangle_union_segment_is_visible(region, nodes[current], nodes[neighbor], tol=tol):
                continue
            weight = float(np.linalg.norm(nodes[current] - nodes[neighbor]))
            candidate = distances[current] + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
    if math.isinf(distances[end_index]):
        raise ValueError("no feasible path found between the supplied points")
    return float(distances[end_index])


def convex_gaussian_kl_bound(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
) -> float:
    """Return `||a-b||^2/(2 sigma^2)`."""

    _validate_sigma(sigma)
    a = np.asarray(center_a, dtype=np.float64)
    b = np.asarray(center_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("centers must have the same shape")
    return float(np.sum((a - b) ** 2) / (2.0 * sigma**2))


def _validate_sigma(sigma: float) -> None:
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")


def _validate_center(center: ArrayLike, dimension: int) -> FloatArray:
    vector = np.asarray(center, dtype=np.float64)
    if vector.shape != (dimension,):
        raise ValueError(f"center must have shape ({dimension},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("center must contain only finite values")
    return vector


def _box_interiors_overlap(left: AxisAlignedBox, right: AxisAlignedBox) -> bool:
    overlap_lower = np.maximum(left.lower, right.lower)
    overlap_upper = np.minimum(left.upper, right.upper)
    return bool(np.all(overlap_lower < overlap_upper))


def _standardized_interval(
    lower: float,
    upper: float,
    center: float,
    sigma: float,
) -> tuple[float, float]:
    return ((lower - center) / sigma, (upper - center) / sigma)


def _interval_probability(
    lower: float,
    upper: float,
    center: float,
    sigma: float,
) -> float:
    alpha, beta = _standardized_interval(lower, upper, center, sigma)
    probability, _, _ = _standardized_interval_moments(alpha, beta)
    return probability


def _interval_first_moment(
    lower: float,
    upper: float,
    center: float,
    sigma: float,
) -> float:
    alpha, beta = _standardized_interval(lower, upper, center, sigma)
    probability, first_standardized, _ = _standardized_interval_moments(alpha, beta)
    return float(center * probability + sigma * first_standardized)


def _interval_second_moment(
    lower: float,
    upper: float,
    center: float,
    sigma: float,
) -> float:
    alpha, beta = _standardized_interval(lower, upper, center, sigma)
    probability, density_difference, standardized_second = _standardized_interval_moments(
        alpha, beta
    )
    return float(
        center * center * probability
        + 2.0 * center * sigma * density_difference
        + sigma * sigma * standardized_second
    )


def _interval_absolute_deviation(
    lower: float,
    upper: float,
    center: float,
    sigma: float,
    location: float,
) -> float:
    """Unnormalized ``int |x-location| phi_sigma(x-center) dx`` on an interval."""

    if location <= lower:
        probability = _interval_probability(lower, upper, center, sigma)
        first = _interval_first_moment(lower, upper, center, sigma)
        return float(max(0.0, first - location * probability))
    if location >= upper:
        probability = _interval_probability(lower, upper, center, sigma)
        first = _interval_first_moment(lower, upper, center, sigma)
        return float(max(0.0, location * probability - first))

    left_probability = _interval_probability(lower, location, center, sigma)
    right_probability = _interval_probability(location, upper, center, sigma)
    left_first = _interval_first_moment(lower, location, center, sigma)
    right_first = _interval_first_moment(location, upper, center, sigma)
    value = (
        location * left_probability
        - left_first
        + right_first
        - location * right_probability
    )
    return float(max(0.0, value))


@lru_cache(maxsize=16384)
def _standardized_interval_moments(
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    """Stable mass and first two raw moments of ``N(0,1)`` on an interval.

    CDF subtraction loses the entire mass in the positive tail, while the
    recurrence for the second moment can cancel to a negative number on very
    narrow intervals.  Use the survival-function branch in the positive tail,
    an ``expm1`` density difference, and deterministic adaptive quadrature only
    when the closed forms are numerically ill-conditioned.
    """

    if not alpha < beta:
        raise ValueError("standardized interval must have positive width")
    midpoint = 0.5 * (alpha + beta)
    width = beta - alpha
    narrow = width <= 1e-7 * (1.0 + abs(midpoint))

    if alpha >= 0.0:
        probability = float(norm.sf(alpha) - norm.sf(beta))
    elif beta <= 0.0:
        probability = float(norm.cdf(beta) - norm.cdf(alpha))
    else:
        probability = float(norm.cdf(beta) - norm.cdf(alpha))

    density_alpha = float(norm.pdf(alpha))
    exponent_delta = -0.5 * (beta * beta - alpha * alpha)
    density_beta = float(norm.pdf(beta))
    if exponent_delta <= 0.0:
        density_difference = float(-density_alpha * math.expm1(exponent_delta))
    else:
        # Here phi(beta) is the larger density.  Re-express the difference as
        # phi(beta) * (exp(-exponent_delta) - 1) so expm1 never overflows when
        # an interval endpoint is hundreds of standard deviations away.
        density_difference = float(density_beta * math.expm1(-exponent_delta))
    second = float(probability + alpha * density_alpha - beta * density_beta)
    cancellation_scale = abs(probability) + abs(alpha * density_alpha) + abs(beta * density_beta)
    ill_conditioned_second = second <= 0.0 or (
        cancellation_scale > 0.0 and abs(second) <= 1e-11 * cancellation_scale
    )

    if narrow or probability <= 0.0 or ill_conditioned_second:
        # On a finite interval these smooth positive integrands are benign for
        # QUADPACK.  Zero absolute tolerance preserves very small tail/narrow
        # moments instead of rounding them to an arbitrary absolute target.
        probability = float(
            quad(lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi),
                 alpha, beta, epsabs=0.0, epsrel=5e-14, limit=100)[0]
        )
        density_difference = float(
            quad(lambda x: x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi),
                 alpha, beta, epsabs=0.0, epsrel=5e-14, limit=100)[0]
        )
        second = float(
            quad(lambda x: x * x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi),
                 alpha, beta, epsabs=0.0, epsrel=5e-14, limit=100)[0]
        )
    if probability <= 0.0 or second < 0.0:
        raise FloatingPointError("failed to evaluate positive Gaussian interval moments")
    return probability, density_difference, second


def _interval_probabilities(
    box: AxisAlignedBox,
    center: FloatArray,
    sigma: float,
) -> FloatArray:
    return np.array(
        [
            _interval_probability(lower, upper, axis_center, sigma)
            for lower, upper, axis_center in zip(box.lower, box.upper, center, strict=True)
        ],
        dtype=np.float64,
    )


def _box_probability(
    box: AxisAlignedBox,
    center: FloatArray,
    sigma: float,
) -> float:
    return float(np.prod(_interval_probabilities(box, center, sigma)))


def _box_second_moment(
    box: AxisAlignedBox,
    center: FloatArray,
    sigma: float,
) -> FloatArray:
    probabilities = _interval_probabilities(box, center, sigma)
    first_moments = np.array(
        [
            _interval_first_moment(lower, upper, axis_center, sigma)
            for lower, upper, axis_center in zip(box.lower, box.upper, center, strict=True)
        ],
        dtype=np.float64,
    )
    second_moments = np.array(
        [
            _interval_second_moment(lower, upper, axis_center, sigma)
            for lower, upper, axis_center in zip(box.lower, box.upper, center, strict=True)
        ],
        dtype=np.float64,
    )
    result = np.zeros((box.dimension, box.dimension), dtype=np.float64)
    for left in range(box.dimension):
        other = np.delete(probabilities, left)
        result[left, left] = second_moments[left] * float(np.prod(other))
        for right in range(left + 1, box.dimension):
            keep = np.ones(box.dimension, dtype=bool)
            keep[[left, right]] = False
            cross = (
                first_moments[left]
                * first_moments[right]
                * float(np.prod(probabilities[keep]))
            )
            result[left, right] = cross
            result[right, left] = cross
    return result


def _box_segment_parameter_interval(
    box: AxisAlignedBox,
    start: FloatArray,
    end: FloatArray,
    *,
    tol: float,
) -> tuple[float, float] | None:
    direction = end - start
    lower = 0.0
    upper = 1.0
    for axis in range(2):
        if abs(float(direction[axis])) <= tol:
            if start[axis] < box.lower[axis] - tol or start[axis] > box.upper[axis] + tol:
                return None
            continue
        first = (box.lower[axis] - start[axis]) / direction[axis]
        second = (box.upper[axis] - start[axis]) / direction[axis]
        axis_lower = min(first, second)
        axis_upper = max(first, second)
        lower = max(lower, float(axis_lower))
        upper = min(upper, float(axis_upper))
        if upper < lower - tol:
            return None
    return max(0.0, lower), min(1.0, upper)


def _unique_points(points: list[FloatArray], *, tol: float) -> list[FloatArray]:
    unique: list[FloatArray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - existing)) <= tol for existing in unique):
            unique.append(np.asarray(point, dtype=np.float64))
    return unique


@lru_cache(maxsize=16)
def _legendre_rule(order: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = leggauss(order)
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _box_projected_probability(
    box: AxisAlignedBox,
    center: FloatArray,
    sigma: float,
    unit_direction: FloatArray,
    threshold: float,
    *,
    quadrature_order: int,
) -> float:
    projections = _box_vertex_projections(box, unit_direction)
    t_min = float(np.min(projections))
    t_max = float(np.max(projections))
    if threshold <= t_min:
        return 0.0
    if threshold >= t_max:
        return _box_probability(box, center, sigma)

    upper = min(float(threshold), t_max)
    breaks = sorted({float(value) for value in projections if t_min < value < upper})
    endpoints = [t_min, *breaks, upper]
    total = 0.0
    for left, right in zip(endpoints[:-1], endpoints[1:], strict=True):
        if right <= left:
            continue
        total += _integrate_projected_box_segment(
            box,
            center,
            sigma,
            unit_direction,
            left,
            right,
            quadrature_order=quadrature_order,
        )
    return float(max(0.0, total))


def _integrate_projected_box_segment(
    box: AxisAlignedBox,
    center: FloatArray,
    sigma: float,
    unit_direction: FloatArray,
    left: float,
    right: float,
    *,
    quadrature_order: int,
) -> float:
    nodes, weights = _legendre_rule(quadrature_order)
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    ts = midpoint + half_width * nodes
    values = np.array(
        [
            _projected_box_density_at_t(box, center, sigma, unit_direction, float(t))
            for t in ts
        ],
        dtype=np.float64,
    )
    return float(half_width * np.dot(weights, values))


def _projected_box_density_at_t(
    box: AxisAlignedBox,
    center: FloatArray,
    sigma: float,
    unit_direction: FloatArray,
    t: float,
) -> float:
    perpendicular = np.array([-unit_direction[1], unit_direction[0]], dtype=np.float64)
    mu_t = float(np.dot(unit_direction, center))
    mu_s = float(np.dot(perpendicular, center))
    interval = _box_cross_section_s_interval(box, unit_direction, perpendicular, t)
    if interval is None:
        return 0.0
    s_lower, s_upper = interval
    if s_upper <= s_lower:
        return 0.0
    t_density = norm.pdf((t - mu_t) / sigma) / sigma
    s_probability = norm.cdf((s_upper - mu_s) / sigma) - norm.cdf((s_lower - mu_s) / sigma)
    return float(t_density * s_probability)


def _box_cross_section_s_interval(
    box: AxisAlignedBox,
    unit_direction: FloatArray,
    perpendicular: FloatArray,
    t: float,
    *,
    tol: float = 1e-14,
) -> tuple[float, float] | None:
    lower = -np.inf
    upper = np.inf
    for axis in range(2):
        base = t * unit_direction[axis]
        slope = perpendicular[axis]
        if abs(slope) <= tol:
            if base < box.lower[axis] - tol or base > box.upper[axis] + tol:
                return None
            continue
        first = (box.lower[axis] - base) / slope
        second = (box.upper[axis] - base) / slope
        axis_lower = min(first, second)
        axis_upper = max(first, second)
        lower = max(lower, axis_lower)
        upper = min(upper, axis_upper)
        if upper < lower:
            return None
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    return float(lower), float(upper)


def _box_vertex_projections(box: AxisAlignedBox, unit_direction: FloatArray) -> FloatArray:
    vertices = np.array(
        [
            [box.lower[0], box.lower[1]],
            [box.lower[0], box.upper[1]],
            [box.upper[0], box.lower[1]],
            [box.upper[0], box.upper[1]],
        ],
        dtype=np.float64,
    )
    return vertices @ unit_direction
