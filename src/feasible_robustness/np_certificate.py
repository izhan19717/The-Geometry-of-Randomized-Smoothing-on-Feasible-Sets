"""Neyman-Pearson certificates for convex rectangle truncations.

The functions here implement the deterministic 2D rectangle case first.  This
is the simplest nontrivial convex setting where the likelihood-ratio ordering
for truncated Gaussians can be evaluated without Monte Carlo: the required CDF
of a linear functional is a one-dimensional quadrature over the rectangle.
"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from scipy.stats import beta, norm

from .kl_exact import gaussian_box_probability


@dataclass(frozen=True)
class RectangleNPCertificate:
    """Pairwise NP certificate diagnostic for one shifted center."""

    lower_top_under_shift: float
    upper_runner_under_shift: float
    margin_under_shift: float
    top_threshold: float
    runner_threshold: float
    direction: np.ndarray


@dataclass(frozen=True)
class RectangleNPRadius:
    """Binary-search NP radius along one direction."""

    radius: float
    iterations: int
    certificate_at_radius: RectangleNPCertificate


@dataclass(frozen=True)
class ConvexPolygonNPRadius:
    """Binary-search NP radius along one direction in a 2D convex polygon."""

    radius: float
    iterations: int
    certificate_at_radius: RectangleNPCertificate


@dataclass(frozen=True)
class RectangleMCNPCertificate:
    """Monte Carlo NP certificate for an arbitrary-dimensional rectangle.

    The samples are exact independent draws from the product truncated normal
    on the rectangle.  Probability bounds use Clopper-Pearson intervals; this
    is a high-confidence numerical certificate, not a deterministic quadrature
    theorem.
    """

    lower_top_under_shift: float
    upper_runner_under_shift: float
    margin_under_shift: float
    top_threshold: float
    runner_threshold: float
    direction: np.ndarray
    sample_count: int
    alpha: float
    confidence: float
    top_successes: int
    runner_successes: int


@dataclass(frozen=True)
class RectangleMCNPRadius:
    """Binary-search MC-NP radius along one direction in a hyperrectangle."""

    radius: float
    iterations: int
    max_radius: float
    certificate_at_radius: RectangleMCNPCertificate


def rectangle_projected_cdf(
    center: ArrayLike,
    sigma: float,
    lower: ArrayLike,
    upper: ArrayLike,
    direction: ArrayLike,
    threshold: float,
    *,
    epsabs: float = 1e-11,
    epsrel: float = 1e-11,
) -> float:
    """CDF of ``<direction, Z>`` for a Gaussian truncated to a 2D rectangle."""

    center_arr = _validate_vector(center, "center")
    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    direction_arr = _validate_vector(direction, "direction")
    if center_arr.shape != (2,) or lower_arr.shape != (2,) or upper_arr.shape != (2,):
        raise ValueError("rectangle_projected_cdf currently supports 2D rectangles")
    if direction_arr.shape != (2,):
        raise ValueError("direction must be two-dimensional")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if np.any(lower_arr >= upper_arr):
        raise ValueError("rectangle lower bounds must be below upper bounds")
    norm_direction = float(np.linalg.norm(direction_arr))
    if norm_direction == 0.0 or not np.isfinite(norm_direction):
        raise ValueError("direction must be finite and nonzero")
    u = direction_arr / norm_direction

    dot_min, dot_max = rectangle_dot_range(lower_arr, upper_arr, u)
    if threshold <= dot_min:
        return 0.0
    if threshold >= dot_max:
        return 1.0

    normalizer = gaussian_box_probability(lower_arr, upper_arr, center_arr, sigma)
    if normalizer <= 0.0:
        raise FloatingPointError("rectangle Gaussian normalizer underflowed")

    ux, uy = float(u[0]), float(u[1])
    cx, cy = float(center_arr[0]), float(center_arr[1])
    lx, ly = float(lower_arr[0]), float(lower_arr[1])
    hx, hy = float(upper_arr[0]), float(upper_arr[1])

    def x_density(x: float) -> float:
        return norm.pdf((x - cx) / sigma) / sigma

    if abs(uy) < 1e-14:
        if ux > 0.0:
            x_hi = min(hx, threshold / ux)
            if x_hi <= lx:
                return 0.0
            x_probability = norm.cdf((x_hi - cx) / sigma) - norm.cdf((lx - cx) / sigma)
        else:
            x_lo = max(lx, threshold / ux)
            if x_lo >= hx:
                return 0.0
            x_probability = norm.cdf((hx - cx) / sigma) - norm.cdf((x_lo - cx) / sigma)
        y_probability = norm.cdf((hy - cy) / sigma) - norm.cdf((ly - cy) / sigma)
        return float(np.clip((x_probability * y_probability) / normalizer, 0.0, 1.0))

    def conditional_y_probability(x: float) -> float:
        y_cut = (threshold - ux * x) / uy
        if uy > 0.0:
            y_hi = min(hy, y_cut)
            if y_hi <= ly:
                return 0.0
            return float(norm.cdf((y_hi - cy) / sigma) - norm.cdf((ly - cy) / sigma))
        y_lo = max(ly, y_cut)
        if y_lo >= hy:
            return 0.0
        return float(norm.cdf((hy - cy) / sigma) - norm.cdf((y_lo - cy) / sigma))

    integral, _error = quad(
        lambda x: x_density(x) * conditional_y_probability(x),
        lx,
        hx,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return float(np.clip(integral / normalizer, 0.0, 1.0))


def rectangle_projected_quantile(
    center: ArrayLike,
    sigma: float,
    lower: ArrayLike,
    upper: ArrayLike,
    direction: ArrayLike,
    probability: float,
) -> float:
    """Quantile of ``<direction,Z>`` for a 2D rectangle-truncated Gaussian."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    direction_arr = _validate_vector(direction, "direction")
    norm_direction = float(np.linalg.norm(direction_arr))
    if norm_direction == 0.0:
        raise ValueError("direction must be nonzero")
    u = direction_arr / norm_direction
    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    dot_min, dot_max = rectangle_dot_range(lower_arr, upper_arr, u)
    if probability <= 0.0:
        return dot_min
    if probability >= 1.0:
        return dot_max

    def shifted_cdf(threshold: float) -> float:
        return rectangle_projected_cdf(center, sigma, lower_arr, upper_arr, u, threshold) - probability

    return float(brentq(shifted_cdf, dot_min, dot_max, xtol=1e-12, rtol=1e-12, maxiter=100))


def rectangle_np_pair_certificate(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    lower: ArrayLike,
    upper: ArrayLike,
    top_lower: float,
    runner_upper: float,
) -> RectangleNPCertificate:
    """Pairwise NP lower/upper bounds from ``q_a`` to ``q_b`` on a rectangle."""

    a = _validate_vector(center_a, "center_a")
    b = _validate_vector(center_b, "center_b")
    if a.shape != b.shape:
        raise ValueError("centers must have the same shape")
    if not 0.0 <= runner_upper <= top_lower <= 1.0:
        raise ValueError("require 0 <= runner_upper <= top_lower <= 1")
    direction = b - a
    norm_direction = float(np.linalg.norm(direction))
    if norm_direction == 0.0:
        return RectangleNPCertificate(
            lower_top_under_shift=top_lower,
            upper_runner_under_shift=runner_upper,
            margin_under_shift=top_lower - runner_upper,
            top_threshold=float("nan"),
            runner_threshold=float("nan"),
            direction=direction,
        )
    unit_direction = direction / norm_direction
    top_threshold = rectangle_projected_quantile(
        a,
        sigma,
        lower,
        upper,
        unit_direction,
        top_lower,
    )
    runner_threshold = rectangle_projected_quantile(
        a,
        sigma,
        lower,
        upper,
        unit_direction,
        1.0 - runner_upper,
    )
    lower_top = rectangle_projected_cdf(
        b,
        sigma,
        lower,
        upper,
        unit_direction,
        top_threshold,
    )
    upper_runner = 1.0 - rectangle_projected_cdf(
        b,
        sigma,
        lower,
        upper,
        unit_direction,
        runner_threshold,
    )
    return RectangleNPCertificate(
        lower_top_under_shift=lower_top,
        upper_runner_under_shift=upper_runner,
        margin_under_shift=lower_top - upper_runner,
        top_threshold=top_threshold,
        runner_threshold=runner_threshold,
        direction=unit_direction,
    )


def rectangle_np_radius_along_direction(
    center: ArrayLike,
    direction: ArrayLike,
    sigma: float,
    lower: ArrayLike,
    upper: ArrayLike,
    top_lower: float,
    runner_upper: float,
    *,
    max_radius: float | None = None,
    iterations: int = 40,
) -> RectangleNPRadius:
    """Binary-search the NP-certified radius along one direction."""

    a = _validate_vector(center, "center")
    direction_arr = _validate_vector(direction, "direction")
    direction_norm = float(np.linalg.norm(direction_arr))
    if direction_norm == 0.0:
        raise ValueError("direction must be nonzero")
    unit_direction = direction_arr / direction_norm
    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    if max_radius is None:
        max_radius = distance_to_rectangle_boundary(a, lower_arr, upper_arr, unit_direction)
    if max_radius < 0.0:
        raise ValueError("max_radius must be non-negative")

    lo = 0.0
    hi = float(max_radius)
    best = rectangle_np_pair_certificate(a, a, sigma, lower_arr, upper_arr, top_lower, runner_upper)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        candidate = rectangle_np_pair_certificate(
            a,
            a + mid * unit_direction,
            sigma,
            lower_arr,
            upper_arr,
            top_lower,
            runner_upper,
        )
        if candidate.margin_under_shift > 0.0:
            lo = mid
            best = candidate
        else:
            hi = mid
    return RectangleNPRadius(radius=lo, iterations=iterations, certificate_at_radius=best)


def rectangle_truncated_gaussian_samples(
    center: ArrayLike,
    sigma: float,
    lower: ArrayLike,
    upper: ArrayLike,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Exact independent samples from a product Gaussian truncated to a rectangle."""

    center_arr = _validate_vector(center, "center")
    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    if center_arr.shape != lower_arr.shape or center_arr.shape != upper_arr.shape:
        raise ValueError("center and rectangle bounds must have the same shape")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if np.any(lower_arr >= upper_arr):
        raise ValueError("rectangle lower bounds must be below upper bounds")

    alpha = (lower_arr - center_arr) / sigma
    beta = (upper_arr - center_arr) / sigma
    cdf_lower = norm.cdf(alpha)
    cdf_upper = norm.cdf(beta)
    if np.any(cdf_upper <= cdf_lower):
        raise FloatingPointError("truncated-normal interval underflowed")

    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(cdf_lower, cdf_upper, size=(count, center_arr.size))
    uniforms = np.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return center_arr + sigma * norm.ppf(uniforms)


def _cp_lower(successes: int, trials: int, alpha: float) -> float:
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def _cp_upper(successes: int, trials: int, alpha: float) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def lower_tail_order_statistic_threshold(
    values: np.ndarray,
    probability: float,
    alpha: float,
) -> float:
    """Conservative lower-tail cutoff for an NP lower envelope.

    If ``X_(k)`` is the kth order statistic from a continuous CDF ``F``, then
    ``F(X_(k)) ~ Beta(k, n + 1 - k)``.  We choose the largest fixed rank whose
    one-sided upper confidence bound is at most ``probability``.  Therefore the
    selected tail has base mass at most the target with probability at least
    ``1-alpha`` and its shifted mass is a conservative lower envelope.
    """

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    if n <= 0:
        raise ValueError("values must be non-empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if probability == 0.0:
        return float("-inf")
    selected_rank = _conservative_lower_tail_rank(n, probability, alpha)
    return float("-inf") if selected_rank == 0 else float(ordered[selected_rank - 1])


def upper_tail_order_statistic_threshold(
    values: np.ndarray,
    probability: float,
    alpha: float,
) -> float:
    """Conservative upper-tail cutoff for an NP upper envelope.

    For the kth order statistic, the continuous upper-tail mass
    ``1-F(X_(k))`` has distribution ``Beta(n + 1 - k, k)``.  We choose the
    largest rank whose one-sided lower confidence bound is at least the target,
    producing a superset of the target-mass extremal tail with confidence
    ``1-alpha``.
    """

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    if n <= 0:
        raise ValueError("values must be non-empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if probability == 0.0:
        return float("inf")
    if probability == 1.0:
        return float("-inf")
    # Requiring upper-tail mass at least p is equivalent to requiring
    # F(X_(k)) at most 1-p, so this is the same lower-quantile rank problem.
    selected_rank = _conservative_lower_tail_rank(n, 1.0 - probability, alpha)
    return float("-inf") if selected_rank == 0 else float(ordered[selected_rank - 1])


@lru_cache(maxsize=512)
def _conservative_lower_tail_rank(n: int, probability: float, alpha: float) -> int:
    """Largest rank with a one-sided Beta upper mass bound below the target."""

    lo = 0
    hi = n
    while lo < hi:
        rank = (lo + hi + 1) // 2
        mass_upper = float(beta.ppf(1.0 - alpha, rank, n + 1 - rank))
        if mass_upper <= probability:
            lo = rank
        else:
            hi = rank - 1
    return lo


def _rectangle_np_mc_from_base_projection(
    shifted_center: np.ndarray,
    sigma: float,
    lower: np.ndarray,
    upper: np.ndarray,
    unit_direction: np.ndarray,
    *,
    base_projection: np.ndarray,
    top_lower: float,
    runner_upper: float,
    sample_count: int,
    seed: int,
    base_alpha: float,
    shift_alpha: float,
    familywise_alpha: float,
) -> RectangleMCNPCertificate:
    top_threshold = lower_tail_order_statistic_threshold(
        base_projection, top_lower, base_alpha
    )
    runner_threshold = upper_tail_order_statistic_threshold(
        base_projection, runner_upper, base_alpha
    )
    shifted_samples = rectangle_truncated_gaussian_samples(
        shifted_center,
        sigma,
        lower,
        upper,
        sample_count,
        seed=seed,
    )
    shifted_projection = shifted_samples @ unit_direction
    top_successes = int(np.sum(shifted_projection <= top_threshold))
    runner_successes = int(np.sum(shifted_projection >= runner_threshold))
    lower_top = _cp_lower(top_successes, sample_count, shift_alpha)
    upper_runner = _cp_upper(runner_successes, sample_count, shift_alpha)
    return RectangleMCNPCertificate(
        lower_top_under_shift=lower_top,
        upper_runner_under_shift=upper_runner,
        margin_under_shift=lower_top - upper_runner,
        top_threshold=top_threshold,
        runner_threshold=runner_threshold,
        direction=unit_direction,
        sample_count=sample_count,
        alpha=familywise_alpha,
        confidence=max(0.0, 1.0 - familywise_alpha),
        top_successes=top_successes,
        runner_successes=runner_successes,
    )


def rectangle_np_pair_certificate_mc(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    lower: ArrayLike,
    upper: ArrayLike,
    top_lower: float,
    runner_upper: float,
    *,
    sample_count: int = 10_000,
    seed: int = 0,
    alpha: float = 0.001,
) -> RectangleMCNPCertificate:
    """High-confidence NP certificate for an arbitrary-dimensional rectangle.

    Unlike `rectangle_np_pair_certificate`, this works in any finite dimension
    but uses independent Monte Carlo samples.  The returned confidence is a
    simple union bound over the two base-threshold and two shifted-tail
    Clopper-Pearson events.
    """

    a = _validate_vector(center_a, "center_a")
    b = _validate_vector(center_b, "center_b")
    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    if a.shape != b.shape or a.shape != lower_arr.shape or a.shape != upper_arr.shape:
        raise ValueError("centers and rectangle bounds must have the same shape")
    if not 0.0 <= runner_upper <= top_lower <= 1.0:
        raise ValueError("require 0 <= runner_upper <= top_lower <= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    direction = b - a
    norm_direction = float(np.linalg.norm(direction))
    if norm_direction == 0.0:
        return RectangleMCNPCertificate(
            lower_top_under_shift=top_lower,
            upper_runner_under_shift=runner_upper,
            margin_under_shift=top_lower - runner_upper,
            top_threshold=float("nan"),
            runner_threshold=float("nan"),
            direction=direction,
            sample_count=0,
            alpha=alpha,
            confidence=1.0,
            top_successes=0,
            runner_successes=0,
        )
    unit_direction = direction / norm_direction
    component_alpha = alpha / 4.0
    base_samples = rectangle_truncated_gaussian_samples(
        a,
        sigma,
        lower_arr,
        upper_arr,
        sample_count,
        seed=seed,
    )
    return _rectangle_np_mc_from_base_projection(
        b,
        sigma,
        lower_arr,
        upper_arr,
        unit_direction,
        base_projection=base_samples @ unit_direction,
        top_lower=top_lower,
        runner_upper=runner_upper,
        sample_count=sample_count,
        seed=seed + 1,
        base_alpha=component_alpha,
        shift_alpha=component_alpha,
        familywise_alpha=alpha,
    )


def rectangle_np_radius_along_direction_mc(
    center: ArrayLike,
    direction: ArrayLike,
    sigma: float,
    lower: ArrayLike,
    upper: ArrayLike,
    top_lower: float,
    runner_upper: float,
    *,
    max_radius: float | None = None,
    iterations: int = 16,
    sample_count: int = 10_000,
    seed: int = 0,
    alpha: float = 0.001,
) -> RectangleMCNPRadius:
    """Binary-search a high-confidence MC-NP radius in any rectangle dimension."""

    a = _validate_vector(center, "center")
    direction_arr = _validate_vector(direction, "direction")
    direction_norm = float(np.linalg.norm(direction_arr))
    if direction_norm == 0.0:
        raise ValueError("direction must be nonzero")
    unit_direction = direction_arr / direction_norm
    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    if a.shape != lower_arr.shape or a.shape != upper_arr.shape:
        raise ValueError("center and rectangle bounds must have the same shape")
    if max_radius is None:
        max_radius = distance_to_rectangle_boundary(a, lower_arr, upper_arr, unit_direction)
    if max_radius < 0.0:
        raise ValueError("max_radius must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    evaluations = iterations + 1
    base_alpha = alpha / 4.0
    shift_alpha = alpha / (4.0 * evaluations)
    base_samples = rectangle_truncated_gaussian_samples(
        a,
        sigma,
        lower_arr,
        upper_arr,
        sample_count,
        seed=seed,
    )
    base_projection = base_samples @ unit_direction
    lo = 0.0
    hi = float(max_radius)
    best = _rectangle_np_mc_from_base_projection(
        a,
        sigma,
        lower_arr,
        upper_arr,
        unit_direction,
        base_projection=base_projection,
        top_lower=top_lower,
        runner_upper=runner_upper,
        sample_count=sample_count,
        seed=seed + 10_000,
        base_alpha=base_alpha,
        shift_alpha=shift_alpha,
        familywise_alpha=alpha,
    )
    for iteration in range(iterations):
        mid = 0.5 * (lo + hi)
        candidate = _rectangle_np_mc_from_base_projection(
            a + mid * unit_direction,
            sigma,
            lower_arr,
            upper_arr,
            unit_direction,
            base_projection=base_projection,
            top_lower=top_lower,
            runner_upper=runner_upper,
            sample_count=sample_count,
            seed=seed + 20_000 + 1009 * iteration,
            base_alpha=base_alpha,
            shift_alpha=shift_alpha,
            familywise_alpha=alpha,
        )
        if candidate.margin_under_shift > 0.0:
            lo = mid
            best = candidate
        else:
            hi = mid
    return RectangleMCNPRadius(
        radius=lo,
        iterations=iterations,
        max_radius=float(max_radius),
        certificate_at_radius=best,
    )


def convex_polygon_projected_cdf(
    center: ArrayLike,
    sigma: float,
    vertices: ArrayLike,
    direction: ArrayLike,
    threshold: float,
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> float:
    """CDF of ``<direction, Z>`` for a Gaussian truncated to a 2D convex polygon.

    The implementation rotates coordinates so the first coordinate is the
    queried linear functional.  Convexity makes every cross-section an interval,
    so the probability reduces to one-dimensional quadrature over that first
    coordinate.
    """

    center_arr = _validate_vector(center, "center")
    polygon = _ordered_convex_polygon(vertices)
    direction_arr = _validate_vector(direction, "direction")
    if center_arr.shape != (2,) or polygon.shape[1] != 2:
        raise ValueError("convex_polygon_projected_cdf currently supports 2D polygons")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    direction_norm = float(np.linalg.norm(direction_arr))
    if direction_norm == 0.0 or not np.isfinite(direction_norm):
        raise ValueError("direction must be finite and nonzero")

    rotated, mu_t, mu_s = _rotate_polygon_problem(center_arr, polygon, direction_arr)
    t_min, t_max = float(np.min(rotated[:, 0])), float(np.max(rotated[:, 0]))
    if threshold <= t_min:
        return 0.0
    if threshold >= t_max:
        return 1.0

    normalizer = _convex_polygon_projected_integral(
        rotated,
        mu_t,
        mu_s,
        sigma,
        t_min,
        t_max,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    if normalizer <= 0.0:
        raise FloatingPointError("convex polygon Gaussian normalizer underflowed")

    numerator = _convex_polygon_projected_integral(
        rotated,
        mu_t,
        mu_s,
        sigma,
        t_min,
        min(threshold, t_max),
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return float(np.clip(numerator / normalizer, 0.0, 1.0))


def convex_polygon_projected_quantile(
    center: ArrayLike,
    sigma: float,
    vertices: ArrayLike,
    direction: ArrayLike,
    probability: float,
) -> float:
    """Quantile of ``<direction, Z>`` for a 2D convex-polygon truncation."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    direction_arr = _validate_vector(direction, "direction")
    direction_norm = float(np.linalg.norm(direction_arr))
    if direction_norm == 0.0:
        raise ValueError("direction must be nonzero")
    polygon = _ordered_convex_polygon(vertices)
    unit_direction = direction_arr / direction_norm
    projections = polygon @ unit_direction
    dot_min, dot_max = float(np.min(projections)), float(np.max(projections))
    if probability <= 0.0:
        return dot_min
    if probability >= 1.0:
        return dot_max

    def shifted_cdf(threshold: float) -> float:
        return convex_polygon_projected_cdf(
            center,
            sigma,
            polygon,
            unit_direction,
            threshold,
        ) - probability

    return float(brentq(shifted_cdf, dot_min, dot_max, xtol=1e-11, rtol=1e-11, maxiter=100))


def convex_polygon_np_pair_certificate(
    center_a: ArrayLike,
    center_b: ArrayLike,
    sigma: float,
    vertices: ArrayLike,
    top_lower: float,
    runner_upper: float,
) -> RectangleNPCertificate:
    """Pairwise NP lower/upper bounds from ``q_a`` to ``q_b`` on a convex polygon."""

    a = _validate_vector(center_a, "center_a")
    b = _validate_vector(center_b, "center_b")
    if a.shape != b.shape:
        raise ValueError("centers must have the same shape")
    if not 0.0 <= runner_upper <= top_lower <= 1.0:
        raise ValueError("require 0 <= runner_upper <= top_lower <= 1")
    polygon = _ordered_convex_polygon(vertices)
    direction = b - a
    norm_direction = float(np.linalg.norm(direction))
    if norm_direction == 0.0:
        return RectangleNPCertificate(
            lower_top_under_shift=top_lower,
            upper_runner_under_shift=runner_upper,
            margin_under_shift=top_lower - runner_upper,
            top_threshold=float("nan"),
            runner_threshold=float("nan"),
            direction=direction,
        )
    unit_direction = direction / norm_direction
    top_threshold = convex_polygon_projected_quantile(
        a,
        sigma,
        polygon,
        unit_direction,
        top_lower,
    )
    runner_threshold = convex_polygon_projected_quantile(
        a,
        sigma,
        polygon,
        unit_direction,
        1.0 - runner_upper,
    )
    lower_top = convex_polygon_projected_cdf(
        b,
        sigma,
        polygon,
        unit_direction,
        top_threshold,
    )
    upper_runner = 1.0 - convex_polygon_projected_cdf(
        b,
        sigma,
        polygon,
        unit_direction,
        runner_threshold,
    )
    return RectangleNPCertificate(
        lower_top_under_shift=lower_top,
        upper_runner_under_shift=upper_runner,
        margin_under_shift=lower_top - upper_runner,
        top_threshold=top_threshold,
        runner_threshold=runner_threshold,
        direction=unit_direction,
    )


def convex_polygon_np_radius_along_direction(
    center: ArrayLike,
    direction: ArrayLike,
    sigma: float,
    vertices: ArrayLike,
    top_lower: float,
    runner_upper: float,
    *,
    max_radius: float | None = None,
    iterations: int = 36,
) -> ConvexPolygonNPRadius:
    """Binary-search the NP-certified radius along one direction in a convex polygon."""

    a = _validate_vector(center, "center")
    polygon = _ordered_convex_polygon(vertices)
    direction_arr = _validate_vector(direction, "direction")
    direction_norm = float(np.linalg.norm(direction_arr))
    if direction_norm == 0.0:
        raise ValueError("direction must be nonzero")
    unit_direction = direction_arr / direction_norm
    if max_radius is None:
        max_radius = distance_to_convex_polygon_boundary(a, polygon, unit_direction)
    if max_radius < 0.0:
        raise ValueError("max_radius must be non-negative")

    lo = 0.0
    hi = float(max_radius)
    best = convex_polygon_np_pair_certificate(a, a, sigma, polygon, top_lower, runner_upper)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        candidate = convex_polygon_np_pair_certificate(
            a,
            a + mid * unit_direction,
            sigma,
            polygon,
            top_lower,
            runner_upper,
        )
        if candidate.margin_under_shift > 0.0:
            lo = mid
            best = candidate
        else:
            hi = mid
    return ConvexPolygonNPRadius(radius=lo, iterations=iterations, certificate_at_radius=best)


def distance_to_rectangle_boundary(
    point: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    direction: ArrayLike,
) -> float:
    """Maximum nonnegative step that keeps ``point + t direction`` in a rectangle."""

    point_arr = _validate_vector(point, "point")
    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    direction_arr = _validate_vector(direction, "direction")
    if point_arr.shape != lower_arr.shape or point_arr.shape != upper_arr.shape:
        raise ValueError("point and rectangle bounds must have the same shape")
    bounds: list[float] = []
    for x, lo, hi, d in zip(point_arr, lower_arr, upper_arr, direction_arr, strict=True):
        if d > 0.0:
            bounds.append((hi - x) / d)
        elif d < 0.0:
            bounds.append((lo - x) / d)
    if not bounds:
        raise ValueError("direction must be nonzero")
    return float(max(0.0, min(bounds)))


def distance_to_convex_polygon_boundary(
    point: ArrayLike,
    vertices: ArrayLike,
    direction: ArrayLike,
    *,
    tol: float = 1e-12,
) -> float:
    """Maximum nonnegative step that keeps ``point + t direction`` in a convex polygon."""

    point_arr = _validate_vector(point, "point")
    polygon = _ordered_convex_polygon(vertices)
    direction_arr = _validate_vector(direction, "direction")
    if point_arr.shape != (2,) or direction_arr.shape != (2,):
        raise ValueError("point and direction must be two-dimensional")
    direction_norm = float(np.linalg.norm(direction_arr))
    if direction_norm == 0.0:
        raise ValueError("direction must be nonzero")
    unit_direction = direction_arr / direction_norm

    candidates: list[float] = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        matrix = np.column_stack((unit_direction, -edge))
        det = float(np.linalg.det(matrix))
        if abs(det) <= tol:
            continue
        rhs = start - point_arr
        t, alpha = np.linalg.solve(matrix, rhs)
        if t >= -tol and -tol <= alpha <= 1.0 + tol:
            candidates.append(max(0.0, float(t)))
    if not candidates:
        raise ValueError("ray did not intersect polygon boundary")
    positive = [value for value in candidates if value > tol]
    return float(min(positive) if positive else 0.0)


def rectangle_dot_range(
    lower: ArrayLike,
    upper: ArrayLike,
    direction: ArrayLike,
) -> tuple[float, float]:
    """Minimum and maximum of a linear functional over a rectangle."""

    lower_arr = _validate_vector(lower, "lower")
    upper_arr = _validate_vector(upper, "upper")
    direction_arr = _validate_vector(direction, "direction")
    min_corner = np.where(direction_arr >= 0.0, lower_arr, upper_arr)
    max_corner = np.where(direction_arr >= 0.0, upper_arr, lower_arr)
    return float(np.dot(direction_arr, min_corner)), float(np.dot(direction_arr, max_corner))


def convex_polygon_dot_range(vertices: ArrayLike, direction: ArrayLike) -> tuple[float, float]:
    """Minimum and maximum of a linear functional over a convex polygon."""

    polygon = _ordered_convex_polygon(vertices)
    direction_arr = _validate_vector(direction, "direction")
    values = polygon @ direction_arr
    return float(np.min(values)), float(np.max(values))


def _ordered_convex_polygon(vertices: ArrayLike) -> np.ndarray:
    polygon = np.asarray(vertices, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("vertices must be an array of at least three 2D points")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("vertices must be finite")
    hull = ConvexHull(polygon)
    ordered = polygon[hull.vertices]
    if ordered.shape[0] < 3 or hull.volume <= 0.0:
        raise ValueError("vertices must span a nondegenerate convex polygon")
    return ordered


def _rotate_polygon_problem(
    center: np.ndarray,
    polygon: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    unit = direction / np.linalg.norm(direction)
    perp = np.array([-unit[1], unit[0]], dtype=np.float64)
    basis = np.vstack([unit, perp])
    rotated = polygon @ basis.T
    center_rot = basis @ center
    return rotated, float(center_rot[0]), float(center_rot[1])


def _convex_polygon_projected_integral(
    rotated_vertices: np.ndarray,
    mu_t: float,
    mu_s: float,
    sigma: float,
    lower_t: float,
    upper_t: float,
    *,
    epsabs: float,
    epsrel: float,
    order: int = 96,
) -> float:
    if upper_t <= lower_t:
        return 0.0

    def cross_section_mass(t_value: float) -> float:
        interval = _polygon_cross_section_interval(rotated_vertices, t_value)
        if interval is None:
            return 0.0
        low_s, high_s = interval
        if high_s <= low_s:
            return 0.0
        return float(norm.cdf((high_s - mu_s) / sigma) - norm.cdf((low_s - mu_s) / sigma))

    def integrand(t_value: float) -> float:
        density_t = norm.pdf((t_value - mu_t) / sigma) / sigma
        return float(density_t * cross_section_mass(t_value))

    breakpoints = sorted(
        {
            lower_t,
            upper_t,
            *(
                float(value)
                for value in rotated_vertices[:, 0]
                if lower_t < float(value) < upper_t
            ),
        }
    )
    nodes, weights = _gauss_legendre_rule(order)
    total = 0.0
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        values = np.array([integrand(float(midpoint + half_width * node)) for node in nodes])
        total += half_width * float(np.dot(weights, values))
    return float(max(0.0, total))


def _polygon_cross_section_interval(
    rotated_vertices: np.ndarray,
    t_value: float,
    *,
    tol: float = 1e-11,
) -> tuple[float, float] | None:
    intersections: list[float] = []
    for start, end in zip(rotated_vertices, np.roll(rotated_vertices, -1, axis=0), strict=True):
        t0, s0 = float(start[0]), float(start[1])
        t1, s1 = float(end[0]), float(end[1])
        low_t, high_t = min(t0, t1), max(t0, t1)
        if t_value < low_t - tol or t_value > high_t + tol:
            continue
        if abs(t1 - t0) <= tol:
            if abs(t_value - t0) <= tol:
                intersections.extend([s0, s1])
            continue
        alpha = (t_value - t0) / (t1 - t0)
        if -tol <= alpha <= 1.0 + tol:
            intersections.append(s0 + alpha * (s1 - s0))
    if len(intersections) < 2:
        return None
    return float(min(intersections)), float(max(intersections))


@lru_cache(maxsize=16)
def _gauss_legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("quadrature order must be at least 8")
    nodes, weights = leggauss(order)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _validate_vector(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return vector
