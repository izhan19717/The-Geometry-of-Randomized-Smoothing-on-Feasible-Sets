"""Exact one-step reach--avoid pilot for feasible smoothing kernels.

The nominal two-dimensional action is interpreted as an affine one-step
displacement.  The action box is ``[-1, 1]^2`` and one axis-aligned rectangle
contains the actions whose next state collides with an obstacle.  Its exact
feasible complement is decomposed into four disjoint convex corridors.

Four kernels are compared.

``moving_conditioned_gaussian``
    An isotropic Gaussian conditioned on the feasible complement.  Its
    occupancy normalizer moves with the center.  KL, label probabilities,
    moments, and rejection cost are evaluated analytically.  An ambient
    Gaussian radius is recorded only as an *invalid, deliberately
    misapplied* audit quantity.

``fixed_collision_repair``
    A center-independent deterministic map clips the Gaussian action to the
    action box and pushes colliding actions past the nearest horizontal face.
    The ordinary Gaussian comparison is valid by data processing.

``anchor_frozen_corridor_projection``
    Corridor probabilities are computed once at the anchor and then kept
    fixed over the certified neighborhood.  Conditional on a corridor, a
    center-independent projection sends the Gaussian draw into a
    clearance-shrunk safe rectangle.  This is again one fixed Markov kernel,
    so the ordinary Gaussian comparison is valid by data processing.  This is
    a projection baseline, not a mixture of conditioned Gaussians.

``anchor_frozen_conditioned_corridor_mixture``
    The anchor occupancy probabilities are kept fixed, but each within-mode
    law is the Gaussian conditioned on the corresponding original convex
    corridor.  This is the mode-stable construction analyzed in the paper: it
    exactly matches whole-set conditioning at the anchor and obeys the
    Gaussian comparison by the convex fixed-weight mixture theorem.  Its
    full-law KL and naive rejection-sampling cost are evaluated exactly.

Neither anchor-frozen result applies to re-anchoring its weights at every
query.

All numerical quantities in this script are deterministic closed-form normal
interval calculations; no Monte Carlo estimate is presented as a result.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


DEFAULT_OUTPUT_STEM = Path("outputs/iclr2027_reach_avoid_kernel_pilot")
DEVELOPMENT_SEED = 1729
EVALUATION_SEED = 20270925
DEFAULT_DEVELOPMENT_LAYOUTS = 24
DEFAULT_EVALUATION_LAYOUTS = 128
METHODS = (
    "moving_conditioned_gaussian",
    "fixed_collision_repair",
    "anchor_frozen_corridor_projection",
    "anchor_frozen_conditioned_corridor_mixture",
)


@dataclass(frozen=True)
class Rectangle:
    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float

    def __post_init__(self) -> None:
        values = (self.x_lower, self.x_upper, self.y_lower, self.y_upper)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rectangle endpoints must be finite")
        if not self.x_lower < self.x_upper or not self.y_lower < self.y_upper:
            raise ValueError("rectangle intervals must have positive width")


@dataclass(frozen=True)
class Layout:
    split: str
    layout_index: int
    box: Rectangle
    obstacle: Rectangle
    sigma: float
    clearance: float
    anchor_x: float
    anchor_y: float

    @property
    def anchor(self) -> np.ndarray:
        return np.asarray([self.anchor_x, self.anchor_y], dtype=np.float64)

    @property
    def paired_center(self) -> np.ndarray:
        """A reflection across the upper/lower avoidance boundary."""

        return np.asarray([self.anchor_x, -self.anchor_y], dtype=np.float64)


@dataclass(frozen=True)
class IntervalStats:
    mass: float
    first: float
    second: float


@dataclass(frozen=True)
class RegionStats:
    mass: float
    first_x: float
    first_y: float
    second_x: float
    second_y: float


def standard_normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def interval_stats(
    lower: float, upper: float, mean: float, sigma: float
) -> IntervalStats:
    """Unnormalized first two moments on one finite Gaussian interval."""

    if not lower < upper:
        raise ValueError("interval endpoints must be strictly ordered")
    if not math.isfinite(mean) or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("mean and positive sigma must be finite")
    alpha = (lower - mean) / sigma
    beta = (upper - mean) / sigma
    phi_alpha = standard_normal_pdf(alpha)
    phi_beta = standard_normal_pdf(beta)
    mass = float(ndtr(beta) - ndtr(alpha))
    first = mean * mass + sigma * (phi_alpha - phi_beta)
    second = (
        (mean * mean + sigma * sigma) * mass
        + 2.0 * mean * sigma * (phi_alpha - phi_beta)
        + sigma * sigma * (alpha * phi_alpha - beta * phi_beta)
    )
    return IntervalStats(float(mass), float(first), float(second))


def rectangle_stats(
    rectangle: Rectangle, center: np.ndarray, sigma: float
) -> RegionStats:
    vector = np.asarray(center, dtype=np.float64)
    if vector.shape != (2,) or np.any(~np.isfinite(vector)):
        raise ValueError("center must be a finite two-vector")
    x_stats = interval_stats(
        rectangle.x_lower, rectangle.x_upper, float(vector[0]), sigma
    )
    y_stats = interval_stats(
        rectangle.y_lower, rectangle.y_upper, float(vector[1]), sigma
    )
    return RegionStats(
        mass=x_stats.mass * y_stats.mass,
        first_x=x_stats.first * y_stats.mass,
        first_y=x_stats.mass * y_stats.first,
        second_x=x_stats.second * y_stats.mass,
        second_y=x_stats.mass * y_stats.second,
    )


def subtract_region(outer: RegionStats, inner: RegionStats) -> RegionStats:
    return RegionStats(
        mass=outer.mass - inner.mass,
        first_x=outer.first_x - inner.first_x,
        first_y=outer.first_y - inner.first_y,
        second_x=outer.second_x - inner.second_x,
        second_y=outer.second_y - inner.second_y,
    )


def feasible_region_stats(layout: Layout, center: np.ndarray) -> RegionStats:
    stats = subtract_region(
        rectangle_stats(layout.box, center, layout.sigma),
        rectangle_stats(layout.obstacle, center, layout.sigma),
    )
    if not 0.0 < stats.mass < 1.0:
        raise FloatingPointError("feasible Gaussian occupancy must lie in (0, 1)")
    return stats


def corridor_rectangles(layout: Layout) -> tuple[Rectangle, ...]:
    """Measure representatives of the exact half-open corridor partition.

    Boundary inclusion is specified by :func:`corridor_memberships`.  The
    closed rectangle representatives here have identical Gaussian masses,
    because their boundary differences have measure zero.
    """

    box = layout.box
    obstacle = layout.obstacle
    return (
        Rectangle(box.x_lower, obstacle.x_lower, box.y_lower, box.y_upper),
        Rectangle(obstacle.x_upper, box.x_upper, box.y_lower, box.y_upper),
        Rectangle(
            obstacle.x_lower,
            obstacle.x_upper,
            box.y_lower,
            obstacle.y_lower,
        ),
        Rectangle(
            obstacle.x_lower,
            obstacle.x_upper,
            obstacle.y_upper,
            box.y_upper,
        ),
    )


def safe_projection_rectangles(layout: Layout) -> tuple[Rectangle, ...]:
    """Clearance-shrunk safe targets for the fixed corridor mixture."""

    box = layout.box
    obstacle = layout.obstacle
    gap = layout.clearance
    return (
        Rectangle(box.x_lower, obstacle.x_lower - gap, box.y_lower, box.y_upper),
        Rectangle(obstacle.x_upper + gap, box.x_upper, box.y_lower, box.y_upper),
        Rectangle(
            obstacle.x_lower,
            obstacle.x_upper,
            box.y_lower,
            obstacle.y_lower - gap,
        ),
        Rectangle(
            obstacle.x_lower,
            obstacle.x_upper,
            obstacle.y_upper + gap,
            box.y_upper,
        ),
    )


def point_in_rectangle_closed(point: np.ndarray, rectangle: Rectangle) -> bool:
    x_value, y_value = (float(value) for value in point)
    return bool(
        rectangle.x_lower <= x_value <= rectangle.x_upper
        and rectangle.y_lower <= y_value <= rectangle.y_upper
    )


def point_is_feasible(point: np.ndarray, layout: Layout) -> bool:
    return point_in_rectangle_closed(point, layout.box) and not point_in_rectangle_closed(
        point, layout.obstacle
    )


def corridor_memberships(point: np.ndarray, layout: Layout) -> tuple[bool, ...]:
    """Exact disjoint partition of ``box \\ closed_obstacle``.

    The left corridor is closed at the box face and open at the obstacle;
    the right corridor is open at the obstacle and closed at the box face.
    Bottom and top own the exterior parts directly below/above both vertical
    obstacle faces.  This convention excludes every point on the closed
    obstacle and assigns every other box point to exactly one corridor.
    """

    x_value, y_value = (float(value) for value in point)
    box = layout.box
    obstacle = layout.obstacle
    inside_box = (
        box.x_lower <= x_value <= box.x_upper
        and box.y_lower <= y_value <= box.y_upper
    )
    if not inside_box:
        return (False, False, False, False)
    left = box.x_lower <= x_value < obstacle.x_lower
    right = obstacle.x_upper < x_value <= box.x_upper
    bottom = (
        obstacle.x_lower <= x_value <= obstacle.x_upper
        and box.y_lower <= y_value < obstacle.y_lower
    )
    top = (
        obstacle.x_lower <= x_value <= obstacle.x_upper
        and obstacle.y_upper < y_value <= box.y_upper
    )
    return (left, right, bottom, top)


def corridor_weights(layout: Layout, center: np.ndarray) -> np.ndarray:
    masses = np.asarray(
        [
            rectangle_stats(rectangle, center, layout.sigma).mass
            for rectangle in corridor_rectangles(layout)
        ],
        dtype=np.float64,
    )
    occupancy = feasible_region_stats(layout, center).mass
    if not math.isclose(
        float(np.sum(masses)), occupancy, rel_tol=2e-13, abs_tol=2e-15
    ):
        raise FloatingPointError("corridor masses do not sum to feasible occupancy")
    return masses / occupancy


def upper_probability_moving_conditioned(layout: Layout, center: np.ndarray) -> float:
    """Exact probability of positive lateral avoidance under conditioning."""

    box = layout.box
    obstacle = layout.obstacle
    if not box.y_lower < 0.0 < box.y_upper:
        raise ValueError("the binary avoidance threshold must lie inside the box")
    if not obstacle.y_lower < 0.0 < obstacle.y_upper:
        raise ValueError("the obstacle must straddle the avoidance threshold")
    box_upper = Rectangle(box.x_lower, box.x_upper, 0.0, box.y_upper)
    obstacle_upper = Rectangle(
        obstacle.x_lower, obstacle.x_upper, 0.0, obstacle.y_upper
    )
    upper_mass = (
        rectangle_stats(box_upper, center, layout.sigma).mass
        - rectangle_stats(obstacle_upper, center, layout.sigma).mass
    )
    return float(upper_mass / feasible_region_stats(layout, center).mass)


def upper_probability_fixed_repair(layout: Layout, center: np.ndarray) -> float:
    return float(ndtr(float(center[1]) / layout.sigma))


def upper_probability_frozen_projection(
    layout: Layout, center: np.ndarray, frozen_weights: np.ndarray
) -> float:
    weights = np.asarray(frozen_weights, dtype=np.float64)
    if weights.shape != (4,) or np.any(weights < 0.0):
        raise ValueError("frozen_weights must contain four nonnegative entries")
    if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("frozen_weights must sum to one")
    raw_upper = float(ndtr(float(center[1]) / layout.sigma))
    return float(weights[3] + (weights[0] + weights[1]) * raw_upper)


def upper_probability_frozen_conditioned_mixture(
    layout: Layout, center: np.ndarray, frozen_weights: np.ndarray
) -> float:
    """Upper-label mass for fixed weights and corridor-conditioned modes."""

    weights = np.asarray(frozen_weights, dtype=np.float64)
    if weights.shape != (4,) or np.any(weights < 0.0):
        raise ValueError("frozen_weights must contain four nonnegative entries")
    if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("frozen_weights must sum to one")
    box = layout.box
    y_box = interval_stats(
        box.y_lower, box.y_upper, float(center[1]), layout.sigma
    ).mass
    y_upper = interval_stats(0.0, box.y_upper, float(center[1]), layout.sigma).mass
    conditional_upper = y_upper / y_box
    return float(
        weights[3] + (weights[0] + weights[1]) * conditional_upper
    )


def moving_conditioned_kl(
    layout: Layout, anchor: np.ndarray, shifted: np.ndarray
) -> float:
    """Exact KL between two equal-scale Gaussians conditioned on one set."""

    anchor_stats = feasible_region_stats(layout, anchor)
    shifted_stats = feasible_region_stats(layout, shifted)
    conditional_mean = np.asarray(
        [anchor_stats.first_x, anchor_stats.first_y], dtype=np.float64
    ) / anchor_stats.mass
    mean_term = (
        2.0 * float(np.dot(conditional_mean, anchor - shifted))
        + float(np.dot(shifted, shifted))
        - float(np.dot(anchor, anchor))
    ) / (2.0 * layout.sigma**2)
    value = mean_term + math.log(shifted_stats.mass / anchor_stats.mass)
    if value < 0.0 and abs(value) < 5e-13:
        return 0.0
    if value < 0.0:
        raise FloatingPointError(f"computed a negative conditioned KL: {value}")
    return float(value)


def rectangle_conditioned_kl(
    rectangle: Rectangle,
    sigma: float,
    first_center: np.ndarray,
    second_center: np.ndarray,
) -> float:
    """Exact KL for equal-scale Gaussians conditioned on one rectangle."""

    first = np.asarray(first_center, dtype=np.float64)
    second = np.asarray(second_center, dtype=np.float64)
    first_stats = rectangle_stats(rectangle, first, sigma)
    second_stats = rectangle_stats(rectangle, second, sigma)
    if first_stats.mass <= 0.0 or second_stats.mass <= 0.0:
        raise FloatingPointError("conditioned rectangle must have positive occupancy")
    conditional_mean = np.asarray(
        [first_stats.first_x, first_stats.first_y], dtype=np.float64
    ) / first_stats.mass
    mean_term = (
        2.0 * float(np.dot(conditional_mean, first - second))
        + float(np.dot(second, second))
        - float(np.dot(first, first))
    ) / (2.0 * sigma**2)
    value = mean_term + math.log(second_stats.mass / first_stats.mass)
    if value < 0.0 and abs(value) < 5e-13:
        return 0.0
    if value < 0.0:
        raise FloatingPointError(f"computed a negative rectangle KL: {value}")
    return float(value)


def frozen_conditioned_mixture_kl(
    layout: Layout,
    first_center: np.ndarray,
    second_center: np.ndarray,
    frozen_weights: np.ndarray,
) -> float:
    """Exact full-law KL for disjoint conditioned modes with fixed weights."""

    weights = np.asarray(frozen_weights, dtype=np.float64)
    if weights.shape != (4,) or np.any(weights < 0.0):
        raise ValueError("frozen_weights must contain four nonnegative entries")
    if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("frozen_weights must sum to one")
    return float(
        sum(
            weight
            * rectangle_conditioned_kl(
                rectangle,
                layout.sigma,
                first_center,
                second_center,
            )
            for weight, rectangle in zip(
                weights, corridor_rectangles(layout), strict=True
            )
        )
    )


def gaussian_kl(layout: Layout, first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return float(np.dot(difference, difference) / (2.0 * layout.sigma**2))


def gaussian_total_variation(
    layout: Layout, first: np.ndarray, second: np.ndarray
) -> float:
    distance = float(np.linalg.norm(np.asarray(first) - np.asarray(second)))
    return float(2.0 * ndtr(distance / (2.0 * layout.sigma)) - 1.0)


def bernoulli_kl(probability: float, other: float) -> float:
    if not 0.0 < probability < 1.0 or not 0.0 < other < 1.0:
        raise ValueError("Bernoulli probabilities must lie strictly inside (0, 1)")
    return float(
        probability * math.log(probability / other)
        + (1.0 - probability) * math.log((1.0 - probability) / (1.0 - other))
    )


def conditional_tracking_mse(
    layout: Layout, center: np.ndarray, target: np.ndarray
) -> float:
    stats = feasible_region_stats(layout, center)
    target_array = np.asarray(target, dtype=np.float64)
    first = np.asarray([stats.first_x, stats.first_y]) / stats.mass
    second_sum = (stats.second_x + stats.second_y) / stats.mass
    return float(
        second_sum - 2.0 * np.dot(target_array, first) + np.dot(target_array, target_array)
    )


def clipped_interval_moments(
    lower: float, upper: float, mean: float, sigma: float
) -> tuple[float, float]:
    """First two moments after clipping a Gaussian to ``[lower, upper]``."""

    interior = interval_stats(lower, upper, mean, sigma)
    lower_mass = float(ndtr((lower - mean) / sigma))
    upper_mass = float(ndtr((mean - upper) / sigma))
    first = lower * lower_mass + interior.first + upper * upper_mass
    second = lower * lower * lower_mass + interior.second + upper * upper * upper_mass
    return float(first), float(second)


def clipped_rectangle_tracking_mse(
    rectangle: Rectangle, center: np.ndarray, sigma: float, target: np.ndarray
) -> float:
    total = 0.0
    for lower, upper, mean, target_value in (
        (rectangle.x_lower, rectangle.x_upper, center[0], target[0]),
        (rectangle.y_lower, rectangle.y_upper, center[1], target[1]),
    ):
        first, second = clipped_interval_moments(
            lower, upper, float(mean), sigma
        )
        total += second - 2.0 * float(target_value) * first + float(target_value) ** 2
    return float(total)


def interval_squared_error_mass(
    lower: float,
    upper: float,
    mean: float,
    sigma: float,
    target: float,
) -> float:
    stats = interval_stats(lower, upper, mean, sigma)
    return float(stats.second - 2.0 * target * stats.first + target * target * stats.mass)


def fixed_repair_tracking_mse(
    layout: Layout, center: np.ndarray, target: np.ndarray
) -> float:
    """Exact tracking MSE of box clipping followed by collision repair."""

    box = layout.box
    obstacle = layout.obstacle
    base = clipped_rectangle_tracking_mse(box, center, layout.sigma, target)
    x_collision_mass = interval_stats(
        obstacle.x_lower,
        obstacle.x_upper,
        float(center[0]),
        layout.sigma,
    ).mass
    bottom_stats = interval_stats(
        obstacle.y_lower, 0.0, float(center[1]), layout.sigma
    )
    top_stats = interval_stats(
        0.0, obstacle.y_upper, float(center[1]), layout.sigma
    )
    safe_bottom = obstacle.y_lower - layout.clearance
    safe_top = obstacle.y_upper + layout.clearance
    repaired_error = (
        bottom_stats.mass * (safe_bottom - float(target[1])) ** 2
        + top_stats.mass * (safe_top - float(target[1])) ** 2
    )
    unrepaired_error = interval_squared_error_mass(
        obstacle.y_lower,
        obstacle.y_upper,
        float(center[1]),
        layout.sigma,
        float(target[1]),
    )
    return float(base + x_collision_mass * (repaired_error - unrepaired_error))


def frozen_projection_tracking_mse(
    layout: Layout,
    center: np.ndarray,
    target: np.ndarray,
    frozen_weights: np.ndarray,
) -> float:
    return float(
        sum(
            weight
            * clipped_rectangle_tracking_mse(
                rectangle, center, layout.sigma, target
            )
            for weight, rectangle in zip(
                frozen_weights, safe_projection_rectangles(layout), strict=True
            )
        )
    )


def conditioned_rectangle_tracking_mse(
    rectangle: Rectangle,
    center: np.ndarray,
    sigma: float,
    target: np.ndarray,
) -> float:
    """Exact tracking MSE under a rectangle-conditioned Gaussian."""

    stats = rectangle_stats(rectangle, center, sigma)
    if stats.mass <= 0.0:
        raise FloatingPointError("conditioned rectangle must have positive occupancy")
    target_array = np.asarray(target, dtype=np.float64)
    first = np.asarray([stats.first_x, stats.first_y], dtype=np.float64)
    squared_error_mass = (
        stats.second_x
        + stats.second_y
        - 2.0 * float(np.dot(target_array, first))
        + float(np.dot(target_array, target_array)) * stats.mass
    )
    return float(squared_error_mass / stats.mass)


def frozen_conditioned_mixture_tracking_mse(
    layout: Layout,
    center: np.ndarray,
    target: np.ndarray,
    frozen_weights: np.ndarray,
) -> float:
    """Exact tracking MSE of the anchor-frozen conditioned-mode mixture."""

    return float(
        sum(
            weight
            * conditioned_rectangle_tracking_mse(
                rectangle, center, layout.sigma, target
            )
            for weight, rectangle in zip(
                frozen_weights, corridor_rectangles(layout), strict=True
            )
        )
    )


def frozen_conditioned_mixture_expected_proposals(
    layout: Layout, center: np.ndarray, frozen_weights: np.ndarray
) -> float:
    """Expected Gaussian draws for naive mode-first rejection sampling."""

    return float(
        sum(
            weight / rectangle_stats(rectangle, center, layout.sigma).mass
            for weight, rectangle in zip(
                frozen_weights, corridor_rectangles(layout), strict=True
            )
        )
    )


def fixed_collision_repair(point: np.ndarray, layout: Layout) -> np.ndarray:
    """Apply the center-independent deterministic repair used in the pilot."""

    value = np.asarray(point, dtype=np.float64)
    if value.shape != (2,) or np.any(~np.isfinite(value)):
        raise ValueError("point must be a finite two-vector")
    box = layout.box
    repaired = np.clip(
        value,
        [box.x_lower, box.y_lower],
        [box.x_upper, box.y_upper],
    )
    if point_in_rectangle_closed(repaired, layout.obstacle):
        if repaired[1] <= 0.0:
            repaired[1] = layout.obstacle.y_lower - layout.clearance
        else:
            repaired[1] = layout.obstacle.y_upper + layout.clearance
    return repaired


def top_label_radius(probability_upper: float, sigma: float) -> float:
    top_probability = max(probability_upper, 1.0 - probability_upper)
    if not 0.5 < top_probability < 1.0:
        raise ValueError("top-label probability must lie strictly inside (0.5, 1)")
    return float(sigma * ndtri(top_probability))


def find_upper_decision_boundary(
    probability_function: Callable[[np.ndarray], float], anchor: np.ndarray, sigma: float
) -> float | None:
    """Find the unique vertical center at which the upper label has mass 1/2."""

    def objective(y_value: float) -> float:
        center = np.asarray([anchor[0], y_value], dtype=np.float64)
        return probability_function(center) - 0.5

    # Five standard deviations are enough for every generated split and keep
    # the finite action-box occupancy away from floating-point underflow.
    radius = 5.0 * sigma
    lower = float(anchor[1] - radius)
    upper = float(anchor[1] + radius)
    lower_value = objective(lower)
    upper_value = objective(upper)
    if lower_value >= 0.0 and upper_value > 0.0:
        # A sufficiently large frozen top-corridor mass can make the upper
        # decision constant for every center.  It then has no finite lower
        # decision boundary.
        return None
    if not lower_value < 0.0 or not upper_value > 0.0:
        raise FloatingPointError("could not bracket the binary decision boundary")
    return float(brentq(objective, lower, upper, xtol=2e-14, rtol=1e-14))


def generate_layouts(split: str, seed: int, count: int) -> list[Layout]:
    """Generate a deterministic split of one-step obstacle preimages."""

    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    box = Rectangle(-1.0, 1.0, -1.0, 1.0)
    layouts: list[Layout] = []
    for layout_index in range(count):
        obstacle_center_x = float(rng.uniform(-0.22, 0.22))
        obstacle_half_width = float(rng.uniform(0.24, 0.52))
        obstacle_half_height = float(rng.uniform(0.24, 0.58))
        obstacle = Rectangle(
            obstacle_center_x - obstacle_half_width,
            obstacle_center_x + obstacle_half_width,
            -obstacle_half_height,
            obstacle_half_height,
        )
        sigma = float(rng.uniform(0.17, 0.33))
        anchor_y = float(sigma * rng.uniform(0.10, 0.38))
        anchor_x = float(obstacle_center_x + sigma * rng.uniform(-0.45, 0.45))
        layouts.append(
            Layout(
                split=split,
                layout_index=layout_index,
                box=box,
                obstacle=obstacle,
                sigma=sigma,
                clearance=0.025,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )
        )
    return layouts


def method_probability_functions(
    layout: Layout, frozen_weights: np.ndarray
) -> dict[str, Callable[[np.ndarray], float]]:
    return {
        "moving_conditioned_gaussian": lambda center: upper_probability_moving_conditioned(
            layout, center
        ),
        "fixed_collision_repair": lambda center: upper_probability_fixed_repair(
            layout, center
        ),
        "anchor_frozen_corridor_projection": (
            lambda center: upper_probability_frozen_projection(
                layout, center, frozen_weights
            )
        ),
        "anchor_frozen_conditioned_corridor_mixture": (
            lambda center: upper_probability_frozen_conditioned_mixture(
                layout, center, frozen_weights
            )
        ),
    }


def evaluate_layout(layout: Layout) -> list[dict[str, Any]]:
    anchor = layout.anchor
    shifted = layout.paired_center
    frozen_weights = corridor_weights(layout, anchor)
    probability_functions = method_probability_functions(layout, frozen_weights)
    gaussian_pair_kl = gaussian_kl(layout, anchor, shifted)
    gaussian_pair_tv = gaussian_total_variation(layout, anchor, shifted)
    naive_full_kl = moving_conditioned_kl(layout, anchor, shifted)
    frozen_conditioned_full_kl = frozen_conditioned_mixture_kl(
        layout, anchor, shifted, frozen_weights
    )
    occupancy_anchor = feasible_region_stats(layout, anchor).mass
    occupancy_shifted = feasible_region_stats(layout, shifted).mass

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        probability_function = probability_functions[method]
        anchor_probability = probability_function(anchor)
        shifted_probability = probability_function(shifted)
        if not anchor_probability > 0.5:
            raise FloatingPointError(
                f"expected the upper label at the anchor for {method}; "
                f"got {anchor_probability}, {shifted_probability}"
            )
        radius = top_label_radius(anchor_probability, layout.sigma)
        boundary_y = find_upper_decision_boundary(
            probability_function, anchor, layout.sigma
        )
        boundary_distance = (
            None if boundary_y is None else float(anchor[1] - boundary_y)
        )
        radius_exceeds_boundary = bool(
            boundary_distance is not None and radius > boundary_distance + 2e-12
        )
        if radius_exceeds_boundary:
            assert boundary_distance is not None
            attack_distance = 0.5 * (radius + boundary_distance)
        else:
            attack_distance = 0.999 * radius
        attack_center = np.asarray(
            [anchor[0], anchor[1] - attack_distance], dtype=np.float64
        )
        attack_probability = probability_function(attack_center)
        attack_flips_label = bool(attack_probability < 0.5)

        if method == "moving_conditioned_gaussian":
            full_output_kl = naive_full_kl
            full_output_kl_ratio = naive_full_kl / gaussian_pair_kl
            full_output_kl_upper_bound = None
            gaussian_radius_applicable = False
            certificate_theoretically_sound = False
            certificate_semantics = (
                "invalid_audit_only_moving_occupancy; no Gaussian radius claimed"
            )
            expected_proposals_anchor = 1.0 / occupancy_anchor
            expected_proposals_shifted = 1.0 / occupancy_shifted
            anchor_tracking_mse = conditional_tracking_mse(layout, anchor, anchor)
            shifted_tracking_mse = conditional_tracking_mse(
                layout, shifted, shifted
            )
            matches_whole_conditioning_at_anchor = True
            sampling_cost_semantics = "whole-set Gaussian rejection"
        elif method == "fixed_collision_repair":
            full_output_kl = None
            full_output_kl_ratio = None
            full_output_kl_upper_bound = gaussian_pair_kl
            gaussian_radius_applicable = True
            certificate_theoretically_sound = True
            certificate_semantics = (
                "valid_fixed_deterministic_pushforward_by_data_processing"
            )
            expected_proposals_anchor = 1.0
            expected_proposals_shifted = 1.0
            anchor_tracking_mse = fixed_repair_tracking_mse(
                layout, anchor, anchor
            )
            shifted_tracking_mse = fixed_repair_tracking_mse(
                layout, shifted, shifted
            )
            matches_whole_conditioning_at_anchor = False
            sampling_cost_semantics = "one Gaussian draw plus deterministic repair"
        elif method == "anchor_frozen_corridor_projection":
            full_output_kl = None
            full_output_kl_ratio = None
            full_output_kl_upper_bound = gaussian_pair_kl
            gaussian_radius_applicable = True
            certificate_theoretically_sound = True
            certificate_semantics = (
                "valid_fixed_randomized_projection_by_data_processing; "
                "anchor_weights_frozen_over_the_neighborhood"
            )
            expected_proposals_anchor = 1.0
            expected_proposals_shifted = 1.0
            anchor_tracking_mse = frozen_projection_tracking_mse(
                layout, anchor, anchor, frozen_weights
            )
            shifted_tracking_mse = frozen_projection_tracking_mse(
                layout, shifted, shifted, frozen_weights
            )
            matches_whole_conditioning_at_anchor = False
            sampling_cost_semantics = (
                "one categorical draw and one Gaussian draw plus projection"
            )
        else:
            full_output_kl = frozen_conditioned_full_kl
            full_output_kl_ratio = frozen_conditioned_full_kl / gaussian_pair_kl
            full_output_kl_upper_bound = gaussian_pair_kl
            gaussian_radius_applicable = True
            certificate_theoretically_sound = True
            certificate_semantics = (
                "valid_convex_fixed_weight_conditioned_mixture_theorem; "
                "anchor_weights_frozen_over_the_neighborhood"
            )
            expected_proposals_anchor = (
                frozen_conditioned_mixture_expected_proposals(
                    layout, anchor, frozen_weights
                )
            )
            expected_proposals_shifted = (
                frozen_conditioned_mixture_expected_proposals(
                    layout, shifted, frozen_weights
                )
            )
            anchor_tracking_mse = frozen_conditioned_mixture_tracking_mse(
                layout, anchor, anchor, frozen_weights
            )
            shifted_tracking_mse = frozen_conditioned_mixture_tracking_mse(
                layout, shifted, shifted, frozen_weights
            )
            matches_whole_conditioning_at_anchor = True
            sampling_cost_semantics = (
                "naive mode-first Gaussian rejection; direct truncated-normal "
                "sampling not timed"
            )

        binary_kl = bernoulli_kl(anchor_probability, shifted_probability)
        event_tv = abs(anchor_probability - shifted_probability)
        row = {
            "split": layout.split,
            "layout_index": layout.layout_index,
            "method": method,
            "sigma": layout.sigma,
            "clearance": layout.clearance,
            "obstacle_x_lower": layout.obstacle.x_lower,
            "obstacle_x_upper": layout.obstacle.x_upper,
            "obstacle_y_lower": layout.obstacle.y_lower,
            "obstacle_y_upper": layout.obstacle.y_upper,
            "anchor_x": layout.anchor_x,
            "anchor_y": layout.anchor_y,
            "paired_center_x": float(shifted[0]),
            "paired_center_y": float(shifted[1]),
            "anchor_feasible_occupancy": occupancy_anchor,
            "paired_feasible_occupancy": occupancy_shifted,
            "frozen_weight_left": float(frozen_weights[0]),
            "frozen_weight_right": float(frozen_weights[1]),
            "frozen_weight_bottom": float(frozen_weights[2]),
            "frozen_weight_top": float(frozen_weights[3]),
            "anchor_upper_probability": anchor_probability,
            "paired_upper_probability": shifted_probability,
            "anchor_label": "upper",
            "paired_label": "upper" if shifted_probability >= 0.5 else "lower",
            "gaussian_pair_kl": gaussian_pair_kl,
            "gaussian_pair_tv": gaussian_pair_tv,
            "full_output_kl_exact": full_output_kl,
            "full_output_kl_ratio": full_output_kl_ratio,
            "full_output_kl_certified_upper_bound": full_output_kl_upper_bound,
            "binary_observable_kl": binary_kl,
            "binary_observable_kl_ratio": binary_kl / gaussian_pair_kl,
            "binary_event_tv": event_tv,
            "binary_event_tv_ratio": event_tv / gaussian_pair_tv,
            "gaussian_radius_applicable": gaussian_radius_applicable,
            "certificate_theoretically_sound": certificate_theoretically_sound,
            "certificate_semantics": certificate_semantics,
            "matches_whole_conditioning_at_anchor": (
                matches_whole_conditioning_at_anchor
            ),
            "anchor_gaussian_formula_radius": radius,
            "distance_to_exact_label_boundary": boundary_distance,
            "formula_radius_exceeds_exact_boundary": radius_exceeds_boundary,
            "attack_distance": attack_distance,
            "attack_center_y": float(attack_center[1]),
            "attack_upper_probability": attack_probability,
            "attack_flips_label": attack_flips_label,
            "misapplied_radius_attack_found_false_positive": bool(
                method == "moving_conditioned_gaussian" and attack_flips_label
            ),
            "targeted_attack_found_certificate_violation": (
                bool(attack_flips_label)
                if certificate_theoretically_sound
                else None
            ),
            "tracking_mse_at_anchor": anchor_tracking_mse,
            "tracking_mse_at_paired_center": shifted_tracking_mse,
            "expected_gaussian_proposals_at_anchor": expected_proposals_anchor,
            "expected_gaussian_proposals_at_paired_center": expected_proposals_shifted,
            "sampling_cost_semantics": sampling_cost_semantics,
            "analytical_collision_probability": 0.0,
        }
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no rows found for split {split!r}")
    summary: dict[str, Any] = {
        "split": split,
        "layout_count": len(selected) // len(METHODS),
    }
    for method in METHODS:
        method_rows = [row for row in selected if row["method"] == method]
        binary_kl_ratios = np.asarray(
            [row["binary_observable_kl_ratio"] for row in method_rows]
        )
        event_tv_ratios = np.asarray(
            [row["binary_event_tv_ratio"] for row in method_rows]
        )
        tracking = np.asarray(
            [row["tracking_mse_at_anchor"] for row in method_rows]
        )
        proposals = np.asarray(
            [row["expected_gaussian_proposals_at_anchor"] for row in method_rows]
        )
        attack_flags = [
            row["targeted_attack_found_certificate_violation"]
            for row in method_rows
            if row["targeted_attack_found_certificate_violation"] is not None
        ]
        method_summary: dict[str, Any] = {
            "misapplied_radius_attack_found_false_positive_rate": float(
                np.mean(
                    [
                        row["misapplied_radius_attack_found_false_positive"]
                        for row in method_rows
                    ]
                )
            ),
            "targeted_attack_found_certificate_violation_rate": float(
                np.mean(attack_flags)
            ) if attack_flags else None,
            "certificate_theoretically_sound": bool(
                all(row["certificate_theoretically_sound"] for row in method_rows)
            ),
            "median_binary_observable_kl_ratio": float(np.median(binary_kl_ratios)),
            "maximum_binary_observable_kl_ratio": float(np.max(binary_kl_ratios)),
            "median_binary_event_tv_ratio": float(np.median(event_tv_ratios)),
            "maximum_binary_event_tv_ratio": float(np.max(event_tv_ratios)),
            "median_tracking_mse_at_anchor": float(np.median(tracking)),
            "median_expected_gaussian_proposals_at_anchor": float(
                np.median(proposals)
            ),
            "maximum_analytical_collision_probability": float(
                max(row["analytical_collision_probability"] for row in method_rows)
            ),
        }
        full_ratios = [
            row["full_output_kl_ratio"]
            for row in method_rows
            if row["full_output_kl_ratio"] is not None
        ]
        if full_ratios:
            method_summary.update(
                {
                    "median_full_output_kl_ratio": float(np.median(full_ratios)),
                    "minimum_full_output_kl_ratio": float(np.min(full_ratios)),
                    "maximum_full_output_kl_ratio": float(np.max(full_ratios)),
                    "fraction_full_output_kl_above_gaussian": float(
                        np.mean(np.asarray(full_ratios) > 1.0)
                    ),
                }
            )
        else:
            method_summary.update(
                {
                    "median_full_output_kl_ratio": None,
                    "minimum_full_output_kl_ratio": None,
                    "maximum_full_output_kl_ratio": None,
                    "fraction_full_output_kl_above_gaussian": None,
                }
            )
        summary[method] = method_summary
    return summary


def build_payload(
    development_layouts: int = DEFAULT_DEVELOPMENT_LAYOUTS,
    evaluation_layouts: int = DEFAULT_EVALUATION_LAYOUTS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layouts = generate_layouts(
        "development", DEVELOPMENT_SEED, development_layouts
    ) + generate_layouts("evaluation", EVALUATION_SEED, evaluation_layouts)
    rows = [row for layout in layouts for row in evaluate_layout(layout)]
    payload = {
        "experiment": "iclr2027_exact_one_step_reach_avoid_kernel_pilot",
        "configuration": {
            "development_seed": DEVELOPMENT_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "development_layout_count": development_layouts,
            "evaluation_layout_count": evaluation_layouts,
            "action_box": asdict(layouts[0].box),
            "layout_distribution": {
                "obstacle_center_x": "Uniform[-0.22, 0.22]",
                "obstacle_half_width": "Uniform[0.24, 0.52]",
                "obstacle_center_y": 0.0,
                "obstacle_half_height": "Uniform[0.24, 0.58]",
                "sigma": "Uniform[0.17, 0.33]",
                "anchor_y_over_sigma": "Uniform[0.10, 0.38]",
                "anchor_x_offset_over_sigma": "Uniform[-0.45, 0.45]",
                "repair_clearance": 0.025,
            },
            "paired_center_rule": "reflect anchor across y=0",
            "binary_label": "upper iff repaired/feasible action has y > 0",
        },
        "geometry_statement": {
            "feasible_set": "closed action box minus closed forbidden rectangle",
            "partition": [
                "left: x < obstacle.x_lower",
                "right: x > obstacle.x_upper",
                "bottom: obstacle.x_lower <= x <= obstacle.x_upper and y < obstacle.y_lower",
                "top: obstacle.x_lower <= x <= obstacle.x_upper and y > obstacle.y_upper",
            ],
            "partition_properties": (
                "exact set equality, pairwise disjointness, and convexity; obstacle "
                "boundary excluded"
            ),
        },
        "certificate_policy": {
            "moving_conditioned_gaussian": (
                "No ambient Gaussian certificate. The displayed Gaussian-formula "
                "radius is an explicitly invalid audit counterfactual because the "
                "occupancy normalizer moves with the center."
            ),
            "fixed_collision_repair": (
                "Valid ambient Gaussian KL/TV and binary radius by data processing "
                "through one center-independent deterministic repair."
            ),
            "anchor_frozen_corridor_projection": (
                "Valid by data processing through a center-independent randomized "
                "projection, only while the anchor corridor weights remain fixed."
            ),
            "anchor_frozen_conditioned_corridor_mixture": (
                "Valid by the convex fixed-weight mixture theorem while the anchor "
                "weights remain fixed. It exactly matches whole-set conditioning at "
                "the anchor; re-anchoring is outside the claim."
            ),
        },
        "development_summary": summarize_rows(rows, "development"),
        "evaluation_summary": summarize_rows(rows, "evaluation"),
        "records": rows,
        "explicit_nonclaims": [
            "This is an exact one-step geometric control proxy, not a learned-policy or trajectory benchmark.",
            "The axis-aligned forbidden action rectangle is an affine one-step obstacle preimage; no simulator result is claimed.",
            "The upper/lower label is a meaningful avoidance decision but is specialized to an obstacle straddling y=0.",
            "No full repair/projection output KL is estimated: those rows report the rigorous Gaussian upper bound and exact binary-observable lower bound.",
            "The projection baseline is not the paper's conditioned-corridor mixture.",
            "Neither anchor-frozen result certifies a deployment that recomputes corridor weights at every query.",
            "The layout generator deliberately centers each obstacle on y=0, chooses a positive anchor, reflects the paired center, and evaluates the upper/lower label; the observed attack rate is an engineered mechanism check, not population prevalence.",
            "Zero attack-found violations are not empirical proof of soundness; soundness follows from data processing or the stated convex fixed-weight mixture theorem.",
        ],
    }
    return payload, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(payload: dict[str, Any]) -> str:
    evaluation = payload["evaluation_summary"]
    naive = evaluation["moving_conditioned_gaussian"]
    repair = evaluation["fixed_collision_repair"]
    projection = evaluation["anchor_frozen_corridor_projection"]
    conditioned = evaluation["anchor_frozen_conditioned_corridor_mixture"]
    return f"""# ICLR 2027 exact one-step reach--avoid kernel pilot

## Evaluation-split mechanism check

The evaluation split contains **{evaluation['layout_count']}** deterministic
random obstacle layouts generated only from seed
`{payload['configuration']['evaluation_seed']}`.  For the moving conditioned
Gaussian, naively reusing the ambient Gaussian binary radius would produce a
targeted-attack false positive on
**{100.0 * naive['misapplied_radius_attack_found_false_positive_rate']:.2f}%**
of layouts.
This radius is explicitly invalid and is reported only to demonstrate the
failure mode.  Its exact full-law KL / Gaussian-KL ratio has median
**{naive['median_full_output_kl_ratio']:.3f}**, range
**[{naive['minimum_full_output_kl_ratio']:.3f},
{naive['maximum_full_output_kl_ratio']:.3f}]**, and exceeds one on
**{100.0 * naive['fraction_full_output_kl_above_gaussian']:.2f}%** of layouts.

The fixed repair, frozen projection, and fixed-weight conditioned mixture had
targeted-attack-found certificate-violation rates of
**{100.0 * repair['targeted_attack_found_certificate_violation_rate']:.2f}%**,
**{100.0 * projection['targeted_attack_found_certificate_violation_rate']:.2f}%**,
and
**{100.0 * conditioned['targeted_attack_found_certificate_violation_rate']:.2f}%**.
Those zeros are not empirical proofs.  Soundness follows independently from
data processing for the two fixed pushforwards and from the convex
fixed-weight mixture theorem for the conditioned-corridor construction.

The conditioned-corridor mixture's exact full-law KL / Gaussian-KL ratio has
median **{conditioned['median_full_output_kl_ratio']:.3f}** and maximum
**{conditioned['maximum_full_output_kl_ratio']:.3f}**.  Unlike the projection
baseline, it exactly matches whole-set conditioning at the anchor.  Its
reported proposal count is the exact expectation for naive mode-first
Gaussian rejection, not a claim about an optimized truncated-normal sampler.

| method | certificate basis | attack-found rate | median full KL ratio | median binary KL ratio | median event-TV ratio | median anchor MSE | median proposals |
|:--|:--|--:|--:|--:|--:|--:|--:|
| moving conditioned Gaussian | invalid ambient-radius audit | {100.0 * naive['misapplied_radius_attack_found_false_positive_rate']:.2f}% | {naive['median_full_output_kl_ratio']:.3f} | {naive['median_binary_observable_kl_ratio']:.3f} | {naive['median_binary_event_tv_ratio']:.3f} | {naive['median_tracking_mse_at_anchor']:.4f} | {naive['median_expected_gaussian_proposals_at_anchor']:.3f} |
| fixed collision repair | data processing | {100.0 * repair['targeted_attack_found_certificate_violation_rate']:.2f}% | -- | {repair['median_binary_observable_kl_ratio']:.3f} | {repair['median_binary_event_tv_ratio']:.3f} | {repair['median_tracking_mse_at_anchor']:.4f} | {repair['median_expected_gaussian_proposals_at_anchor']:.3f} |
| anchor-frozen corridor projection | data processing | {100.0 * projection['targeted_attack_found_certificate_violation_rate']:.2f}% | -- | {projection['median_binary_observable_kl_ratio']:.3f} | {projection['median_binary_event_tv_ratio']:.3f} | {projection['median_tracking_mse_at_anchor']:.4f} | {projection['median_expected_gaussian_proposals_at_anchor']:.3f} |
| anchor-frozen conditioned-corridor mixture | convex fixed-weight theorem | {100.0 * conditioned['targeted_attack_found_certificate_violation_rate']:.2f}% | {conditioned['median_full_output_kl_ratio']:.3f} | {conditioned['median_binary_observable_kl_ratio']:.3f} | {conditioned['median_binary_event_tv_ratio']:.3f} | {conditioned['median_tracking_mse_at_anchor']:.4f} | {conditioned['median_expected_gaussian_proposals_at_anchor']:.3f} |

All four methods have analytical collision probability zero under the stated
closed-obstacle model.  Whole-set conditioning costs `1 / feasible occupancy`
in expectation.  Repair and projection use one Gaussian draw.  The
conditioned-corridor mixture first samples a fixed mode and has expected cost
`sum_j w_j / GaussianMass(center, corridor_j)` under naive rejection.

## Exact geometry and computations

The feasible action set is the closed box `[-1,1]^2` minus a closed
axis-aligned collision rectangle.  A half-open boundary convention partitions
it exactly into left, right, bottom, and top convex corridors.  Tests enumerate
all boundary combinations and randomized convex combinations.  Gaussian
rectangle masses and first two unnormalized moments are closed-form normal-CDF
expressions.  The moving-law KL uses the exact common-support
exponential-family identity.  Disjoint corridor support makes the fixed-weight
mixture KL exactly the fixed-weight sum of its four within-corridor KL values.
No histogram or Monte Carlo KL is used.

The binary outcome is `upper` iff the final feasible action has positive
lateral coordinate.  This is a function of the final action and is not a
corridor identifier supplied to the classifier.  It represents the one-step
choice to pass above rather than below a centered obstacle.

## Certificate boundary

- **Moving conditioning:** occupancy changes with the center.  The ambient
  Gaussian radius is not a certificate.  The reported false-positive rate is
  the result of deliberately misapplying it.
- **Fixed repair:** clipping and collision repair form one deterministic map
  independent of the Gaussian center, so KL, TV, and the binary smoothing
  radius follow by data processing.
- **Anchor-frozen projection:** the categorical corridor weights are computed
  at the anchor and held fixed.  Sampling a corridor followed by projection is
  one center-independent randomized map.  This is a baseline, not the paper's
  conditioned-corridor construction.
- **Anchor-frozen conditioned-corridor mixture:** the same fixed weights are
  combined with Gaussians conditioned within the original convex corridors.
  It equals whole-set conditioning at the anchor and obeys the Gaussian event,
  TV, and KL comparisons by the convex fixed-weight theorem.  Recomputing its
  weights at each center is outside this claim.

## Limitations

This is an exact **one-step** reach--avoid geometry.  It is not a learned-policy
or trajectory benchmark, nor evidence from long-horizon dynamics, a perception
stack, or a physical robot.  The generator deliberately centers every
obstacle on `y=0`, takes a positive anchor, reflects the paired center, and
tests the upper/lower decision.  The 100% moving-law attack rate is therefore
an engineered mechanism check, not prevalence in a target population and not
confirmatory evidence.  Axis alignment makes exact auditing possible but is
narrower than general collision preimages.  Tracking MSE is a utility proxy,
not task return.  The split establishes deterministic holdout discipline, not
population generalization.  Full output KL is exact only for the moving and
fixed-weight conditioned laws; repaired/projection output KL is represented by
its rigorous Gaussian upper bound and an exact binary-observable lower bound.
"""


def write_outputs(
    output_stem: Path, payload: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, str]:
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.parent / f"{output_stem.name}_records.csv"
    markdown_path = output_stem.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "json": str(json_path),
        "records_csv": str(csv_path),
        "markdown": str(markdown_path),
    }
    output_payload = {**payload, "artifacts": artifacts}
    json_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    markdown_path.write_text(markdown_report(output_payload), encoding="utf-8")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact ICLR 2027 one-step reach--avoid kernel pilot."
    )
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--development-layouts", type=int, default=DEFAULT_DEVELOPMENT_LAYOUTS
    )
    parser.add_argument(
        "--evaluation-layouts", type=int, default=DEFAULT_EVALUATION_LAYOUTS
    )
    args = parser.parse_args()
    payload, rows = build_payload(args.development_layouts, args.evaluation_layouts)
    artifacts = write_outputs(args.output_stem, payload, rows)
    print(
        json.dumps(
            {
                "artifacts": artifacts,
                "evaluation_summary": payload["evaluation_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
