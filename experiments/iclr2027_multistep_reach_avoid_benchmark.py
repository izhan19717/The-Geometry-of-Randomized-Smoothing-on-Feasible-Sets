"""Frozen-split multi-step reach--avoid benchmark for ICLR 2027.

This benchmark is deliberately separate from the symmetric one-step mechanism
pilot.  A discrete-time point robot visits seven forward waypoint stages.  At
each stage the admissible next-state set is an axis-aligned box with one
nonsymmetrically placed closed rectangular obstacle removed.  A conventional
one-step receding-horizon controller supplies the Gaussian action center.

The experiment compares moving whole-set conditioning with three fixed-kernel
repairs and two exact convex partitions.  Certificate claims are *per state*:
the support and anchor weights are held fixed while a center perturbation is
audited.  Recomputing weights after perturbing a query is never certified.

Collision means membership of the sampled next-state waypoint in the current
closed obstacle.  The script makes no swept-segment, perception, observation-
space, learned-policy, or physical-robot claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri

from experiments.iclr2027_reach_avoid_kernel_pilot import (
    Rectangle,
    RegionStats,
    bernoulli_kl,
    interval_stats,
    rectangle_conditioned_kl,
    rectangle_stats,
    subtract_region,
)


DEFAULT_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "REACH_AVOID_CONTROLLER_PROTOCOL_V2_20260830.json"
)
DEFAULT_OUTPUT_STEM = Path("outputs/iclr2027_multistep_reach_avoid_benchmark")
ANONYMOUS_SOURCE_PATH = Path("experiments/iclr2027_multistep_reach_avoid_benchmark.py")
METHODS = (
    "native_controller",
    "moving_conditioned_gaussian",
    "fixed_vertical_collision_repair",
    "anchor_frozen_corridor_projection",
    "fixed_uniform_conditioned_coarse4",
    "anchor_frozen_conditioned_coarse4",
    "anchor_frozen_conditioned_refined8",
)
SOUND_METHODS = frozenset(
    {
        "fixed_vertical_collision_repair",
        "anchor_frozen_corridor_projection",
        "fixed_uniform_conditioned_coarse4",
        "anchor_frozen_conditioned_coarse4",
        "anchor_frozen_conditioned_refined8",
    }
)
AUDIT_METHODS = METHODS[1:] + ("convex_box_conditioned_control",)
SUMMARY_METRICS = (
    "success",
    "task_return",
    "path_length",
    "final_distance",
    "actual_gaussian_proposals",
    "naive_rejection_equivalent_proposals",
    "primitive_random_variates",
    "inverse_cdf_calls",
    "runtime_ms",
)


@dataclass(frozen=True)
class Stage:
    stage_index: int
    box: Rectangle
    obstacle: Rectangle
    sigma: float
    clearance: float

    @property
    def center_x(self) -> float:
        return 0.5 * (self.box.x_lower + self.box.x_upper)

    @property
    def obstacle_center_x(self) -> float:
        return 0.5 * (self.obstacle.x_lower + self.obstacle.x_upper)

    @property
    def obstacle_center_y(self) -> float:
        return 0.5 * (self.obstacle.y_lower + self.obstacle.y_upper)


@dataclass(frozen=True)
class NavigationLayout:
    split: str
    layout_index: int
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float
    stages: tuple[Stage, ...]

    @property
    def start(self) -> np.ndarray:
        return np.asarray([self.start_x, self.start_y], dtype=np.float64)

    @property
    def goal(self) -> np.ndarray:
        return np.asarray([self.goal_x, self.goal_y], dtype=np.float64)


@dataclass(frozen=True)
class SampleCost:
    actual_gaussian_proposals: int
    primitive_random_variates: int
    inverse_cdf_calls: int
    naive_rejection_equivalent_proposals: float


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") not in {1, 2}:
        raise ValueError("only protocol versions 1 and 2 are supported")
    return payload


def point_in_closed_rectangle(point: np.ndarray, rectangle: Rectangle) -> bool:
    x_value, y_value = (float(value) for value in point)
    return bool(
        rectangle.x_lower <= x_value <= rectangle.x_upper
        and rectangle.y_lower <= y_value <= rectangle.y_upper
    )


def point_is_feasible(point: np.ndarray, stage: Stage) -> bool:
    return point_in_closed_rectangle(point, stage.box) and not point_in_closed_rectangle(
        point, stage.obstacle
    )


def support_stats(stage: Stage, center: np.ndarray) -> RegionStats:
    stats = subtract_region(
        rectangle_stats(stage.box, center, stage.sigma),
        rectangle_stats(stage.obstacle, center, stage.sigma),
    )
    if not 0.0 < stats.mass < 1.0:
        raise FloatingPointError("stage support occupancy must lie in (0,1)")
    return stats


def coarse_partition(stage: Stage) -> tuple[Rectangle, ...]:
    box = stage.box
    obstacle = stage.obstacle
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


def refined_partition(stage: Stage) -> tuple[Rectangle, ...]:
    """Eight-piece exact refinement, predeclared independently of the label.

    Left and right are split at global ``y=0``.  Bottom and top are split at
    the obstacle center in ``x``.  All splits lie strictly inside their parent
    rectangles under the frozen layout distribution.
    """

    left, right, bottom, top = coarse_partition(stage)
    split_y = 0.0
    split_x = stage.obstacle_center_x
    return (
        Rectangle(left.x_lower, left.x_upper, left.y_lower, split_y),
        Rectangle(left.x_lower, left.x_upper, split_y, left.y_upper),
        Rectangle(right.x_lower, right.x_upper, right.y_lower, split_y),
        Rectangle(right.x_lower, right.x_upper, split_y, right.y_upper),
        Rectangle(bottom.x_lower, split_x, bottom.y_lower, bottom.y_upper),
        Rectangle(split_x, bottom.x_upper, bottom.y_lower, bottom.y_upper),
        Rectangle(top.x_lower, split_x, top.y_lower, top.y_upper),
        Rectangle(split_x, top.x_upper, top.y_lower, top.y_upper),
    )


def projection_rectangles(stage: Stage) -> tuple[Rectangle, ...]:
    left, right, bottom, top = coarse_partition(stage)
    gap = stage.clearance
    return (
        Rectangle(left.x_lower, left.x_upper - gap, left.y_lower, left.y_upper),
        Rectangle(right.x_lower + gap, right.x_upper, right.y_lower, right.y_upper),
        Rectangle(
            bottom.x_lower,
            bottom.x_upper,
            bottom.y_lower,
            bottom.y_upper - gap,
        ),
        Rectangle(
            top.x_lower,
            top.x_upper,
            top.y_lower + gap,
            top.y_upper,
        ),
    )


def partition_weights(
    stage: Stage, center: np.ndarray, rectangles: Sequence[Rectangle]
) -> np.ndarray:
    log_masses = np.asarray(
        [rectangle_logmass(rectangle, center, stage.sigma) for rectangle in rectangles],
        dtype=np.float64,
    )
    log_partition_mass = float(logsumexp(log_masses))
    occupancy = support_stats(stage, center).mass
    if not math.isclose(
        log_partition_mass,
        math.log(occupancy),
        rel_tol=5e-11,
        abs_tol=5e-12,
    ):
        raise FloatingPointError("partition masses do not equal support occupancy")
    weights = np.exp(log_masses - log_partition_mass)
    return weights / np.sum(weights)


def moving_conditioned_kl(stage: Stage, first: np.ndarray, second: np.ndarray) -> float:
    first_stats = support_stats(stage, first)
    second_stats = support_stats(stage, second)
    conditional_mean = np.asarray(
        [first_stats.first_x, first_stats.first_y], dtype=np.float64
    ) / first_stats.mass
    mean_term = (
        2.0 * float(np.dot(conditional_mean, first - second))
        + float(np.dot(second, second))
        - float(np.dot(first, first))
    ) / (2.0 * stage.sigma**2)
    value = mean_term + math.log(second_stats.mass / first_stats.mass)
    if value < 0.0 and abs(value) < 2e-11:
        return 0.0
    if value < 0.0:
        raise FloatingPointError(f"negative moving-conditioned KL: {value}")
    return float(value)


def gaussian_kl(stage: Stage, first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first) - np.asarray(second)
    return float(np.dot(difference, difference) / (2.0 * stage.sigma**2))


def fixed_mixture_kl(
    stage: Stage,
    first: np.ndarray,
    second: np.ndarray,
    rectangles: Sequence[Rectangle],
    weights: np.ndarray,
) -> float:
    return float(
        sum(
            float(weight)
            * rectangle_conditioned_kl(rectangle, stage.sigma, first, second)
            for rectangle, weight in zip(rectangles, weights, strict=True)
        )
    )


def log_subtract_exp(log_larger: float, log_smaller: float) -> float:
    """Return ``log(exp(log_larger) - exp(log_smaller))`` stably."""

    if math.isinf(log_smaller) and log_smaller < 0.0:
        return float(log_larger)
    difference = float(log_smaller - log_larger)
    if difference > 5e-13:
        raise FloatingPointError("log-domain subtraction has reversed operands")
    if difference >= 0.0:
        return -math.inf
    return float(log_larger + math.log1p(-math.exp(difference)))


def normal_interval_logmass(
    lower: float, upper: float, mean: float, sigma: float
) -> float:
    """Stable log probability of a one-dimensional Gaussian interval."""

    if not upper > lower:
        return -math.inf
    standardized_lower = (lower - mean) / sigma
    standardized_upper = (upper - mean) / sigma
    if standardized_upper <= 0.0:
        return log_subtract_exp(
            float(log_ndtr(standardized_upper)),
            float(log_ndtr(standardized_lower)),
        )
    if standardized_lower >= 0.0:
        return log_subtract_exp(
            float(log_ndtr(-standardized_lower)),
            float(log_ndtr(-standardized_upper)),
        )
    probability = float(ndtr(standardized_upper) - ndtr(standardized_lower))
    if probability <= 0.0:
        raise FloatingPointError("central Gaussian interval has nonpositive mass")
    return float(math.log(probability))


def rectangle_logmass(
    rectangle: Rectangle, center: np.ndarray, sigma: float
) -> float:
    return float(
        normal_interval_logmass(
            rectangle.x_lower, rectangle.x_upper, float(center[0]), sigma
        )
        + normal_interval_logmass(
            rectangle.y_lower, rectangle.y_upper, float(center[1]), sigma
        )
    )


def upper_rectangle_logmass(
    rectangle: Rectangle, center: np.ndarray, sigma: float, threshold: float
) -> float:
    lower = max(rectangle.y_lower, threshold)
    if lower >= rectangle.y_upper:
        return -math.inf
    return float(
        normal_interval_logmass(
            rectangle.x_lower, rectangle.x_upper, float(center[0]), sigma
        )
        + normal_interval_logmass(lower, rectangle.y_upper, float(center[1]), sigma)
    )


def upper_mass_rectangle(
    rectangle: Rectangle, center: np.ndarray, sigma: float, threshold: float
) -> float:
    lower = max(rectangle.y_lower, threshold)
    if lower >= rectangle.y_upper:
        return 0.0
    upper_rectangle = Rectangle(
        rectangle.x_lower, rectangle.x_upper, lower, rectangle.y_upper
    )
    return float(rectangle_stats(upper_rectangle, center, sigma).mass)


def moving_upper_probability(stage: Stage, center: np.ndarray) -> float:
    threshold = stage.obstacle_center_y
    log_support = log_subtract_exp(
        rectangle_logmass(stage.box, center, stage.sigma),
        rectangle_logmass(stage.obstacle, center, stage.sigma),
    )
    log_upper_support = log_subtract_exp(
        upper_rectangle_logmass(stage.box, center, stage.sigma, threshold),
        upper_rectangle_logmass(stage.obstacle, center, stage.sigma, threshold),
    )
    if math.isinf(log_upper_support) and log_upper_support < 0.0:
        return 0.0
    return float(np.clip(math.exp(log_upper_support - log_support), 0.0, 1.0))


def convex_box_upper_probability(stage: Stage, center: np.ndarray) -> float:
    log_total = normal_interval_logmass(
        stage.box.y_lower, stage.box.y_upper, float(center[1]), stage.sigma
    )
    log_upper = normal_interval_logmass(
        max(stage.box.y_lower, stage.obstacle_center_y),
        stage.box.y_upper,
        float(center[1]),
        stage.sigma,
    )
    return float(np.clip(math.exp(log_upper - log_total), 0.0, 1.0))


def fixed_repair_upper_probability(stage: Stage, center: np.ndarray) -> float:
    return float(ndtr((float(center[1]) - stage.obstacle_center_y) / stage.sigma))


def clipped_rectangle_upper_probability(
    rectangle: Rectangle, center: np.ndarray, sigma: float, threshold: float
) -> float:
    if rectangle.y_upper <= threshold:
        return 0.0
    if rectangle.y_lower >= threshold:
        return 1.0
    log_total = normal_interval_logmass(
        rectangle.y_lower, rectangle.y_upper, float(center[1]), sigma
    )
    log_upper = normal_interval_logmass(
        threshold, rectangle.y_upper, float(center[1]), sigma
    )
    return float(np.clip(math.exp(log_upper - log_total), 0.0, 1.0))


def projected_rectangle_upper_probability(
    rectangle: Rectangle, center: np.ndarray, sigma: float, threshold: float
) -> float:
    """Exact upper-event probability after coordinatewise clipping.

    If the clipping interval straddles ``threshold``, clipping preserves the
    event ``Y > threshold`` and its probability is the unconditioned Gaussian
    tail.  This is intentionally different from conditioning the Gaussian on
    the interval.  Protocol v2 corrects the original diagnostic helper, which
    used the conditioned probability for this projected law.
    """

    if rectangle.y_upper <= threshold:
        return 0.0
    if rectangle.y_lower > threshold:
        return 1.0
    return float(ndtr((float(center[1]) - threshold) / sigma))


def frozen_projection_upper_probability(
    stage: Stage, center: np.ndarray, weights: np.ndarray
) -> float:
    return float(
        sum(
            float(weight)
            * projected_rectangle_upper_probability(
                rectangle,
                center,
                stage.sigma,
                stage.obstacle_center_y,
            )
            for rectangle, weight in zip(
                projection_rectangles(stage), weights, strict=True
            )
        )
    )


def frozen_conditioned_upper_probability(
    stage: Stage,
    center: np.ndarray,
    rectangles: Sequence[Rectangle],
    weights: np.ndarray,
) -> float:
    probability = 0.0
    for rectangle, weight in zip(rectangles, weights, strict=True):
        conditional_probability = clipped_rectangle_upper_probability(
            rectangle, center, stage.sigma, stage.obstacle_center_y
        )
        probability += float(weight) * conditional_probability
    return float(probability)


def truncated_normal_sample(
    lower: float, upper: float, mean: float, sigma: float, rng: np.random.Generator
) -> float:
    standardized_lower = (lower - mean) / sigma
    standardized_upper = (upper - mean) / sigma
    draw = float(rng.random())
    if standardized_lower >= 0.0:
        survival_lower = float(ndtr(-standardized_lower))
        survival_upper = float(ndtr(-standardized_upper))
        probability = survival_upper + draw * (survival_lower - survival_upper)
        standardized_sample = -float(ndtri(probability))
    else:
        lower_cdf = float(ndtr(standardized_lower))
        upper_cdf = float(ndtr(standardized_upper))
        probability = lower_cdf + draw * (upper_cdf - lower_cdf)
        standardized_sample = float(ndtri(probability))
    if not math.isfinite(standardized_sample):
        raise FloatingPointError("truncated-normal interval has zero numerical mass")
    value = float(mean + sigma * standardized_sample)
    return float(np.clip(value, lower, upper))


def sample_conditioned_rectangle(
    rectangle: Rectangle,
    center: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray(
        [
            truncated_normal_sample(
                rectangle.x_lower, rectangle.x_upper, float(center[0]), sigma, rng
            ),
            truncated_normal_sample(
                rectangle.y_lower, rectangle.y_upper, float(center[1]), sigma, rng
            ),
        ],
        dtype=np.float64,
    )


def fixed_vertical_repair(raw: np.ndarray, stage: Stage) -> np.ndarray:
    repaired = np.clip(
        np.asarray(raw, dtype=np.float64),
        [stage.box.x_lower, stage.box.y_lower],
        [stage.box.x_upper, stage.box.y_upper],
    )
    if point_in_closed_rectangle(repaired, stage.obstacle):
        if repaired[1] <= stage.obstacle_center_y:
            repaired[1] = stage.obstacle.y_lower - stage.clearance
        else:
            repaired[1] = stage.obstacle.y_upper + stage.clearance
    if not point_is_feasible(repaired, stage):
        raise FloatingPointError("fixed repair returned an infeasible waypoint")
    return repaired


def categorical_index(weights: np.ndarray, rng: np.random.Generator) -> int:
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
    draw = float(rng.random())
    return int(np.searchsorted(cumulative, draw, side="right").clip(0, len(weights) - 1))


def sample_kernel(
    method: str, stage: Stage, center: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, SampleCost]:
    occupancy = support_stats(stage, center).mass
    if method == "native_controller":
        if not point_is_feasible(center, stage):
            raise FloatingPointError("native controller returned an infeasible waypoint")
        return np.asarray(center, dtype=np.float64), SampleCost(0, 0, 0, 0.0)

    if method == "moving_conditioned_gaussian":
        proposals = 0
        while True:
            proposals += 1
            if proposals > 100_000:
                raise RuntimeError("moving-law rejection sampler exceeded cap")
            sample = rng.normal(center, stage.sigma, size=2)
            if point_is_feasible(sample, stage):
                return sample, SampleCost(
                    actual_gaussian_proposals=proposals,
                    primitive_random_variates=2 * proposals,
                    inverse_cdf_calls=0,
                    naive_rejection_equivalent_proposals=1.0 / occupancy,
                )

    if method == "fixed_vertical_collision_repair":
        sample = fixed_vertical_repair(rng.normal(center, stage.sigma, size=2), stage)
        return sample, SampleCost(1, 2, 0, 1.0)

    if method == "anchor_frozen_corridor_projection":
        weights = partition_weights(stage, center, coarse_partition(stage))
        mode = categorical_index(weights, rng)
        raw = rng.normal(center, stage.sigma, size=2)
        rectangle = projection_rectangles(stage)[mode]
        sample = np.clip(
            raw,
            [rectangle.x_lower, rectangle.y_lower],
            [rectangle.x_upper, rectangle.y_upper],
        )
        if not point_is_feasible(sample, stage):
            raise FloatingPointError("corridor projection returned infeasible waypoint")
        return sample, SampleCost(1, 3, 0, 1.0)

    if method in {
        "fixed_uniform_conditioned_coarse4",
        "anchor_frozen_conditioned_coarse4",
        "anchor_frozen_conditioned_refined8",
    }:
        rectangles = (
            coarse_partition(stage)
            if method.endswith("coarse4")
            else refined_partition(stage)
        )
        weights = (
            np.full(len(rectangles), 1.0 / len(rectangles), dtype=np.float64)
            if method.startswith("fixed_uniform")
            else partition_weights(stage, center, rectangles)
        )
        mode = categorical_index(weights, rng)
        sample = sample_conditioned_rectangle(
            rectangles[mode], center, stage.sigma, rng
        )
        if not point_is_feasible(sample, stage):
            raise FloatingPointError("conditioned mixture returned infeasible waypoint")
        if method.startswith("fixed_uniform"):
            log_masses = np.asarray(
                [rectangle_logmass(rectangle, center, stage.sigma) for rectangle in rectangles]
            )
            equivalent = float(
                np.sum(weights * np.exp(np.minimum(-log_masses, 700.0)))
            )
        else:
            # At the declared anchor, w_j=G_a(K_j)/G_a(K), so the exact
            # mode-first rejection expectation is sum_j w_j/G_a(K_j)=m/G_a(K).
            # This algebraic form avoids an artificial 0/0 when a remote
            # component mass underflows in ordinary floating-point arithmetic.
            equivalent = float(len(rectangles) / occupancy)
        return sample, SampleCost(0, 3, 2, equivalent)

    raise ValueError(f"unknown method: {method}")


def generate_layouts(
    split: str, seed: int, count: int, protocol: dict[str, Any]
) -> list[NavigationLayout]:
    environment = protocol["environment"]
    horizon = int(environment["horizon"])
    spacing = float(environment["stage_spacing"])
    half_width_x = float(environment["stage_box_half_width_x"])
    y_lower, y_upper = (float(value) for value in environment["stage_box_y_bounds"])
    clearance = float(environment["repair_clearance"])
    rng = np.random.default_rng(seed)
    layouts: list[NavigationLayout] = []
    for layout_index in range(count):
        start_y = float(rng.uniform(-0.30, 0.30))
        goal_y = float(rng.uniform(-0.35, 0.35))
        goal_x = horizon * spacing
        stages: list[Stage] = []
        for stage_index in range(horizon):
            center_x = (stage_index + 1) * spacing
            box = Rectangle(center_x - half_width_x, center_x + half_width_x, y_lower, y_upper)
            for attempt in range(10_000):
                obstacle_center_x = center_x + float(rng.uniform(-0.12, 0.12))
                obstacle_center_y = float(rng.uniform(-0.50, 0.50))
                half_width = float(rng.uniform(0.11, 0.20))
                half_height = float(rng.uniform(0.18, 0.38))
                obstacle = Rectangle(
                    obstacle_center_x - half_width,
                    obstacle_center_x + half_width,
                    obstacle_center_y - half_height,
                    obstacle_center_y + half_height,
                )
                final_goal_clear = not (
                    stage_index == horizon - 1
                    and obstacle.x_lower - clearance <= goal_x <= obstacle.x_upper + clearance
                    and obstacle.y_lower - clearance <= goal_y <= obstacle.y_upper + clearance
                )
                if final_goal_clear:
                    break
            else:
                raise RuntimeError("could not generate a feasible terminal goal")
            sigma = float(rng.uniform(0.15, 0.23))
            stages.append(Stage(stage_index, box, obstacle, sigma, clearance))
        layouts.append(
            NavigationLayout(
                split=split,
                layout_index=layout_index,
                start_x=0.0,
                start_y=start_y,
                goal_x=goal_x,
                goal_y=goal_y,
                stages=tuple(stages),
            )
        )
    return layouts


def controller_center(
    state: np.ndarray, layout: NavigationLayout, stage: Stage
) -> np.ndarray:
    """One-step receding-horizon goal-directed feasible controller."""

    remaining = len(layout.stages) - stage.stage_index
    desired = np.asarray(
        [
            stage.center_x,
            float(state[1]) + (layout.goal_y - float(state[1])) / remaining,
        ],
        dtype=np.float64,
    )
    desired = np.clip(
        desired,
        [stage.box.x_lower, stage.box.y_lower],
        [stage.box.x_upper, stage.box.y_upper],
    )
    if point_is_feasible(desired, stage):
        return desired

    obstacle = stage.obstacle
    gap = stage.clearance
    candidates = (
        np.asarray([obstacle.x_lower - gap, desired[1]]),
        np.asarray([obstacle.x_upper + gap, desired[1]]),
        np.asarray([desired[0], obstacle.y_lower - gap]),
        np.asarray([desired[0], obstacle.y_upper + gap]),
    )
    feasible_candidates = [
        np.clip(
            candidate,
            [stage.box.x_lower, stage.box.y_lower],
            [stage.box.x_upper, stage.box.y_upper],
        )
        for candidate in candidates
    ]
    feasible_candidates = [
        candidate for candidate in feasible_candidates if point_is_feasible(candidate, stage)
    ]
    if not feasible_candidates:
        raise FloatingPointError("controller found no one-step feasible detour")

    def objective(candidate: np.ndarray) -> float:
        immediate = float(np.linalg.norm(candidate - state))
        terminal = float(np.linalg.norm(layout.goal - candidate))
        lateral_change = abs(float(candidate[1] - state[1]))
        return immediate + terminal + 0.05 * lateral_change

    selected = min(
        feasible_candidates,
        key=lambda candidate: (objective(candidate), float(candidate[1]), float(candidate[0])),
    )
    if not point_is_feasible(selected, stage):
        raise AssertionError("controller center must be exactly feasible")
    return np.asarray(selected, dtype=np.float64)


def episode_seed_sequence(
    base_seed: int, layout_index: int, episode_index: int, stream_index: int
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [int(base_seed), int(layout_index), int(episode_index), int(stream_index)]
    )


def run_episode(
    layout: NavigationLayout,
    episode_index: int,
    episode_seed: int,
    method: str,
) -> dict[str, Any]:
    method_index = METHODS.index(method)
    start_rng = np.random.default_rng(
        episode_seed_sequence(episode_seed, layout.layout_index, episode_index, 10_000)
    )
    method_rng = np.random.default_rng(
        episode_seed_sequence(episode_seed, layout.layout_index, episode_index, method_index)
    )
    state = layout.start.copy()
    state[1] = float(np.clip(state[1] + start_rng.uniform(-0.12, 0.12), -1.2, 1.2))
    initial_state = state.copy()
    path_length = 0.0
    collision_count = 0
    actual_proposals = 0
    primitive_variates = 0
    inverse_calls = 0
    equivalent_proposals = 0.0
    controller_detours = 0
    started = time.perf_counter()
    for stage in layout.stages:
        direct_y = float(state[1]) + (layout.goal_y - float(state[1])) / (
            len(layout.stages) - stage.stage_index
        )
        direct = np.asarray([stage.center_x, direct_y])
        center = controller_center(state, layout, stage)
        controller_detours += int(not np.allclose(center, direct, atol=1e-14, rtol=0.0))
        next_state, cost = sample_kernel(method, stage, center, method_rng)
        collision_count += int(not point_is_feasible(next_state, stage))
        path_length += float(np.linalg.norm(next_state - state))
        state = next_state
        actual_proposals += cost.actual_gaussian_proposals
        primitive_variates += cost.primitive_random_variates
        inverse_calls += cost.inverse_cdf_calls
        equivalent_proposals += cost.naive_rejection_equivalent_proposals
    runtime_seconds = time.perf_counter() - started
    final_distance = float(np.linalg.norm(state - layout.goal))
    success = int(final_distance <= 0.40)
    task_return = 10.0 * success - path_length - 2.0 * final_distance
    return {
        "split": layout.split,
        "layout_index": layout.layout_index,
        "episode_index": episode_index,
        "method": method,
        "start_x": float(initial_state[0]),
        "start_y": float(initial_state[1]),
        "goal_x": layout.goal_x,
        "goal_y": layout.goal_y,
        "final_x": float(state[0]),
        "final_y": float(state[1]),
        "success": success,
        "task_return": float(task_return),
        "path_length": float(path_length),
        "final_distance": final_distance,
        "forbidden_waypoint_count": collision_count,
        "controller_detour_count": controller_detours,
        "actual_gaussian_proposals": actual_proposals,
        "naive_rejection_equivalent_proposals": float(equivalent_proposals),
        "primitive_random_variates": primitive_variates,
        "inverse_cdf_calls": inverse_calls,
        "runtime_seconds": float(runtime_seconds),
        "runtime_ms": float(1000.0 * runtime_seconds),
    }


def deterministic_controller_states(
    layout: NavigationLayout,
) -> list[tuple[Stage, np.ndarray]]:
    state = layout.start.copy()
    result: list[tuple[Stage, np.ndarray]] = []
    for stage in layout.stages:
        center = controller_center(state, layout, stage)
        result.append((stage, center))
        state = center
    return result


def probability_functions(
    stage: Stage, anchor: np.ndarray
) -> dict[str, Callable[[np.ndarray], float]]:
    coarse = coarse_partition(stage)
    refined = refined_partition(stage)
    coarse_weights = partition_weights(stage, anchor, coarse)
    refined_weights = partition_weights(stage, anchor, refined)
    uniform_weights = np.full(len(coarse), 1.0 / len(coarse), dtype=np.float64)
    return {
        "moving_conditioned_gaussian": lambda center: moving_upper_probability(stage, center),
        "fixed_vertical_collision_repair": lambda center: fixed_repair_upper_probability(stage, center),
        "anchor_frozen_corridor_projection": lambda center: frozen_projection_upper_probability(stage, center, coarse_weights),
        "fixed_uniform_conditioned_coarse4": lambda center: frozen_conditioned_upper_probability(stage, center, coarse, uniform_weights),
        "anchor_frozen_conditioned_coarse4": lambda center: frozen_conditioned_upper_probability(stage, center, coarse, coarse_weights),
        "anchor_frozen_conditioned_refined8": lambda center: frozen_conditioned_upper_probability(stage, center, refined, refined_weights),
        "convex_box_conditioned_control": lambda center: convex_box_upper_probability(stage, center),
    }


def nearest_event_boundary(
    probability_function: Callable[[np.ndarray], float],
    anchor: np.ndarray,
    sigma: float,
) -> float | None:
    anchor_value = probability_function(anchor) - 0.5
    if abs(anchor_value) <= 1e-13:
        return 0.0
    grid = np.linspace(float(anchor[1] - 8.0 * sigma), float(anchor[1] + 8.0 * sigma), 321)
    values = [
        probability_function(np.asarray([anchor[0], y_value])) - 0.5
        for y_value in grid
    ]
    roots: list[float] = []
    for left_y, right_y, left_value, right_value in zip(
        grid[:-1], grid[1:], values[:-1], values[1:], strict=True
    ):
        if left_value == 0.0:
            roots.append(float(left_y))
        elif left_value * right_value < 0.0:
            root = brentq(
                lambda y_value: probability_function(
                    np.asarray([anchor[0], y_value], dtype=np.float64)
                )
                - 0.5,
                float(left_y),
                float(right_y),
                xtol=2e-13,
                rtol=2e-13,
            )
            roots.append(float(root))
    if not roots:
        return None
    return float(min(abs(root - float(anchor[1])) for root in roots))


def perturbation_centers(stage: Stage, anchor: np.ndarray) -> Iterable[np.ndarray]:
    for radius_multiple in (0.25, 0.50, 0.75):
        radius = radius_multiple * stage.sigma
        for direction_index in range(16):
            angle = 2.0 * math.pi * direction_index / 16.0
            yield anchor + radius * np.asarray([math.cos(angle), math.sin(angle)])


def audit_state(
    layout: NavigationLayout, stage: Stage, anchor: np.ndarray
) -> list[dict[str, Any]]:
    functions = probability_functions(stage, anchor)
    coarse = coarse_partition(stage)
    refined = refined_partition(stage)
    coarse_weights = partition_weights(stage, anchor, coarse)
    refined_weights = partition_weights(stage, anchor, refined)
    uniform_weights = np.full(len(coarse), 1.0 / len(coarse), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for method in AUDIT_METHODS:
        probability_function = functions[method]
        anchor_probability = probability_function(anchor)
        top_probability = max(anchor_probability, 1.0 - anchor_probability)
        gaussian_formula_radius = float(stage.sigma * ndtri(top_probability))
        boundary_distance = nearest_event_boundary(
            probability_function, anchor, stage.sigma
        )
        radius_attack_found = bool(
            boundary_distance is not None
            and gaussian_formula_radius > boundary_distance + 2e-10
        )
        maximum_full_kl_ratio: float | None = None
        maximum_binary_kl_ratio = 0.0
        maximizing_center: np.ndarray | None = None
        for second in perturbation_centers(stage, anchor):
            denominator = gaussian_kl(stage, anchor, second)
            other_probability = probability_function(second)
            binary_ratio = bernoulli_kl(anchor_probability, other_probability) / denominator
            maximum_binary_kl_ratio = max(maximum_binary_kl_ratio, binary_ratio)
            full_value: float | None
            if method == "moving_conditioned_gaussian":
                full_value = moving_conditioned_kl(stage, anchor, second)
            elif method == "fixed_uniform_conditioned_coarse4":
                full_value = fixed_mixture_kl(
                    stage, anchor, second, coarse, uniform_weights
                )
            elif method == "anchor_frozen_conditioned_coarse4":
                full_value = fixed_mixture_kl(
                    stage, anchor, second, coarse, coarse_weights
                )
            elif method == "anchor_frozen_conditioned_refined8":
                full_value = fixed_mixture_kl(
                    stage, anchor, second, refined, refined_weights
                )
            elif method == "convex_box_conditioned_control":
                full_value = rectangle_conditioned_kl(
                    stage.box, stage.sigma, anchor, second
                )
            else:
                full_value = None
            if full_value is not None:
                ratio = full_value / denominator
                if maximum_full_kl_ratio is None or ratio > maximum_full_kl_ratio:
                    maximum_full_kl_ratio = float(ratio)
                    maximizing_center = second
        rows.append(
            {
                "split": layout.split,
                "layout_index": layout.layout_index,
                "stage_index": stage.stage_index,
                "method": method,
                "controller_center_x": float(anchor[0]),
                "controller_center_y": float(anchor[1]),
                "obstacle_center_x": stage.obstacle_center_x,
                "obstacle_center_y": stage.obstacle_center_y,
                "sigma": stage.sigma,
                "anchor_upper_probability": float(anchor_probability),
                "ambient_gaussian_formula_radius": gaussian_formula_radius,
                "distance_to_exact_event_boundary": boundary_distance,
                "radius_attack_found": radius_attack_found,
                "gaussian_radius_applicable": method in SOUND_METHODS
                or method == "convex_box_conditioned_control",
                "certificate_theoretically_sound": method in SOUND_METHODS
                or method == "convex_box_conditioned_control",
                "maximum_binary_kl_ratio_on_declared_grid": float(maximum_binary_kl_ratio),
                "maximum_full_kl_ratio_on_declared_grid": maximum_full_kl_ratio,
                "maximizing_center_x": None if maximizing_center is None else float(maximizing_center[0]),
                "maximizing_center_y": None if maximizing_center is None else float(maximizing_center[1]),
            }
        )
    return rows


def cluster_bootstrap_interval(
    layout_values: np.ndarray, seed: int, resamples: int
) -> tuple[float, float, float]:
    values = np.asarray(layout_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("cluster bootstrap requires at least two layout values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(np.mean(values)), float(lower), float(upper)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for independent binary units."""

    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    z_value = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    radius = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def summarize_episodes(
    records: list[dict[str, Any]], protocol: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    selected = [record for record in records if record["split"] == split]
    bootstrap_seed = int(protocol["bootstrap_seed"])
    resamples = int(protocol["bootstrap_resamples"])
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        method_records = [record for record in selected if record["method"] == method]
        layout_indices = sorted({int(record["layout_index"]) for record in method_records})
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            layout_values = np.asarray(
                [
                    np.mean(
                        [
                            float(record[metric])
                            for record in method_records
                            if int(record["layout_index"]) == layout_index
                        ]
                    )
                    for layout_index in layout_indices
                ],
                dtype=np.float64,
            )
            estimate, lower, upper = cluster_bootstrap_interval(
                layout_values,
                bootstrap_seed + 1000 * method_index + metric_index,
                resamples,
            )
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "independent_layout_count": len(layout_indices),
                    "episode_count": len(method_records),
                    "cluster_unit": "layout",
                }
            )
    return rows


def summarize_audit(
    audit_rows: list[dict[str, Any]], protocol: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    selected = [row for row in audit_rows if row["split"] == split]
    rows: list[dict[str, Any]] = []
    bootstrap_seed = int(protocol["bootstrap_seed"])
    resamples = int(protocol["bootstrap_resamples"])
    for method_index, method in enumerate(AUDIT_METHODS):
        method_rows = [row for row in selected if row["method"] == method]
        layout_indices = sorted({int(row["layout_index"]) for row in method_rows})
        layout_attack_rates = np.asarray(
            [
                np.mean(
                    [
                        float(row["radius_attack_found"])
                        for row in method_rows
                        if int(row["layout_index"]) == layout_index
                    ]
                )
                for layout_index in layout_indices
            ],
            dtype=np.float64,
        )
        attack_rate, attack_lower, attack_upper = cluster_bootstrap_interval(
            layout_attack_rates,
            bootstrap_seed + 50_000 + method_index,
            resamples,
        )
        layout_any_attack = np.asarray(layout_attack_rates > 0.0, dtype=np.float64)
        layout_all_attack = np.asarray(layout_attack_rates >= 1.0, dtype=np.float64)
        any_attack_lower, any_attack_upper = wilson_interval(
            int(np.sum(layout_any_attack)), len(layout_any_attack)
        )
        all_attack_lower, all_attack_upper = wilson_interval(
            int(np.sum(layout_all_attack)), len(layout_all_attack)
        )
        full_ratios = [
            float(row["maximum_full_kl_ratio_on_declared_grid"])
            for row in method_rows
            if row["maximum_full_kl_ratio_on_declared_grid"] is not None
        ]
        if full_ratios:
            layout_exceedance_rates = np.asarray(
                [
                    np.mean(
                        [
                            float(row["maximum_full_kl_ratio_on_declared_grid"] > 1.0 + 1e-10)
                            for row in method_rows
                            if int(row["layout_index"]) == layout_index
                            and row["maximum_full_kl_ratio_on_declared_grid"] is not None
                        ]
                    )
                    for layout_index in layout_indices
                ],
                dtype=np.float64,
            )
            exceedance_rate, exceedance_lower, exceedance_upper = cluster_bootstrap_interval(
                layout_exceedance_rates,
                bootstrap_seed + 60_000 + method_index,
                resamples,
            )
        else:
            exceedance_rate = exceedance_lower = exceedance_upper = None
        rows.append(
            {
                "split": split,
                "method": method,
                "audited_state_count": len(method_rows),
                "independent_layout_count": len(layout_indices),
                "radius_attack_found_rate": attack_rate,
                "radius_attack_ci95_lower": attack_lower,
                "radius_attack_ci95_upper": attack_upper,
                "layout_any_radius_attack_rate": float(np.mean(layout_any_attack)),
                "layout_any_radius_attack_wilson95_lower": any_attack_lower,
                "layout_any_radius_attack_wilson95_upper": any_attack_upper,
                "layout_all_states_attacked_rate": float(np.mean(layout_all_attack)),
                "layout_all_states_attacked_wilson95_lower": all_attack_lower,
                "layout_all_states_attacked_wilson95_upper": all_attack_upper,
                "maximum_binary_kl_ratio": float(
                    max(row["maximum_binary_kl_ratio_on_declared_grid"] for row in method_rows)
                ),
                "median_maximum_binary_kl_ratio": float(
                    np.median(
                        [row["maximum_binary_kl_ratio_on_declared_grid"] for row in method_rows]
                    )
                ),
                "maximum_full_kl_ratio": None if not full_ratios else float(max(full_ratios)),
                "median_maximum_full_kl_ratio": None if not full_ratios else float(np.median(full_ratios)),
                "fraction_full_kl_ratio_above_one": exceedance_rate,
                "fraction_full_kl_ratio_above_one_ci95_lower": exceedance_lower,
                "fraction_full_kl_ratio_above_one_ci95_upper": exceedance_upper,
                "certificate_theoretically_sound": all(
                    bool(row["certificate_theoretically_sound"])
                    for row in method_rows
                ),
            }
        )
    return rows


def summarize_paired_effects(
    records: list[dict[str, Any]], protocol: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    selected = [record for record in records if record["split"] == split]
    bootstrap_seed = int(protocol["bootstrap_seed"])
    resamples = int(protocol["bootstrap_resamples"])
    metrics = (
        "success",
        "task_return",
        "path_length",
        "final_distance",
        "actual_gaussian_proposals",
        "naive_rejection_equivalent_proposals",
    )
    references = ("native_controller", "moving_conditioned_gaussian")
    layout_indices = sorted({int(record["layout_index"]) for record in selected})
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        for reference_index, reference in enumerate(references):
            if method == reference:
                continue
            for metric_index, metric in enumerate(metrics):
                layout_differences = []
                for layout_index in layout_indices:
                    method_values = [
                        float(record[metric])
                        for record in selected
                        if record["method"] == method
                        and int(record["layout_index"]) == layout_index
                    ]
                    reference_values = [
                        float(record[metric])
                        for record in selected
                        if record["method"] == reference
                        and int(record["layout_index"]) == layout_index
                    ]
                    if len(method_values) != len(reference_values):
                        raise AssertionError("paired method/reference episode counts differ")
                    layout_differences.append(
                        float(np.mean(method_values) - np.mean(reference_values))
                    )
                estimate, lower, upper = cluster_bootstrap_interval(
                    np.asarray(layout_differences),
                    bootstrap_seed
                    + 70_000
                    + 1000 * method_index
                    + 100 * reference_index
                    + metric_index,
                    resamples,
                )
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "reference": reference,
                        "metric": metric,
                        "paired_difference": estimate,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "independent_layout_count": len(layout_indices),
                        "cluster_unit": "layout",
                    }
                )
    return rows


def layout_rows(layouts: Sequence[NavigationLayout]) -> list[dict[str, Any]]:
    return [
        {
            "split": layout.split,
            "layout_index": layout.layout_index,
            "stage_index": stage.stage_index,
            "start_y": layout.start_y,
            "goal_y": layout.goal_y,
            "sigma": stage.sigma,
            "box_x_lower": stage.box.x_lower,
            "box_x_upper": stage.box.x_upper,
            "box_y_lower": stage.box.y_lower,
            "box_y_upper": stage.box.y_upper,
            "obstacle_x_lower": stage.obstacle.x_lower,
            "obstacle_x_upper": stage.obstacle.x_upper,
            "obstacle_y_lower": stage.obstacle.y_lower,
            "obstacle_y_upper": stage.obstacle.y_upper,
        }
        for layout in layouts
        for stage in layout.stages
    ]


def deterministic_digest(
    episode_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    layouts_as_rows: list[dict[str, Any]],
) -> str:
    deterministic_episodes = [
        {
            key: value
            for key, value in row.items()
            if key not in {"runtime_seconds", "runtime_ms"}
        }
        for row in episode_rows
    ]
    serialized = json.dumps(
        {
            "episodes": deterministic_episodes,
            "audit": audit_rows,
            "layouts": layouts_as_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_payload(
    protocol_path: Path,
    splits: Sequence[str],
    freeze_manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    protocol = load_protocol(protocol_path)
    script_path = Path(__file__).resolve()
    source_sha = sha256_path(script_path)
    protocol_sha = sha256_path(protocol_path)
    if "confirmation" in splits:
        if freeze_manifest is None:
            raise ValueError("confirmation requires --freeze-manifest")
        freeze = json.loads(freeze_manifest.read_text(encoding="utf-8"))
        if freeze.get("source_sha256") != source_sha:
            raise ValueError("source hash does not match the frozen confirmation manifest")
        if freeze.get("protocol_sha256") != protocol_sha:
            raise ValueError("protocol hash does not match the frozen confirmation manifest")

    all_layouts: list[NavigationLayout] = []
    episode_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for split in splits:
        split_config = protocol["splits"][split]
        layouts = generate_layouts(
            split,
            int(split_config["layout_seed"]),
            int(split_config["layout_count"]),
            protocol,
        )
        all_layouts.extend(layouts)
        for layout in layouts:
            for episode_index in range(int(split_config["episodes_per_layout"])):
                for method in METHODS:
                    episode_rows.append(
                        run_episode(
                            layout,
                            episode_index,
                            int(split_config["episode_seed"]),
                            method,
                        )
                    )
            for stage, anchor in deterministic_controller_states(layout):
                audit_rows.extend(audit_state(layout, stage, anchor))

    summaries = [
        row
        for split in splits
        for row in summarize_episodes(episode_rows, protocol, split)
    ]
    audit_summaries = [
        row
        for split in splits
        for row in summarize_audit(audit_rows, protocol, split)
    ]
    paired_summaries = [
        row
        for split in splits
        for row in summarize_paired_effects(episode_rows, protocol, split)
    ]
    layouts_as_rows = layout_rows(all_layouts)
    if any(row["forbidden_waypoint_count"] != 0 for row in episode_rows):
        raise AssertionError("a supposedly feasible kernel sampled a forbidden waypoint")
    payload = {
        "experiment": "iclr2027_multistep_discrete_time_reach_avoid_controller_benchmark",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "source_path": ANONYMOUS_SOURCE_PATH.as_posix(),
        "source_sha256": source_sha,
        "splits_executed": list(splits),
        "methods": list(METHODS),
        "controller": protocol["controller"],
        "certificate_scope": (
            "Each audit freezes the current support and anchor weights before varying "
            "the Gaussian action center. Across trajectory states a new per-state "
            "anchor is allowed, but no observation-space certificate is claimed. "
            "The audited states are the seven deterministic no-noise reference-"
            "trajectory controller states per layout, not all stochastic rollout states."
        ),
        "deterministic_result_sha256_excluding_runtime": deterministic_digest(
            episode_rows, audit_rows, layouts_as_rows
        ),
        "episode_summary": summaries,
        "paired_effect_summary": paired_summaries,
        "certificate_audit_summary": audit_summaries,
        "explicit_nonclaims": protocol["explicit_nonclaims"],
        "records": {
            "episodes": episode_rows,
            "certificate_audit": audit_rows,
            "layouts": layouts_as_rows,
        },
    }
    return payload, {
        "episodes": episode_rows,
        "certificate_audit": audit_rows,
        "summary": summaries,
        "paired_summary": paired_summaries,
        "audit_summary": audit_summaries,
        "layouts": layouts_as_rows,
    }


def summary_lookup(
    summary_rows: Sequence[dict[str, Any]], split: str, method: str, metric: str
) -> dict[str, Any]:
    matches = [
        row
        for row in summary_rows
        if row["split"] == split and row["method"] == method and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError("summary lookup is not unique")
    return matches[0]


def markdown_report(payload: dict[str, Any]) -> str:
    summary_rows = payload["episode_summary"]
    audit_rows = payload["certificate_audit_summary"]
    sections = [
        "# ICLR 2027 multi-step reach--avoid controller benchmark\n",
        "This is a frozen-split discrete-time waypoint benchmark. The controller is "
        "a standard one-step receding-horizon goal-directed controller, not a learned "
        "policy. The smoothed variable is its **action center**. No observation-space "
        "or end-to-end policy certificate is claimed.\n",
        "Each sampled next-state waypoint is checked exactly against the current "
        "box-minus-rectangle support. Collision does not include the swept segment "
        "between waypoints.\n",
    ]
    for split in payload["splits_executed"]:
        sections.append(f"## {split.capitalize()} results\n")
        sections.append(
            "| method | success (95% layout CI) | return (95% CI) | path length | final distance | actual proposals | rejection-equivalent proposals | runtime ms |\n"
            "|:--|--:|--:|--:|--:|--:|--:|--:|\n"
        )
        for method in METHODS:
            values = {
                metric: summary_lookup(summary_rows, split, method, metric)
                for metric in (
                    "success",
                    "task_return",
                    "path_length",
                    "final_distance",
                    "actual_gaussian_proposals",
                    "naive_rejection_equivalent_proposals",
                    "runtime_ms",
                )
            }
            sections.append(
                f"| {method} | "
                f"{values['success']['estimate']:.3f} [{values['success']['ci95_lower']:.3f}, {values['success']['ci95_upper']:.3f}] | "
                f"{values['task_return']['estimate']:.3f} [{values['task_return']['ci95_lower']:.3f}, {values['task_return']['ci95_upper']:.3f}] | "
                f"{values['path_length']['estimate']:.3f} | "
                f"{values['final_distance']['estimate']:.3f} | "
                f"{values['actual_gaussian_proposals']['estimate']:.2f} | "
                f"{values['naive_rejection_equivalent_proposals']['estimate']:.2f} | "
                f"{values['runtime_ms']['estimate']:.2f} |\n"
            )
        sections.append("\n### Per-state certificate audit\n")
        sections.append(
            "| method | theorem-backed? | states attacked | layouts with any attack (Wilson 95%) | median max full-KL ratio | max full-KL ratio |\n"
            "|:--|:--:|--:|--:|--:|--:|\n"
        )
        for method in AUDIT_METHODS:
            row = next(
                item for item in audit_rows if item["split"] == split and item["method"] == method
            )
            median_full = row["median_maximum_full_kl_ratio"]
            maximum_full = row["maximum_full_kl_ratio"]
            sections.append(
                f"| {method} | {'yes' if row['certificate_theoretically_sound'] else 'no'} | "
                f"{row['audited_state_count']} ({100.0 * row['radius_attack_found_rate']:.1f}%) | "
                f"{100.0 * row['layout_any_radius_attack_rate']:.1f}% "
                f"[{100.0 * row['layout_any_radius_attack_wilson95_lower']:.1f}, "
                f"{100.0 * row['layout_any_radius_attack_wilson95_upper']:.1f}] | "
                f"{'--' if median_full is None else f'{median_full:.3f}'} | "
                f"{'--' if maximum_full is None else f'{maximum_full:.3f}'} |\n"
            )
        sections.append("\n")
    sections.extend(
        [
            "## Interpretation boundary\n",
            "The moving-law Gaussian radius is deliberately audited although it is "
            "not valid: occupancy weights move with the action center. The five "
            "fixed alternatives are sound from data processing or the convex "
            "fixed-weight theorem; a zero attack-found rate is not used as proof. "
            "The convex-box row is a negative-control audit and is not deployed as "
            "an obstacle-avoidance controller.\n\n",
            "The anchor-occupancy coarse4 and refined8 laws both exactly partition "
            "the same feasible support and match whole-set conditioning at their anchor. "
            "Their away-from-anchor "
            "laws differ, so their utility/cost comparison is a partition-sensitivity "
            "result rather than a claim that one partition is canonical.\n\n",
            "Wall-clock timing is secondary and implementation-specific. Primitive "
            "random-variate counts, inverse-CDF calls, actual moving rejection draws, "
            "and exact rejection-equivalent expectations are reported separately.\n\n",
            f"Deterministic result SHA-256 excluding wall clock: `{payload['deterministic_result_sha256_excluding_runtime']}`.\n",
        ]
    )
    return "".join(sections)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_stem: Path,
    payload: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_stem.with_suffix(".json"),
        "markdown": output_stem.with_suffix(".md"),
        "episode_csv": output_stem.parent / f"{output_stem.name}_episode_records.csv",
        "certificate_csv": output_stem.parent / f"{output_stem.name}_certificate_audit.csv",
        "summary_csv": output_stem.parent / f"{output_stem.name}_summary.csv",
        "paired_summary_csv": output_stem.parent / f"{output_stem.name}_paired_summary.csv",
        "audit_summary_csv": output_stem.parent / f"{output_stem.name}_audit_summary.csv",
        "layout_csv": output_stem.parent / f"{output_stem.name}_layouts.csv",
    }
    payload_with_artifacts = {
        **payload,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(
        json.dumps(payload_with_artifacts, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(markdown_report(payload_with_artifacts), encoding="utf-8")
    write_csv(paths["episode_csv"], tables["episodes"])
    write_csv(paths["certificate_csv"], tables["certificate_audit"])
    write_csv(paths["summary_csv"], tables["summary"])
    write_csv(paths["paired_summary_csv"], tables["paired_summary"])
    write_csv(paths["audit_summary_csv"], tables["audit_summary"])
    write_csv(paths["layout_csv"], tables["layouts"])
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("development", "validation", "confirmation"),
        default=("development", "validation"),
    )
    parser.add_argument("--freeze-manifest", type=Path)
    args = parser.parse_args()
    payload, tables = build_payload(
        args.protocol, args.splits, freeze_manifest=args.freeze_manifest
    )
    artifacts = write_outputs(args.output_stem, payload, tables)
    print(
        json.dumps(
            {
                "splits": args.splits,
                "deterministic_result_sha256_excluding_runtime": payload[
                    "deterministic_result_sha256_excluding_runtime"
                ],
                "artifacts": artifacts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
