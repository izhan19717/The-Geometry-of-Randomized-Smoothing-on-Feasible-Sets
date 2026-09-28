"""Seeded analytic stress ensemble for Gaussian conditioning on box unions.

The generator in this module is intentionally narrow.  It samples finite
unions of product boxes from a declared synthetic distribution; it is not a
model of naturally occurring feasible sets.  Box interiors are disjoint by
construction because every mode lies in a separate first-coordinate cell.
All probability and KL calculations use products of one-dimensional Gaussian
interval integrals through :mod:`feasible_robustness.kl_exact`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .convex_patch import (
    anchored_tempered_weights,
    box_conditioned_kl,
    categorical_kl,
    component_statistics,
    mixture_weights,
)
from .kl_exact import AxisAlignedBox, RectangleUnion, truncated_gaussian_kl_rectangle_union
from .kl_exact import rectangle_union_covariance


FloatArray = NDArray[np.float64]
ENSEMBLE_VERSION = "random-product-box-v2"
DEFAULT_ROOT_SEED = 20_260_713
DEFAULT_DIRECTION_ROOT_SEED = 73_102_062
DEFAULT_BOOTSTRAP_SEED = 91_726_031


@dataclass(frozen=True)
class RandomBoxGeometry:
    """One reproducible box union and one independently seeded direction."""

    region: RectangleUnion
    direction: FloatArray
    geometry_family: str
    dimension: int
    mode_count: int
    replicate: int
    volume_ratio_cap: float
    geometry_seed: int
    direction_seed: int


def _derived_seed(
    root_seed: int,
    dimension: int,
    mode_count: int,
    replicate: int,
    stream: int = 0,
) -> int:
    if root_seed < 0 or replicate < 0:
        raise ValueError("root_seed and replicate must be nonnegative")
    sequence = np.random.SeedSequence(
        [int(root_seed), int(dimension), int(mode_count), int(replicate), int(stream)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _box_volume(box: AxisAlignedBox) -> float:
    return float(np.prod(box.upper - box.lower))


def box_volumes(region: RectangleUnion) -> FloatArray:
    """Return Lebesgue volumes for the full-dimensional product boxes."""

    return np.asarray([_box_volume(box) for box in region.boxes], dtype=np.float64)


def exact_union_diameter(region: RectangleUnion) -> float:
    """Return the exact Euclidean diameter of a finite product-box union."""

    maximum_squared = 0.0
    for left in region.boxes:
        for right in region.boxes:
            coordinate_separations = np.maximum(
                np.abs(left.lower - right.upper),
                np.abs(left.upper - right.lower),
            )
            maximum_squared = max(
                maximum_squared,
                float(np.dot(coordinate_separations, coordinate_separations)),
            )
    return math.sqrt(maximum_squared)


def make_random_box_geometry(
    dimension: int,
    mode_count: int,
    replicate: int,
    volume_ratio_cap: float,
    *,
    geometry_family: str = "diffuse",
    root_seed: int = DEFAULT_ROOT_SEED,
    direction_root_seed: int = DEFAULT_DIRECTION_ROOT_SEED,
    first_axis_fill_fraction: float = 0.7,
    auxiliary_width: float = 1.2,
    auxiliary_center_limit: float = 0.29,
) -> RandomBoxGeometry:
    """Generate one paired random geometry under a fixed support envelope.

    The first coordinate is partitioned into ``mode_count`` equal cells in
    ``[-1,1]``.  Every box occupies one cell, so interiors cannot overlap.  The
    first and last boxes touch the enclosing endpoints and the two central
    boxes meet at zero.  Auxiliary interval centers are sampled in a fixed
    range.  The two central intervals contain zero in every coordinate, which
    makes the declared anchor feasible.

    ``geometry_family='diffuse'`` draws auxiliary interval centers
    independently.  ``geometry_family='coherent'`` draws a random rank-one
    sign orientation, random mode loadings, and small independent jitter.  The
    latter is a structured multimodality stress family, not a claim about
    typical feasible sets.

    ``volume_ratio_cap`` changes only the width of the first auxiliary
    coordinate for noncentral modes.  Random centers and all other widths are
    paired across imbalance levels, so this factor does not silently redraw a
    new geometry.  The realized ratio is at most the supplied cap, not exactly
    equal to it for every finite sample.
    """

    if dimension < 2:
        raise ValueError("the random ensemble requires dimension at least two")
    if mode_count < 4 or mode_count % 2:
        raise ValueError("mode_count must be an even integer of at least four")
    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    if not math.isfinite(volume_ratio_cap) or volume_ratio_cap < 1.0:
        raise ValueError("volume_ratio_cap must be finite and at least one")
    if geometry_family not in {"diffuse", "coherent"}:
        raise ValueError("geometry_family must be 'diffuse' or 'coherent'")
    if not 0.0 < first_axis_fill_fraction < 1.0:
        raise ValueError("first_axis_fill_fraction must lie in (0,1)")
    if not 0.0 < auxiliary_width < 2.0:
        raise ValueError("auxiliary_width must lie in (0,2)")
    if auxiliary_center_limit < 0.0:
        raise ValueError("auxiliary_center_limit must be nonnegative")
    if auxiliary_center_limit + auxiliary_width / 2.0 > 1.0:
        raise ValueError("auxiliary intervals must fit inside the fixed cube")

    family_stream = 0 if geometry_family == "diffuse" else 1
    geometry_seed = _derived_seed(
        root_seed,
        dimension,
        mode_count,
        replicate,
        family_stream,
    )
    direction_seed = _derived_seed(
        direction_root_seed,
        dimension,
        mode_count,
        replicate,
    )
    rng = np.random.default_rng(geometry_seed)
    direction_rng = np.random.default_rng(direction_seed)

    pitch = 2.0 / mode_count
    first_width = first_axis_fill_fraction * pitch
    central_left = mode_count // 2 - 1
    central_right = mode_count // 2

    # Draw all latent variables even when the cap is one.  Consequently the
    # balanced and imbalanced configurations are an explicitly paired design.
    first_locations = rng.uniform(size=mode_count)
    if geometry_family == "diffuse":
        auxiliary_centers = rng.uniform(
            -auxiliary_center_limit,
            auxiliary_center_limit,
            size=(mode_count, dimension - 1),
        )
    else:
        orientation = rng.choice((-1.0, 1.0), size=dimension - 1)
        loadings = rng.uniform(-1.0, 1.0, size=mode_count)
        coherent_amplitude = max(0.0, auxiliary_center_limit - 0.04)
        jitter = rng.uniform(-0.04, 0.04, size=(mode_count, dimension - 1))
        auxiliary_centers = (
            coherent_amplitude * loadings[:, None] * orientation[None, :] + jitter
        )
    log_shrink_latent = rng.uniform(size=mode_count)
    auxiliary_centers[[central_left, central_right], :] = 0.0
    shrink = np.exp(-math.log(volume_ratio_cap) * log_shrink_latent)
    shrink[[central_left, central_right]] = 1.0

    boxes: list[AxisAlignedBox] = []
    for mode in range(mode_count):
        cell_lower = -1.0 + mode * pitch
        cell_upper = cell_lower + pitch
        if mode == 0:
            first_lower = -1.0
        elif mode == mode_count - 1:
            first_lower = 1.0 - first_width
        elif mode == central_left:
            first_lower = -first_width
        elif mode == central_right:
            first_lower = 0.0
        else:
            first_lower = cell_lower + first_locations[mode] * (pitch - first_width)
        first_upper = first_lower + first_width
        if first_lower < cell_lower - 1e-14 or first_upper > cell_upper + 1e-14:
            raise AssertionError("first-axis interval escaped its assigned cell")

        lower = np.empty(dimension, dtype=np.float64)
        upper = np.empty(dimension, dtype=np.float64)
        lower[0] = first_lower
        upper[0] = first_upper
        for coordinate in range(1, dimension):
            width = auxiliary_width
            if coordinate == 1:
                width *= float(shrink[mode])
            center = float(auxiliary_centers[mode, coordinate - 1])
            lower[coordinate] = center - width / 2.0
            upper[coordinate] = center + width / 2.0
        boxes.append(AxisAlignedBox.from_bounds(lower, upper))

    raw_direction = direction_rng.normal(size=dimension)
    raw_direction[0] = abs(float(raw_direction[0]))
    norm = float(np.linalg.norm(raw_direction))
    if norm <= 0.0:
        raise FloatingPointError("sampled a zero direction")
    direction = np.asarray(raw_direction / norm, dtype=np.float64)
    region = RectangleUnion(tuple(boxes))
    if not region.contains(np.zeros(dimension, dtype=np.float64)):
        raise AssertionError("the constructed anchor must be feasible")
    return RandomBoxGeometry(
        region=region,
        direction=direction,
        geometry_family=geometry_family,
        dimension=dimension,
        mode_count=mode_count,
        replicate=replicate,
        volume_ratio_cap=float(volume_ratio_cap),
        geometry_seed=geometry_seed,
        direction_seed=direction_seed,
    )


def spectral_stress_direction(region: RectangleUnion, sigma: float) -> FloatArray:
    """Return a top-covariance direction for a declared local stress test.

    This direction is geometry- and noise-adaptive.  It must therefore be
    reported separately from an independent random direction; it is an attack
    search diagnostic, not an i.i.d. direction draw.
    """

    covariance = rectangle_union_covariance(
        region,
        np.zeros(region.dimension, dtype=np.float64),
        sigma,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("conditioned covariance eigensystem is not finite")
    direction = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    if direction[0] < 0.0:
        direction = -direction
    direction /= float(np.linalg.norm(direction))
    return direction


def _entropy_effective_count(probabilities: FloatArray) -> float:
    return float(math.exp(-float(np.dot(probabilities, np.log(probabilities)))))


def _method_gates(
    occupancy_a: FloatArray,
    occupancy_b: FloatArray,
    anchored_taus: Iterable[float],
) -> list[tuple[str, float | None, FloatArray, FloatArray]]:
    uniform = np.full_like(occupancy_a, 1.0 / occupancy_a.size)
    methods: list[tuple[str, float | None, FloatArray, FloatArray]] = [
        ("uniform_fixed", None, uniform, uniform),
        ("anchor_fixed", 0.0, occupancy_a, occupancy_a),
    ]
    for tau in anchored_taus:
        value = float(tau)
        if not 0.0 < value < 1.0:
            raise ValueError("anchored temperatures must lie strictly in (0,1)")
        methods.append(
            (
                f"anchored_tau_{value:g}",
                value,
                occupancy_a,
                anchored_tempered_weights(occupancy_a, occupancy_b, value),
            )
        )
    methods.append(("whole_conditioning", 1.0, occupancy_a, occupancy_b))
    return methods


def evaluate_random_box_pair(
    geometry: RandomBoxGeometry,
    sigma: float,
    separation_ratio: float,
    *,
    anchored_taus: Iterable[float] = (0.25, 0.5, 0.75),
    direction: FloatArray | None = None,
    direction_protocol: str = "iid",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate exact KL decompositions for one geometry and center pair."""

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(separation_ratio) or separation_ratio <= 0.0:
        raise ValueError("separation_ratio must be finite and positive")
    if direction_protocol not in {"iid", "spectral_stress"}:
        raise ValueError("unknown direction_protocol")
    if direction is None:
        selected_direction = geometry.direction
    else:
        selected_direction = np.asarray(direction, dtype=np.float64)
        if selected_direction.shape != (geometry.dimension,):
            raise ValueError("direction has the wrong dimension")
        direction_norm = float(np.linalg.norm(selected_direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError("direction must be finite and nonzero")
        selected_direction = selected_direction / direction_norm
    if selected_direction[0] < -1e-14:
        raise ValueError("direction must have a nonnegative first coordinate")

    anchor = np.zeros(geometry.dimension, dtype=np.float64)
    displacement = sigma * separation_ratio
    center_b = displacement * selected_direction
    if not geometry.region.contains(anchor) or not geometry.region.contains(center_b):
        raise ValueError(
            "the requested displacement leaves the center-bearing mode; "
            "reduce sigma or separation_ratio"
        )

    stats_a = component_statistics(geometry.region, anchor, sigma)
    stats_b = component_statistics(geometry.region, center_b, sigma)
    occupancy_a = mixture_weights(stats_a)
    occupancy_b = mixture_weights(stats_b)
    component_kls = np.asarray(
        [
            box_conditioned_kl(box, anchor, center_b, sigma, left, right)
            for box, left, right in zip(
                geometry.region.boxes,
                stats_a,
                stats_b,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    if np.any(component_kls < -1e-9):
        raise FloatingPointError("a component KL became materially negative")
    component_kls = np.maximum(component_kls, 0.0)

    gaussian_kl = separation_ratio**2 / 2.0
    diameter = exact_union_diameter(geometry.region)
    volumes = box_volumes(geometry.region)
    volume_ratio = float(np.max(volumes) / np.min(volumes))
    whole_independent = float(
        truncated_gaussian_kl_rectangle_union(
            anchor,
            center_b,
            sigma,
            geometry.region,
        ).kl
    )
    whole_gate = categorical_kl(occupancy_a, occupancy_b)
    whole_within = float(np.dot(occupancy_a, component_kls))
    chain_error = abs(whole_independent - whole_gate - whole_within)

    shared = {
        "ensemble_version": ENSEMBLE_VERSION,
        "geometry_family": geometry.geometry_family,
        "direction_protocol": direction_protocol,
        "dimension": geometry.dimension,
        "mode_count": geometry.mode_count,
        "replicate": geometry.replicate,
        "geometry_seed": geometry.geometry_seed,
        "direction_seed": geometry.direction_seed,
        "volume_ratio_cap": geometry.volume_ratio_cap,
        "realized_volume_ratio": volume_ratio,
        "sigma": float(sigma),
        "separation_ratio": float(separation_ratio),
        "displacement_norm": displacement,
        "direction_first_coordinate": float(selected_direction[0]),
        "union_diameter": diameter,
        "bounding_cube_diameter": 2.0 * math.sqrt(geometry.dimension),
        "gaussian_kl": gaussian_kl,
        "whole_independent_kl": whole_independent,
        "whole_chain_abs_error": chain_error,
        "occupancy_effective_modes_anchor": _entropy_effective_count(occupancy_a),
        "occupancy_effective_modes_b": _entropy_effective_count(occupancy_b),
        "occupancy_max_min_ratio_anchor": float(np.max(occupancy_a) / np.min(occupancy_a)),
        "occupancy_max_min_ratio_b": float(np.max(occupancy_b) / np.min(occupancy_b)),
        "union_gaussian_mass_anchor": float(sum(item.normalizer for item in stats_a)),
        "union_gaussian_mass_b": float(sum(item.normalizer for item in stats_b)),
    }

    rows: list[dict[str, Any]] = []
    for method, tau, gate_a, gate_b in _method_gates(
        occupancy_a,
        occupancy_b,
        anchored_taus,
    ):
        gate_kl = categorical_kl(gate_a, gate_b)
        within_kl = float(np.dot(gate_a, component_kls))
        total_kl = gate_kl + within_kl
        tv_b = 0.5 * float(np.sum(np.abs(gate_b - occupancy_b)))
        if method in {"uniform_fixed", "anchor_fixed"}:
            theorem_bound = gaussian_kl
        elif method.startswith("anchored_tau_"):
            assert tau is not None
            theorem_bound = (
                1.0 / (2.0 * sigma**2)
                + tau**2 * diameter**2 / (8.0 * sigma**4)
            ) * displacement**2
        else:
            theorem_bound = math.nan
        rows.append(
            {
                **shared,
                "method": method,
                "tau": tau,
                "within_kl": within_kl,
                "gate_kl": gate_kl,
                "total_kl": total_kl,
                "within_ratio": within_kl / gaussian_kl,
                "gate_ratio": gate_kl / gaussian_kl,
                "total_ratio": total_kl / gaussian_kl,
                "numeric_gaussian_contraction": bool(total_kl <= gaussian_kl + 1e-10),
                "theorem_bound": theorem_bound,
                "theorem_bound_residual": (
                    total_kl - theorem_bound if math.isfinite(theorem_bound) else math.nan
                ),
                "nominal_tv_from_whole": 0.5
                * float(np.sum(np.abs(gate_a - occupancy_a))),
                "tv_from_whole_at_b": tv_b,
                "tagged_untagged_abs_gap": 0.0,
            }
        )

    summary = {
        **shared,
        "direction": selected_direction.tolist(),
        "anchor": anchor.tolist(),
        "center_b": center_b.tolist(),
        "occupancy_anchor": occupancy_a.tolist(),
        "occupancy_b": occupancy_b.tolist(),
        "component_kls": component_kls.tolist(),
        "box_volumes": volumes.tolist(),
        "region": geometry.region.to_json_dict(),
    }
    return rows, summary


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials with trials positive")
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_median_interval(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    resamples: int = 2_000,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for the median."""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a nonempty finite one-dimensional sequence")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    medians = np.median(array[indices], axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)
