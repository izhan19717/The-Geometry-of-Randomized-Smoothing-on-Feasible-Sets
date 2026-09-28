"""Run the seeded analytic random product-box union stress ensemble.

The experiment broadens the deterministic mode-scaling audit with a declared
distribution over a.e.-disjoint product-box geometries.  It separates
independent random directions from geometry-adaptive spectral stress
directions.  It remains a synthetic theorem stress test, not a systems or
learned-controller benchmark.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
PRINT_PLOT_STYLE = {
    "font.size": 12.3,
    "axes.titlesize": 13.1,
    "axes.labelsize": 12.2,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.5,
    "legend.framealpha": 0.94,
}
plt.rcParams.update(PRINT_PLOT_STYLE)

from feasible_robustness.random_box_ensemble import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_DIRECTION_ROOT_SEED,
    DEFAULT_ROOT_SEED,
    ENSEMBLE_VERSION,
    bootstrap_median_interval,
    evaluate_random_box_pair,
    make_random_box_geometry,
    spectral_stress_direction,
    wilson_interval,
)


DEFAULT_DIMENSIONS = (2, 5, 10, 20)
DEFAULT_MODE_COUNTS = (4, 8, 16)
DEFAULT_SIGMAS = (0.25, 0.5)
DEFAULT_SEPARATION_RATIOS = (0.05, 0.15)
DEFAULT_VOLUME_RATIO_CAPS = (1.0, 4.0)
DEFAULT_GEOMETRY_FAMILIES = ("diffuse", "coherent")
DEFAULT_DIRECTION_PROTOCOLS = ("iid", "spectral_stress")
DEFAULT_ANCHORED_TAUS = (0.25, 0.5, 0.75)
DEFAULT_REPLICATES = 16


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(dict.fromkeys(cast(item) for item in raw.split(",") if item.strip()))
    if not values:
        raise ValueError("the list must be nonempty")
    return values


def run_ensemble(
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    mode_counts: Iterable[int] = DEFAULT_MODE_COUNTS,
    sigmas: Iterable[float] = DEFAULT_SIGMAS,
    separation_ratios: Iterable[float] = DEFAULT_SEPARATION_RATIOS,
    volume_ratio_caps: Iterable[float] = DEFAULT_VOLUME_RATIO_CAPS,
    *,
    replicates: int = DEFAULT_REPLICATES,
    anchored_taus: Iterable[float] = DEFAULT_ANCHORED_TAUS,
    geometry_families: Iterable[str] = DEFAULT_GEOMETRY_FAMILIES,
    direction_protocols: Iterable[str] = DEFAULT_DIRECTION_PROTOCOLS,
    root_seed: int = DEFAULT_ROOT_SEED,
    direction_root_seed: int = DEFAULT_DIRECTION_ROOT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate the paired factorial ensemble in deterministic loop order."""

    dimensions = tuple(int(value) for value in dimensions)
    mode_counts = tuple(int(value) for value in mode_counts)
    sigmas = tuple(float(value) for value in sigmas)
    separation_ratios = tuple(float(value) for value in separation_ratios)
    volume_ratio_caps = tuple(float(value) for value in volume_ratio_caps)
    anchored_taus = tuple(float(value) for value in anchored_taus)
    geometry_families = tuple(str(value) for value in geometry_families)
    direction_protocols = tuple(str(value) for value in direction_protocols)
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for dimension in dimensions:
        for mode_count in mode_counts:
            for replicate in range(replicates):
                for geometry_family in geometry_families:
                    for volume_ratio_cap in volume_ratio_caps:
                        geometry = make_random_box_geometry(
                            dimension,
                            mode_count,
                            replicate,
                            volume_ratio_cap,
                            geometry_family=geometry_family,
                            root_seed=root_seed,
                            direction_root_seed=direction_root_seed,
                        )
                        geometry_record = {
                            "ensemble_version": ENSEMBLE_VERSION,
                            "geometry_family": geometry.geometry_family,
                            "dimension": geometry.dimension,
                            "mode_count": geometry.mode_count,
                            "replicate": geometry.replicate,
                            "volume_ratio_cap": geometry.volume_ratio_cap,
                            "geometry_seed": geometry.geometry_seed,
                            "direction_seed": geometry.direction_seed,
                            "iid_direction": geometry.direction.tolist(),
                            "spectral_directions_by_sigma": {},
                            "region": geometry.region.to_json_dict(),
                        }
                        for sigma in sigmas:
                            spectral = spectral_stress_direction(geometry.region, sigma)
                            geometry_record["spectral_directions_by_sigma"][str(sigma)] = (
                                spectral.tolist()
                            )
                            for direction_protocol in direction_protocols:
                                if direction_protocol == "iid":
                                    direction = geometry.direction
                                elif direction_protocol == "spectral_stress":
                                    direction = spectral
                                else:
                                    raise ValueError(
                                        f"unknown direction protocol: {direction_protocol}"
                                    )
                                for separation_ratio in separation_ratios:
                                    pair_rows, _ = evaluate_random_box_pair(
                                        geometry,
                                        sigma,
                                        separation_ratio,
                                        anchored_taus=anchored_taus,
                                        direction=direction,
                                        direction_protocol=direction_protocol,
                                    )
                                    rows.extend(pair_rows)
                        provenance.append(geometry_record)
    return rows, provenance


STRATUM_KEYS = (
    "geometry_family",
    "direction_protocol",
    "dimension",
    "mode_count",
    "sigma",
    "separation_ratio",
    "volume_ratio_cap",
)


def summarize_whole_strata(
    rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = 2_000,
) -> list[dict[str, Any]]:
    """Summarize independent replicate distributions within fixed strata."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["method"] == "whole_conditioning":
            groups[tuple(row[key] for key in STRATUM_KEYS)].append(row)
    rng = np.random.default_rng(bootstrap_seed)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        ratios = np.asarray([float(row["total_ratio"]) for row in group])
        successes = int(np.sum(ratios > 1.0 + 1e-9))
        wilson_low, wilson_high = wilson_interval(successes, len(group))
        median_low, median_high = bootstrap_median_interval(
            ratios,
            rng=rng,
            resamples=bootstrap_resamples,
        )
        summary = dict(zip(STRATUM_KEYS, key, strict=True))
        summary.update(
            {
                "replicates": len(group),
                "violation_count": successes,
                "violation_rate": successes / len(group),
                "violation_wilson_95_low": wilson_low,
                "violation_wilson_95_high": wilson_high,
                "ratio_median": float(np.median(ratios)),
                "ratio_bootstrap_median_95_low": median_low,
                "ratio_bootstrap_median_95_high": median_high,
                "ratio_q10": float(np.quantile(ratios, 0.1)),
                "ratio_q90": float(np.quantile(ratios, 0.9)),
                "ratio_max": float(np.max(ratios)),
                "gate_ratio_median": float(
                    np.median([float(row["gate_ratio"]) for row in group])
                ),
                "within_ratio_median": float(
                    np.median([float(row["within_ratio"]) for row in group])
                ),
            }
        )
        summaries.append(summary)
    return summaries


def paired_imbalance_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cap-4 minus cap-1 whole-ratio differences for matched pairs."""

    whole = [row for row in rows if row["method"] == "whole_conditioning"]
    levels = sorted({float(row["volume_ratio_cap"]) for row in whole})
    if levels != [1.0, 4.0]:
        return []
    keys = (
        "geometry_family",
        "direction_protocol",
        "dimension",
        "mode_count",
        "replicate",
        "sigma",
        "separation_ratio",
        "geometry_seed",
        "direction_seed",
    )
    lookup = {
        (tuple(row[key] for key in keys), float(row["volume_ratio_cap"])): row
        for row in whole
    }
    differences: list[dict[str, Any]] = []
    identities = sorted({identity for identity, _ in lookup})
    for identity in identities:
        balanced = lookup[(identity, 1.0)]
        imbalanced = lookup[(identity, 4.0)]
        differences.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "ratio_cap_1": float(balanced["total_ratio"]),
                "ratio_cap_4": float(imbalanced["total_ratio"]),
                "paired_ratio_difference": float(imbalanced["total_ratio"])
                - float(balanced["total_ratio"]),
                "realized_volume_ratio_cap_4": float(
                    imbalanced["realized_volume_ratio"]
                ),
            }
        )
    return differences


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path_stem: Path, rows: list[dict[str, Any]]) -> None:
    """Write a print-safe four-panel summary from the supplied frozen rows."""

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    # Keep typography deterministic when this helper is imported after another
    # plotting module that has changed Matplotlib's process-global defaults.
    plt.rcParams.update(PRINT_PLOT_STYLE)
    whole = [row for row in rows if row["method"] == "whole_conditioning"]
    dimensions = sorted({int(row["dimension"]) for row in whole})
    mode_counts = sorted({int(row["mode_count"]) for row in whole})
    methods = [
        "anchor_fixed",
        "anchored_tau_0.25",
        "anchored_tau_0.5",
        "anchored_tau_0.75",
        "whole_conditioning",
    ]
    available = {str(row["method"]) for row in rows}
    methods = [method for method in methods if method in available]

    # The compact canvas avoids the severe font reduction caused by the old
    # 11.5-inch landscape figure at its 0.92-linewidth manuscript setting.
    fig = plt.figure(figsize=(8.4, 8.25), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 0.18, 1.0))
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    axes[0, 1] = fig.add_subplot(grid[0, 1])
    axes[1, 0] = fig.add_subplot(grid[2, 0])
    axes[1, 1] = fig.add_subplot(grid[2, 1])
    box_legend_axis = fig.add_subplot(grid[1, :])
    box_legend_axis.axis("off")

    positions = np.arange(len(dimensions), dtype=np.float64)
    iid_by_dimension = [
        [
            float(row["total_ratio"])
            for row in whole
            if int(row["dimension"]) == dimension and row["direction_protocol"] == "iid"
        ]
        for dimension in dimensions
    ]
    spectral_by_dimension = [
        [
            float(row["total_ratio"])
            for row in whole
            if int(row["dimension"]) == dimension
            and row["direction_protocol"] == "spectral_stress"
        ]
        for dimension in dimensions
    ]
    iid_plot = axes[0, 0].boxplot(
        iid_by_dimension,
        positions=positions - 0.17,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        boxprops={"linewidth": 1.15},
        whiskerprops={"linewidth": 1.15},
        capprops={"linewidth": 1.15},
        medianprops={"color": "#1b1b1b", "linewidth": 1.5},
    )
    spectral_plot = axes[0, 0].boxplot(
        spectral_by_dimension,
        positions=positions + 0.17,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        boxprops={"linewidth": 1.15},
        whiskerprops={"linewidth": 1.15},
        capprops={"linewidth": 1.15},
        medianprops={"color": "#1b1b1b", "linewidth": 1.5},
    )
    for patch in iid_plot["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_hatch("///")
    for patch in spectral_plot["boxes"]:
        patch.set_facecolor("#E69F00")
        patch.set_hatch("\\\\")
    violation_handle = None
    for index, dimension in enumerate(dimensions):
        violations = [
            float(row["total_ratio"])
            for row in whole
            if int(row["dimension"]) == dimension
            and row["direction_protocol"] == "spectral_stress"
            and float(row["total_ratio"]) > 1.0 + 1e-9
        ]
        if violations:
            violation_handle = axes[0, 0].scatter(
                np.full(len(violations), positions[index] + 0.17),
                violations,
                marker="x",
                s=34,
                linewidths=1.4,
                color="#D55E00",
                zorder=4,
            )
    axes[0, 0].set_xticks(positions, labels=dimensions)
    legend_handles = [iid_plot["boxes"][0], spectral_plot["boxes"][0]]
    legend_labels = ["independent direction", "spectral stress direction"]
    if violation_handle is not None:
        legend_handles.append(violation_handle)
        legend_labels.append("spectral ratio > 1")
    box_legend_axis.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=3,
        handlelength=2.0,
        columnspacing=1.2,
    )
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.3)
    axes[0, 0].set_xlabel("ambient dimension")
    axes[0, 0].set_ylabel("KL ratio (whole / Gaussian)")
    axes[0, 0].set_title("A. Whole-conditioning ratio")

    heatmap = np.zeros((len(dimensions), len(mode_counts)), dtype=np.float64)
    for i, dimension in enumerate(dimensions):
        for j, mode_count in enumerate(mode_counts):
            group = [
                row
                for row in whole
                if int(row["dimension"]) == dimension
                and int(row["mode_count"]) == mode_count
                and row["direction_protocol"] == "spectral_stress"
            ]
            heatmap[i, j] = np.mean(
                [float(row["total_ratio"]) > 1.0 + 1e-9 for row in group]
            )
    # Keep a fixed minimum visual range so zero and the small observed rates
    # are distinguishable without silently normalizing the maximum to one.
    heatmap_vmax = max(0.05, float(np.max(heatmap)))
    image = axes[0, 1].imshow(
        heatmap,
        vmin=0.0,
        vmax=heatmap_vmax,
        cmap="magma",
        aspect="auto",
    )
    axes[0, 1].set_xticks(range(len(mode_counts)), labels=mode_counts)
    axes[0, 1].set_yticks(range(len(dimensions)), labels=dimensions)
    axes[0, 1].set_xlabel("number of modes")
    axes[0, 1].set_ylabel("ambient dimension")
    axes[0, 1].set_title("B. Spectral-stress exceedance")
    for i in range(len(dimensions)):
        for j in range(len(mode_counts)):
            axes[0, 1].text(
                j,
                i,
                f"{heatmap[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if heatmap[i, j] < 0.6 * heatmap_vmax else "black",
                fontsize=11.5,
            )
    colorbar = fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)
    colorbar.set_label("fraction with KL ratio > 1")
    colorbar.ax.tick_params(labelsize=11.3)

    median_ratio = []
    median_tv = []
    labels = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        median_ratio.append(float(np.median([float(row["total_ratio"]) for row in group])))
        median_tv.append(
            float(np.median([float(row["tv_from_whole_at_b"]) for row in group]))
        )
        labels.append(
            {
                "anchor_fixed": "fixed anchor",
                "anchored_tau_0.25": r"$\tau=0.25$",
                "anchored_tau_0.5": r"$\tau=0.50$",
                "anchored_tau_0.75": r"$\tau=0.75$",
                "whole_conditioning": "whole",
            }.get(method, method.replace("_", " "))
        )
    axes[1, 0].plot(
        median_tv,
        median_ratio,
        color="#4d4d4d",
        linewidth=1.8,
        zorder=2,
    )
    point_colors = ("#CC79A7", "#E69F00", "#009E73", "#D55E00", "#0072B2")
    point_markers = ("P", "D", "^", "s", "o")
    point_handles = []
    for x, y, label, color, marker in zip(
        median_tv,
        median_ratio,
        labels,
        point_colors,
        point_markers,
        strict=True,
    ):
        point_handles.append(
            axes[1, 0].scatter(
                [x],
                [y],
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.7,
                s=58,
                label=label,
                zorder=3,
            )
        )
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.3)
    axes[1, 0].set_xlabel("median TV from whole conditioning")
    axes[1, 0].set_ylabel("median KL / Gaussian KL")
    axes[1, 0].set_title("C. Anchored-weight trade-off")
    axes[1, 0].set_xlim(-0.0005, max(median_tv) * 1.07)
    axes[1, 0].set_ylim(0.32, 1.03)
    axes[1, 0].set_xticks(np.linspace(0.0, max(median_tv), 4))
    axes[1, 0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    axes[1, 0].legend(
        point_handles[::-1],
        labels[::-1],
        loc="upper right",
        handlelength=1.0,
        handletextpad=0.5,
        borderaxespad=0.5,
    )

    differences = paired_imbalance_differences(rows)
    differences_by_dimension = [
        [
            float(row["paired_ratio_difference"])
            for row in differences
            if int(row["dimension"]) == dimension
        ]
        for dimension in dimensions
    ]
    if differences:
        axes[1, 1].boxplot(
            differences_by_dimension,
            tick_labels=dimensions,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d9d9d9", "linewidth": 1.15},
            whiskerprops={"linewidth": 1.15},
            capprops={"linewidth": 1.15},
            medianprops={"color": "#D55E00", "linewidth": 1.5},
        )
        axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.3)
    axes[1, 1].set_xlabel("ambient dimension")
    axes[1, 1].set_ylabel("paired KL-ratio change\n(cap 4 minus cap 1)")
    axes[1, 1].set_title("D. Paired imbalance contrast")

    for axis in axes.flat:
        axis.grid(True, color="#bdbdbd", alpha=0.34, linewidth=0.7)
        axis.tick_params(direction="out", width=0.8, length=4)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    fig.savefig(path_stem.with_suffix(".pdf"), metadata={"Creator": "Matplotlib"})
    plt.close(fig)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    strata: list[dict[str, Any]],
    differences: list[dict[str, Any]],
    *,
    design: dict[str, Any],
) -> None:
    """Write a bounded interpretation with reproducibility instructions."""

    whole = [row for row in rows if row["method"] == "whole_conditioning"]
    fixed = [
        row for row in rows if row["method"] in {"uniform_fixed", "anchor_fixed"}
    ]
    anchored = [row for row in rows if str(row["method"]).startswith("anchored_tau_")]
    violation_count = sum(float(row["total_ratio"]) > 1.0 + 1e-9 for row in whole)
    violation_geometry_count = len(
        {
            int(row["geometry_seed"])
            for row in whole
            if float(row["total_ratio"]) > 1.0 + 1e-9
        }
    )
    maximum = max(whole, key=lambda row: float(row["total_ratio"]))
    max_fixed = max(float(row["total_ratio"]) for row in fixed)
    max_anchored = max(float(row["total_ratio"]) for row in anchored)
    max_chain_error = max(float(row["whole_chain_abs_error"]) for row in rows)
    max_bound_residual = max(
        float(row["theorem_bound_residual"])
        for row in rows
        if math.isfinite(float(row["theorem_bound_residual"]))
    )
    strata_with_violations = sum(int(row["violation_count"]) > 0 for row in strata)
    paired_values = np.asarray(
        [float(row["paired_ratio_difference"]) for row in differences],
        dtype=np.float64,
    )
    paired_median = float(np.median(paired_values)) if paired_values.size else math.nan

    by_dimension = []
    for protocol in design["direction_protocols"]:
        for dimension in design["dimensions"]:
            group = [
                row
                for row in whole
                if int(row["dimension"]) == int(dimension)
                and row["direction_protocol"] == protocol
            ]
            ratios = np.asarray([float(row["total_ratio"]) for row in group])
            by_dimension.append(
                (
                    protocol,
                    int(dimension),
                    int(np.sum(ratios > 1.0 + 1e-9)),
                    int(ratios.size),
                    float(np.median(ratios)),
                    float(np.quantile(ratios, 0.1)),
                    float(np.quantile(ratios, 0.9)),
                )
            )
    table = "\n".join(
        [
            "| direction protocol | dimension | violations / evaluated cells | median ratio | 10th--90th percentile |",
            "|:---|---:|---:|---:|---:|",
            *[
                f"| {protocol} | {dimension} | {violations} / {count} | {median:.6f} | {low:.6f}--{high:.6f} |"
                for protocol, dimension, violations, count, median, low, high in by_dimension
            ],
        ]
    )

    text = f"""# Seeded analytic random product-box union ensemble

## What this adds

This experiment asks whether the deterministic scaling witness survives a
declared distributional stress test.  It samples a.e.-disjoint axis-aligned
box unions, evaluates both independent and adaptive perturbation directions,
and computes every Gaussian occupancy and KL term with analytic product
formulas.  It is a
synthetic breadth check.  It is **not** evidence that these geometries are
typical of applications, and it is not a learned-controller or systems result.

## Declared computational design

This generated report records the complete executed design, but it does not
by itself establish preregistration or untouched-holdout status.  Any
development/holdout selection history must be recorded in a separate frozen
run manifest.

- Ensemble version: `{ENSEMBLE_VERSION}`.
- Dimensions: `{design['dimensions']}`; mode counts: `{design['mode_counts']}`.
- Geometry families: `{design['geometry_families']}`.  `diffuse` uses
  independent auxiliary centers; `coherent` uses a randomly oriented rank-one
  mode pattern plus jitter.
- Direction protocols: `{design['direction_protocols']}`.  `iid` is independent
  of the sampled geometry.  `spectral_stress` uses the top exact conditioned-
  covariance direction at the anchor and is explicitly an adaptive attack
  diagnostic, not a random-direction draw.
- Gaussian scales: `{design['sigmas']}`; normalized center separations
  `||a-b||/sigma`: `{design['separation_ratios']}`.
- Volume-imbalance caps: `{design['volume_ratio_caps']}`.  The cap-1 and cap-4
  cases use paired latent positions and directions; cap 4 only shrinks the
  first auxiliary width of noncentral modes.
- Independent geometry/direction replicates per fixed factorial stratum:
  `{design['replicates']}`.
- Geometry root seed `{design['root_seed']}`, direction root seed
  `{design['direction_root_seed']}`, bootstrap seed
  `{design['bootstrap_seed']}`.
- Every box occupies a different first-coordinate cell in `[-1,1]`, so box
  interiors are disjoint.  All auxiliary intervals remain in `[-1,1]`; the
  first and last modes fix the first-coordinate envelope at `[-1,1]`.
- The two central boxes meet at the feasible anchor.  The independently drawn
  direction has nonnegative first coordinate and the tested displacement is
  small enough that the second center remains in the right central box.

The `{len([row for row in rows if row['method'] == 'whole_conditioning'])}`
whole-conditioning cells are **not** that many independent samples.  They are
factorial reuse of
`{len(design['dimensions']) * len(design['mode_counts']) * len(design['geometry_families']) * design['replicates']}`
independently seeded geometry-family draws: imbalance caps are paired, the same
i.i.d. direction is reused across paired factors, and the spectral direction
is a deterministic function of each geometry and `sigma`.

For each fixed `(geometry family, direction protocol, d, M, sigma, separation
ratio, imbalance cap)` stratum, the
report includes a Wilson 95% interval for the violation fraction and a
deterministic percentile-bootstrap 95% interval for the median ratio.  These
pointwise intervals are not simultaneous over the fixed strata and quantify
the declared synthetic ensemble only.  Aggregated counts below are descriptive
because factor cells from the same seeded geometry are paired.

## Results

Whole conditioning exceeded the ambient Gaussian KL comparator in
`{violation_count}` of `{len(whole)}` evaluated factor cells, across
`{strata_with_violations}` of `{len(strata)}` fixed strata.  The maximum ratio
was `{float(maximum['total_ratio']):.9f}` at `d={maximum['dimension']}`,
`M={maximum['mode_count']}`, `sigma={maximum['sigma']}`,
`||a-b||/sigma={maximum['separation_ratio']}`, imbalance cap
`{maximum['volume_ratio_cap']}`, replicate `{maximum['replicate']}`, in the
`{maximum['geometry_family']}` geometry family under the
`{maximum['direction_protocol']}` direction protocol.  At that
cell the within-mode and categorical-weight contributions were
`{float(maximum['within_ratio']):.9f}` and
`{float(maximum['gate_ratio']):.9f}` Gaussian units.

The violating factor cells arose from `{violation_geometry_count}` of the
`{design['independent_geometry_draw_count']}` independently seeded base
geometries.  Repeated separation, noise-scale, and paired-factor evaluations
must not be interpreted as independent occurrences.

{table}

Both fixed-weight controls contracted in every evaluated cell; their largest
ratio was `{max_fixed:.9f}`.  This is a numerical cross-check of the proved
fixed-weight statement, not its evidentiary basis.  Across the three tested
intermediate anchored temperatures, the largest evaluated ratio was
`{max_anchored:.9f}`; those finite-cell values must not be restated as an
unqualified Gaussian contraction theorem.

The median paired change in whole-conditioning ratio after moving from volume
cap 1 to cap 4 was `{paired_median:.9f}` over `{len(differences)}` matched
factor cells.  This paired perturbation isolates the declared width change,
but it does not make volume imbalance a universal causal explanation: Gaussian
occupancy also depends on mode position and noise scale.

The maximum independent-union versus sequential-weight reconstruction error was
`{max_chain_error:.3e}`.  The largest evaluated total-minus-theorem-bound
residual among fixed and anchored methods was `{max_bound_residual:.3e}`.
Calculations use IEEE double precision, not outward-rounded interval
arithmetic.

## Interpretation boundary

The ensemble tests whether conclusions drawn from one hand-selected L-shape or
one symmetric scaling family persist across the recorded geometry family.  It
does not estimate a population frequency for real feasible sets.  Dimensions,
mode counts, noise levels, and imbalance levels are design factors, while the
box locations and i.i.d. directions are random under the recorded generator;
spectral directions are adaptive stress searches.  The deterministic witness
establishes existence, while this study reports behavior under the recorded
generator.

Dimension and mode-count contrasts are not isolated causal effects.  The
coordinatewise envelope stays in `[-1,1]^d`, but its Euclidean diameter grows
as `2 sqrt(d)`, and product occupancies concentrate differently with `d`.
Holding the first-coordinate envelope fixed while changing `M` changes cell
pitch and box width.  In addition, the spectral direction is recomputed when
`sigma` changes.  These design couplings are recorded in the CSV and rule out
claims such as “violations become more prevalent with dimension/mode count.”

## Reproduction and artifacts

```sh
PYTHONPATH=src .venv/bin/python experiments/e_random_box_union_ensemble.py
.venv/bin/python -m pytest -q tests/test_random_box_union_ensemble.py
```

- Method-level rows: `outputs/e_random_box_union_ensemble.csv`
- Fixed-stratum uncertainty: `outputs/e_random_box_union_ensemble_strata.csv`
- Paired imbalance contrasts: `outputs/e_random_box_union_ensemble_paired.csv`
- Seeds and exact generated boxes: `outputs/e_random_box_union_ensemble.json`
- Figure: `outputs/e_random_box_union_ensemble.png` and `.pdf`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default=",".join(map(str, DEFAULT_DIMENSIONS)))
    parser.add_argument("--mode-counts", default=",".join(map(str, DEFAULT_MODE_COUNTS)))
    parser.add_argument("--sigmas", default=",".join(map(str, DEFAULT_SIGMAS)))
    parser.add_argument(
        "--separation-ratios",
        default=",".join(map(str, DEFAULT_SEPARATION_RATIOS)),
    )
    parser.add_argument(
        "--volume-ratio-caps",
        default=",".join(map(str, DEFAULT_VOLUME_RATIO_CAPS)),
    )
    parser.add_argument(
        "--geometry-families",
        default=",".join(DEFAULT_GEOMETRY_FAMILIES),
    )
    parser.add_argument(
        "--direction-protocols",
        default=",".join(DEFAULT_DIRECTION_PROTOCOLS),
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument(
        "--direction-root-seed",
        type=int,
        default=DEFAULT_DIRECTION_ROOT_SEED,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--report", type=Path, default=Path("docs/e_random_box_union_ensemble.md")
    )
    args = parser.parse_args()

    dimensions = _parse_values(args.dimensions, int)
    mode_counts = _parse_values(args.mode_counts, int)
    sigmas = _parse_values(args.sigmas, float)
    separation_ratios = _parse_values(args.separation_ratios, float)
    volume_ratio_caps = _parse_values(args.volume_ratio_caps, float)
    geometry_families = _parse_values(args.geometry_families, str)
    direction_protocols = _parse_values(args.direction_protocols, str)
    design = {
        "ensemble_version": ENSEMBLE_VERSION,
        "dimensions": list(dimensions),
        "mode_counts": list(mode_counts),
        "sigmas": list(sigmas),
        "separation_ratios": list(separation_ratios),
        "volume_ratio_caps": list(volume_ratio_caps),
        "geometry_families": list(geometry_families),
        "direction_protocols": list(direction_protocols),
        "anchored_taus": list(DEFAULT_ANCHORED_TAUS),
        "replicates": args.replicates,
        "independent_geometry_draw_count": (
            len(dimensions)
            * len(mode_counts)
            * len(geometry_families)
            * args.replicates
        ),
        "root_seed": args.root_seed,
        "direction_root_seed": args.direction_root_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "numpy_version": np.__version__,
    }
    rows, provenance = run_ensemble(
        dimensions,
        mode_counts,
        sigmas,
        separation_ratios,
        volume_ratio_caps,
        replicates=args.replicates,
        geometry_families=geometry_families,
        direction_protocols=direction_protocols,
        root_seed=args.root_seed,
        direction_root_seed=args.direction_root_seed,
    )
    strata = summarize_whole_strata(
        rows,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    differences = paired_imbalance_differences(rows)

    stem = args.output_dir / "e_random_box_union_ensemble"
    write_csv(stem.with_suffix(".csv"), rows)
    write_csv(args.output_dir / "e_random_box_union_ensemble_strata.csv", strata)
    if differences:
        write_csv(args.output_dir / "e_random_box_union_ensemble_paired.csv", differences)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "design": design,
                "strata": strata,
                "paired_imbalance": differences,
                "geometries": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_results(stem, rows)
    write_report(args.report, rows, strata, differences, design=design)


if __name__ == "__main__":
    main()
