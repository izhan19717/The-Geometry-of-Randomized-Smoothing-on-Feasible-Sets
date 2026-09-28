"""Fresh targeted holdout for the conditional-covariance KL mechanism.

The design is fixed in a protocol file before the holdout seeds are used.  It
repeats a stratum identified in an earlier random-box study with new geometry
and direction seeds.  The unit of analysis is a dimension-specific geometry.
Noise scales, displacements, and direction rules are repeated measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, spearmanr

from feasible_robustness import (
    rectangle_union_covariance,
    rectangle_union_path_integrated_kl,
    truncated_gaussian_kl_rectangle_union,
)
from feasible_robustness.random_box_ensemble import (
    make_random_box_geometry,
    spectral_stress_direction,
    wilson_interval,
)


SCRIPT_VERSION = "covariance-mechanism-holdout-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path, *, verify_source: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_holdout_execution":
        raise ValueError("protocol is not frozen for holdout execution")
    if payload.get("experiment") != SCRIPT_VERSION:
        raise ValueError("unknown covariance holdout version")
    if verify_source:
        expected = payload["provenance"]["experiment_script_sha256"]
        if sha256_path(Path(__file__).resolve()) != expected:
            raise ValueError("experiment script digest does not match protocol")
    return payload


def run_holdout(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    design = protocol["design"]
    rows: list[dict[str, Any]] = []
    for dimension in design["dimensions"]:
        for replicate in range(int(design["replicates_per_dimension"])):
            geometry = make_random_box_geometry(
                int(dimension),
                int(design["mode_count"]),
                replicate,
                float(design["volume_ratio_cap"]),
                geometry_family=str(design["geometry_family"]),
                root_seed=int(protocol["seeds"]["geometry_root"]),
                direction_root_seed=int(protocol["seeds"]["direction_root"]),
            )
            anchor = np.zeros(int(dimension), dtype=np.float64)
            for sigma in design["sigmas"]:
                scale = float(sigma)
                covariance = rectangle_union_covariance(
                    geometry.region,
                    anchor,
                    scale,
                )
                directions = {
                    "independent": geometry.direction,
                    "top covariance": spectral_stress_direction(
                        geometry.region,
                        scale,
                    ),
                }
                for direction_name, direction in directions.items():
                    local_ratio = float(direction @ covariance @ direction) / scale**2
                    for normalized_distance in design["normalized_distances"]:
                        distance_ratio = float(normalized_distance)
                        endpoint = scale * distance_ratio * direction
                        direct_kl = truncated_gaussian_kl_rectangle_union(
                            anchor,
                            endpoint,
                            scale,
                            geometry.region,
                        ).kl
                        integrated_kl = rectangle_union_path_integrated_kl(
                            anchor,
                            endpoint,
                            scale,
                            geometry.region,
                            quadrature_order=int(design["quadrature_order"]),
                        )
                        gaussian_kl = distance_ratio**2 / 2.0
                        rows.append(
                            {
                                "dimension": int(dimension),
                                "replicate": replicate,
                                "geometry_seed": geometry.geometry_seed,
                                "direction_seed": geometry.direction_seed,
                                "sigma": scale,
                                "normalized_distance": distance_ratio,
                                "direction_protocol": direction_name,
                                "anchor_covariance_ratio": local_ratio,
                                "finite_kl_ratio": direct_kl / gaussian_kl,
                                "integrated_covariance_ratio": integrated_kl
                                / gaussian_kl,
                                "path_identity_abs_error": abs(direct_kl - integrated_kl),
                            }
                        )
    return rows


def summarize_holdout(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    design = protocol["design"]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["dimension"]), int(row["replicate"])), []).append(row)

    unit_rows: list[dict[str, Any]] = []
    for (dimension, replicate), group in sorted(grouped.items()):
        independent_max = max(
            float(row["finite_kl_ratio"])
            for row in group
            if row["direction_protocol"] == "independent"
        )
        spectral_max = max(
            float(row["finite_kl_ratio"])
            for row in group
            if row["direction_protocol"] == "top covariance"
        )
        unit_rows.append(
            {
                "dimension": dimension,
                "replicate": replicate,
                "independent_max_ratio": independent_max,
                "spectral_max_ratio": spectral_max,
                "paired_max_difference": spectral_max - independent_max,
                "independent_exceeds_one": independent_max > 1.0,
                "spectral_exceeds_one": spectral_max > 1.0,
            }
        )

    spectral_only = sum(
        bool(row["spectral_exceeds_one"]) and not bool(row["independent_exceeds_one"])
        for row in unit_rows
    )
    independent_only = sum(
        bool(row["independent_exceeds_one"]) and not bool(row["spectral_exceeds_one"])
        for row in unit_rows
    )
    discordant = spectral_only + independent_only
    paired_pvalue = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                spectral_only,
                discordant,
                p=0.5,
                alternative="greater",
            ).pvalue
        )
    )

    fixed_cells: list[dict[str, Any]] = []
    for dimension in design["dimensions"]:
        for sigma in design["sigmas"]:
            for normalized_distance in design["normalized_distances"]:
                for direction_protocol in ("independent", "top covariance"):
                    group = [
                        row
                        for row in rows
                        if int(row["dimension"]) == int(dimension)
                        and float(row["sigma"]) == float(sigma)
                        and float(row["normalized_distance"])
                        == float(normalized_distance)
                        and row["direction_protocol"] == direction_protocol
                    ]
                    successes = sum(float(row["finite_kl_ratio"]) > 1.0 for row in group)
                    interval = wilson_interval(successes, len(group))
                    local = np.asarray(
                        [float(row["anchor_covariance_ratio"]) for row in group]
                    )
                    finite = np.asarray([float(row["finite_kl_ratio"]) for row in group])
                    fixed_cells.append(
                        {
                            "dimension": int(dimension),
                            "sigma": float(sigma),
                            "normalized_distance": float(normalized_distance),
                            "direction_protocol": direction_protocol,
                            "replicate_count": len(group),
                            "finite_ratio_exceeds_one_count": successes,
                            "finite_ratio_exceeds_one_fraction": successes / len(group),
                            "wilson_95_low": interval[0],
                            "wilson_95_high": interval[1],
                            "spearman_local_finite": float(
                                spearmanr(local, finite).statistic
                            ),
                            "sign_concordance_at_one": float(
                                np.mean((local > 1.0) == (finite > 1.0))
                            ),
                            "median_abs_local_finite_difference": float(
                                np.median(np.abs(local - finite))
                            ),
                            "maximum_finite_ratio": float(np.max(finite)),
                        }
                    )

    local_all = np.asarray([float(row["anchor_covariance_ratio"]) for row in rows])
    finite_all = np.asarray([float(row["finite_kl_ratio"]) for row in rows])
    integrated_all = np.asarray(
        [float(row["integrated_covariance_ratio"]) for row in rows]
    )
    return {
        "status": "fresh_targeted_holdout_complete",
        "protocol_sha256": protocol.get("protocol_sha256_at_execution"),
        "analysis_unit": "dimension-specific independently seeded geometry",
        "factor_row_count": len(rows),
        "independent_geometry_count": len(unit_rows),
        "spectral_only_exceedance_count": spectral_only,
        "independent_only_exceedance_count": independent_only,
        "discordant_geometry_count": discordant,
        "one_sided_paired_sign_pvalue": paired_pvalue,
        "spearman_local_finite_all_repeated_cells": float(
            spearmanr(local_all, finite_all).statistic
        ),
        "maximum_path_identity_abs_error": float(
            max(float(row["path_identity_abs_error"]) for row in rows)
        ),
        "maximum_ratio_identity_abs_error": float(
            np.max(np.abs(finite_all - integrated_all))
        ),
        "unit_rows": unit_rows,
        "fixed_cells": fixed_cells,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_holdout(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.4,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(4.7, 3.0), constrained_layout=True)
    styles = {
        "independent": ("#0072B2", "o"),
        "top covariance": ("#D55E00", "^"),
    }
    for name, (color, marker) in styles.items():
        group = [row for row in rows if row["direction_protocol"] == name]
        axis.scatter(
            [float(row["anchor_covariance_ratio"]) for row in group],
            [float(row["finite_kl_ratio"]) for row in group],
            s=18,
            alpha=0.48,
            color=color,
            marker=marker,
            linewidths=0.0,
            label=name,
        )
    bounds = [
        min(
            min(float(row["anchor_covariance_ratio"]) for row in rows),
            min(float(row["finite_kl_ratio"]) for row in rows),
        ),
        max(
            max(float(row["anchor_covariance_ratio"]) for row in rows),
            max(float(row["finite_kl_ratio"]) for row in rows),
        ),
    ]
    padding = 0.03 * (bounds[1] - bounds[0])
    limits = (bounds[0] - padding, bounds[1] + padding)
    axis.plot(limits, limits, color="black", linewidth=1.1, linestyle="--")
    axis.axhline(1.0, color="#555555", linewidth=0.85, linestyle=":")
    axis.axvline(1.0, color="#555555", linewidth=0.85, linestyle=":")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel(r"anchor directional covariance $u^\top C_a u/\sigma^2$")
    axis.set_ylabel("finite KL / Gaussian KL")
    axis.set_title("Local covariance predicts finite divergence")
    axis.legend(loc="lower right", frameon=True)
    axis.grid(True, color="#bdbdbd", alpha=0.32, linewidth=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Matplotlib"})
    fig.savefig(path.with_suffix(".png"), dpi=320)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    protocol["protocol_sha256_at_execution"] = sha256_path(args.protocol)
    rows = run_holdout(protocol)
    summary = summarize_holdout(rows, protocol)
    output_dir = args.output_dir
    write_csv(output_dir / "factor_rows.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_holdout(output_dir / "covariance_mechanism_holdout", rows, summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"unit_rows", "fixed_cells"}}, indent=2))


if __name__ == "__main__":
    main()
