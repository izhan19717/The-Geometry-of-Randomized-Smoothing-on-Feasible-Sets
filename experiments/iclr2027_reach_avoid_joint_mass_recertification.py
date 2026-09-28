"""Joint-mass recertification of the retained reach--avoid reference states.

This analysis reuses the already opened protocol-v2 confirmation population.
It is descriptive rather than a new held-out confirmation.  All probabilities
are analytic Gaussian rectangle calculations from the retained benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtri

from experiments.iclr2027_multistep_reach_avoid_benchmark import (
    deterministic_controller_states,
    generate_layouts,
    load_protocol,
    moving_upper_probability,
    nearest_event_boundary,
    support_stats,
)


DEFAULT_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "REACH_AVOID_CONTROLLER_PROTOCOL_V2_20260830.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/iclr2027_reach_avoid_joint_mass_recertification.json"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def recertify(protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    confirmation = protocol["splits"]["confirmation"]
    layouts = generate_layouts(
        "confirmation",
        int(confirmation["layout_seed"]),
        int(confirmation["layout_count"]),
        protocol,
    )
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        for stage, center in deterministic_controller_states(layout):
            occupancy = float(support_stats(stage, center).mass)
            upper_probability = float(moving_upper_probability(stage, center))
            selected_probability = max(upper_probability, 1.0 - upper_probability)
            runner_probability = min(upper_probability, 1.0 - upper_probability)
            selected_mass = occupancy * selected_probability
            runner_mass = occupancy * runner_probability
            joint_mass_radius = float(
                0.5
                * stage.sigma
                * (ndtri(selected_mass) - ndtri(runner_mass))
            )
            imported_radius = float(stage.sigma * ndtri(selected_probability))
            diameter = float(
                math.hypot(
                    stage.box.x_upper - stage.box.x_lower,
                    stage.box.y_upper - stage.box.y_lower,
                )
            )
            diameter_pinsker_radius = float(
                2.0
                * stage.sigma**2
                * (selected_probability - runner_probability)
                / diameter
            )
            diameter_log_odds_radius = float(
                stage.sigma**2
                * math.log(selected_probability / runner_probability)
                / diameter
            )
            boundary_distance = nearest_event_boundary(
                lambda query: moving_upper_probability(stage, query),
                center,
                stage.sigma,
            )
            if boundary_distance is None:
                raise RuntimeError("no vertical event boundary was located")
            rows.append(
                {
                    "layout_index": int(layout.layout_index),
                    "stage_index": int(stage.stage_index),
                    "sigma": float(stage.sigma),
                    "occupancy": occupancy,
                    "upper_probability": upper_probability,
                    "selected_probability": selected_probability,
                    "runner_probability": runner_probability,
                    "selected_joint_mass": selected_mass,
                    "runner_joint_mass": runner_mass,
                    "imported_radius": imported_radius,
                    "joint_mass_radius": joint_mass_radius,
                    "diameter_pinsker_radius": diameter_pinsker_radius,
                    "diameter_log_odds_radius": diameter_log_odds_radius,
                    "vertical_boundary_distance": float(boundary_distance),
                    "joint_to_imported_ratio": joint_mass_radius / imported_radius,
                    "joint_to_boundary_ratio": joint_mass_radius / boundary_distance,
                    "imported_radius_crosses_boundary": bool(
                        imported_radius > boundary_distance + 2e-10
                    ),
                    "joint_mass_radius_crosses_boundary": bool(
                        joint_mass_radius > boundary_distance + 2e-10
                    ),
                }
            )

    arrays = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "occupancy",
            "imported_radius",
            "joint_mass_radius",
            "diameter_pinsker_radius",
            "diameter_log_odds_radius",
            "vertical_boundary_distance",
            "joint_to_imported_ratio",
            "joint_to_boundary_ratio",
        )
    }
    return {
        "status": "descriptive_reanalysis_of_retained_confirmation_population",
        "scope": (
            "analytic population probabilities at the 448 retained deterministic "
            "reference states; no finite-sample confidence interval"
        ),
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_path(protocol_path),
        "state_count": len(rows),
        "layout_count": len(layouts),
        "summary": {
            key: quantiles(value) for key, value in arrays.items()
        },
        "imported_radius_crossing_count": sum(
            bool(row["imported_radius_crosses_boundary"]) for row in rows
        ),
        "joint_mass_radius_crossing_count": sum(
            bool(row["joint_mass_radius_crosses_boundary"]) for row in rows
        ),
        "positive_joint_mass_radius_count": sum(
            float(row["joint_mass_radius"]) > 0.0 for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = recertify(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
