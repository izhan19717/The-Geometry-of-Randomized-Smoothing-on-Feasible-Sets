"""Summarize completed AuditVotes recertification records.

This post-processing script is intentionally independent of CUDA and of the
upstream AuditVotes checkout.  It consumes the immutable per-image JSONL file,
checks its index set, applies the strict certified-radius convention used by
the released evaluation, and records a dense certified-accuracy curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, norm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
    if not rows:
        raise ValueError("the per-image file is empty")
    indices = [int(row["index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("the per-image file contains duplicate indices")
    if sorted(indices) != list(range(len(rows))):
        raise ValueError("expected one row for every consecutive test index")
    return rows


def _certified_accuracy(
    rows: list[dict[str, Any]],
    correct_field: str,
    radius_field: str,
    radii: list[float],
) -> dict[str, float]:
    return {
        f"{radius:.2f}": float(
            np.mean(
                [
                    bool(row[correct_field])
                    and float(row[radius_field]) > radius
                    for row in rows
                ]
            )
        )
        for radius in radii
    }


def _one_sided_lower(successes: int, trials: int, error: float) -> float:
    if successes == 0:
        return 0.0
    return float(beta.ppf(error, successes, trials - successes + 1))


def _one_sided_upper(successes: int, trials: int, error: float) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - error, successes + 1, trials - successes))


def _upper_endpoint_grid(trials: int, error: float) -> np.ndarray:
    successes = np.arange(trials, dtype=int)
    endpoints = np.empty(trials + 1, dtype=float)
    endpoints[:-1] = beta.ppf(
        1.0 - error,
        successes + 1,
        trials - successes,
    )
    endpoints[-1] = 1.0
    return endpoints


def _recover_success_count(endpoint: float, endpoints: np.ndarray) -> int:
    insertion = int(np.searchsorted(endpoints, endpoint))
    candidates = {
        max(0, min(len(endpoints) - 1, insertion - 1)),
        max(0, min(len(endpoints) - 1, insertion)),
    }
    recovered = min(candidates, key=lambda count: abs(endpoints[count] - endpoint))
    if not np.isclose(endpoints[recovered], endpoint, rtol=0.0, atol=1e-9):
        raise ValueError("stored competitor endpoint does not identify an integer count")
    return recovered


def _gaussian_radius(selected: float, runner: float, sigma: float) -> float:
    if selected <= runner:
        return 0.0
    return float(0.5 * sigma * (norm.ppf(selected) - norm.ppf(runner)))


def summarize(
    rows_path: Path,
    radii: list[float],
    alpha: float,
    sigma: float,
    num_classes: int = 10,
) -> dict[str, Any]:
    rows = _load_rows(rows_path)
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    released = np.asarray([float(row["released_radius"]) for row in rows])
    joint = np.asarray([float(row["joint_radius"]) for row in rows])
    gaussian = np.asarray([float(row["gaussian_radius"]) for row in rows])
    complement = []
    simultaneous = []
    endpoint_grids: dict[int, np.ndarray] = {}
    endpoint_reconstruction_errors: list[float] = []
    for row in rows:
        proposals = int(row["estimation_proposals"])
        selected_count = int(row["joint_top_count"])
        competing_accepted = int(row["estimation_retained"]) - selected_count
        selected_lower = _one_sided_lower(
            selected_count, proposals, alpha / 2.0
        )
        selected_or_rejected_lower = _one_sided_lower(
            proposals - competing_accepted, proposals, alpha / 2.0
        )
        complement.append(
            _gaussian_radius(
                selected_lower, 1.0 - selected_or_rejected_lower, sigma
            )
        )
        original_competitor_error = alpha / (2.0 * (num_classes - 1))
        if proposals not in endpoint_grids:
            endpoint_grids[proposals] = _upper_endpoint_grid(
                proposals, original_competitor_error
            )
        maximum_competitor_count = _recover_success_count(
            float(row["joint_runner_up_upper"]), endpoint_grids[proposals]
        )
        endpoint_reconstruction_errors.append(
            abs(
                endpoint_grids[proposals][maximum_competitor_count]
                - float(row["joint_runner_up_upper"])
            )
        )
        if maximum_competitor_count > competing_accepted:
            raise ValueError("recovered competitor count exceeds all competitor counts")
        family_error = alpha / 3.0
        simultaneous_selected_lower = _one_sided_lower(
            selected_count, proposals, family_error
        )
        simultaneous_explicit_upper = _one_sided_upper(
            maximum_competitor_count,
            proposals,
            family_error / (num_classes - 1),
        )
        simultaneous_union_lower = _one_sided_lower(
            proposals - competing_accepted,
            proposals,
            family_error,
        )
        simultaneous.append(
            _gaussian_radius(
                simultaneous_selected_lower,
                min(
                    simultaneous_explicit_upper,
                    1.0 - simultaneous_union_lower,
                ),
                sigma,
            )
        )
    complement_array = np.asarray(complement)
    simultaneous_array = np.asarray(simultaneous)
    released_positive = released > 0.0
    ratio = np.divide(
        joint,
        released,
        out=np.zeros_like(joint),
        where=released_positive,
    )
    released_curve = _certified_accuracy(
        rows, "conditional_correct", "released_radius", radii
    )
    joint_curve = _certified_accuracy(
        rows, "conditional_correct", "joint_radius", radii
    )
    gaussian_curve = _certified_accuracy(
        rows, "gaussian_correct", "gaussian_radius", radii
    )
    complement_curve = {
        f"{radius:.2f}": float(
            np.mean(
                [
                    bool(row["conditional_correct"]) and value > radius
                    for row, value in zip(rows, complement_array)
                ]
            )
        )
        for radius in radii
    }
    simultaneous_curve = {
        f"{radius:.2f}": float(
            np.mean(
                [
                    bool(row["conditional_correct"]) and value > radius
                    for row, value in zip(rows, simultaneous_array)
                ]
            )
        )
        for radius in radii
    }
    overstatement = {
        key: released_curve[key] - joint_curve[key] for key in released_curve
    }

    return {
        "images": len(rows),
        "index_start": int(rows[0]["index"]),
        "index_stop_exclusive": int(rows[-1]["index"]) + 1,
        "per_image_sha256": _sha256(rows_path),
        "strict_radius_comparison": True,
        "alpha": alpha,
        "sigma": sigma,
        "mean_acceptance_rate": float(
            np.mean([float(row["acceptance_rate"]) for row in rows])
        ),
        "conditional_accuracy": float(
            np.mean([bool(row["conditional_correct"]) for row in rows])
        ),
        "gaussian_accuracy": float(
            np.mean([bool(row["gaussian_correct"]) for row in rows])
        ),
        "released_positive_fraction": float(np.mean(released_positive)),
        "joint_positive_fraction": float(np.mean(joint > 0.0)),
        "gaussian_positive_fraction": float(np.mean(gaussian > 0.0)),
        "median_released_radius": float(np.median(released)),
        "median_joint_radius": float(np.median(joint)),
        "median_gaussian_radius": float(np.median(gaussian)),
        "median_rejection_complement_radius": float(
            np.median(complement_array)
        ),
        "median_simultaneous_joint_radius": float(
            np.median(simultaneous_array)
        ),
        "median_joint_to_released_ratio_for_released_positive": float(
            np.median(ratio[released_positive])
        ),
        "fraction_joint_radius_above_released_radius": float(
            np.mean(joint > released + 1e-12)
        ),
        "released_certified_accuracy": released_curve,
        "joint_certified_accuracy": joint_curve,
        "gaussian_certified_accuracy": gaussian_curve,
        "rejection_complement_certified_accuracy": complement_curve,
        "simultaneous_joint_certified_accuracy": simultaneous_curve,
        "rejection_complement_positive_fraction": float(
            np.mean(complement_array > 0.0)
        ),
        "simultaneous_joint_positive_fraction": float(
            np.mean(simultaneous_array > 0.0)
        ),
        "simultaneous_joint_reconstruction": {
            "num_classes": num_classes,
            "source_field": "joint_runner_up_upper",
            "source_competitor_error": alpha / (2.0 * (num_classes - 1)),
            "maximum_endpoint_reconstruction_error": max(
                endpoint_reconstruction_errors
            ),
        },
        "released_minus_joint_certified_accuracy": overstatement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[float(value) for value in np.arange(0.0, 0.751, 0.05)],
    )
    args = parser.parse_args()
    summary = summarize(
        args.rows.resolve(),
        list(args.radii),
        args.alpha,
        args.sigma,
        args.num_classes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
