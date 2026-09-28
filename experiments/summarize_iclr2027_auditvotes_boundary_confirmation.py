#!/usr/bin/env python3
"""Validate and summarize the frozen AuditVotes boundary confirmation.

The search output fixes six candidate pairs before confirmation. This script
binds the confirmation to the frozen files and applies one Bonferroni family
to every endpoint comparison and every nominal conditional-probability lower
bound. A reported population-radius violation requires both endpoint labels
to be verified and the stored displacement to lie below the fresh lower bound
on the substituted conditional radius.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from scipy.stats import binomtest, norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feasible_robustness.filtered_certificate import one_sided_binomial_lower  # noqa: E402


def sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_winner_pvalues(counts: Iterable[int], winner: int) -> list[float]:
    values = [int(value) for value in counts]
    if len(values) < 2 or not 0 <= winner < len(values) or any(value < 0 for value in values):
        raise ValueError("invalid retained-label counts")
    output: list[float] = []
    for competitor, count in enumerate(values):
        if competitor == winner:
            continue
        total = values[winner] + count
        output.append(
            1.0
            if total == 0
            else float(
                binomtest(
                    values[winner], total, p=0.5, alternative="greater"
                ).pvalue
            )
        )
    return output


def binary_substituted_radius_lower(
    winner_count: int, retained: int, sigma: float, error: float
) -> tuple[float, float]:
    lower = one_sided_binomial_lower(winner_count, retained, error)
    radius = 0.0 if lower <= 0.5 else float(sigma * norm.ppf(lower))
    return lower, radius


def retained_margin(
    counts: list[int], retained: int, original_label: int, alternative_label: int
) -> float:
    if retained <= 0:
        raise ValueError("retained count must be positive")
    return float(counts[original_label] - counts[alternative_label]) / retained


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def summarize(
    freeze_path: Path,
    protocol_path: Path,
    search_summary_path: Path,
    confirmation_path: Path,
    assets_dir: Path,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    search = json.loads(search_summary_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())

    _require_equal(freeze["status"], "frozen_before_confirmation", "invalid freeze status")
    _require_equal(protocol["status"], "frozen_before_boundary_search", "invalid protocol status")
    _require_equal(confirmation["status"], "confirmation_complete", "confirmation incomplete")
    _require_equal(
        sha256_path(protocol_path), freeze["inputs"]["protocol_sha256"], "protocol digest changed"
    )
    _require_equal(
        sha256_path(search_summary_path),
        freeze["inputs"]["search_summary_sha256"],
        "search summary digest changed",
    )
    _require_equal(
        confirmation["protocol_sha256"],
        freeze["inputs"]["protocol_sha256"],
        "confirmation uses another protocol",
    )
    _require_equal(
        confirmation["search_summary_sha256"],
        freeze["inputs"]["search_summary_sha256"],
        "confirmation uses another search summary",
    )

    frozen_selected = freeze["selected"]
    searched_selected = search["selected"]
    confirmed_rows = confirmation["rows"]
    maximum_pairs = int(freeze["confirmation"]["maximum_pairs"])
    _require_equal(len(frozen_selected), len(searched_selected), "selected-pair count changed")
    _require_equal(len(frozen_selected), len(confirmed_rows), "confirmation row count changed")
    if len(frozen_selected) > maximum_pairs:
        raise ValueError("too many selected pairs")

    for frozen, searched, confirmed in zip(
        frozen_selected, searched_selected, confirmed_rows
    ):
        identity = (
            int(frozen["index"]),
            int(frozen["original_label"]),
            int(frozen["alternative_label"]),
        )
        _require_equal(
            identity,
            (
                int(searched["index"]),
                int(searched["original_label"]),
                int(searched["alternative_label"]),
            ),
            "search selection order changed",
        )
        _require_equal(
            identity,
            (
                int(confirmed["index"]),
                int(confirmed["original_label"]),
                int(confirmed["alternative_label"]),
            ),
            "confirmation selection order changed",
        )
        _require_equal(
            searched["asset"], frozen["asset"], "selected asset name changed"
        )
        _require_equal(
            searched["asset_sha256"],
            frozen["asset_sha256"],
            "selected asset digest changed",
        )
        _require_equal(
            sha256_path(assets_dir / frozen["asset"]),
            frozen["asset_sha256"],
            "selected asset file changed",
        )

    num_classes = len(confirmed_rows[0]["endpoints"][0]["counts"])
    family_size = maximum_pairs * (2 * (num_classes - 1) + 1)
    familywise_alpha = float(freeze["confirmation"]["familywise_alpha"])
    per_inference_alpha = familywise_alpha / family_size
    sigma = float(protocol["model"]["sigma"])

    rows: list[dict[str, Any]] = []
    for row in confirmed_rows:
        endpoints = {endpoint["endpoint"]: endpoint for endpoint in row["endpoints"]}
        _require_equal(set(endpoints), {"original", "perturbed"}, "endpoint names changed")
        original_label = int(row["original_label"])
        alternative_label = int(row["alternative_label"])
        endpoint_results: dict[str, dict[str, Any]] = {}
        for endpoint_name, expected_winner in (
            ("original", original_label),
            ("perturbed", alternative_label),
        ):
            endpoint = endpoints[endpoint_name]
            counts = [int(value) for value in endpoint["counts"]]
            retained = int(endpoint["retained"])
            _require_equal(len(counts), num_classes, "class count changed")
            _require_equal(sum(counts), retained, "retained total does not match counts")
            _require_equal(int(endpoint["winner"]), expected_winner, "declared winner changed")
            pvalues = exact_winner_pvalues(counts, expected_winner)
            stored = [float(value) for value in endpoint["pairwise_pvalues"]]
            if len(stored) != len(pvalues) or any(
                not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-300)
                for a, b in zip(stored, pvalues)
            ):
                raise ValueError("stored pairwise p-values do not recompute")
            endpoint_results[endpoint_name] = {
                "retained": retained,
                "maximum_pairwise_pvalue": max(pvalues),
                "winner_verified_in_joint_family": max(pvalues) < per_inference_alpha,
                "original_minus_alternative_share": retained_margin(
                    counts, retained, original_label, alternative_label
                ),
            }

        nominal = endpoints["original"]
        nominal_lower, radius_lower = binary_substituted_radius_lower(
            int(nominal["counts"][original_label]),
            int(nominal["retained"]),
            sigma,
            per_inference_alpha,
        )
        distance = float(row["l2_distance"])
        inside_fresh_lower = distance < radius_lower
        verified = bool(
            original_label != alternative_label
            and endpoint_results["original"]["winner_verified_in_joint_family"]
            and endpoint_results["perturbed"]["winner_verified_in_joint_family"]
            and inside_fresh_lower
        )
        rows.append(
            {
                "index": int(row["index"]),
                "original_label": original_label,
                "alternative_label": alternative_label,
                "l2_distance": distance,
                "stored_released_radius": float(row["released_radius"]),
                "fresh_conditional_probability_lower": nominal_lower,
                "fresh_substituted_radius_lower": radius_lower,
                "distance_below_fresh_radius_lower": radius_lower - distance,
                "inside_fresh_substituted_radius_lower": inside_fresh_lower,
                "endpoints": endpoint_results,
                "population_substitution_violation_verified": verified,
            }
        )

    return {
        "status": "confirmation_analysis_complete",
        "freeze_sha256": sha256_path(freeze_path),
        "protocol_sha256": sha256_path(protocol_path),
        "search_summary_sha256": sha256_path(search_summary_path),
        "confirmation_sha256": sha256_path(confirmation_path),
        "analysis_script_sha256": sha256_path(Path(__file__).resolve()),
        "familywise_alpha": familywise_alpha,
        "simultaneous_family_size": family_size,
        "per_inference_alpha": per_inference_alpha,
        "selected_pair_count": len(rows),
        "verified_population_substitution_violation_count": sum(
            bool(row["population_substitution_violation_verified"]) for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--search-summary", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = summarize(
        args.freeze,
        args.protocol,
        args.search_summary,
        args.confirmation,
        args.assets_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
