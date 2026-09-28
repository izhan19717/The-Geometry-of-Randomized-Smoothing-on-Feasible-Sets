"""Recompute conditional Renyi certificates from the retained CIFAR-10.2 counts.

This is a deterministic reanalysis, developed after the original evaluation.
It adds no model evaluations and does not replace the original experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from experiments.plot_iclr2027_global_band_cifar102_confirmation import (
    load_final_result,
)
from feasible_robustness.conditional_renyi import (
    DEFAULT_ORDERS,
    conditioned_covariance_renyi_radius,
)
from feasible_robustness.filtered_certificate import (
    gaussian_joint_mass_radius,
    one_sided_binomial_lower,
    one_sided_binomial_upper,
)

DEFAULT_SOURCE = Path("outputs/iclr2027_global_band_cifar102_confirmation_v2")
DEFAULT_OUTPUT = Path("outputs/iclr2027_conditional_renyi_reanalysis_20260924")
EXPECTED_ROWS_SHA256 = (
    "976a5b7d458b708cd556efa64a2c02acad10e71e858123988834807947e514df"
)
METHODS = ("forward_kl", "reverse_kl", "renyi", "joint_mass", "unfiltered")


def analyze_rows(rows: list[dict], *, orders=DEFAULT_ORDERS) -> tuple[list[dict], dict]:
    """Use the existing simultaneous confidence bounds for every order."""
    records = []
    for row in rows:
        p = row["conditional_selected_probability_lower"]
        q = row["conditional_largest_competitor_upper"]
        n = sum(row["filtered_estimation_counts"])
        eta = row["bonferroni_tail_error"]
        selected = row["filtered_selected_label"]
        recomputed_p = one_sided_binomial_lower(
            row["filtered_estimation_counts"][selected], n, eta
        )
        recomputed_q = max(
            one_sided_binomial_upper(c, n, eta)
            for j, c in enumerate(row["filtered_estimation_counts"])
            if j != selected
        )
        if not (
            math.isclose(p, recomputed_p, abs_tol=1e-13)
            and math.isclose(q, recomputed_q, abs_tol=1e-13)
        ):
            raise ValueError("stored conditional bounds disagree with retained counts")
        cert = conditioned_covariance_renyi_radius(p, q, 0.25, 1.0, orders=orders)
        records.append(
            {
                "image_index": row["image_index"],
                "true_label": row["true_label"],
                "filtered_correct": row["filtered_correct"],
                "unfiltered_correct": row["unfiltered_correct"],
                "selected_lower": p,
                "competitor_upper": q,
                "forward_kl": row["r_cov_L"],
                "reverse_kl": cert.reverse_kl_radius,
                "renyi": cert.radius,
                "selected_order": cert.order,
                "joint_mass": row["r_mass_L"],
                "joint_mass_upper": row["r_mass_U"],
                "joint_mass_upper_is_finite": row["r_mass_U_is_finite"],
                "unfiltered": row["r_unfiltered_L"],
            }
        )
    curves = []
    for tick in range(81):
        radius = tick / 100
        counts = {
            method: sum(
                r[
                    (
                        "unfiltered_correct"
                        if method == "unfiltered"
                        else "filtered_correct"
                    )
                ]
                and r[method] > 0
                and r[method] >= radius
                for r in records
            )
            for method in METHODS
        }
        curves.append({"radius": radius, **counts})
    strong = {
        method: sum(
            r["filtered_correct"]
            and r["joint_mass_upper_is_finite"]
            and r[method] - r["joint_mass_upper"] >= 0.01
            for r in records
        )
        for method in ("forward_kl", "reverse_kl", "renyi")
    }
    classes = []
    for label in range(10):
        group = [r for r in records if r["true_label"] == label]
        classes.append(
            {
                "label": label,
                "images": len(group),
                **{
                    method: sum(
                        r[
                            (
                                "unfiltered_correct"
                                if method == "unfiltered"
                                else "filtered_correct"
                            )
                        ]
                        and r[method] >= 0.2
                        for r in group
                    )
                    for method in METHODS
                },
            }
        )
    model_calls = {
        name: sum(
            r[f"{name}_label_selection_model_calls"]
            + r[f"{name}_estimation_model_calls"]
            for r in rows
        )
        for name in ("filtered", "unfiltered")
    }
    summary = {
        "status": "completed_reanalysis_of_retained_counts",
        "record_count": len(records),
        "new_model_evaluations": 0,
        "orders": list(orders),
        "sigma": 0.25,
        "covariance_factor_upper": 1.0,
        "familywise_error": 0.001,
        "confidence_scope": "same 64000 original binomial tails; order choice adds no confidence events",
        "study_scope": "one training-selected filter, checkpoint and scale; no new independent confirmation",
        "curves": curves,
        "strong_separation_counts": strong,
        "classwise_at_radius_0_2": classes,
        "model_calls": model_calls,
        "raw_proposals_per_method": 220_000_000,
        "selected_order_counts": dict(
            sorted(Counter(r["selected_order"] for r in records).items())
        ),
        "minimum_reverse_minus_forward": min(
            r["reverse_kl"] - r["forward_kl"] for r in records
        ),
    }
    for radius in (0.2, 0.3, 0.5):
        summary[f"paired_at_{radius}"] = {
            "renyi_only": sum(
                r["filtered_correct"]
                and r["renyi"] >= radius
                and r["joint_mass"] < radius
                for r in records
            ),
            "joint_only": sum(
                r["filtered_correct"]
                and r["joint_mass"] >= radius
                and r["renyi"] < radius
                for r in records
            ),
        }
    return records, summary


def matched_call_baseline(rows: list[dict], seed: int = 20260924) -> dict:
    """Random subsets reconstructed from IID counts by hypergeometric sampling.

    Subset sizes come from independent filtered streams. They therefore do
    not select on unfiltered labels. This comparison has its own confidence
    family, separate from the original full-count analysis.
    """
    rng = np.random.default_rng(seed)
    radii = []
    correct = []
    selection_changed = 0
    for row in rows:
        eta = row["bonferroni_tail_error"]
        m0 = row["filtered_label_selection_model_calls"]
        m = row["filtered_estimation_model_calls"]
        c0 = rng.multivariate_hypergeometric(
            row["unfiltered_label_selection_counts"], m0
        )
        c = rng.multivariate_hypergeometric(row["unfiltered_estimation_counts"], m)
        selected = int(np.argmax(c0))
        selection_changed += selected != row["unfiltered_selected_label"]
        lower = one_sided_binomial_lower(int(c[selected]), m, eta)
        upper = max(
            one_sided_binomial_upper(int(v), m, eta)
            for j, v in enumerate(c)
            if j != selected
        )
        radii.append(gaussian_joint_mass_radius(lower, upper, 0.25))
        correct.append(selected == row["true_label"])
    return {
        "status": "random_subset_reanalysis",
        "seed": seed,
        "familywise_error": math.fsum(10 * r["bonferroni_tail_error"] for r in rows),
        "tail_error_rule": "same per-tail error as original row",
        "confidence_scope": "separate family for matched-call baseline only",
        "subset_law": "multivariate hypergeometric from retained unfiltered IID label counts",
        "model_calls": sum(
            r["filtered_label_selection_model_calls"]
            + r["filtered_estimation_model_calls"]
            for r in rows
        ),
        "label_selection_changes": selection_changed,
        "correct_counts": {
            str(t): sum(c and r >= t and r > 0 for c, r in zip(correct, radii))
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        },
        "interpretation": "one randomized count-level subset, not a fresh model or timing run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    row_path = args.source_directory / "per_image.jsonl"
    if sha256(row_path.read_bytes()).hexdigest() != EXPECTED_ROWS_SHA256:
        raise ValueError("the specified completed row file has changed")
    result = load_final_result(
        args.source_directory / "summary.json",
        row_path,
        args.source_directory / "manifest.json",
    )
    records, summary = analyze_rows(list(result.rows))
    summary["source_rows_sha256"] = EXPECTED_ROWS_SHA256
    summary["analysis_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    summary["matched_call_unfiltered"] = matched_call_baseline(list(result.rows))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "per_image.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    )
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "record_count": len(records),
                "selected_thresholds": [
                    r for r in summary["curves"] if r["radius"] in [0.2, 0.3, 0.5]
                ],
                "strong_separation_counts": summary["strong_separation_counts"],
                "model_calls": summary["model_calls"],
                "matched_call_unfiltered": summary["matched_call_unfiltered"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
