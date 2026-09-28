"""Validate CIFAR-10.2 count-to-radius calculations with outward Arb bounds.

The input is the completed original evaluation. No new inference is performed.
This checks statistical postprocessing, not the finite-precision noise sampler.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import flint
import feasible_robustness.validated_numerics as numerics
from feasible_robustness.conditional_renyi import DEFAULT_ORDERS

from experiments.iclr2027_conditional_renyi_reanalysis import (
    DEFAULT_SOURCE,
    EXPECTED_ROWS_SHA256,
)
from experiments.plot_iclr2027_global_band_cifar102_confirmation import (
    load_final_result,
)
from feasible_robustness.validated_numerics import (
    binomial_endpoint,
    conditional_renyi_radius,
    gaussian_radius,
)

ETA = Fraction(1, 64_000_000)


def validate_row(row: dict) -> dict:
    counts = row["filtered_estimation_counts"]
    selected = row["filtered_selected_label"]
    runner = row["filtered_runner_label"]
    n = sum(counts)
    proposals = row["filtered_estimation_raw_proposals"]
    competitor = max(c for j, c in enumerate(counts) if j != selected)
    conditional_lower = binomial_endpoint(counts[selected], n, ETA, "lower")
    conditional_upper = binomial_endpoint(competitor, n, ETA, "upper")
    renyi = conditional_renyi_radius(conditional_lower, conditional_upper, 0.25, 1.0)
    joint_lower = binomial_endpoint(counts[selected], proposals, ETA, "lower")
    joint_upper = binomial_endpoint(competitor, proposals, ETA, "upper")
    selected_upper = binomial_endpoint(counts[selected], proposals, ETA, "upper")
    runner_lower = binomial_endpoint(counts[runner], proposals, ETA, "lower")
    joint_radius_upper = gaussian_radius(
        selected_upper, runner_lower, 0.25, side="upper"
    )
    unfiltered = row["unfiltered_estimation_counts"]
    unfiltered_selected = row["unfiltered_selected_label"]
    unfiltered_lower = binomial_endpoint(
        unfiltered[unfiltered_selected], sum(unfiltered), ETA, "lower"
    )
    unfiltered_upper = binomial_endpoint(
        max(c for j, c in enumerate(unfiltered) if j != unfiltered_selected),
        sum(unfiltered),
        ETA,
        "upper",
    )
    return {
        "image_index": row["image_index"],
        "filtered_correct": row["filtered_correct"],
        "unfiltered_correct": row["unfiltered_correct"],
        "conditional_probability_lower": conditional_lower,
        "conditional_competitor_upper": conditional_upper,
        "conditional_renyi_radius_lower": renyi.radius_lower,
        "conditional_reverse_kl_radius_lower": renyi.reverse_kl_radius_lower,
        "selected_order": renyi.selected_order,
        "joint_mass_radius_lower": gaussian_radius(joint_lower, joint_upper, 0.25),
        "joint_mass_radius_upper": (
            joint_radius_upper if math.isfinite(joint_radius_upper) else None
        ),
        "unfiltered_radius_lower": gaussian_radius(
            unfiltered_lower, unfiltered_upper, 0.25
        ),
        "max_conditional_endpoint_change": max(
            abs(conditional_lower - row["conditional_selected_probability_lower"]),
            abs(conditional_upper - row["conditional_largest_competitor_upper"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/iclr2027_validated_cifar102_20260924"),
    )
    args = parser.parse_args()
    rows_path = args.source_directory / "per_image.jsonl"
    if sha256(rows_path.read_bytes()).hexdigest() != EXPECTED_ROWS_SHA256:
        raise ValueError("original row file does not match the recorded evaluation")
    result = load_final_result(
        args.source_directory / "summary.json",
        rows_path,
        args.source_directory / "manifest.json",
    )
    start = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(
            pool.map(validate_row, result.rows, chunksize=10), 1
        ):
            records.append(record)
            if index % 200 == 0:
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "elapsed_seconds": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
    counts = []
    for radius in (0.2, 0.3, 0.5):
        exact_threshold = Fraction(str(radius))
        counts.append(
            {
                "radius": radius,
                **{
                    method: sum(
                        r[
                            (
                                "unfiltered_correct"
                                if method == "unfiltered"
                                else "filtered_correct"
                            )
                        ]
                        and Fraction(r[key]) >= exact_threshold
                        for r in records
                    )
                    for method, key in (
                        ("renyi", "conditional_renyi_radius_lower"),
                        ("reverse_kl", "conditional_reverse_kl_radius_lower"),
                        ("joint_mass", "joint_mass_radius_lower"),
                        ("unfiltered", "unfiltered_radius_lower"),
                    )
                },
            }
        )
    strong = sum(
        r["filtered_correct"]
        and r["joint_mass_radius_upper"] is not None
        and Fraction(r["conditional_renyi_radius_lower"])
        - Fraction(r["joint_mass_radius_upper"])
        >= Fraction(1, 100)
        for r in records
    )
    summary = {
        "status": "completed_outward_count_to_radius_validation",
        "records": len(records),
        "tail_error_rational": str(ETA),
        "familywise_error": "1/1000",
        "arithmetic": "Arb real balls",
        "python_flint_version": flint.__version__,
        "precision_bits": 128,
        "binomial_method": "positive binomial recurrence with certified geometric remainder",
        "sigma": 0.25,
        "covariance_factor": 1.0,
        "renyi_orders": list(DEFAULT_ORDERS),
        "one_sided_tails_per_image": 32,
        "endpoint_method": "SciPy candidate accepted only after rigorous binomial-tail test",
        "radius_method": "outward Gaussian quantiles and finite-order Renyi formulas",
        "correct_certified_counts": counts,
        "strong_separation_count": strong,
        "max_conditional_endpoint_change": max(
            r["max_conditional_endpoint_change"] for r in records
        ),
        "new_model_evaluations": 0,
        "source_rows_sha256": EXPECTED_ROWS_SHA256,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "numerics_source_sha256": sha256(
            Path(numerics.__file__).read_bytes()
        ).hexdigest(),
        "scope": "statistical postprocessing only; does not validate floating-point Gaussian sampling, filtering or neural inference",
        "elapsed_seconds": time.monotonic() - start,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "per_image.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    )
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
