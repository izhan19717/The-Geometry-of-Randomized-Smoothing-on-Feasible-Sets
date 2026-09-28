"""Focused tests for the completed CIFAR-10.2 confirmation figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.plot_iclr2027_global_band_cifar102_confirmation import (
    EXPECTED_RECORDS,
    build_figures,
    certified_fraction_curve,
    load_final_result,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(label: int, total: int) -> list[int]:
    values = [0] * 10
    values[label] = total
    return values


def _write_complete_result(directory: Path) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text('{"study":"synthetic test"}\n', encoding="utf-8")
    manifest_digest = _sha256(manifest)
    rows_path = directory / "per_image.jsonl"
    rows = []
    for index in range(EXPECTED_RECORDS):
        true_label = index // 200
        runner = 1 if true_label == 0 else 0
        covariance_radius = 0.25 if index < 1_200 else 0.10
        mass_radius = 0.23 if index < 800 else 0.05
        unfiltered_radius = 0.22 if index < 1_000 else 0.08
        rows.append(
            {
                "ordinal": index + 1,
                "image_index": index,
                "true_label": true_label,
                "manifest_sha256": manifest_digest,
                "record_complete": True,
                "filtered_selected_label": true_label,
                "filtered_runner_label": runner,
                "unfiltered_selected_label": true_label,
                "filtered_correct": True,
                "unfiltered_correct": True,
                "filtered_label_selection_counts": _counts(true_label, 4_000),
                "filtered_estimation_counts": _counts(true_label, 40_000),
                "unfiltered_label_selection_counts": _counts(true_label, 10_000),
                "unfiltered_estimation_counts": _counts(true_label, 100_000),
                "filtered_label_selection_raw_proposals": 10_000,
                "filtered_label_selection_model_calls": 4_000,
                "filtered_estimation_raw_proposals": 100_000,
                "filtered_estimation_model_calls": 40_000,
                "unfiltered_label_selection_raw_proposals": 10_000,
                "unfiltered_label_selection_model_calls": 10_000,
                "unfiltered_estimation_raw_proposals": 100_000,
                "unfiltered_estimation_model_calls": 100_000,
                "label_selection_retention_rate": 0.4,
                "estimation_retention_rate": 0.4,
                "raw_filter_applied_before_classifier": True,
                "clipping_performed": False,
                "top_up_proposals_drawn": False,
                "r_cov_L": covariance_radius,
                "r_mass_L": mass_radius,
                "r_mass_U": 0.20,
                "r_unfiltered_L": unfiltered_radius,
            }
        )
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    observed = {
        "records": EXPECTED_RECORDS,
        "denominator": EXPECTED_RECORDS,
        "all_records_complete": True,
        "mean_estimation_retention": 0.4,
        "mean_label_selection_retention": 0.4,
        "filtered_selection_correct": EXPECTED_RECORDS,
        "filtered_selection_accuracy": 1.0,
        "unfiltered_selection_correct": EXPECTED_RECORDS,
        "unfiltered_selection_accuracy": 1.0,
        "covariance_correct_certified_count_at_radius_0_20": 1_200,
        "covariance_correct_certified_fraction_at_radius_0_20": 0.6,
        "joint_correct_certified_count_at_radius_0_20": 800,
        "joint_correct_certified_fraction_at_radius_0_20": 0.4,
        "unfiltered_correct_certified_count_at_radius_0_20": 1_000,
        "unfiltered_correct_certified_fraction_at_radius_0_20": 0.5,
        "correct_strong_margin_count": 1_200,
    }
    summary = {
        "status": "external_confirmation_passed_predeclared_success_rule",
        "manifest_sha256": manifest_digest,
        "per_image_jsonl_sha256": _sha256(rows_path),
        "all_images_in_official_fixed_order": True,
        "all_images_remain_in_denominator": True,
        "all_records_run_without_outcome_based_early_stopping": True,
        "raw_filter_applied_before_classifier": True,
        "clipping_performed": False,
        "top_up_proposals_drawn": False,
        "predeclared_success_rule": {
            "passed_predeclared_success_rule": True,
            "checks": {"complete": True, "primary": True},
            "thresholds": {"denominator": EXPECTED_RECORDS, "target_radius": 0.2},
            "observed": observed,
        },
    }
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary_path, rows_path, manifest


def _update_summary_rows_digest(summary_path: Path, rows_path: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["per_image_jsonl_sha256"] = _sha256(rows_path)
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_certified_fraction_curve_uses_full_denominator() -> None:
    correct = np.asarray([True, True, False, True])
    radii = np.asarray([0.3, 0.1, 0.8, 0.2])
    grid = np.asarray([0.0, 0.15, 0.25])
    assert certified_fraction_curve(correct, radii, grid).tolist() == [0.75, 0.5, 0.25]


def test_zero_radius_abstention_is_not_a_positive_certificate() -> None:
    curve = certified_fraction_curve(np.array([True, True]),
                                     np.array([0., .2]), np.array([0., .1]))
    assert curve.tolist() == [.5, .5]


def test_complete_result_emits_vector_and_raster_figures(tmp_path: Path) -> None:
    summary, rows, manifest = _write_complete_result(tmp_path / "result")
    result = load_final_result(summary, rows, manifest)
    assert result.mean_retention == pytest.approx(0.4)
    assert int(np.sum(result.filtered_correct)) == EXPECTED_RECORDS

    output = tmp_path / "figures"
    main_stem, appendix_stem = build_figures(summary, rows, output, manifest)
    for stem in (main_stem, appendix_stem):
        pdf = stem.with_suffix(".pdf")
        png = stem.with_suffix(".png")
        assert pdf.read_bytes().startswith(b"%PDF")
        assert png.read_bytes().startswith(b"\x89PNG")
        assert pdf.stat().st_size > 5_000
        assert png.stat().st_size > 15_000


def test_loader_rejects_partial_even_with_updated_digest(tmp_path: Path) -> None:
    summary, rows, manifest = _write_complete_result(tmp_path / "result")
    lines = rows.read_text(encoding="utf-8").splitlines()
    rows.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    _update_summary_rows_digest(summary, rows)
    with pytest.raises(ValueError, match="exactly 2000"):
        load_final_result(summary, rows, manifest)


def test_loader_rejects_reordered_or_mismatched_rows(tmp_path: Path) -> None:
    summary, rows, manifest = _write_complete_result(tmp_path / "result")
    lines = rows.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _update_summary_rows_digest(summary, rows)
    with pytest.raises(ValueError, match="fixed index order"):
        load_final_result(summary, rows, manifest)


def test_loader_rejects_summary_quantity_not_reproduced_by_rows(tmp_path: Path) -> None:
    summary, rows, manifest = _write_complete_result(tmp_path / "result")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["predeclared_success_rule"]["observed"][
        "covariance_correct_certified_count_at_radius_0_20"
    ] = 1_199
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match the row file"):
        load_final_result(summary, rows, manifest)


def test_loader_rejects_corrupted_row_digest(tmp_path: Path) -> None:
    summary, rows, manifest = _write_complete_result(tmp_path / "result")
    rows.write_bytes(rows.read_bytes() + b" \n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_final_result(summary, rows, manifest)
