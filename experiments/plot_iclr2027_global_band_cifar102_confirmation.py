"""Plot the completed CIFAR-10.2 fixed-band external confirmation.

The plotting path is deliberately separate from the experimental runner.  It
accepts only a completed 2,000-image result, verifies the row-file digest and
recomputes the plotted quantities before writing either figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_RECORDS = 2_000
CLASS_COUNT = 10
TARGET_RADIUS = 0.20
STRONG_SEPARATION = 0.01
PLOT_MAX_RADIUS = 0.30

DEFAULT_RESULT_DIRECTORY = Path(
    "outputs/iclr2027_global_band_cifar102_confirmation_v2"
)
DEFAULT_SUMMARY = DEFAULT_RESULT_DIRECTORY / "summary.json"
DEFAULT_ROWS = DEFAULT_RESULT_DIRECTORY / "per_image.jsonl"
DEFAULT_MANIFEST = DEFAULT_RESULT_DIRECTORY / "manifest.json"
DEFAULT_OUTPUT_DIRECTORY = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/figures"
)

GREEN = "#009E73"
BLUE = "#0072B2"
DARK_GREY = "#4B5563"
LIGHT_GREY = "#9CA3AF"
GRID_GREY = "#CBD5E1"


@dataclass(frozen=True)
class ConfirmationResult:
    """Verified values needed by the two figures."""

    summary: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    filtered_correct: np.ndarray
    unfiltered_correct: np.ndarray
    covariance_radius_lower: np.ndarray
    joint_mass_radius_lower: np.ndarray
    joint_mass_radius_upper: np.ndarray
    unfiltered_radius_lower: np.ndarray
    mean_retention: float


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _loads_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_complete_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("per-image JSONL is not newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("per-image JSONL is not valid UTF-8") from error
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("per-image JSONL contains a blank record")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        value = _loads_strict(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record {line_number} is not an object")
        rows.append(value)
    return rows


def _bool_field(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"row field {key!r} must be Boolean")
    return value


def _int_field(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"row field {key!r} must be an integer")
    return value


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"row field {key!r} must be finite")
    return result


def _counts(row: Mapping[str, Any], key: str) -> list[int]:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != CLASS_COUNT:
        raise ValueError(f"row field {key!r} must contain ten counts")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"row field {key!r} contains an invalid count")
    return value


def _selected_and_runner(counts: Sequence[int]) -> tuple[int, int]:
    order = sorted(range(CLASS_COUNT), key=lambda label: (-counts[label], label))
    return order[0], order[1]


def _assert_close(observed: float, expected: Any, name: str) -> None:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError(f"summary field {name!r} must be numeric")
    if not math.isclose(observed, float(expected), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"summary field {name!r} does not match the row file")


def _verify_row_accounting(row: Mapping[str, Any]) -> None:
    stream_specs = (
        ("filtered_label_selection", 10_000, False),
        ("filtered_estimation", 100_000, False),
        ("unfiltered_label_selection", 10_000, True),
        ("unfiltered_estimation", 100_000, True),
    )
    stream_counts: dict[str, list[int]] = {}
    for prefix, expected_raw, is_unfiltered in stream_specs:
        counts = _counts(row, f"{prefix}_counts")
        raw = _int_field(row, f"{prefix}_raw_proposals")
        calls = _int_field(row, f"{prefix}_model_calls")
        if raw != expected_raw or calls != sum(counts):
            raise ValueError(f"{prefix} stream accounting is inconsistent")
        if calls < 0 or calls > raw or (is_unfiltered and calls != raw):
            raise ValueError(f"{prefix} model-call count is inconsistent")
        stream_counts[prefix] = counts

    filtered_selected, filtered_runner = _selected_and_runner(
        stream_counts["filtered_label_selection"]
    )
    unfiltered_selected, _ = _selected_and_runner(
        stream_counts["unfiltered_label_selection"]
    )
    if _int_field(row, "filtered_selected_label") != filtered_selected:
        raise ValueError("filtered selected label does not match selection counts")
    if _int_field(row, "filtered_runner_label") != filtered_runner:
        raise ValueError("filtered runner label does not match selection counts")
    if _int_field(row, "unfiltered_selected_label") != unfiltered_selected:
        raise ValueError("unfiltered selected label does not match selection counts")

    selection_retention = _finite_float(row, "label_selection_retention_rate")
    estimation_retention = _finite_float(row, "estimation_retention_rate")
    expected_selection = _int_field(
        row, "filtered_label_selection_model_calls"
    ) / 10_000
    expected_estimation = _int_field(row, "filtered_estimation_model_calls") / 100_000
    if not math.isclose(selection_retention, expected_selection, abs_tol=1e-12):
        raise ValueError("label-selection retention is inconsistent")
    if not math.isclose(estimation_retention, expected_estimation, abs_tol=1e-12):
        raise ValueError("estimation retention is inconsistent")


def load_final_result(
    summary_path: Path,
    rows_path: Path,
    manifest_path: Path | None = None,
) -> ConfirmationResult:
    """Load a completed result and reject partial or inconsistent outcomes."""

    summary_path = summary_path.resolve()
    rows_path = rows_path.resolve()
    summary = _read_json_object(summary_path)
    expected_statuses = {
        "external_confirmation_passed_predeclared_success_rule",
        "external_confirmation_failed_predeclared_success_rule",
    }
    if summary.get("status") not in expected_statuses:
        raise ValueError("summary is not a completed external confirmation")
    if summary.get("per_image_jsonl_sha256") != sha256_path(rows_path):
        raise ValueError("per-image JSONL SHA-256 does not match the summary")

    manifest_digest = summary.get("manifest_sha256")
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        raise ValueError("summary manifest SHA-256 is missing or malformed")
    candidate_manifest = manifest_path
    if candidate_manifest is None:
        sibling = summary_path.parent / "manifest.json"
        candidate_manifest = sibling if sibling.is_file() else None
    if candidate_manifest is not None:
        candidate_manifest = candidate_manifest.resolve()
        if not candidate_manifest.is_file():
            raise FileNotFoundError(candidate_manifest)
        if sha256_path(candidate_manifest) != manifest_digest:
            raise ValueError("manifest SHA-256 does not match the summary")

    rows = _read_complete_jsonl(rows_path)
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError(f"expected exactly {EXPECTED_RECORDS} completed records")
    if [row.get("image_index") for row in rows] != list(range(EXPECTED_RECORDS)):
        raise ValueError("per-image records are not in fixed index order")
    if [row.get("ordinal") for row in rows] != list(range(1, EXPECTED_RECORDS + 1)):
        raise ValueError("per-image ordinals are not a complete fixed sequence")

    filtered_correct: list[bool] = []
    unfiltered_correct: list[bool] = []
    covariance_lower: list[float] = []
    mass_lower: list[float] = []
    mass_upper: list[float] = []
    unfiltered_lower: list[float] = []
    retentions: list[float] = []
    selection_retentions: list[float] = []
    true_labels: list[int] = []
    for row in rows:
        if row.get("manifest_sha256") != manifest_digest:
            raise ValueError("a per-image record has the wrong manifest digest")
        if not _bool_field(row, "record_complete"):
            raise ValueError("a per-image record is marked incomplete")
        if not _bool_field(row, "raw_filter_applied_before_classifier"):
            raise ValueError("a row does not apply the filter before the classifier")
        if _bool_field(row, "clipping_performed"):
            raise ValueError("a row reports clipped proposals")
        if _bool_field(row, "top_up_proposals_drawn"):
            raise ValueError("a row reports replacement proposals")
        _verify_row_accounting(row)

        true_label = _int_field(row, "true_label")
        if not 0 <= true_label < CLASS_COUNT:
            raise ValueError("a true label lies outside 0 through 9")
        filtered_is_correct = _bool_field(row, "filtered_correct")
        unfiltered_is_correct = _bool_field(row, "unfiltered_correct")
        if filtered_is_correct != (
            _int_field(row, "filtered_selected_label") == true_label
        ):
            raise ValueError("filtered correctness does not match the selected label")
        if unfiltered_is_correct != (
            _int_field(row, "unfiltered_selected_label") == true_label
        ):
            raise ValueError("unfiltered correctness does not match the selected label")

        radii = (
            _finite_float(row, "r_cov_L"),
            _finite_float(row, "r_mass_L"),
            _finite_float(row, "r_unfiltered_L"),
        )
        if any(radius < 0.0 for radius in radii):
            raise ValueError("certificate radii must be nonnegative")
        upper_value = row.get("r_mass_U")
        if upper_value is None:
            upper = math.nan
        else:
            upper = _finite_float(row, "r_mass_U")
            if upper < 0.0:
                raise ValueError("joint-mass upper radius must be nonnegative")

        retention = _finite_float(row, "estimation_retention_rate")
        selection_retention = _finite_float(row, "label_selection_retention_rate")
        if not 0.0 <= retention <= 1.0 or not 0.0 <= selection_retention <= 1.0:
            raise ValueError("retention rates must lie in [0, 1]")
        true_labels.append(true_label)
        filtered_correct.append(filtered_is_correct)
        unfiltered_correct.append(unfiltered_is_correct)
        covariance_lower.append(radii[0])
        mass_lower.append(radii[1])
        mass_upper.append(upper)
        unfiltered_lower.append(radii[2])
        retentions.append(retention)
        selection_retentions.append(selection_retention)

    label_counts = np.bincount(np.asarray(true_labels), minlength=CLASS_COUNT)
    if label_counts.tolist() != [EXPECTED_RECORDS // CLASS_COUNT] * CLASS_COUNT:
        raise ValueError("per-image rows are not the balanced ten-class census")

    filtered_array = np.asarray(filtered_correct, dtype=bool)
    unfiltered_array = np.asarray(unfiltered_correct, dtype=bool)
    covariance_array = np.asarray(covariance_lower)
    mass_lower_array = np.asarray(mass_lower)
    mass_upper_array = np.asarray(mass_upper)
    unfiltered_array_radius = np.asarray(unfiltered_lower)
    retention_array = np.asarray(retentions)
    selection_retention_array = np.asarray(selection_retentions)

    success = summary.get("predeclared_success_rule")
    if not isinstance(success, dict):
        raise ValueError("summary lacks the predeclared success result")
    observed = success.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("summary lacks observed confirmation quantities")
    if observed.get("records") != EXPECTED_RECORDS or observed.get("denominator") != EXPECTED_RECORDS:
        raise ValueError("summary does not use all 2,000 records")
    if observed.get("all_records_complete") is not True:
        raise ValueError("summary does not mark all records complete")
    if success.get("thresholds", {}).get("denominator") != EXPECTED_RECORDS:
        raise ValueError("success rule does not retain the full denominator")
    if success.get("thresholds", {}).get("target_radius") != TARGET_RADIUS:
        raise ValueError("success rule uses an unexpected target radius")

    covariance_at_target = filtered_array & (covariance_array >= TARGET_RADIUS)
    mass_at_target = filtered_array & (mass_lower_array >= TARGET_RADIUS)
    unfiltered_at_target = unfiltered_array & (
        unfiltered_array_radius >= TARGET_RADIUS
    )
    finite_upper = np.isfinite(mass_upper_array)
    correct_strong = (
        filtered_array
        & finite_upper
        & ((covariance_array - mass_upper_array) >= STRONG_SEPARATION)
    )
    recomputed: dict[str, float | int] = {
        "mean_estimation_retention": float(np.mean(retention_array)),
        "mean_label_selection_retention": float(np.mean(selection_retention_array)),
        "filtered_selection_correct": int(np.sum(filtered_array)),
        "filtered_selection_accuracy": float(np.mean(filtered_array)),
        "unfiltered_selection_correct": int(np.sum(unfiltered_array)),
        "unfiltered_selection_accuracy": float(np.mean(unfiltered_array)),
        "covariance_correct_certified_count_at_radius_0_20": int(
            np.sum(covariance_at_target)
        ),
        "covariance_correct_certified_fraction_at_radius_0_20": float(
            np.mean(covariance_at_target)
        ),
        "joint_correct_certified_count_at_radius_0_20": int(np.sum(mass_at_target)),
        "joint_correct_certified_fraction_at_radius_0_20": float(
            np.mean(mass_at_target)
        ),
        "unfiltered_correct_certified_count_at_radius_0_20": int(
            np.sum(unfiltered_at_target)
        ),
        "unfiltered_correct_certified_fraction_at_radius_0_20": float(
            np.mean(unfiltered_at_target)
        ),
        "correct_strong_margin_count": int(np.sum(correct_strong)),
    }
    integer_fields = {
        "filtered_selection_correct",
        "unfiltered_selection_correct",
        "covariance_correct_certified_count_at_radius_0_20",
        "joint_correct_certified_count_at_radius_0_20",
        "unfiltered_correct_certified_count_at_radius_0_20",
        "correct_strong_margin_count",
    }
    for key, value in recomputed.items():
        if key in integer_fields:
            if observed.get(key) != value:
                raise ValueError(f"summary field {key!r} does not match the row file")
        else:
            _assert_close(float(value), observed.get(key), key)

    passed = success.get("passed_predeclared_success_rule")
    checks = success.get("checks")
    if type(passed) is not bool or not isinstance(checks, dict):
        raise ValueError("predeclared success decision is malformed")
    if passed != all(value is True for value in checks.values()):
        raise ValueError("predeclared success decision does not match its checks")
    expected_pass = summary["status"].startswith("external_confirmation_passed")
    if passed != expected_pass:
        raise ValueError("summary status and predeclared success decision disagree")
    for key in (
        "all_images_in_official_fixed_order",
        "all_images_remain_in_denominator",
        "all_records_run_without_outcome_based_early_stopping",
        "raw_filter_applied_before_classifier",
    ):
        if summary.get(key) is not True:
            raise ValueError(f"summary completion flag {key!r} is not true")
    if summary.get("clipping_performed") is not False:
        raise ValueError("summary reports clipped proposals")
    if summary.get("top_up_proposals_drawn") is not False:
        raise ValueError("summary reports replacement proposals")

    return ConfirmationResult(
        summary=summary,
        rows=tuple(rows),
        filtered_correct=filtered_array,
        unfiltered_correct=unfiltered_array,
        covariance_radius_lower=covariance_array,
        joint_mass_radius_lower=mass_lower_array,
        joint_mass_radius_upper=mass_upper_array,
        unfiltered_radius_lower=unfiltered_array_radius,
        mean_retention=float(np.mean(retention_array)),
    )


def certified_fraction_curve(
    correct: np.ndarray, radii: np.ndarray, radius_grid: np.ndarray
) -> np.ndarray:
    """Return correct-and-certified fractions over the full fixed census."""

    correct = np.asarray(correct, dtype=bool)
    radii = np.asarray(radii, dtype=float)
    radius_grid = np.asarray(radius_grid, dtype=float)
    if correct.ndim != 1 or radii.shape != correct.shape:
        raise ValueError("correctness and radius vectors must have the same shape")
    if radius_grid.ndim != 1 or np.any(np.diff(radius_grid) < 0):
        raise ValueError("radius grid must be one-dimensional and nondecreasing")
    if np.any(~np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("certificate radii must be finite and nonnegative")
    return np.asarray(
        [np.mean(correct & (radii > 0) & (radii >= radius)) for radius in radius_grid]
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 320,
        }
    )


def _save(figure: plt.Figure, stem: Path, title: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": title,
        "Author": "Anonymous",
        "Subject": "ICLR 2027 submission figure",
        "Creator": "Matplotlib",
        "CreationDate": datetime(2026, 9, 20, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 9, 20, tzinfo=timezone.utc),
    }
    figure.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(stem.with_suffix(".png"), dpi=320)
    plt.close(figure)


def plot_certified_fraction(
    result: ConfirmationResult, output_stem: Path
) -> None:
    radius_grid = np.linspace(0.0, PLOT_MAX_RADIUS, 121)
    methods = (
        (
            "filtered covariance",
            result.filtered_correct,
            result.covariance_radius_lower,
            GREEN,
            "-",
            "o",
        ),
        (
            "filtered joint mass",
            result.filtered_correct,
            result.joint_mass_radius_lower,
            BLUE,
            "--",
            "s",
        ),
        (
            "unfiltered Gaussian",
            result.unfiltered_correct,
            result.unfiltered_radius_lower,
            DARK_GREY,
            ":",
            "^",
        ),
    )
    figure, axis = plt.subplots(figsize=(5.5, 3.0))
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.18, top=0.78)
    figure.suptitle(
        "CIFAR-10.2 external confirmation",
        x=0.13,
        y=0.985,
        ha="left",
        fontsize=9.3,
    )
    for label, correct, radii, color, linestyle, marker in methods:
        curve = certified_fraction_curve(correct, radii, radius_grid)
        axis.step(
            radius_grid,
            curve,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=20,
            markerfacecolor="white",
            markeredgewidth=0.9,
            where="post",
        )
        target_fraction = float(np.mean(correct & (radii >= TARGET_RADIUS)))
        axis.scatter(
            [TARGET_RADIUS],
            [target_fraction],
            color=color,
            marker=marker,
            s=28,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

    axis.axvline(TARGET_RADIUS, color=LIGHT_GREY, lw=1.0, ls="--", zorder=0)
    axis.text(
        0.018,
        0.035,
        f"mean retained fraction  {result.mean_retention:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=DARK_GREY,
        fontsize=7.3,
        bbox={"boxstyle": "square,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.88},
    )
    axis.set(
        xlim=(0.0, PLOT_MAX_RADIUS),
        ylim=(0.0, 1.0),
        xlabel=r"required $\ell_2$ radius",
        ylabel="correct certified fraction",
    )
    axis.set_xticks(np.arange(0.0, PLOT_MAX_RADIUS + 0.001, 0.05))
    axis.set_yticks(np.arange(0.0, 1.01, 0.20))
    axis.grid(color=GRID_GREY, alpha=0.55, lw=0.55)
    axis.set_axisbelow(True)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
        mode="expand",
        ncol=3,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.94,
        edgecolor="#D1D5DB",
        columnspacing=1.0,
        handlelength=2.4,
    )
    _save(figure, output_stem, "CIFAR-10.2 external confirmation")


def plot_strong_separation(
    result: ConfirmationResult, output_stem: Path
) -> None:
    finite = result.filtered_correct & np.isfinite(result.joint_mass_radius_upper)
    if not np.any(finite):
        raise ValueError("no filtered-correct row has a finite joint-mass upper bound")
    x = result.joint_mass_radius_upper[finite]
    y = result.covariance_radius_lower[finite]
    separated = (y - x) >= STRONG_SEPARATION
    high = max(PLOT_MAX_RADIUS, float(np.max(np.concatenate((x, y)))) * 1.04)
    low = 0.0
    line_x = np.linspace(low, high, 300)

    detail_high = 0.32
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.19,
        top=0.76,
        wspace=0.25,
    )

    handles: list[Any] = []
    labels: list[str] = []
    for panel_index, (axis, panel_high, title) in enumerate(
        zip(axes, (high, detail_high), ("A   Full range", "B   Detail"))
    ):
        other_points = axis.scatter(
            x[~separated],
            y[~separated],
            s=11,
            color=LIGHT_GREY,
            alpha=0.46,
            linewidth=0,
            label=r"difference $<.01$",
            zorder=2,
        )
        separated_points = axis.scatter(
            x[separated],
            y[separated],
            s=12,
            color=GREEN,
            alpha=0.74,
            linewidth=0,
            label=r"difference $\geq .01$",
            zorder=3,
        )
        equality_line = axis.plot(
            line_x,
            line_x,
            color="black",
            lw=1.0,
            ls="--",
            label="equality",
            zorder=4,
        )[0]
        valid = line_x + STRONG_SEPARATION <= panel_high
        separation_line = axis.plot(
            line_x[valid],
            line_x[valid] + STRONG_SEPARATION,
            color=BLUE,
            lw=1.2,
            ls=":",
            label=r"equality $+.01$",
            zorder=4,
        )[0]
        axis.set(
            xlim=(low, panel_high),
            ylim=(low, panel_high),
        )
        axis.set_title(title, loc="left", pad=5.0, fontweight="bold")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color=GRID_GREY, alpha=0.52, lw=0.55)
        axis.set_axisbelow(True)
        if panel_index == 0:
            handles = [other_points, separated_points, equality_line, separation_line]
            labels = [item.get_label() for item in handles]
        else:
            axis.text(
                0.965,
                0.055,
                f"{int(np.sum(separated))} of {len(x)}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color=DARK_GREY,
                fontsize=7.4,
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "fc": "white",
                    "ec": GRID_GREY,
                    "lw": 0.55,
                    "alpha": 0.94,
                },
            )

    axes[0].set_xticks(np.arange(0.0, high + 0.001, 0.2))
    axes[0].set_yticks(np.arange(0.0, high + 0.001, 0.2))
    axes[1].set_xticks(np.arange(0.0, detail_high + 0.001, 0.1))
    axes[1].set_yticks(np.arange(0.0, detail_high + 0.001, 0.1))
    figure.supxlabel("joint-mass radius upper bound", y=0.035, fontsize=8.5)
    figure.supylabel("covariance radius lower bound", x=0.015, fontsize=8.5)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.94,
        edgecolor="#D1D5DB",
        columnspacing=1.15,
        handlelength=1.8,
        fontsize=7.4,
    )
    _save(figure, output_stem, "CIFAR-10.2 simultaneous radius comparison")


def build_figures(
    summary_path: Path,
    rows_path: Path,
    output_directory: Path,
    manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    configure_style()
    result = load_final_result(summary_path, rows_path, manifest_path)
    main_stem = output_directory / "cifar102_external_confirmation"
    appendix_stem = output_directory / "cifar102_external_strong_separation"
    plot_certified_fraction(result, main_stem)
    plot_strong_separation(result, appendix_stem)
    return main_stem, appendix_stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional manifest; a sibling manifest.json is checked automatically",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_figures(
        args.summary,
        args.rows,
        args.output_directory,
        args.manifest,
    )


if __name__ == "__main__":
    main()
