"""Create the focused filtered-smoothing figures for the ICLR paper."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from experiments.plot_iclr2027_global_band_cifar102_confirmation import (
    ConfirmationResult,
    certified_fraction_curve,
    load_final_result,
)


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#6B7280"
LIGHT_BLUE = "#DCEAF4"
LIGHT_ORANGE = "#FBE7C0"

DEFAULT_OUTPUT = Path("_ICLR_2027__Feasibility_Breaks_Smoothing/figures")
DEFAULT_REACH = Path("outputs/iclr2027_reach_avoid_joint_mass_recertification.json")
DEFAULT_AUDITVOTES_SUMMARY = Path(
    "outputs/auditvotes_recertification_full_20260906/corrected_summary.json"
)
DEFAULT_CIFAR101_SUMMARY = Path(
    "outputs/auditvotes_cifar101_v4_fixed_cohort_20260909/evaluation/summary.json"
)
DEFAULT_CIFAR101_CONFIRMATION = Path(
    "outputs/auditvotes_cifar101_v4_fixed_cohort_20260909/confirmation.json"
)
DEFAULT_RANDOM_BOX = Path(
    "outputs/e_random_box_union_ensemble_holdout/e_random_box_union_ensemble.csv"
)
DEFAULT_COVARIANCE_SUMMARY = Path(
    "outputs/iclr2027_covariance_mechanism_holdout_20260912/summary.json"
)
DEFAULT_CIFAR102_DIRECTORY = Path(
    "outputs/iclr2027_global_band_cifar102_confirmation_v2"
)
DEFAULT_CIFAR102_SUMMARY = DEFAULT_CIFAR102_DIRECTORY / "summary.json"
DEFAULT_CIFAR102_ROWS = DEFAULT_CIFAR102_DIRECTORY / "per_image.jsonl"
DEFAULT_CIFAR102_MANIFEST = DEFAULT_CIFAR102_DIRECTORY / "manifest.json"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    axis.text(
        0.0,
        1.04,
        letter,
        transform=axis.transAxes,
        fontsize=10.2,
        fontweight="bold",
        va="bottom",
    )
    artist = axis.set_title(title, loc="left", pad=7.0)
    artist.set_position((0.105, 1.0))


def save(figure: plt.Figure, stem: Path) -> None:
    metadata = {
        "Title": stem.name.replace("_", " "),
        "Author": "Anonymous",
        "Subject": "ICLR 2027 submission figure",
        "CreationDate": datetime(2026, 9, 7, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 9, 7, tzinfo=timezone.utc),
    }
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in raw.items():
                if value in {"", "None"}:
                    parsed[key] = None
                elif key in {
                    "split",
                    "method",
                    "metric",
                    "reference",
                    "cluster_unit",
                    "ensemble_version",
                    "geometry_family",
                    "direction_protocol",
                }:
                    parsed[key] = value
                elif value in {"True", "False"}:
                    parsed[key] = value == "True"
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditioning_failures(
    output_dir: Path,
    cifar102_summary_path: Path = DEFAULT_CIFAR102_SUMMARY,
    cifar102_rows_path: Path = DEFAULT_CIFAR102_ROWS,
    cifar102_manifest_path: Path = DEFAULT_CIFAR102_MANIFEST,
    *,
    external_result: ConfirmationResult | None = None,
    conditional_radii: np.ndarray | None = None,
    conditional_label: str = "covariance",
    radius_limit: float = 0.30,
) -> None:
    center = -0.1
    attacked = 0.01

    def conditional_left(location: np.ndarray) -> np.ndarray:
        left = ndtr(-1.0 - location)
        right = ndtr(location - 1.0)
        return left / (left + right)

    nominal_probability = float(conditional_left(np.asarray(center)))
    conditional_radius = float(ndtri(nominal_probability))
    joint_radius = 0.1

    figure = plt.figure(figsize=(5.5, 2.35), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.42, 1.0))

    axis = figure.add_subplot(grid[0, 0])
    locations = np.linspace(-0.32, 0.14, 600)
    axis.axvspan(
        center - conditional_radius,
        center + conditional_radius,
        color=RED,
        alpha=0.11,
        label="substituted interval",
        zorder=0,
    )
    axis.axvspan(
        center - joint_radius,
        center + joint_radius,
        color=BLUE,
        alpha=0.19,
        label="joint-mass interval",
        zorder=1,
    )
    axis.plot(locations, conditional_left(locations), color="#202020", zorder=3)
    axis.axhline(0.5, color=GREY, lw=1.0, ls="--")
    axis.axvline(0.0, color=GREY, lw=1.0, ls=":")
    axis.scatter(
        [center],
        [nominal_probability],
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    attacked_probability = float(conditional_left(np.asarray(attacked)))
    axis.scatter(
        [attacked],
        [attacked_probability],
        color=RED,
        marker="D",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    axis.annotate(
        "nominal  0.576",
        (center, nominal_probability),
        xytext=(-0.275, 0.589),
        arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.8},
        color=BLUE,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.92},
    )
    axis.annotate(
        "label flips\nat 0.01",
        (attacked, attacked_probability),
        xytext=(0.118, 0.523),
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8},
        color=RED,
        ha="right",
        va="center",
        linespacing=0.92,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.92},
    )
    axis.text(
        -0.10,
        0.466,
        "substituted interval",
        color=RED,
        ha="center",
        va="bottom",
        fontsize=7.0,
        bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.82},
    )
    axis.text(
        -0.10,
        0.482,
        "joint-mass interval",
        color=BLUE,
        ha="center",
        va="bottom",
        fontsize=7.0,
        bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.82},
    )
    axis.text(
        -0.006,
        0.615,
        "decision\nboundary",
        color=GREY,
        ha="right",
        va="top",
        fontsize=6.8,
        linespacing=0.88,
        clip_on=True,
        bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.82},
    )
    axis.set(
        xlim=(-0.32, 0.14),
        ylim=(0.46, 0.62),
        xlabel="Gaussian center",
        ylabel="conditional probability of label 0",
    )
    axis.set_yticks([0.46, 0.50, 0.54, 0.58, 0.62])
    axis.grid(alpha=0.22)
    panel_title(axis, "A", "Confidence filtering")

    result = external_result
    if result is None:
        result = load_final_result(
            cifar102_summary_path,
            cifar102_rows_path,
            cifar102_manifest_path,
        )

    certificate = figure.add_subplot(grid[0, 1])
    radius_grid = np.linspace(0.0, radius_limit, 701)
    curve_specs = (
        (
            result.filtered_correct,
            result.covariance_radius_lower if conditional_radii is None else conditional_radii,
            GREEN,
            "-",
            conditional_label,
        ),
        (
            result.filtered_correct,
            result.joint_mass_radius_lower,
            BLUE,
            "--",
            "joint mass",
        ),
        (
            result.unfiltered_correct,
            result.unfiltered_radius_lower,
            "#4B5563",
            ":",
            "unfiltered",
        ),
    )
    for correct, radii, color, line_style, label in curve_specs:
        values = certified_fraction_curve(correct, radii, radius_grid)
        certificate.plot(
            radius_grid,
            values,
            color=color,
            ls=line_style,
            lw=1.65,
            drawstyle="steps-post",
            label=label,
        )
        target_value = float(
            certified_fraction_curve(
                correct,
                radii,
                np.asarray([0.20]),
            )[0]
        )
        certificate.scatter(
            [0.20],
            [target_value],
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
    certificate.axvline(0.20, color="#9CA3AF", lw=0.9, ls="--", zorder=0)
    certificate.text(
        (0.03 if conditional_radii is not None else 0.96) * radius_limit,
        0.035,
        f"retained {result.mean_retention:.3f}",
        ha="left" if conditional_radii is not None else "right",
        va="bottom",
        color="#4B5563",
        fontsize=6.8,
        bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.88},
    )
    certificate.set(
        xlim=(0.0, radius_limit),
        ylim=(0.0, 0.68),
        xlabel=r"required $\ell_2$ radius",
        ylabel="correct certified fraction",
    )
    certificate.set_xticks(np.arange(0.0, radius_limit + 0.001, 0.1 if radius_limit <= 0.3 else 0.2))
    certificate.set_yticks([0.0, 0.2, 0.4, 0.6])
    certificate.grid(alpha=0.22)
    certificate.legend(
        loc="upper right" if conditional_radii is not None else "lower left",
        bbox_to_anchor=(0.99, 0.99) if conditional_radii is not None else (0.02, 0.06),
        frameon=False if conditional_radii is not None else True,
        handlelength=2.1,
        borderpad=0.35,
        labelspacing=0.25,
        fontsize=7.2,
    )
    panel_title(certificate, "B", "External certificate")
    save(figure, output_dir / "conditioning_failures")


def conditioning_supports(output_dir: Path) -> None:
    """Render the compact L-shaped witness without repeating Figure 1."""

    figure = plt.figure(figsize=(5.2, 2.3))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.05),
        left=0.08,
        right=0.97,
        bottom=0.18,
        top=0.82,
        wspace=0.30,
    )

    support = figure.add_subplot(grid[0, 0])
    event_rectangles = (
        Rectangle((0.0, 0.0), 0.633, 0.5),
        Rectangle((0.0, 0.5), 0.5, 0.5),
    )
    for rectangle in event_rectangles:
        rectangle.set_facecolor(LIGHT_ORANGE)
        rectangle.set_edgecolor(ORANGE)
        rectangle.set_linewidth(0.8)
        rectangle.set_hatch("//")
        support.add_patch(rectangle)
    support.add_patch(
        Rectangle(
            (0.633, 0.0),
            0.367,
            0.5,
            facecolor=LIGHT_BLUE,
            edgecolor="none",
        )
    )
    support.plot(
        [0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0],
        color=BLUE,
        lw=1.8,
    )
    support.plot([0.633, 0.633], [0.0, 0.5], color=ORANGE, lw=1.0)
    support.text(0.22, 0.72, r"$E$", color="#9A6700", fontsize=13)
    support.text(0.76, 0.22, r"$K\setminus E$", color=BLUE, fontsize=11, ha="center")
    support.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xticks=[0.0, 0.5, 1.0],
        yticks=[0.0, 0.5, 1.0],
        aspect="equal",
    )
    panel_title(support, "A", "Retained support and event")

    displacement = figure.add_subplot(grid[0, 1])
    nominal = np.asarray((0.65, 0.49))
    shifted = np.asarray((0.66, 0.49))
    sigma = 0.15

    def interval_mass(left: float, right: float, location: float) -> float:
        return float(ndtr((right - location) / sigma) - ndtr((left - location) / sigma))

    lower_y_mass = interval_mass(0.0, 0.5, nominal[1])
    upper_y_mass = interval_mass(0.5, 1.0, nominal[1])

    def event_probability(location: float) -> float:
        event_mass = (
            interval_mass(0.0, 0.633, location) * lower_y_mass
            + interval_mass(0.0, 0.5, location) * upper_y_mass
        )
        support_mass = (
            interval_mass(0.0, 1.0, location) * lower_y_mass
            + interval_mass(0.0, 0.5, location) * upper_y_mass
        )
        return event_mass / support_mass

    conditional_radius = sigma * float(ndtri(event_probability(nominal[0])))
    boundary_x = brentq(
        lambda location: event_probability(location) - 0.5,
        nominal[0],
        shifted[0],
    )
    displacement.axvspan(
        nominal[0] - conditional_radius,
        nominal[0] + conditional_radius,
        facecolor=LIGHT_ORANGE,
        edgecolor=RED,
        linestyle="--",
        linewidth=1.2,
        alpha=0.72,
        label=r"substituted interval at $a$",
    )
    displacement.axvline(boundary_x, color=GREY, linestyle=":", linewidth=1.1)
    displacement.plot(
        [nominal[0], shifted[0]],
        [0.50, 0.50],
        color="#202020",
        linewidth=1.2,
        zorder=2,
    )
    displacement.scatter(
        [nominal[0]],
        [0.50],
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        s=42,
        zorder=3,
    )
    displacement.scatter(
        [shifted[0]],
        [0.50],
        color=RED,
        marker="D",
        edgecolor="white",
        linewidth=0.6,
        s=40,
        zorder=3,
    )
    displacement.text(
        boundary_x - 0.00035,
        0.94,
        "decision\nboundary",
        color=GREY,
        ha="right",
        va="top",
        fontsize=7.0,
        linespacing=0.90,
    )
    displacement.text(
        nominal[0],
        0.58,
        r"nominal $a$",
        color=BLUE,
        ha="right",
        va="bottom",
        fontsize=7.2,
    )
    displacement.text(
        shifted[0],
        0.42,
        r"shifted $b$",
        color=RED,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    displacement.annotate(
        "",
        xy=(shifted[0], 0.29),
        xytext=(nominal[0], 0.29),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 1.0},
    )
    displacement.text(
        0.50,
        0.05,
        "$\\|a-b\\|_2=0.01$\n$<r_{\\rm cond}=0.01034$",
        transform=displacement.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.1,
        linespacing=0.90,
    )
    displacement.set(
        xlim=(0.637, 0.663),
        ylim=(0.0, 1.0),
        xlabel=r"first coordinate $z_1$",
        yticks=[],
    )
    displacement.grid(axis="x", alpha=0.20)
    panel_title(displacement, "B", "Certified displacement")
    save(figure, output_dir / "conditioning_failures_exact_support")


def two_band_covariance_radius_figure(output_dir: Path) -> None:
    """Plot the end-to-end certificate ordering for the nonconvex band family."""

    distances = np.linspace(0.0, 0.5, 251)
    selected = ndtr(1.0 - distances) - ndtr(0.9 - distances)
    runner = ndtr(1.0 + distances) - ndtr(0.9 + distances)
    occupancy = selected + runner
    selected_probability = selected / occupancy
    runner_probability = runner / occupancy

    substituted = ndtri(selected_probability)
    joint_mass = 0.5 * (ndtri(selected) - ndtri(runner))
    categorical_kl = (
        selected_probability * np.log(2.0 * selected_probability)
        + runner_probability * np.log(2.0 * runner_probability)
    )
    covariance = np.sqrt(np.maximum(0.0, 2.0 * categorical_kl))

    figure, axis = plt.subplots(figsize=(4.7, 2.55), constrained_layout=True)
    axis.plot(
        distances,
        substituted,
        color=RED,
        ls="--",
        lw=1.9,
        label="substituted conditional",
    )
    axis.plot(
        distances,
        distances,
        color="#202020",
        lw=1.7,
        label="exact boundary",
    )
    axis.plot(
        distances,
        covariance,
        color=BLUE,
        lw=1.9,
        label="conditional forward KL",
    )
    axis.plot(
        distances,
        joint_mass,
        color=ORANGE,
        ls="-.",
        lw=1.9,
        label="joint retained mass",
    )

    audited_index = int(np.argmin(np.abs(distances - 0.1)))
    for values, color in (
        (substituted, RED),
        (distances, "#202020"),
        (covariance, BLUE),
        (joint_mass, ORANGE),
    ):
        axis.scatter(
            [distances[audited_index]],
            [values[audited_index]],
            s=35,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    axis.axvline(0.1, color=GREY, ls=":", lw=0.9, zorder=0)
    axis.text(
        0.97,
        0.06,
        r"retention $5.08\%$ at $\delta=0.1$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        color="#202020",
    )
    axis.set(
        xlim=(0.0, 0.5),
        ylim=(0.0, 0.62),
        xlabel=r"nominal distance $\delta$",
        ylabel="radius",
    )
    axis.set_xticks(np.arange(0.0, 0.51, 0.1))
    axis.grid(alpha=0.22)
    axis.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        columnspacing=0.9,
        handlelength=2.2,
    )
    save(figure, output_dir / "two_band_covariance_radius")


def reach_avoid_figure(output_dir: Path, result_path: Path) -> None:
    payload = json.loads(result_path.read_text())
    rows = payload["rows"]
    imported_ratio = np.asarray(
        [row["imported_radius"] / row["vertical_boundary_distance"] for row in rows]
    )
    joint_ratio = np.asarray([row["joint_to_boundary_ratio"] for row in rows])
    boundary = np.asarray([row["vertical_boundary_distance"] for row in rows])
    imported = np.asarray([row["imported_radius"] for row in rows])
    joint = np.asarray([row["joint_mass_radius"] for row in rows])
    percentile = 100.0 * (np.arange(len(rows)) + 0.5) / len(rows)

    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), constrained_layout=True)
    left, right = axes
    left.plot(percentile, np.sort(imported_ratio), color=RED, label="substituted")
    left.plot(percentile, np.sort(joint_ratio), color=BLUE, label="joint mass")
    left.axhline(1.0, color="black", ls="--", lw=1.0, label="located boundary")
    left.set(
        xlabel="reference-state percentile",
        ylabel="radius / boundary distance",
        xlim=(0, 100),
        ylim=(0.55, max(1.38, float(imported_ratio.max()) + 0.03)),
    )
    left.grid(alpha=0.22)
    left.legend(loc="upper left", frameon=True)
    left.text(98, 1.025, "unsafe side", color=RED, ha="right", va="bottom", fontsize=7.3)
    panel_title(left, "A", "All 448 fixed states")

    right.scatter(boundary, imported, s=11, alpha=0.45, color=RED, label="substituted")
    right.scatter(boundary, joint, s=11, alpha=0.45, color=BLUE, label="joint mass")
    limit = max(float(boundary.max()), float(imported.max())) * 1.03
    right.plot([0, limit], [0, limit], color="black", ls="--", lw=1.0)
    right.set(
        xlabel="located boundary distance",
        ylabel="certified radius",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    right.grid(alpha=0.22)
    right.legend(loc="upper left", frameon=True)
    panel_title(right, "B", "Absolute radii")
    save(figure, output_dir / "reach_avoid_recertification")


def random_box_figure(output_dir: Path, result_path: Path) -> None:
    """Render the frozen product-box stress test and paired fixed control."""

    rows = read_rows(result_path)
    whole_rows = [row for row in rows if row["method"] == "whole_conditioning"]
    fixed_rows = [row for row in rows if row["method"] == "anchor_fixed"]
    if len(whole_rows) != 6_144 or len(fixed_rows) != 6_144:
        raise ValueError("unexpected product-box factor-cell count")

    protocols = ("iid", "spectral_stress")
    dimensions = (2, 5, 10, 20)
    grouped = {
        (protocol, dimension): np.asarray(
            [
                float(row["total_ratio"])
                for row in whole_rows
                if row["direction_protocol"] == protocol
                and int(row["dimension"]) == dimension
            ]
        )
        for protocol in protocols
        for dimension in dimensions
    }
    if sum(value.size for value in grouped.values()) != 6_144:
        raise ValueError("incomplete product-box grouping")
    iid_exceedances = sum(
        float(row["total_ratio"]) > 1.0
        for row in whole_rows
        if row["direction_protocol"] == "iid"
    )
    spectral_exceedances = sum(
        float(row["total_ratio"]) > 1.0
        for row in whole_rows
        if row["direction_protocol"] == "spectral_stress"
    )
    if (iid_exceedances, spectral_exceedances) != (0, 10):
        raise ValueError("frozen exceedance counts changed")

    key_fields = (
        "ensemble_version",
        "geometry_family",
        "direction_protocol",
        "dimension",
        "mode_count",
        "replicate",
        "geometry_seed",
        "direction_seed",
        "volume_ratio_cap",
        "sigma",
        "separation_ratio",
    )
    paired: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in whole_rows + fixed_rows:
        key = tuple(row[field] for field in key_fields)
        paired.setdefault(key, {})[row["method"]] = float(row["total_ratio"])
    if len(paired) != 6_144 or any(len(values) != 2 for values in paired.values()):
        raise ValueError("product-box pairing failed")
    whole_values = np.asarray(
        [values["whole_conditioning"] for values in paired.values()]
    )
    fixed_values = np.asarray([values["anchor_fixed"] for values in paired.values()])
    if not np.all(fixed_values < whole_values):
        raise ValueError("frozen paired ordering changed")
    if not np.isclose(whole_values.max(), 1.044797665628306):
        raise ValueError("unexpected whole-conditioning maximum")
    if not np.isclose(fixed_values.max(), 0.8906681563287747):
        raise ValueError("unexpected fixed-weight maximum")

    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.62), constrained_layout=True)
    ratio_axis, paired_axis = axes
    offsets = (-0.16, 0.16)
    colors = ("#56B4E9", ORANGE)
    hatches = ("//", "\\\\")
    widths = 0.26
    for protocol, offset, color, hatch in zip(
        protocols, offsets, colors, hatches, strict=True
    ):
        values = [grouped[(protocol, dimension)] for dimension in dimensions]
        positions = np.arange(len(dimensions), dtype=float) + offset
        artists = ratio_axis.boxplot(
            values,
            positions=positions,
            widths=widths,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": RED, "lw": 1.0},
            boxprops={"color": "black", "lw": 0.8},
            whiskerprops={"color": "black", "lw": 0.8},
            capprops={"color": "black", "lw": 0.8},
        )
        for box in artists["boxes"]:
            box.set_facecolor(color)
            box.set_hatch(hatch)
        for position, dimension in zip(positions, dimensions, strict=True):
            exceedances = grouped[(protocol, dimension)]
            exceedances = exceedances[exceedances > 1.0]
            if exceedances.size:
                ratio_axis.scatter(
                    np.full(exceedances.size, position),
                    exceedances,
                    marker="x",
                    color=RED,
                    s=22,
                    linewidth=1.0,
                    zorder=4,
                )
    ratio_axis.axhline(1.0, color="black", ls="--", lw=1.0)
    ratio_axis.set_xticks(np.arange(len(dimensions)), dimensions)
    ratio_axis.set_xlabel("ambient dimension")
    ratio_axis.set_ylabel("KL ratio to Gaussian")
    ratio_axis.set_ylim(0.0, 1.07)
    panel_title(ratio_axis, "A", "Whole conditioning")
    ratio_axis.grid(axis="y", color="#D1D5DB", lw=0.5, alpha=0.7)
    legend_handles = [
        Rectangle((0, 0), 1, 1, fc=colors[0], ec="black", hatch=hatches[0]),
        Rectangle((0, 0), 1, 1, fc=colors[1], ec="black", hatch=hatches[1]),
        plt.Line2D([], [], color=RED, marker="x", ls="none"),
    ]
    figure.legend(
        legend_handles,
        ["independent", "covariance direction", "ratio above one"],
        loc="outside lower center",
        ncol=3,
        columnspacing=0.9,
        handlelength=1.25,
        handletextpad=0.35,
        fontsize=7.2,
        frameon=False,
    )

    protocol_by_key = [key[2] for key in paired]
    for protocol, color, marker in zip(protocols, colors, ("o", "^"), strict=True):
        mask = np.asarray([value == protocol for value in protocol_by_key])
        paired_axis.scatter(
            whole_values[mask],
            fixed_values[mask],
            color=color,
            marker=marker,
            s=7,
            alpha=0.23,
            linewidth=0.0,
            rasterized=True,
            label="independent" if protocol == "iid" else "covariance direction",
        )
    paired_axis.plot([0.25, 1.06], [0.25, 1.06], color="black", ls="--", lw=0.9)
    paired_axis.axvline(1.0, color=RED, ls=":", lw=0.9)
    paired_axis.set_xlim(0.25, 1.06)
    paired_axis.set_ylim(0.0, 0.93)
    paired_axis.set_xlabel("whole-conditioning ratio")
    paired_axis.set_ylabel("fixed-anchor ratio")
    panel_title(paired_axis, "B", "Fixed-anchor control")
    paired_axis.grid(color="#D1D5DB", lw=0.5, alpha=0.7)

    save(figure, output_dir / "random_box_breadth")


def retained_pair_margin(
    counts: list[int], retained: int, original_label: int, alternative_label: int
) -> float:
    """Return the original-minus-alternative share among retained proposals."""
    if retained <= 0:
        raise ValueError("retained count must be positive")
    if len(counts) <= max(original_label, alternative_label):
        raise ValueError("label index is outside the count vector")
    return float(counts[original_label] - counts[alternative_label]) / float(retained)


def _plot_accuracy_curves(
    axis: plt.Axes,
    curves: dict[str, dict[str, float]],
    *,
    title_letter: str,
    title: str,
    show_ylabel: bool,
) -> list[plt.Line2D]:
    methods = (
        ("released", "conditional substitution", RED, "--", "D"),
        ("algorithm_one", "Algorithm 1", BLUE, "-", "o"),
        ("unfiltered", "unfiltered", GREEN, "-.", "^"),
    )
    artists: list[plt.Line2D] = []
    for key, label, color, linestyle, marker in methods:
        ordered = sorted((float(radius), float(value)) for radius, value in curves[key].items())
        radii = np.asarray([item[0] for item in ordered])
        values = np.asarray([item[1] for item in ordered])
        (artist,) = axis.plot(
            radii,
            values,
            color=color,
            ls=linestyle,
            marker=marker,
            markevery=max(1, len(radii) // 5),
            label=label,
        )
        artists.append(artist)
    axis.set(
        xlim=(-0.015, 0.815),
        ylim=(0.0, 0.8),
        xlabel=r"required $\ell_2$ radius",
        ylabel="correct test fraction" if show_ylabel else None,
    )
    axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.grid(alpha=0.22)
    panel_title(axis, title_letter, title)
    return artists


def auditvotes_figure(
    output_dir: Path,
    summary_path: Path,
    cifar101_summary_path: Path,
    confirmation_path: Path,
) -> None:
    """Plot cross-dataset recertification and the fixed-cohort attack audit."""

    cifar10 = json.loads(summary_path.read_text())
    cifar101 = json.loads(cifar101_summary_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    if int(cifar10.get("images", -1)) != 10_000:
        raise ValueError("CIFAR-10 summary does not cover 10,000 images")
    if cifar10.get("strict_radius_comparison") is not True:
        raise ValueError("CIFAR-10 summary does not use strict radius comparisons")
    cifar10_rows_path = summary_path.parent / "per_image.jsonl"
    if cifar10.get("per_image_sha256") != sha256_path(cifar10_rows_path):
        raise ValueError("CIFAR-10 per-image digest changed")
    cifar101_manifest_path = cifar101_summary_path.parent / "manifest.json"
    cifar101_manifest = json.loads(cifar101_manifest_path.read_text())
    if (
        cifar101_manifest.get("status") != "evaluation_complete"
        or int(cifar101_manifest.get("completed", -1)) != 2_021
        or int(cifar101.get("images", -1)) != 2_021
    ):
        raise ValueError("CIFAR-10.1 evaluation is incomplete")
    if cifar101_manifest.get("summary_sha256") != sha256_path(cifar101_summary_path):
        raise ValueError("CIFAR-10.1 summary digest changed")
    cifar101_rows_path = cifar101_summary_path.parent / "per_image.jsonl"
    if cifar101_manifest.get("per_image_sha256") != sha256_path(cifar101_rows_path):
        raise ValueError("CIFAR-10.1 per-image digest changed")
    if confirmation.get("status") != "confirmation_complete":
        raise ValueError("fixed-cohort confirmation is not complete")
    confirmation_rows = confirmation.get("rows", [])
    confirmation_analysis = confirmation.get("analysis", {})
    if (
        len(confirmation_rows) != 128
        or int(
            confirmation_analysis.get("primary_fixed_cohort_yield", {}).get(
                "trials", -1
            )
        )
        != 128
    ):
        raise ValueError("fixed-cohort confirmation does not cover 128 images")
    if int(confirmation_analysis.get("positive_screen_count", -1)) != sum(
        bool(row.get("screen_positive", False)) for row in confirmation_rows
    ):
        raise ValueError("fixed-cohort screen count is inconsistent")
    if int(confirmation_analysis.get("confirmed_screen_count", -1)) != sum(
        bool(row.get("confirmation_performed", False))
        for row in confirmation_rows
    ):
        raise ValueError("fixed-cohort confirmation count is inconsistent")
    if int(confirmation_analysis.get("verified_count", -1)) != sum(
        bool(row.get("population_violation_verified", False))
        for row in confirmation_rows
    ):
        raise ValueError("fixed-cohort verified count is inconsistent")
    if int(
        confirmation_analysis.get("primary_fixed_cohort_yield", {}).get(
            "successes", -1
        )
    ) != int(confirmation_analysis["verified_count"]):
        raise ValueError("fixed-cohort primary success count is inconsistent")

    figure = plt.figure(figsize=(5.5, 4.45))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.82),
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.865,
        hspace=0.62,
        wspace=0.36,
    )
    cifar10_axis = figure.add_subplot(grid[0, 0])
    cifar101_axis = figure.add_subplot(grid[0, 1])
    cohort_axis = figure.add_subplot(grid[1, 0])
    endpoint_axis = figure.add_subplot(grid[1, 1])

    legend_artists = _plot_accuracy_curves(
        cifar10_axis,
        {
            "released": cifar10["released_certified_accuracy"],
            "algorithm_one": cifar10["simultaneous_joint_certified_accuracy"],
            "unfiltered": cifar10["gaussian_certified_accuracy"],
        },
        title_letter="A",
        title="CIFAR-10",
        show_ylabel=True,
    )
    _plot_accuracy_curves(
        cifar101_axis,
        {
            "released": cifar101["methods"]["released"]["certified_accuracy"],
            "algorithm_one": cifar101["methods"]["algorithm_one_hybrid"]["certified_accuracy"],
            "unfiltered": cifar101["methods"]["unfiltered"]["certified_accuracy"],
        },
        title_letter="B",
        title="CIFAR-10.1 v4",
        show_ylabel=False,
    )
    figure.legend(
        legend_artists,
        [artist.get_label() for artist in legend_artists],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=7.4,
        handlelength=2.6,
        columnspacing=1.5,
    )

    analysis = confirmation["analysis"]
    cohort_count = int(analysis["primary_fixed_cohort_yield"]["trials"])
    positive_count = sum(
        bool(row["released_positive"]) for row in confirmation["rows"]
    )
    screen_count = int(analysis["positive_screen_count"])
    verified_count = int(analysis["verified_count"])
    stages = ["fixed\ncohort", "positive\nradius", "positive\nscreen", "verified"]
    counts = [cohort_count, positive_count, screen_count, verified_count]
    bars = cohort_axis.bar(
        np.arange(4),
        counts,
        width=0.68,
        color=[GREY, ORANGE, PURPLE, BLUE],
        edgecolor="white",
        linewidth=0.55,
    )
    cohort_axis.bar_label(bars, labels=[str(value) for value in counts], padding=2, fontsize=7.8)
    cohort_axis.set(
        ylim=(0.0, max(1.0, cohort_count * 1.18)),
        ylabel="images",
    )
    cohort_axis.set_xticks(np.arange(4), stages)
    cohort_axis.grid(axis="y", alpha=0.22)
    cohort_axis.spines[["top", "right"]].set_visible(False)
    panel_title(cohort_axis, "C", "Fixed-cohort accounting")

    confirmed = [
        row for row in confirmation["rows"] if bool(row.get("confirmation_performed", False))
    ]
    ratios: list[tuple[float, bool, bool]] = []
    for row in confirmed:
        radius = float(row["fresh_substituted_radius_lower"])
        distance = float(row["l2_distance_recomputed"])
        endpoint_winners = all(bool(endpoint["winner_verified"]) for endpoint in row["endpoints"])
        ratio = np.inf if radius <= 0.0 else distance / radius
        ratios.append((ratio, bool(row["population_violation_verified"]), endpoint_winners))
    ratios.sort(key=lambda item: item[0])
    finite_ratios = np.asarray([item[0] for item in ratios if np.isfinite(item[0])])
    if len(ratios) == 0:
        endpoint_axis.text(
            0.5,
            0.5,
            "No positive screens",
            ha="center",
            va="center",
            transform=endpoint_axis.transAxes,
            color=GREY,
        )
        endpoint_axis.set(xticks=[], yticks=[])
    else:
        for rank, (ratio, verified, endpoint_winners) in enumerate(ratios, start=1):
            if not np.isfinite(ratio):
                continue
            if verified:
                endpoint_axis.scatter(
                    rank,
                    ratio,
                    s=28,
                    marker="o",
                    facecolor=BLUE,
                    edgecolor=BLUE,
                    linewidth=0.9,
                    zorder=3,
                )
            elif endpoint_winners:
                endpoint_axis.scatter(
                    rank,
                    ratio,
                    s=30,
                    marker="D",
                    facecolor="white",
                    edgecolor=ORANGE,
                    linewidth=0.9,
                    zorder=3,
                )
            else:
                endpoint_axis.scatter(
                    rank,
                    ratio,
                    s=28,
                    marker="x",
                    color=GREY,
                    linewidth=0.9,
                    zorder=3,
                )
        endpoint_axis.axhline(1.0, color="black", ls="--", lw=0.9, zorder=0)
        maximum = max(1.15, float(finite_ratios.max()) * 1.12) if finite_ratios.size else 1.15
        endpoint_axis.set(
            xlim=(0.2, len(ratios) + 0.8),
            ylim=(0.0, maximum),
            xlabel="confirmed pair rank",
            ylabel="shift / fresh radius bound",
        )
        endpoint_axis.set_xticks(
            sorted(
                set(
                    np.linspace(
                        1,
                        len(ratios),
                        min(4, len(ratios)),
                        dtype=int,
                    )
                )
            )
        )
        endpoint_axis.grid(alpha=0.22)
    panel_title(endpoint_axis, "D", "Fresh endpoint tests")
    endpoint_axis.text(
        0.98,
        0.04,
        r"familywise $\alpha\leq 0.001$",
        transform=endpoint_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=GREY,
    )
    save(figure, output_dir / "auditvotes_recertification")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reach", type=Path, default=DEFAULT_REACH)
    parser.add_argument(
        "--auditvotes-summary", type=Path, default=DEFAULT_AUDITVOTES_SUMMARY
    )
    parser.add_argument(
        "--cifar101-summary", type=Path, default=DEFAULT_CIFAR101_SUMMARY
    )
    parser.add_argument(
        "--cifar101-confirmation", type=Path, default=DEFAULT_CIFAR101_CONFIRMATION
    )
    parser.add_argument("--random-box", type=Path, default=DEFAULT_RANDOM_BOX)
    args = parser.parse_args()
    configure_style()
    conditioning_failures(args.output)
    conditioning_supports(args.output)
    two_band_covariance_radius_figure(args.output)
    reach_avoid_figure(args.output, args.reach)
    auditvotes_figure(
        args.output,
        args.auditvotes_summary,
        args.cifar101_summary,
        args.cifar101_confirmation,
    )
    random_box_figure(args.output, args.random_box)
    print(args.output / "conditioning_failures.pdf")
    print(args.output / "conditioning_failures_exact_support.pdf")
    print(args.output / "two_band_covariance_radius.pdf")
    print(args.output / "reach_avoid_recertification.pdf")
    print(args.output / "auditvotes_recertification.pdf")
    print(args.output / "random_box_breadth.pdf")


if __name__ == "__main__":
    main()
