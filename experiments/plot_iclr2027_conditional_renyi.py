"""Plot conditional divergence certificates from verified CIFAR-10.2 counts."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.iclr2027_conditional_renyi_reanalysis import (
    DEFAULT_SOURCE,
    EXPECTED_ROWS_SHA256,
    analyze_rows,
)
from experiments.plot_iclr2027_filtered_smoothing import (
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    conditioning_failures,
    configure_style,
    panel_title,
    save,
)
from experiments.plot_iclr2027_global_band_cifar102_confirmation import (
    load_final_result,
)


def comparison_figure(records: list[dict], output_directory: Path) -> None:
    """Show the full radius curves and paired lower-versus-upper bounds."""
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 3.05))
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.89, wspace=0.38)
    grid = np.linspace(0, 0.8, 801)
    methods = (
        ("renyi", "Conditional Rényi", GREEN, "-"),
        ("reverse_kl", "Reverse KL", PURPLE, "-."),
        ("forward_kl", "Forward KL", ORANGE, (0, (1, 1))),
        ("joint_mass", "Joint mass", BLUE, "--"),
        ("unfiltered", "Unfiltered", GREY, ":"),
    )
    for method, label, color, style in methods:
        key = "unfiltered_correct" if method == "unfiltered" else "filtered_correct"
        radii = np.asarray([r[method] for r in records])
        correct = np.asarray([r[key] for r in records])
        fractions = np.asarray(
            [np.mean(correct & (radii > 0) & (radii >= x)) for x in grid]
        )
        axes[0].plot(grid, fractions, color=color, ls=style, label=label, lw=1.5)
    axes[0].set(
        xlim=(0, 0.8),
        ylim=(0, 0.66),
        xlabel=r"required $\ell_2$ radius",
        ylabel="correct certified fraction",
    )
    axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axes[0].set_yticks([0, 0.2, 0.4, 0.6])
    axes[0].grid(alpha=0.22)
    panel_title(axes[0], "A", "Divergence comparison")

    eligible = [
        r for r in records if r["filtered_correct"] and r["joint_mass_upper_is_finite"]
    ]
    x = np.asarray([r["joint_mass_upper"] for r in eligible])
    y = np.asarray([r["renyi"] for r in eligible])
    strong = y - x >= 0.01
    limit = float(np.ceil(max(x.max(), y.max()) * 10) / 10)
    for mask, color in ((~strong, "#B9C0C8"), (strong, GREEN)):
        axes[1].scatter(
            x[mask],
            y[mask],
            color=color,
            s=8,
            alpha=0.55,
            linewidths=0,
            rasterized=False,
        )
    axes[1].plot([0, limit], [0, limit], color=GREY, ls="--", lw=0.85)
    axes[1].set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="joint-mass upper radius",
        ylabel="conditional Rényi lower radius",
    )
    axes[1].text(
        0.04,
        0.97,
        f"{strong.sum():,} of {len(eligible):,} pairs\nexceed the upper bound\nby at least 0.01",
        transform=axes[1].transAxes,
        va="top",
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    axes[1].grid(alpha=0.18)
    panel_title(axes[1], "B", "Paired certificate bounds")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.005),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        columnspacing=1.4,
        handlelength=2.5,
    )
    save(figure, output_directory / "conditional_renyi_reanalysis")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("_ICLR_2027__Feasibility_Breaks_Smoothing/figures"),
    )
    args = parser.parse_args()
    rows_path = args.source_directory / "per_image.jsonl"
    if sha256(rows_path.read_bytes()).hexdigest() != EXPECTED_ROWS_SHA256:
        raise ValueError("the specified completed row file has changed")
    result = load_final_result(
        args.source_directory / "summary.json",
        rows_path,
        args.source_directory / "manifest.json",
    )
    records, _ = analyze_rows(list(result.rows))
    configure_style()
    conditioning_failures(
        args.output_directory,
        external_result=result,
        conditional_radii=np.asarray([r["renyi"] for r in records]),
        conditional_label="conditional Rényi",
        radius_limit=0.8,
    )
    comparison_figure(records, args.output_directory)


if __name__ == "__main__":
    main()
