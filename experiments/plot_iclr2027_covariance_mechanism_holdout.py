"""Plot the frozen ICLR covariance holdout without rerunning its experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROWS = Path(
    "outputs/iclr2027_covariance_mechanism_holdout_20260912/factor_rows.csv"
)
DEFAULT_OUTPUT = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/figures/covariance_mechanism_holdout"
)
PROTOCOLS = (
    ("independent", "#0072B2", "o"),
    ("top covariance", "#D55E00", "^"),
)


def load_frozen_rows(path: Path) -> list[dict[str, str]]:
    """Check provenance and the factor-row counts used in the manuscript."""

    if not hashlib.sha256(path.read_bytes()).hexdigest().startswith("3624c187e480"):
        raise ValueError("the frozen covariance holdout row file has changed")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1_024:
        raise ValueError("expected 1,024 frozen factor rows")
    for name, _, _ in PROTOCOLS:
        if sum(row["direction_protocol"] == name for row in rows) != 512:
            raise ValueError(f"unexpected {name} factor-row count")
    exceedances = [row for row in rows if float(row["finite_kl_ratio"]) > 1.0]
    base_geometries = {
        (row["dimension"], row["geometry_seed"]) for row in exceedances
    }
    if (
        len(exceedances) != 27
        or len(base_geometries) != 14
        or any(row["direction_protocol"] != "top covariance" for row in exceedances)
    ):
        raise ValueError("the frozen exceedance result has changed")
    return rows


def plot_frozen_rows(rows: list[dict[str, str]], output_stem: Path) -> None:
    """Show both the full comparison and its otherwise hidden signed residual."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.1,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (comparison, residual) = plt.subplots(
        1, 2, figsize=(5.5, 2.52), constrained_layout=True
    )
    for name, color, marker in PROTOCOLS:
        group = [row for row in rows if row["direction_protocol"] == name]
        anchor = np.asarray([float(row["anchor_covariance_ratio"]) for row in group])
        finite = np.asarray([float(row["finite_kl_ratio"]) for row in group])
        comparison.scatter(
            anchor,
            finite,
            s=9,
            alpha=0.48,
            color=color,
            marker=marker,
            linewidths=0,
            rasterized=True,
            label=name,
        )
        residual.scatter(
            finite,
            finite - anchor,
            s=9,
            alpha=0.48,
            color=color,
            marker=marker,
            linewidths=0,
            rasterized=True,
        )

    limits = (0.37, 1.09)
    comparison.plot(limits, limits, color="black", ls="--", lw=0.9)
    comparison.axhline(1.0, color="#6B7280", ls=":", lw=0.85)
    comparison.axvline(1.0, color="#6B7280", ls=":", lw=0.85)
    comparison.set(
        xlim=limits,
        ylim=limits,
        xlabel="anchor covariance / Gaussian",
        ylabel="finite KL / Gaussian",
        title="A  Full comparison",
    )
    residual.axhline(0.0, color="black", ls="--", lw=0.9)
    residual.axvline(1.0, color="#6B7280", ls=":", lw=0.85)
    residual.set(
        xlim=limits,
        ylim=(-0.011, 0.019),
        xlabel="finite KL / Gaussian",
        ylabel="finite - anchor ratio",
        title="B  Signed residual",
    )
    for axis in (comparison, residual):
        axis.title.set_horizontalalignment("left")
        axis.title.set_position((0.0, 1.0))
        axis.grid(color="#CBD5E1", alpha=0.5, lw=0.5)
        axis.set_axisbelow(True)
    figure.legend(
        *comparison.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        metadata={
            "Title": "Covariance mechanism holdout",
            "Author": "Anonymous",
            "Subject": "ICLR 2027 submission figure",
            "Creator": "Matplotlib",
            "CreationDate": datetime(2026, 9, 12, tzinfo=timezone.utc),
            "ModDate": datetime(2026, 9, 12, tzinfo=timezone.utc),
        },
    )
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_frozen_rows(load_frozen_rows(args.rows), args.output)


if __name__ == "__main__":
    main()
