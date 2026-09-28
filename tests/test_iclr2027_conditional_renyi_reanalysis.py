"""Regression checks of the retained-count reanalysis and plotting."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from experiments.iclr2027_conditional_renyi_reanalysis import (
    DEFAULT_SOURCE,
    EXPECTED_ROWS_SHA256,
    analyze_rows,
    matched_call_baseline,
)
from experiments.plot_iclr2027_conditional_renyi import (
    comparison_figure,
    configure_style,
)


@pytest.fixture(scope="module")
def rows():
    path = DEFAULT_SOURCE / "per_image.jsonl"
    assert sha256(path.read_bytes()).hexdigest() == EXPECTED_ROWS_SHA256
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def result(rows):
    return analyze_rows(rows)


def test_all_images_and_counts_reproduce(result):
    records, summary = result
    assert len(records) == summary["record_count"] == 2000
    curves = {r["radius"]: r for r in summary["curves"]}
    for radius, expected in (
        (0.2, (829, 888, 892, 725, 948)),
        (0.3, (0, 698, 713, 417, 814)),
        (0.5, (0, 329, 381, 0, 529)),
    ):
        assert (
            tuple(
                curves[radius][m]
                for m in (
                    "forward_kl",
                    "reverse_kl",
                    "renyi",
                    "joint_mass",
                    "unfiltered",
                )
            )
            == expected
        )
    assert summary["strong_separation_counts"] == {
        "forward_kl": 329,
        "reverse_kl": 869,
        "renyi": 877,
    }
    assert summary["new_model_evaluations"] == 0
    assert summary["minimum_reverse_minus_forward"] >= -1e-12


def test_modified_probability_bound_is_rejected(rows):
    altered = deepcopy(rows[:1])
    altered[0]["conditional_selected_probability_lower"] += 0.01
    with pytest.raises(ValueError, match="disagree"):
        analyze_rows(altered)


def test_matched_calls_reproduce(rows):
    result = matched_call_baseline(rows)
    assert result["model_calls"] == 94_064_290
    assert result["correct_counts"]["0.2"] == 945
    assert result["correct_counts"]["0.5"] == 522
    assert result["label_selection_changes"] == 6
    assert result["familywise_error"] == pytest.approx(0.0003125)
    assert (
        result["confidence_scope"] == "separate family for matched-call baseline only"
    )


def test_vector_comparison_figure(result, tmp_path):
    configure_style()
    comparison_figure(result[0], tmp_path)
    assert (
        (tmp_path / "conditional_renyi_reanalysis.pdf").read_bytes().startswith(b"%PDF")
    )
    assert (tmp_path / "conditional_renyi_reanalysis.png").stat().st_size > 20000
