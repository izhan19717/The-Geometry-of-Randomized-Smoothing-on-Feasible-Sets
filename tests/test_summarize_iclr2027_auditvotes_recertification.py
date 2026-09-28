from pathlib import Path

import pytest

from experiments.summarize_iclr2027_auditvotes_recertification import summarize


ROWS = Path("outputs/auditvotes_recertification_full_20260906/per_image.jsonl")


def test_retained_counts_reproduce_algorithm_one_summary():
    result = summarize(
        ROWS,
        radii=[0.0, 0.3, 0.5, 0.7],
        alpha=0.001,
        sigma=0.25,
        num_classes=10,
    )

    assert result["simultaneous_joint_positive_fraction"] == pytest.approx(0.8875)
    assert result["median_simultaneous_joint_radius"] == pytest.approx(
        0.3330324, abs=1e-7
    )
    assert result["simultaneous_joint_certified_accuracy"] == {
        "0.00": pytest.approx(0.7423),
        "0.30": pytest.approx(0.5153),
        "0.50": pytest.approx(0.3202),
        "0.70": pytest.approx(0.1394),
    }
    assert (
        result["simultaneous_joint_reconstruction"][
            "maximum_endpoint_reconstruction_error"
        ]
        < 1e-12
    )
