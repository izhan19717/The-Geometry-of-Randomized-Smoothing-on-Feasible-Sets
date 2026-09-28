import json
from types import SimpleNamespace

import numpy as np

from experiments.plot_iclr2027_filtered_smoothing import (
    DEFAULT_AUDITVOTES_SUMMARY,
    DEFAULT_CIFAR101_SUMMARY,
    DEFAULT_RANDOM_BOX,
    DEFAULT_REACH,
    auditvotes_figure,
    conditioning_failures,
    conditioning_supports,
    configure_style,
    random_box_figure,
    reach_avoid_figure,
    two_band_covariance_radius_figure,
)


def test_submission_figure_build_emits_six_vector_figures(tmp_path):
    configure_style()
    synthetic_external = SimpleNamespace(
        filtered_correct=np.asarray([True, True, False, True]),
        unfiltered_correct=np.asarray([True, True, True, False]),
        covariance_radius_lower=np.asarray([0.28, 0.22, 0.30, 0.08]),
        joint_mass_radius_lower=np.asarray([0.24, 0.17, 0.25, 0.05]),
        unfiltered_radius_lower=np.asarray([0.30, 0.21, 0.18, 0.10]),
        mean_retention=0.428,
    )
    conditioning_failures(tmp_path, external_result=synthetic_external)
    conditioning_supports(tmp_path)
    two_band_covariance_radius_figure(tmp_path)
    confirmation = tmp_path / "confirmation.json"
    confirmation_rows = [
        {
            "released_positive": True,
            "screen_positive": True,
            "confirmation_performed": True,
            "fresh_substituted_radius_lower": 0.4,
            "l2_distance_recomputed": 0.3,
            "population_violation_verified": True,
            "endpoints": [
                {"winner_verified": True},
                {"winner_verified": True},
            ],
        },
        {
            "released_positive": True,
            "screen_positive": True,
            "confirmation_performed": True,
            "fresh_substituted_radius_lower": 0.35,
            "l2_distance_recomputed": 0.4,
            "population_violation_verified": False,
            "endpoints": [
                {"winner_verified": True},
                {"winner_verified": False},
            ],
        },
    ]
    confirmation_rows.extend(
        {
            "released_positive": index % 7 != 0,
            "screen_positive": False,
            "confirmation_performed": False,
            "population_violation_verified": False,
        }
        for index in range(126)
    )
    confirmation.write_text(
        json.dumps(
            {
                "status": "confirmation_complete",
                "analysis": {
                    "primary_fixed_cohort_yield": {"trials": 128, "successes": 1},
                    "positive_screen_count": 2,
                    "confirmed_screen_count": 2,
                    "verified_count": 1,
                },
                "rows": confirmation_rows,
            }
        )
    )
    auditvotes_figure(
        tmp_path,
        DEFAULT_AUDITVOTES_SUMMARY,
        DEFAULT_CIFAR101_SUMMARY,
        confirmation,
    )
    reach_avoid_figure(tmp_path, DEFAULT_REACH)
    random_box_figure(tmp_path, DEFAULT_RANDOM_BOX)

    for stem in (
        "conditioning_failures",
        "conditioning_failures_exact_support",
        "two_band_covariance_radius",
        "auditvotes_recertification",
        "reach_avoid_recertification",
        "random_box_breadth",
    ):
        pdf = tmp_path / f"{stem}.pdf"
        png = tmp_path / f"{stem}.png"
        assert pdf.read_bytes().startswith(b"%PDF")
        assert pdf.stat().st_size > 10_000
        assert png.read_bytes().startswith(b"\x89PNG")
        assert png.stat().st_size > 20_000
