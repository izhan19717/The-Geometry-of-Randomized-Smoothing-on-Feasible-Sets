import numpy as np

from experiments.iclr2027_covariance_mechanism_holdout import (
    run_holdout,
    summarize_holdout,
)


def _small_protocol():
    return {
        "design": {
            "dimensions": [10],
            "replicates_per_dimension": 2,
            "mode_count": 16,
            "volume_ratio_cap": 1.0,
            "geometry_family": "coherent",
            "sigmas": [0.25],
            "normalized_distances": [0.05],
            "quadrature_order": 16,
        },
        "seeds": {
            "geometry_root": 20260713,
            "direction_root": 73102062,
        },
    }


def test_small_holdout_has_paired_rows_and_exact_path_identity():
    protocol = _small_protocol()
    rows = run_holdout(protocol)
    assert len(rows) == 4
    assert {row["direction_protocol"] for row in rows} == {
        "independent",
        "top covariance",
    }
    assert max(float(row["path_identity_abs_error"]) for row in rows) < 1e-12
    assert max(
        abs(
            float(row["finite_kl_ratio"])
            - float(row["integrated_covariance_ratio"])
        )
        for row in rows
    ) < 1e-8

    summary = summarize_holdout(rows, protocol)
    assert summary["independent_geometry_count"] == 2
    assert len(summary["unit_rows"]) == 2
    assert np.isfinite(summary["spearman_local_finite_all_repeated_cells"])
