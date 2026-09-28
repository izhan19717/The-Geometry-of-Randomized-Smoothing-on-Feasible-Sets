import math

import numpy as np

from experiments.e_random_box_union_ensemble import (
    paired_imbalance_differences,
    run_ensemble,
    summarize_whole_strata,
)
from feasible_robustness.random_box_ensemble import (
    evaluate_random_box_pair,
    make_random_box_geometry,
    wilson_interval,
)


def test_seeded_geometry_is_deterministic_disjoint_and_support_controlled():
    left = make_random_box_geometry(10, 8, 3, 4.0)
    right = make_random_box_geometry(10, 8, 3, 4.0)

    assert left.geometry_seed == right.geometry_seed
    assert left.direction_seed == right.direction_seed
    assert np.array_equal(left.direction, right.direction)
    for left_box, right_box in zip(left.region.boxes, right.region.boxes, strict=True):
        assert np.array_equal(left_box.lower, right_box.lower)
        assert np.array_equal(left_box.upper, right_box.upper)
        assert np.all(left_box.lower >= -1.0)
        assert np.all(left_box.upper <= 1.0)

    bounding = left.region.bounding_box()
    assert bounding.lower[0] == -1.0
    assert bounding.upper[0] == 1.0
    assert left.region.contains(np.zeros(10))


def test_imbalance_factor_is_paired_and_only_shrinks_declared_widths():
    balanced = make_random_box_geometry(5, 8, 7, 1.0)
    imbalanced = make_random_box_geometry(5, 8, 7, 4.0)

    assert np.array_equal(balanced.direction, imbalanced.direction)
    assert balanced.geometry_seed == imbalanced.geometry_seed
    for mode, (base, changed) in enumerate(
        zip(balanced.region.boxes, imbalanced.region.boxes, strict=True)
    ):
        assert np.array_equal(base.lower[[0, 2, 3, 4]], changed.lower[[0, 2, 3, 4]])
        assert np.array_equal(base.upper[[0, 2, 3, 4]], changed.upper[[0, 2, 3, 4]])
        assert np.isclose((base.lower[1] + base.upper[1]) / 2.0, (changed.lower[1] + changed.upper[1]) / 2.0)
        assert changed.upper[1] - changed.lower[1] <= base.upper[1] - base.lower[1] + 1e-15
        if mode in {3, 4}:
            assert np.isclose(changed.upper[1] - changed.lower[1], 1.2)


def test_maximum_requested_displacement_keeps_both_centers_feasible():
    for dimension in (2, 5, 10, 20):
        for mode_count in (4, 8, 16):
            geometry = make_random_box_geometry(dimension, mode_count, 0, 4.0)
            center_b = 0.5 * 0.15 * geometry.direction
            assert geometry.region.contains(np.zeros(dimension))
            assert geometry.region.contains(center_b)


def test_exact_chain_and_fixed_and_anchored_bounds_hold():
    geometry = make_random_box_geometry(20, 16, 2, 4.0)
    rows, summary = evaluate_random_box_pair(geometry, 0.5, 0.15)

    assert summary["whole_chain_abs_error"] < 1e-10
    for row in rows:
        assert row["tagged_untagged_abs_gap"] == 0.0
        if row["method"] in {"uniform_fixed", "anchor_fixed"}:
            assert row["total_ratio"] <= 1.0 + 1e-9
        if row["method"] != "whole_conditioning":
            assert row["theorem_bound_residual"] <= 1e-9


def test_small_factorial_run_has_independent_stratum_counts_and_pairs():
    rows, provenance = run_ensemble(
        dimensions=(2,),
        mode_counts=(4,),
        sigmas=(0.5,),
        separation_ratios=(0.15,),
        volume_ratio_caps=(1.0, 4.0),
        replicates=3,
        geometry_families=("diffuse",),
        direction_protocols=("iid",),
    )
    assert len(rows) == 3 * 2 * 6
    assert len(provenance) == 3 * 2

    strata = summarize_whole_strata(rows, bootstrap_resamples=100)
    assert len(strata) == 2
    assert all(row["replicates"] == 3 for row in strata)
    differences = paired_imbalance_differences(rows)
    assert len(differences) == 3
    assert all(math.isfinite(row["paired_ratio_difference"]) for row in differences)


def test_wilson_interval_known_boundary_cases():
    lower_zero, upper_zero = wilson_interval(0, 16)
    lower_full, upper_full = wilson_interval(16, 16)

    assert lower_zero == 0.0
    assert 0.15 < upper_zero < 0.25
    assert 0.75 < lower_full < 0.85
    assert upper_full == 1.0
