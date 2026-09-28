"""Pure checks for the global stable-band development study."""

from argparse import Namespace
import math

import numpy as np
import pytest

from experiments.iclr2027_global_stable_band_filter_development import (
    BONFERRONI_TAIL_EVENTS,
    CLASS_COUNT,
    DEVELOPMENT_BUDGETS,
    DEVELOPMENT_TARGET_RADIUS,
    GlobalProposalBatch,
    MIN_DIRECTION_BANK_SIZE,
    PAPER_ELIGIBILITY,
    StableBandCandidate,
    deterministic_three_way_split,
    empirical_conditional_radius,
    evaluate_development_gate,
    global_rademacher_direction_bank,
    multiclass_certified_radii_from_counts,
    multiclass_tie_kl,
    rank_stable_directions,
    resolve_budget,
    score_candidate,
    select_candidate,
    selected_and_runner,
    summarize_evaluation,
)


def test_three_way_split_is_balanced_deterministic_and_disjoint() -> None:
    labels = np.repeat(np.arange(CLASS_COUNT), 25).tolist()
    first = deterministic_three_way_split(
        labels,
        fit_per_class=4,
        search_per_class=3,
        evaluation_per_class=2,
        seed=19,
    )
    second = deterministic_three_way_split(
        labels,
        fit_per_class=4,
        search_per_class=3,
        evaluation_per_class=2,
        seed=19,
    )
    changed = deterministic_three_way_split(
        labels,
        fit_per_class=4,
        search_per_class=3,
        evaluation_per_class=2,
        seed=20,
    )
    assert first == second
    assert first != changed
    for label, cohorts in first.items():
        assert len(cohorts["fit"]) == 4
        assert len(cohorts["filter_search"]) == 3
        assert len(cohorts["evaluation"]) == 2
        sets = [set(cohorts[name]) for name in ("fit", "filter_search", "evaluation")]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])
        assert all(labels[index] == label for values in sets for index in values)


def test_global_direction_bank_is_unit_hash_stable_and_large_enough() -> None:
    first = global_rademacher_direction_bank(seed=7, bank_size=128, dimension=31)
    second = global_rademacher_direction_bank(seed=7, bank_size=128, dimension=31)
    changed = global_rademacher_direction_bank(seed=8, bank_size=128, dimension=31)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.unique(first) == pytest.approx([-1 / np.sqrt(31), 1 / np.sqrt(31)])
    with pytest.raises(ValueError):
        global_rademacher_direction_bank(
            seed=7, bank_size=MIN_DIRECTION_BANK_SIZE - 1, dimension=31
        )


def test_stability_ranking_uses_only_dispersion_and_fixed_ties() -> None:
    bank = np.eye(4)
    fit = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 2.0, 3.0]],
            [[0.0, 2.0, 4.0, 6.0]],
        ]
    )
    indices, rows = rank_stable_directions(bank, fit, retain=2)
    assert indices == (0, 1)
    assert rows[0]["clean_projection_dispersion"] == 0.0
    assert rows[0]["retained_for_search"] is True
    assert rows[2]["retained_for_search"] is False


def test_multiclass_tie_divergence_matches_binary_special_case() -> None:
    p_a, p_b = 0.9, 0.1
    expected = p_a * math.log(2.0 * p_a) + p_b * math.log(2.0 * p_b)
    assert multiclass_tie_kl(p_a, p_b) == pytest.approx(expected)
    sub_a, sub_b = 0.45, 0.05
    sub_mass = sub_a + sub_b
    sub_expected = sub_a * math.log(2.0 * sub_a / sub_mass) + sub_b * math.log(
        2.0 * sub_b / sub_mass
    )
    assert multiclass_tie_kl(sub_a, sub_b) == pytest.approx(sub_expected)
    assert multiclass_tie_kl(0.4, 0.4) == 0.0
    counts = [90, 10] + [0] * 8
    assert empirical_conditional_radius(counts, 0.25) == pytest.approx(
        0.25 * math.sqrt(2.0 * expected)
    )
    assert selected_and_runner([4, 4, 3, 2, 1, 0, 0, 0, 0, 0]) == (0, 1)


def test_multiclass_certificates_use_one_complete_bonferroni_family() -> None:
    retained = [27_000, 1_000] + [250] * 8
    unfiltered = [75_000, 5_000] + [2_500] * 8
    radii = multiclass_certified_radii_from_counts(
        filtered_selected_label=0,
        filtered_runner_label=1,
        unfiltered_selected_label=0,
        retained_counts=retained,
        unfiltered_counts=unfiltered,
        proposal_count=100_000,
        sigma=0.25,
        delta=0.01,
        min_retained=64,
    )
    assert sum(retained) == 30_000
    assert sum(unfiltered) == 100_000
    assert radii["bonferroni_tail_events"] == BONFERRONI_TAIL_EVENTS == 32
    assert radii["bonferroni_total_error"] == pytest.approx(0.01)
    assert len(radii["conditional_competitor_probability_uppers"]) == CLASS_COUNT
    assert len(radii["joint_competitor_mass_uppers"]) == CLASS_COUNT
    assert len(radii["unfiltered_competitor_probability_uppers"]) == CLASS_COUNT
    assert radii["r_cov_L"] > 0.0
    assert radii["r_mass_L"] > 0.0
    assert radii["r_unfiltered_L"] > 0.0
    assert radii["r_mass_U"] is not None
    assert radii["r_mass_L"] <= radii["r_mass_U"] < radii["r_cov_L"]
    assert radii["strong_separation"] is True


def test_all_competitors_control_lower_radii_but_frozen_runner_controls_upper() -> None:
    # Class 1 was frozen as the runner.  Class 2 is larger on the independent
    # estimation stream.  The lower certificates must therefore use class 2,
    # while the valid population upper comparison remains tied to class 1.
    retained = [50_000, 100, 49_000] + [100] * 7
    unfiltered = [10_000, 5_000, 70_000] + [2_142] * 6 + [2_148]
    radii = multiclass_certified_radii_from_counts(
        filtered_selected_label=0,
        filtered_runner_label=1,
        unfiltered_selected_label=2,
        retained_counts=retained,
        unfiltered_counts=unfiltered,
        proposal_count=100_000,
        sigma=0.25,
        delta=0.01,
        min_retained=64,
    )
    assert sum(retained) == 99_800
    assert sum(unfiltered) == 100_000
    assert (
        radii["joint_largest_competitor_upper"]
        == radii["joint_competitor_mass_uppers"][2]
    )
    assert (
        radii["joint_largest_competitor_upper"]
        > radii["joint_competitor_mass_uppers"][1]
    )
    assert radii["r_mass_L"] == 0.0
    assert radii["r_mass_U"] is not None
    assert radii["r_mass_U"] > 0.0
    assert radii["r_unfiltered_L"] > 0.0


def test_search_objective_is_predeclared_and_selection_fails_closed() -> None:
    sigma = 0.25
    candidate = StableBandCandidate(0, 4, 0.0, 0.05, 0.25)
    labels = np.array([0] * 90 + [1] * 10)
    accepted = np.full((1, 100), 0.10)
    batches = [
        GlobalProposalBatch(i, 0, labels.copy(), accepted.copy()) for i in range(10)
    ]
    summary = score_candidate(candidate, batches, sigma=sigma, min_retained=20)
    assert summary["feasible"] is True
    assert summary["objective_radius"] == DEVELOPMENT_TARGET_RADIUS
    assert summary["objective_correct_empirical_certificate_accuracy"] == 1.0
    assert summary["mean_retention"] == 1.0
    selected, selected_summary, _ = select_candidate(
        (candidate,), batches, sigma=sigma, min_retained=20
    )
    assert selected == candidate
    assert selected_summary == summary

    rejected = np.full((1, 100), 1.0)
    bad_batches = [
        GlobalProposalBatch(i, 0, labels.copy(), rejected.copy()) for i in range(10)
    ]
    failed, failed_summary, rows = select_candidate(
        (candidate,), bad_batches, sigma=sigma, min_retained=20
    )
    assert failed is None
    assert failed_summary is None
    assert rows[0]["feasible"] is False


def test_evaluation_summary_reports_all_radii_and_model_fraction() -> None:
    base = {
        "retention_rate": 0.4,
        "label_selection_retention_rate": 0.5,
        "evaluation_retention_adequate": True,
        "label_selection_retention_adequate": True,
        "filtered_correct": True,
        "unfiltered_correct": True,
        "strong_separation": True,
        "r_mass_U": 0.1,
        "r_cov_L": 0.21,
        "r_mass_L": 0.12,
        "r_unfiltered_L": 0.18,
    }
    summary = summarize_evaluation([base, {**base, "filtered_correct": False}])
    assert summary["model_evaluation_fraction"] == pytest.approx(0.4)
    expected_keys = {"0.00", "0.05", "0.10", "0.15", "0.20", "0.25"}
    for method in summary["radii"].values():
        assert set(method["correct_certified_accuracy"]) == expected_keys


def test_predeclared_development_gate_passes_and_fails_only_fixed_checks() -> None:
    passing_row = {
        "retention_rate": 0.4,
        "label_selection_retention_rate": 0.4,
        "evaluation_retention_adequate": True,
        "label_selection_retention_adequate": True,
        "filtered_correct": True,
        "unfiltered_correct": True,
        "strong_separation": True,
        "r_mass_U": 0.18,
        "r_cov_L": 0.21,
        "r_mass_L": 0.10,
        "r_unfiltered_L": 0.21,
    }
    summary = summarize_evaluation([passing_row] * 20)
    gate = evaluate_development_gate(summary)
    assert gate["passed"] is True
    assert all(gate["checks"].values())

    failing_summary = summarize_evaluation(
        [{**passing_row, "r_cov_L": 0.19}] * 20
    )
    failed = evaluate_development_gate(failing_summary)
    assert failed["passed"] is False
    assert (
        failed["checks"][
            "covariance_accuracy_exceeds_joint_by_five_points_at_radius_0_20"
        ]
        is False
    )


def test_budgets_are_global_development_only() -> None:
    assert PAPER_ELIGIBILITY == "none"
    assert set(DEVELOPMENT_BUDGETS) == {"smoke", "default"}
    assert DEVELOPMENT_BUDGETS["smoke"].direction_bank_size >= 128
    assert DEVELOPMENT_BUDGETS["smoke"].stable_direction_count == 12
    args = Namespace(budget="smoke", search_proposals=40)
    budget = resolve_budget(args)
    assert budget.search_proposals == 40
    assert budget.evaluation_per_class == 1
