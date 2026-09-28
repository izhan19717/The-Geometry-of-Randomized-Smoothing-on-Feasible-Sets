"""Focused checks for stable-direction bounded-band development."""

from argparse import Namespace

import numpy as np
import pytest

from experiments.iclr2027_learned_stable_band_filter_development import (
    BONFERRONI_TAIL_EVENTS,
    MIN_DIRECTION_BANK_SIZE,
    MultiDirectionProposalBatch,
    PAPER_ELIGIBILITY,
    REMOTE_BUDGETS,
    StableBandCandidate,
    bounded_two_band_mask,
    candidate_grid,
    certified_radii_from_counts,
    covariance_proof,
    deterministic_class_split,
    rademacher_direction_bank,
    rank_stable_directions,
    resolve_budget,
    select_candidate,
)


def test_fit_dev_split_is_deterministic_balanced_and_disjoint() -> None:
    labels = np.repeat(np.arange(3), 20).tolist()
    first = deterministic_class_split(
        labels, fit_per_class=5, dev_per_class=4, seed=17, classes=(0, 1, 2)
    )
    second = deterministic_class_split(
        labels, fit_per_class=5, dev_per_class=4, seed=17, classes=(0, 1, 2)
    )
    changed = deterministic_class_split(
        labels, fit_per_class=5, dev_per_class=4, seed=18, classes=(0, 1, 2)
    )
    assert first == second
    assert first != changed
    for label, parts in first.items():
        assert len(parts["fit"]) == 5
        assert len(parts["dev"]) == 4
        assert set(parts["fit"]).isdisjoint(parts["dev"])
        assert all(labels[index] == label for index in parts["fit"] + parts["dev"])


def test_rademacher_bank_is_version_stable_unit_norm_and_pair_specific() -> None:
    first = rademacher_direction_bank(pair=(3, 5), seed=11, bank_size=64, dimension=31)
    second = rademacher_direction_bank(pair=(3, 5), seed=11, bank_size=64, dimension=31)
    changed = rademacher_direction_bank(pair=(2, 4), seed=11, bank_size=64, dimension=31)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.unique(first) == pytest.approx(
        [-1 / np.sqrt(31), 1 / np.sqrt(31)]
    )
    with pytest.raises(ValueError):
        rademacher_direction_bank(
            pair=(3, 5), seed=11, bank_size=MIN_DIRECTION_BANK_SIZE - 1, dimension=31
        )


def test_stability_ranking_uses_projection_dispersion_with_fixed_ties() -> None:
    bank = np.eye(3)
    fit = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 1.0, 2.0]],
            [[0.0, 2.0, 4.0]],
        ]
    )
    indices, rows = rank_stable_directions(bank, fit, retain=2)
    assert indices == (0, 1)
    assert rows[0]["clean_projection_dispersion"] == 0.0
    assert rows[0]["retained_for_search"] is True
    assert rows[2]["retained_for_search"] is False


def test_filter_and_covariance_proof_are_global_and_raw_proposal_based() -> None:
    sigma = 0.25
    candidate = StableBandCandidate(
        direction_slot=0,
        direction_bank_index=7,
        offset=0.02,
        inner=0.05,
        outer=0.25,
    )
    projections = np.array([-0.28, -0.27, -0.07, -0.02, 0.03, 0.23, 0.24])
    shifted = np.abs(projections + candidate.offset)
    assert np.array_equal(
        bounded_two_band_mask(projections, candidate, sigma),
        (shifted >= candidate.inner) & (shifted <= candidate.outer),
    )
    proof = covariance_proof(np.array([1.0, 0.0]), candidate, sigma)
    assert proof["raw_proposal_before_clipping"] is True
    assert proof["clipping_performed"] is False
    assert proof["global_over_all_gaussian_centers"] is True
    assert proof["normalized_covariance_factor"] == 1.0
    assert proof["directional_variance_upper_by_popoviciu"] <= proof["lambda_max_upper"]
    with pytest.raises(ValueError):
        covariance_proof(
            np.array([1.0, 0.0]),
            StableBandCandidate(0, 7, 0.0, 0.05, 0.251),
            sigma,
        )


def test_grid_and_selection_fail_closed_when_retention_is_inadequate() -> None:
    sigma = 0.25
    grid = candidate_grid(
        stable_bank_indices=(4,),
        offsets_by_slot=((0.0,),),
        sigma=sigma,
        outer_fractions=(1.0,),
        inner_fractions=(0.2,),
    )
    assert len(grid) == 1
    labels = np.tile(np.array([0, 1]), 50)
    rejected = np.full((1, labels.size), 1.0)
    batches = [
        MultiDirectionProposalBatch(i, 0, labels, rejected.copy()) for i in range(10)
    ]
    selected, summary, rows = select_candidate(
        grid, batches, sigma=sigma, min_retained=20
    )
    assert selected is None
    assert summary is None
    assert rows[0]["feasible"] is False


def test_selection_respects_retention_and_uses_correct_only_objective() -> None:
    sigma = 0.25
    candidate = StableBandCandidate(0, 4, 0.0, 0.05, 0.25)
    labels = np.array([0] * 45 + [1] * 5 + [0] * 20 + [1] * 30)
    projections = np.array([0.10] * 50 + [0.50] * 50)[None, :]
    batches = [
        MultiDirectionProposalBatch(i, 0, labels.copy(), projections.copy())
        for i in range(10)
    ]
    selected, summary, _ = select_candidate(
        (candidate,), batches, sigma=sigma, min_retained=20
    )
    assert selected == candidate
    assert summary is not None
    assert summary["feasible"] is True
    assert summary["mean_retention"] == pytest.approx(0.5)
    assert summary["correct_adequate_fraction"] == 1.0
    assert summary["objective_mean_correct_empirical_covariance_radius"] > 0.0


def test_population_joint_upper_bound_makes_strong_separation_checkable() -> None:
    radii = certified_radii_from_counts(
        filtered_selected_label=0,
        unfiltered_selected_label=0,
        retained_counts=(27_000, 3_000),
        unfiltered_counts=(75_000, 25_000),
        proposal_count=100_000,
        sigma=0.25,
        delta=0.01,
        min_retained=64,
    )
    assert radii["bonferroni_tail_events"] == BONFERRONI_TAIL_EVENTS
    assert radii["bonferroni_total_error"] == pytest.approx(0.01)
    assert radii["r_cov_L"] > 0.0
    assert radii["r_mass_L"] > 0.0
    assert radii["r_mass_U"] is not None
    assert radii["r_mass_L"] <= radii["r_mass_U"] < radii["r_cov_L"]
    assert radii["strong_separation"] is True

    no_retention = certified_radii_from_counts(
        filtered_selected_label=0,
        unfiltered_selected_label=0,
        retained_counts=(0, 0),
        unfiltered_counts=(51_000, 49_000),
        proposal_count=100_000,
        sigma=0.25,
        delta=0.01,
        min_retained=64,
    )
    assert no_retention["r_cov_L"] == 0.0
    assert no_retention["r_mass_L"] == 0.0
    assert no_retention["r_mass_U"] is None
    assert no_retention["strong_separation"] is False


def test_remote_budgets_are_bounded_and_development_only() -> None:
    assert PAPER_ELIGIBILITY == "none"
    assert set(REMOTE_BUDGETS) == {"smoke", "default"}
    assert REMOTE_BUDGETS["smoke"].direction_bank_size >= MIN_DIRECTION_BANK_SIZE
    assert (
        REMOTE_BUDGETS["smoke"].evaluation_proposals
        < REMOTE_BUDGETS["default"].evaluation_proposals
    )
    args = Namespace(budget="smoke", search_proposals=40)
    resolved = resolve_budget(args)
    assert resolved.search_proposals == 40
    assert resolved.pair_count == 1
