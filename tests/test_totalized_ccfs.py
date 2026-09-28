import numpy as np
import pytest

from feasible_robustness.totalized_ccfs import (
    TotalizedCCFSSelection,
    select_first_verified_or_abstain,
    totalized_ccfs_step,
)


def test_first_verified_proposal_skips_fallback_library() -> None:
    result = select_first_verified_or_abstain(
        proposals=np.array([[-2.0], [0.25], [0.75]]),
        is_feasible=lambda action: 0.0 <= action[0] <= 1.0,
        fallback_library=np.array([[0.5]]),
    )
    np.testing.assert_array_equal(result.action, np.array([0.25]))
    assert result.selected_index == 1
    assert result.fallback_index is None
    assert result.proposal_inspections == 2
    assert result.fallback_inspections == 0
    assert not result.used_fallback
    assert not result.abstained


def test_first_verified_fallback_is_reported_separately() -> None:
    result = select_first_verified_or_abstain(
        proposals=np.array([[-2.0], [-1.0]]),
        is_feasible=lambda action: 0.0 <= action[0] <= 1.0,
        fallback_library=np.array([[-0.5], [0.4], [0.8]]),
    )
    np.testing.assert_array_equal(result.action, np.array([0.4]))
    assert result.selected_index is None
    assert result.fallback_index == 1
    assert result.proposal_inspections == 2
    assert result.fallback_inspections == 2
    assert result.used_fallback
    assert not result.abstained


def test_exhaustion_returns_explicit_abstention_not_infeasible_action() -> None:
    result = select_first_verified_or_abstain(
        proposals=np.array([[-2.0], [-1.0]]),
        is_feasible=lambda action: 0.0 <= action[0] <= 1.0,
        fallback_library=np.array([[-0.5], [-0.25]]),
    )
    assert result.action is None
    assert result.selected_index is None
    assert result.fallback_index is None
    assert result.proposal_inspections == 2
    assert result.fallback_inspections == 2
    assert not result.used_fallback
    assert result.abstained


def test_empty_fallback_library_is_finite_and_allowed() -> None:
    result = select_first_verified_or_abstain(
        proposals=np.array([[-1.0, -1.0]]),
        is_feasible=lambda _: False,
        fallback_library=np.empty((0, 2)),
    )
    assert result.abstained
    assert result.fallback_inspections == 0


def test_library_exhaustion_does_not_assert_continuous_fibre_is_empty() -> None:
    # The continuous feasible set contains 0.123, but the declared finite
    # library misses it.  The selector correctly abstains without making an
    # emptiness claim about the underlying set.
    result = select_first_verified_or_abstain(
        proposals=np.array([[-1.0]]),
        is_feasible=lambda action: abs(action[0] - 0.123) < 1e-12,
        fallback_library=np.array([[0.0], [0.25]]),
    )
    assert result.abstained


def test_totalized_step_is_deterministic_at_fixed_rng_seed() -> None:
    kwargs = dict(
        center=[-2.0],
        sigma=0.01,
        cap=4,
        correlation=0.5,
        is_feasible=lambda action: action[0] >= 0.0,
        fallback_library=np.array([[0.0]]),
    )
    first = totalized_ccfs_step(rng=np.random.default_rng(71), **kwargs)
    second = totalized_ccfs_step(rng=np.random.default_rng(71), **kwargs)
    np.testing.assert_array_equal(first.action, second.action)
    assert first.selected_index == second.selected_index
    assert first.fallback_index == second.fallback_index
    assert first.proposal_inspections == second.proposal_inspections
    assert first.fallback_inspections == second.fallback_inspections
    assert first.used_fallback == second.used_fallback
    assert first.abstained == second.abstained
    assert first.used_fallback


def test_invalid_matrices_and_inconsistent_result_are_rejected() -> None:
    with pytest.raises(ValueError, match="proposals"):
        select_first_verified_or_abstain(
            proposals=np.array([1.0]),
            is_feasible=lambda _: True,
            fallback_library=np.empty((0, 1)),
        )
    with pytest.raises(ValueError, match="dimension"):
        select_first_verified_or_abstain(
            proposals=np.zeros((1, 2)),
            is_feasible=lambda _: True,
            fallback_library=np.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="inconsistent"):
        TotalizedCCFSSelection(
            action=np.array([0.0]),
            selected_index=None,
            fallback_index=None,
            proposal_inspections=1,
            fallback_inspections=0,
            used_fallback=False,
            abstained=True,
        )
