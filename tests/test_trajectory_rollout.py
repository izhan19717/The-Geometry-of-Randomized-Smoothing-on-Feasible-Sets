from dataclasses import fields
import math

import numpy as np
import pytest

from feasible_robustness.ccfs import comparison_multiplier
from feasible_robustness.totalized_ccfs import TotalizedCCFSSelection
from feasible_robustness.trajectory_rollout import (
    CCFSStepParameters,
    PublicStepRecord,
    TransitionResult,
    run_totalized_ccfs_trajectory,
)


def _run(**overrides):
    arguments = {
        "initial_history": 0,
        "schedule": (CCFSStepParameters(0.1, 2, 0.5),),
        "center_fn": lambda _step, _history: np.array([0.0]),
        "shift_policy": lambda _step, _history, _remaining, _rng: np.array([0.0]),
        "verifier": lambda _step, _history, _action: True,
        "fallback_library_fn": lambda _step, _history: np.empty((0, 1)),
        "transition": lambda _step, history, action, _rng: TransitionResult(
            next_history=history + 1,
            public_outcome=float(action[0]),
            stopped=False,
        ),
        "normalized_energy_budget": 0.0,
        "random_seed": 101,
    }
    arguments.update(overrides)
    return run_totalized_ccfs_trajectory(**arguments)


def test_abstention_never_calls_transition_and_preserves_history() -> None:
    calls = 0

    def forbidden_transition(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("transition must not be called after abstention")

    result = _run(
        verifier=lambda _step, _history, _action: False,
        transition=forbidden_transition,
    )
    assert calls == 0
    assert result.stop_reason == "abstention"
    assert result.final_history == 0
    assert len(result.public_trace) == 1
    record = result.public_trace[0]
    assert record.abstained
    assert record.action is None
    assert record.transition_outcome is None


def test_first_verified_proposal_records_selection_and_outcome() -> None:
    result = _run(
        transition=lambda _step, history, action, _rng: TransitionResult(
            next_history=history + 1,
            public_outcome={"action": float(action[0])},
            stopped=True,
        )
    )
    assert result.stop_reason == "transition"
    assert result.final_history == 1
    record = result.public_trace[0]
    assert record.selected_index == 0
    assert record.fallback_index is None
    assert record.proposal_inspections == 1
    assert not record.used_fallback
    assert record.transition_outcome == {"action": float(record.action[0])}


def test_fallback_selection_records_fallback_index() -> None:
    calls = 0

    def transition(_step, history, action, _rng):
        nonlocal calls
        calls += 1
        return TransitionResult(history + 1, float(action[0]), True)

    result = _run(
        verifier=lambda _step, _history, action: bool(action[0] == 5.0),
        fallback_library_fn=lambda _step, _history: np.array([[-1.0], [5.0]]),
        transition=transition,
    )
    assert calls == 1
    record = result.public_trace[0]
    assert record.selected_index is None
    assert record.fallback_index == 1
    assert record.proposal_inspections == 2
    assert record.fallback_inspections == 2
    assert record.used_fallback
    np.testing.assert_array_equal(record.action, np.array([5.0]))


def test_transition_stop_prevents_later_blocks() -> None:
    calls = 0

    def transition(_step, history, action, _rng):
        nonlocal calls
        calls += 1
        return TransitionResult(history + 1, float(action[0]), True)

    result = _run(
        schedule=(CCFSStepParameters(0.1, 2, 0.5),) * 3,
        transition=transition,
    )
    assert calls == 1
    assert len(result.public_trace) == 1
    assert len(result.private_energy_audit) == 1
    assert result.stop_reason == "transition"


def test_energy_is_limited_before_each_block_and_never_exceeds_budget() -> None:
    budget = 0.5
    parameters = CCFSStepParameters(1.0, 4, 0.75)
    seen_remaining = []

    def shift_policy(_step, _history, remaining, _rng):
        seen_remaining.append(remaining)
        return np.array([3.0, 4.0])

    result = _run(
        schedule=(parameters,) * 3,
        center_fn=lambda _step, _history: np.zeros(2),
        shift_policy=shift_policy,
        fallback_library_fn=lambda _step, _history: np.empty((0, 2)),
        transition=lambda _step, history, _action, _rng: TransitionResult(
            history + 1,
            history,
            False,
        ),
        normalized_energy_budget=budget,
    )
    remaining_before = budget**2
    actual_charges = []
    for record in result.private_energy_audit:
        assert record.charged_energy_squared <= remaining_before
        actual = (
            record.comparison_multiplier
            * float(record.applied_shift @ record.applied_shift)
            / parameters.sigma**2
        )
        assert actual <= record.charged_energy_squared
        actual_charges.append(actual)
        remaining_before = record.remaining_energy_squared
    assert math.fsum(r.charged_energy_squared for r in result.private_energy_audit) <= budget**2
    assert math.fsum(actual_charges) <= budget**2
    assert result.normalized_energy_used <= budget
    assert seen_remaining == sorted(seen_remaining, reverse=True)


def test_nominal_zero_shift_uses_no_energy() -> None:
    result = _run()
    assert result.normalized_energy_used == 0.0
    assert result.private_energy_audit[0].charged_energy_squared == 0.0
    np.testing.assert_array_equal(
        result.private_energy_audit[0].applied_shift,
        np.array([0.0]),
    )


def test_subnormal_squared_budget_fails_conservatively() -> None:
    budget = 1.7e-162
    result = _run(
        normalized_energy_budget=budget,
        shift_policy=lambda _step, _history, _remaining, _rng: np.array([1.0]),
    )
    assert result.normalized_energy_used <= budget


def test_public_trace_excludes_private_shift_and_attack_seed() -> None:
    names = {field.name for field in fields(PublicStepRecord)}
    assert "requested_shift" not in names
    assert "applied_shift" not in names
    assert "attack_seed" not in names


def test_shift_commits_before_proposal_block(monkeypatch) -> None:
    import feasible_robustness.trajectory_rollout as rollout_module

    order = []

    def shift_policy(_step, _history, _remaining, _rng):
        order.append("shift")
        return np.array([0.0])

    def fake_step(**_kwargs):
        order.append("block")
        return TotalizedCCFSSelection(
            action=np.array([0.0]),
            selected_index=0,
            fallback_index=None,
            proposal_inspections=1,
            fallback_inspections=0,
            used_fallback=False,
            abstained=False,
        )

    monkeypatch.setattr(rollout_module, "totalized_ccfs_step", fake_step)
    _run(
        shift_policy=shift_policy,
        transition=lambda _step, history, _action, _rng: (
            order.append("transition")
            or TransitionResult(history + 1, None, False)
        ),
    )
    assert order == ["shift", "block", "transition"]


def test_random_streams_are_spawned_internally_and_reproducibly(monkeypatch) -> None:
    import feasible_robustness.trajectory_rollout as rollout_module

    first_ids = []
    original_step = rollout_module.totalized_ccfs_step

    def recording_step(**arguments):
        first_ids.append(id(arguments["rng"]))
        return original_step(**arguments)

    attack_ids = []
    transition_ids = []
    monkeypatch.setattr(rollout_module, "totalized_ccfs_step", recording_step)

    def shift_policy(_step, _history, _remaining, rng):
        attack_ids.append(id(rng))
        return np.array([0.0])

    def transition(_step, history, action, rng):
        transition_ids.append(id(rng))
        return TransitionResult(history + 1, float(action[0]), False)

    first = _run(shift_policy=shift_policy, transition=transition, random_seed=7)
    first_action = float(first.public_trace[0].action[0])
    ids = {first_ids[-1], attack_ids[-1], transition_ids[-1]}
    assert len(ids) == 3

    second = _run(random_seed=7)
    assert float(second.public_trace[0].action[0]) == first_action


def test_shift_policy_cannot_mutate_live_history() -> None:
    history = {"value": 0}

    def mutating_shift(_step, private_history, _remaining, _rng):
        private_history["value"] = 99
        return np.array([0.0])

    result = _run(
        initial_history=history,
        shift_policy=mutating_shift,
        verifier=lambda _step, live_history, _action: live_history["value"] == 0,
        transition=lambda _step, live_history, action, _rng: TransitionResult(
            {"value": live_history["value"] + 1},
            float(action[0]),
            False,
        ),
    )
    assert history == {"value": 0}
    assert result.final_history == {"value": 1}


def test_invalid_random_seed_fails_closed() -> None:
    with pytest.raises(TypeError, match="random_seed"):
        _run(random_seed=True)


def test_shift_dimension_must_match_center() -> None:
    with pytest.raises(ValueError, match="dimension"):
        _run(
            shift_policy=lambda _step, _history, _remaining, _rng: np.zeros(2)
        )


def test_action_dimension_must_remain_fixed_across_schedule() -> None:
    with pytest.raises(ValueError, match="remain fixed"):
        _run(
            schedule=(
                CCFSStepParameters(0.1, 2, 0.5),
                CCFSStepParameters(0.1, 2, 0.5),
            ),
            center_fn=lambda step, _history: np.zeros(step + 1),
            shift_policy=lambda step, _history, _remaining, _rng: np.zeros(step + 1),
            fallback_library_fn=lambda step, _history: np.empty((0, step + 1)),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"schedule": ()},
        {"schedule": (object(),)},
        {"normalized_energy_budget": -1.0},
        {"normalized_energy_budget": math.inf},
    ],
)
def test_invalid_schedule_and_budget_fail_closed(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        _run(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"sigma": 0.0, "cap": 1, "correlation": 0.0},
        {"sigma": 1.0, "cap": 0, "correlation": 0.0},
        {"sigma": 1.0, "cap": 2, "correlation": 1.0},
    ],
)
def test_invalid_step_parameters_fail_closed(arguments) -> None:
    with pytest.raises(ValueError):
        CCFSStepParameters(**arguments)


def test_energy_multiplier_in_audit_matches_declared_step() -> None:
    parameters = CCFSStepParameters(0.7, 8, 0.6)
    result = _run(schedule=(parameters,))
    assert result.private_energy_audit[0].comparison_multiplier == pytest.approx(
        comparison_multiplier(8, 0.6)
    )
