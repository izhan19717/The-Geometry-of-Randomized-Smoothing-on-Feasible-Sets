"""Development cohort and summary checks for Goal2 safety controllers."""

import pytest

from experiments.iclr2027_goal2_safe_controller_open_pilot import (
    development_seeds,
    summarize,
)
from experiments.iclr2027_goal2_trajectory_confirmation import validate_protocol

import json
from experiments.iclr2027_goal2_trajectory_confirmation import PROTOCOL


def test_opened_development_seeds_are_distinct_from_confirmation() -> None:
    opened = development_seeds(32)
    held_out = validate_protocol(json.loads(PROTOCOL.read_text()))
    assert len(opened) == len(set(opened)) == 32
    assert set(opened).isdisjoint(held_out)
    with pytest.raises(ValueError):
        development_seeds(33)


def test_summary_accounts_for_all_outcomes() -> None:
    records = [
        {"controller_seed": 1, "method": "matched_single", "event": 1,
         "native_cost_hit": 0, "fallback_steps": 2, "executed_steps": 5},
        {"controller_seed": 1, "method": "matched_single", "event": 0,
         "native_cost_hit": 1, "fallback_steps": 4, "executed_steps": 7},
    ]
    row = summarize(records)["seed=1/matched_single"]
    assert row == {
        "trajectories": 2,
        "event_count": 1,
        "cost_count": 1,
        "mean_fallback_steps": 3.0,
        "mean_executed_steps": 6.0,
    }
