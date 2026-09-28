"""Frozen Goal2 safety-event protocol and confidence checks."""

import json
import math

import pytest

from experiments.iclr2027_goal2_safe_controller_open_pilot import development_seeds
from experiments.iclr2027_goal2_safe_trajectory_confirmation import (
    PROTOCOL,
    one_sided_hoeffding_lower,
    safety_event,
    validate_protocol,
)
from experiments.iclr2027_goal2_trajectory_confirmation import (
    PROTOCOL as PREVIOUS_PROTOCOL,
    validate_protocol as previous_seeds,
)


def test_frozen_seed_cohort_is_distinct() -> None:
    seeds = validate_protocol(json.loads(PROTOCOL.read_text()))
    old = previous_seeds(json.loads(PREVIOUS_PROTOCOL.read_text()))
    assert len(seeds) == len(set(seeds)) == 256
    assert set(seeds).isdisjoint(old)
    assert set(seeds).isdisjoint(development_seeds(32))


def test_safety_event_excludes_cost_and_early_stop() -> None:
    assert safety_event({"native_cost_hit": 0, "stop_reason": "horizon"}) == 1
    assert safety_event({"native_cost_hit": 0, "stop_reason": "first_goal_without_cost"}) == 1
    assert safety_event({"native_cost_hit": 1, "stop_reason": "native_cost"}) == 0
    assert safety_event({"native_cost_hit": 0, "stop_reason": "environment_stop"}) == 0
    assert safety_event({"native_cost_hit": 0, "stop_reason": "nonfinite_transition"}) == 0


def test_hoeffding_allocation_is_for_six_nominal_arms() -> None:
    alpha = 0.001 / 6
    expected = 0.8 - math.sqrt(math.log(1 / alpha) / (2 * 256))
    assert one_sided_hoeffding_lower(205, 256, alpha) == pytest.approx(
        205 / 256 - math.sqrt(math.log(1 / alpha) / (2 * 256))
    )
    assert expected > 0.5
    with pytest.raises(ValueError):
        one_sided_hoeffding_lower(257, 256, alpha)
