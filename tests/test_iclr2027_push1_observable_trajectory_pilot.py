"""Checks for the observable Push1 trajectory pilot."""

import numpy as np
import pytest

from experiments.iclr2027_push1_observable_trajectory_pilot import (
    KAPPA,
    RHO_FOUR,
    is_admissible,
    observation_action_caps,
    predictable_shift,
    proposal_block,
)


def test_declared_block_multiplier() -> None:
    assert 4 / (1 + 3 * RHO_FOUR) == pytest.approx(KAPPA)
    for method in ("matched_single", "correlated_four"):
        block, multiplier = proposal_block(
            np.array([0.2, -0.1]), np.array([0.03, 0.04]), 0.2,
            method, np.random.default_rng(17),
        )
        assert block.shape == ((1, 2) if method == "matched_single" else (4, 2))
        assert multiplier == pytest.approx(KAPPA / 0.2**2)


def test_observable_verifier_and_fallback() -> None:
    observation = np.zeros(76)
    observation[28:44] = 1
    forward, steering = observation_action_caps(observation)
    assert (forward, steering) == pytest.approx((0.88, 1.0))
    assert is_admissible(observation, np.array([forward, steering]))
    assert not is_admissible(observation, np.array([forward + 0.01, 0]))


def test_predictable_shift_energy() -> None:
    budget, horizon = 0.5, 100
    shift = predictable_shift(np.array([0.2, -0.1]), budget, horizon)
    assert np.linalg.norm(shift) == pytest.approx(budget / np.sqrt(horizon))
    assert KAPPA * horizon * float(shift @ shift) / 0.2**2 == pytest.approx(
        KAPPA * budget**2 / 0.2**2
    )
