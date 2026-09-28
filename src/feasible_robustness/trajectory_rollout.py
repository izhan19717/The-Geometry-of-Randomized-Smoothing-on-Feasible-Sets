"""Finite-horizon orchestration for capped causal feasible smoothing.

The wrapper in this module fixes the order required by the adaptive trajectory
comparison.  A shift policy commits from the pre-block history, the shift is
limited against the remaining normalized energy, and only then is the current
Gaussian proposal block drawn.  An exhausted finite fallback library records
abstention and ends the rollout without calling the transition function.

The Gaussian comparison applies to ``public_trace`` and the public final
history.  ``private_energy_audit`` contains the attack request and applied
shift, so it must be forgotten before evaluating a certified event.  Python
cannot verify measurability or inspect callback closures.  The controller,
verifier, fallback construction, transition, initial-law procedure, and
stopping semantics must be the same programs under both compared laws.  They
must not receive the shift or attack seed through another input.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Callable, Generic, Literal, Sequence, TypeVar

import numpy as np

from feasible_robustness.ccfs import comparison_multiplier
from feasible_robustness.totalized_ccfs import totalized_ccfs_step
from feasible_robustness.trajectory_certificate import (
    limit_shift_to_remaining_energy,
)


Array = np.ndarray
HistoryT = TypeVar("HistoryT")
OutcomeT = TypeVar("OutcomeT")


@dataclass(frozen=True)
class CCFSStepParameters:
    """Predeclared Gaussian parameters for one trajectory step."""

    sigma: float
    cap: int
    correlation: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        comparison_multiplier(self.cap, self.correlation)


@dataclass(frozen=True)
class TransitionResult(Generic[HistoryT, OutcomeT]):
    """Result returned by the common physical transition kernel."""

    next_history: HistoryT
    public_outcome: OutcomeT
    stopped: bool = False


@dataclass(frozen=True, eq=False)
class PublicStepRecord(Generic[OutcomeT]):
    """One certifiable public selection and transition record."""

    step_index: int
    action: Array | None
    selected_index: int | None
    fallback_index: int | None
    proposal_inspections: int
    fallback_inspections: int
    used_fallback: bool
    abstained: bool
    transition_outcome: OutcomeT | None
    transition_stopped: bool


@dataclass(frozen=True, eq=False)
class PrivateEnergyRecord:
    """Private shift metadata that must not enter a certified event."""

    step_index: int
    requested_shift: Array
    applied_shift: Array
    comparison_multiplier: float
    charged_energy_squared: float
    remaining_energy_squared: float
    was_scaled: bool


@dataclass(frozen=True)
class CausalTrajectoryRollout(Generic[HistoryT, OutcomeT]):
    """Public trajectory plus a separate private energy-accounting record.

    ``normalized_energy_used`` is the square root of the conservative sum of
    charged energies.  It upper-bounds the normalized energy of the applied
    floating-point shifts.
    """

    public_trace: tuple[PublicStepRecord[OutcomeT], ...]
    private_energy_audit: tuple[PrivateEnergyRecord, ...]
    final_history: HistoryT
    stop_reason: Literal["horizon", "transition", "abstention"]
    normalized_energy_used: float


def _finite_vector(value: Array | Sequence[float], *, name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite nonempty vector")
    return vector.copy()


def _spawn_rngs(
    random_seed: int | Sequence[int] | np.random.SeedSequence,
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    if isinstance(random_seed, (bool, np.bool_)):
        raise TypeError("random_seed must be valid SeedSequence entropy")
    try:
        root = (
            random_seed
            if isinstance(random_seed, np.random.SeedSequence)
            else np.random.SeedSequence(random_seed)
        )
    except (TypeError, ValueError) as error:
        raise TypeError("random_seed must be valid SeedSequence entropy") from error
    proposal_seed, attack_seed, transition_seed = root.spawn(3)
    return (
        np.random.default_rng(proposal_seed),
        np.random.default_rng(attack_seed),
        np.random.default_rng(transition_seed),
    )


def run_totalized_ccfs_trajectory(
    *,
    initial_history: HistoryT,
    schedule: Sequence[CCFSStepParameters],
    center_fn: Callable[[int, HistoryT], Array | Sequence[float]],
    shift_policy: Callable[
        [int, HistoryT, float, np.random.Generator],
        Array | Sequence[float],
    ],
    verifier: Callable[[int, HistoryT, Array], bool],
    fallback_library_fn: Callable[
        [int, HistoryT],
        Array | Sequence[Sequence[float]],
    ],
    transition: Callable[
        [int, HistoryT, Array, np.random.Generator],
        TransitionResult[HistoryT, OutcomeT],
    ],
    normalized_energy_budget: float,
    random_seed: int | Sequence[int] | np.random.SeedSequence,
) -> CausalTrajectoryRollout[HistoryT, OutcomeT]:
    """Run one finite capped causal trajectory.

    ``shift_policy`` receives only the step, pre-block history, remaining
    squared normalized energy, and its private RNG.  It must not close over the
    proposal RNG.  ``verifier`` must be a total Boolean query that leaves the
    live physical system unchanged.  ``fallback_library_fn`` must return a
    finite ordered matrix.  ``transition`` is the only callback authorized to
    step the physical system and is called exactly once after a verified
    non-abstaining selection.

    The schedules and every non-attack callback are caller-declared common
    programs.  This runtime ordering enforces the visible control flow, but it
    cannot detect a hidden parameter or random-stream side channel in a
    callback closure.  Proposal, attack, and transition streams are spawned
    internally from distinct children of ``random_seed``.  The shift policy
    receives a defensive copy of history and must not mutate shared external
    state.
    """

    if not isinstance(schedule, Sequence):
        raise TypeError("schedule must be a finite sequence")
    step_parameters = tuple(schedule)
    if not step_parameters:
        raise ValueError("schedule must contain at least one step")
    if not all(isinstance(item, CCFSStepParameters) for item in step_parameters):
        raise TypeError("every schedule item must be CCFSStepParameters")
    if (
        isinstance(normalized_energy_budget, (bool, np.bool_))
        or not math.isfinite(normalized_energy_budget)
        or normalized_energy_budget < 0.0
    ):
        raise ValueError("normalized_energy_budget must be finite and nonnegative")
    budget_value = float(normalized_energy_budget)
    budget_squared = budget_value**2
    if not math.isfinite(budget_squared):
        raise ValueError("normalized_energy_budget is too large to square")
    if budget_squared > 0.0:
        budget_squared = float(np.nextafter(budget_squared, 0.0))
    proposal_rng, attack_rng, transition_rng = _spawn_rngs(random_seed)

    history = initial_history
    remaining_energy_squared = budget_squared
    charged_energy_values: list[float] = []
    charged_energy_squared = 0.0
    public_records: list[PublicStepRecord[OutcomeT]] = []
    private_records: list[PrivateEnergyRecord] = []
    stop_reason: Literal["horizon", "transition", "abstention"] = "horizon"
    action_dimension: int | None = None

    for step_index, parameters in enumerate(step_parameters):
        baseline = _finite_vector(
            center_fn(step_index, history),
            name="center",
        )
        if action_dimension is None:
            action_dimension = int(baseline.size)
        elif baseline.size != action_dimension:
            raise ValueError("center dimension must remain fixed across the schedule")
        fallback_library = np.asarray(
            fallback_library_fn(step_index, history),
            dtype=np.float64,
        ).copy()

        requested = _finite_vector(
            shift_policy(
                step_index,
                copy.deepcopy(history),
                remaining_energy_squared,
                attack_rng,
            ),
            name="shift",
        )
        if requested.shape != baseline.shape:
            raise ValueError("shift dimension must match center dimension")

        limited = limit_shift_to_remaining_energy(
            requested,
            parameters.sigma,
            parameters.cap,
            parameters.correlation,
            remaining_energy_squared,
        )
        if limited.charged_energy_squared > remaining_energy_squared:
            raise RuntimeError("energy limiter exceeded the remaining allowance")
        candidate_total = math.fsum(
            (*charged_energy_values, limited.charged_energy_squared)
        )
        for _ in range(16):
            if candidate_total <= budget_squared:
                break
            previous_total = math.fsum(charged_energy_values)
            smaller_allowance = max(0.0, budget_squared - previous_total)
            if smaller_allowance > 0.0:
                smaller_allowance = float(np.nextafter(smaller_allowance, 0.0))
            limited = limit_shift_to_remaining_energy(
                requested,
                parameters.sigma,
                parameters.cap,
                parameters.correlation,
                smaller_allowance,
            )
            candidate_total = math.fsum(
                (*charged_energy_values, limited.charged_energy_squared)
            )
        if candidate_total > budget_squared:
            limited = limit_shift_to_remaining_energy(
                requested,
                parameters.sigma,
                parameters.cap,
                parameters.correlation,
                0.0,
            )
            candidate_total = math.fsum(
                (*charged_energy_values, limited.charged_energy_squared)
            )
        charged_energy_values.append(limited.charged_energy_squared)
        charged_energy_squared = candidate_total
        remaining_energy_squared = max(
            0.0,
            budget_squared - charged_energy_squared,
        )
        multiplier = comparison_multiplier(
            parameters.cap,
            parameters.correlation,
        )
        private_records.append(
            PrivateEnergyRecord(
                step_index=step_index,
                requested_shift=requested.copy(),
                applied_shift=limited.shift.copy(),
                comparison_multiplier=multiplier,
                charged_energy_squared=limited.charged_energy_squared,
                remaining_energy_squared=remaining_energy_squared,
                was_scaled=limited.was_scaled,
            )
        )

        attacked_center = baseline + limited.shift
        if not np.all(np.isfinite(attacked_center)):
            raise ValueError("center plus applied shift must remain finite")

        selection = totalized_ccfs_step(
            center=attacked_center,
            sigma=parameters.sigma,
            cap=parameters.cap,
            correlation=parameters.correlation,
            rng=proposal_rng,
            is_feasible=lambda action, index=step_index, current=history: verifier(
                index,
                current,
                action,
            ),
            fallback_library=fallback_library,
        )

        if selection.abstained:
            public_records.append(
                PublicStepRecord(
                    step_index=step_index,
                    action=None,
                    selected_index=selection.selected_index,
                    fallback_index=selection.fallback_index,
                    proposal_inspections=selection.proposal_inspections,
                    fallback_inspections=selection.fallback_inspections,
                    used_fallback=selection.used_fallback,
                    abstained=True,
                    transition_outcome=None,
                    transition_stopped=False,
                )
            )
            stop_reason = "abstention"
            break

        if selection.action is None:
            raise RuntimeError("a non-abstaining selection must contain an action")
        transitioned = transition(
            step_index,
            history,
            selection.action.copy(),
            transition_rng,
        )
        if not isinstance(transitioned, TransitionResult):
            raise TypeError("transition must return TransitionResult")
        public_records.append(
            PublicStepRecord(
                step_index=step_index,
                action=selection.action.copy(),
                selected_index=selection.selected_index,
                fallback_index=selection.fallback_index,
                proposal_inspections=selection.proposal_inspections,
                fallback_inspections=selection.fallback_inspections,
                used_fallback=selection.used_fallback,
                abstained=False,
                transition_outcome=transitioned.public_outcome,
                transition_stopped=bool(transitioned.stopped),
            )
        )
        history = transitioned.next_history
        if transitioned.stopped:
            stop_reason = "transition"
            break

    return CausalTrajectoryRollout(
        public_trace=tuple(public_records),
        private_energy_audit=tuple(private_records),
        final_history=history,
        stop_reason=stop_reason,
        normalized_energy_used=math.sqrt(charged_energy_squared),
    )
