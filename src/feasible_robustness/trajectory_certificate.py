"""Finite-sample certification helpers for capped causal trajectories.

These routines implement the scalar calculations in the adaptive trajectory
theorem.  They do not establish the theorem's caller contract.  In particular,
the event must be fixed independently of the nominal sample used for its bound,
and the normalized shift bound must hold for every reachable pre-block history
and every private attack seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import beta

from feasible_robustness.ccfs import (
    comparison_multiplier,
)


Array = np.ndarray
_MIN_POSITIVE_FLOAT = float(np.nextafter(0.0, 1.0))
_MAX_FLOAT = float(np.finfo(np.float64).max)
_ENERGY_ROUNDOFF_MARGIN = 1e-12


@dataclass(frozen=True)
class TrajectoryEventCertificate:
    """Certificate obtained from independent nominal trajectory outcomes."""

    successes: int
    trials: int
    alpha: float
    target_probability: float
    lower_probability: float
    certified: bool
    normalized_energy_radius: float | None
    center_energy_radius: float | None
    comparison_multiplier: float | None


@dataclass(frozen=True, eq=False)
class EnergyLimitedShift:
    """A pre-block shift projected onto the remaining normalized energy."""

    shift: Array
    charged_energy_squared: float
    was_scaled: bool


def _normalized_energy_squared_from_norm(
    vector_norm: float,
    sigma: float,
    multiplier: float,
) -> float:
    """Evaluate ``multiplier * vector_norm**2 / sigma**2`` safely.

    A positive value below the floating-point range is rounded upward to the
    least positive float.  Other positive finite results are also moved one
    representable value upward so budget accounting remains conservative.
    """

    if vector_norm == 0.0:
        return 0.0
    log_charge = (
        math.log(multiplier)
        + 2.0 * math.log(vector_norm)
        - 2.0 * math.log(sigma)
    )
    if log_charge > math.log(_MAX_FLOAT):
        return math.inf
    if log_charge < math.log(_MIN_POSITIVE_FLOAT):
        return _MIN_POSITIVE_FLOAT
    charge = math.exp(log_charge) * (1.0 + _ENERGY_ROUNDOFF_MARGIN)
    if not math.isfinite(charge):
        return math.inf
    return float(np.nextafter(charge, math.inf))


def _validate_count_inputs(successes: int, trials: int, alpha: float) -> None:
    if isinstance(successes, (bool, np.bool_)) or not isinstance(
        successes, (int, np.integer)
    ):
        raise TypeError("successes must be an integer")
    if isinstance(trials, (bool, np.bool_)) or not isinstance(
        trials, (int, np.integer)
    ):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")


def one_sided_binomial_lower_bound(
    successes: int,
    trials: int,
    alpha: float,
) -> float:
    """Return the exact one-sided Clopper--Pearson lower bound.

    The coverage is at least ``1-alpha`` for a fixed Bernoulli event.  A caller
    that reports several events or selects an event from the same data must use
    a valid simultaneous adjustment instead of reusing ``alpha`` for each one.
    """

    _validate_count_inputs(successes, trials, alpha)
    if successes == 0:
        return 0.0
    lower = float(beta.ppf(alpha, successes, trials - successes + 1))
    if not math.isfinite(lower) or lower < 0.0 or lower > 1.0:
        raise FloatingPointError("binomial lower-bound calculation was nonfinite")
    if lower == 1.0:
        lower = float(np.nextafter(1.0, 0.0))
    return lower


def certify_trajectory_event_from_counts(
    successes: int,
    trials: int,
    alpha: float,
    target_probability: float,
    *,
    sigma: float | None = None,
    cap: int | None = None,
    correlation: float | None = None,
) -> TrajectoryEventCertificate:
    """Certify a predeclared trajectory event from nominal IID outcomes.

    The general output is a radius in normalized path energy.  Supplying the
    constant schedule ``sigma``, ``cap``, and ``correlation`` also returns the
    Euclidean center-energy radius
    ``sigma * normalized_energy_radius / sqrt(kappa)``.  The three schedule
    arguments must be supplied together.  A varying schedule retains the
    ellipsoid stated in the theorem and should use the normalized radius.

    ``certified`` is false when the probability lower bound is below the target.
    In that case both radii are ``None`` because even zero shift is not certified.
    """

    _validate_count_inputs(successes, trials, alpha)
    if not math.isfinite(target_probability) or not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must lie in (0,1)")
    schedule = (sigma, cap, correlation)
    if any(value is None for value in schedule) and not all(
        value is None for value in schedule
    ):
        raise ValueError("sigma, cap, and correlation must be supplied together")

    multiplier: float | None = None
    if sigma is not None:
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        assert cap is not None and correlation is not None
        multiplier = comparison_multiplier(cap, correlation)

    lower = one_sided_binomial_lower_bound(successes, trials, alpha)
    if lower < target_probability:
        return TrajectoryEventCertificate(
            successes=int(successes),
            trials=int(trials),
            alpha=float(alpha),
            target_probability=float(target_probability),
            lower_probability=lower,
            certified=False,
            normalized_energy_radius=None,
            center_energy_radius=None,
            comparison_multiplier=multiplier,
        )

    normalized_radius = float(ndtri(lower) - ndtri(target_probability))
    center_radius = (
        None
        if multiplier is None
        else float(sigma * normalized_radius / math.sqrt(multiplier))
    )
    return TrajectoryEventCertificate(
        successes=int(successes),
        trials=int(trials),
        alpha=float(alpha),
        target_probability=float(target_probability),
        lower_probability=lower,
        certified=True,
        normalized_energy_radius=normalized_radius,
        center_energy_radius=center_radius,
        comparison_multiplier=multiplier,
    )


def trajectory_shift_distance(
    shifts: Array,
    sigmas: Array | Sequence[float],
    caps: Sequence[int],
    correlations: Array | Sequence[float],
) -> float:
    """Return normalized energy for one supplied sequence of pre-block shifts.

    Certification requires an upper bound on this value for every attack seed
    and reachable history.  Computing it on one observed trace is only a trace
    diagnostic.  Center differences from separately generated feedback
    rollouts are not the conditional shifts required by the theorem.
    """

    shift_array = np.asarray(shifts, dtype=np.float64)
    scale = np.asarray(sigmas, dtype=np.float64)
    rho = np.asarray(correlations, dtype=np.float64)
    cap_values = tuple(caps)
    if shift_array.ndim != 2:
        raise ValueError("shifts must have shape (horizon, action_dimension)")
    horizon = shift_array.shape[0]
    if horizon < 1 or shift_array.shape[1] < 1:
        raise ValueError("shifts must have nonempty horizon and action dimension")
    if scale.shape != (horizon,) or rho.shape != (horizon,):
        raise ValueError("sigmas and correlations must have one value per time block")
    if len(cap_values) != horizon:
        raise ValueError("caps must have one value per time block")
    if not np.all(np.isfinite(shift_array)):
        raise ValueError("shifts must be finite")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("sigmas must be finite and positive")

    distance = 0.0
    for index, cap in enumerate(cap_values):
        multiplier = comparison_multiplier(cap, float(rho[index]))
        for coordinate in shift_array[index]:
            magnitude = abs(float(coordinate))
            if magnitude == 0.0:
                continue
            ratio = magnitude / float(scale[index])
            if math.isfinite(ratio) and ratio > 0.0:
                component = ratio * math.sqrt(multiplier)
            elif math.isinf(ratio):
                component = math.inf
            else:
                log_component = (
                    math.log(magnitude)
                    - math.log(float(scale[index]))
                    + 0.5 * math.log(multiplier)
                )
                if log_component < math.log(_MIN_POSITIVE_FLOAT):
                    component = 0.0
                else:
                    component = math.exp(log_component)
            distance = math.hypot(distance, component)
    return float(distance)


def limit_shift_to_remaining_energy(
    shift: Array | Sequence[float],
    sigma: float,
    cap: int,
    correlation: float,
    remaining_energy_squared: float,
) -> EnergyLimitedShift:
    """Project a proposed pre-block shift onto a remaining energy allowance.

    This pure calculation must be called before the current Gaussian block is
    drawn.  It limits the shift channel only.  It does not prevent an attack
    from entering a verifier, feasible set, fallback, controller, or dynamics
    through another input.
    """

    vector = np.asarray(shift, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("shift must be a finite nonempty vector")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(remaining_energy_squared) or remaining_energy_squared < 0.0:
        raise ValueError("remaining_energy_squared must be finite and nonnegative")

    multiplier = comparison_multiplier(cap, correlation)
    vector_norm = math.hypot(*(abs(float(value)) for value in vector))
    if vector_norm == 0.0:
        return EnergyLimitedShift(
            shift=vector.copy(),
            charged_energy_squared=0.0,
            was_scaled=False,
        )
    if remaining_energy_squared == 0.0:
        return EnergyLimitedShift(
            shift=np.zeros_like(vector),
            charged_energy_squared=0.0,
            was_scaled=True,
        )

    proposed_charge = _normalized_energy_squared_from_norm(
        vector_norm,
        sigma,
        multiplier,
    )
    if math.isfinite(proposed_charge) and proposed_charge <= remaining_energy_squared:
        return EnergyLimitedShift(
            shift=vector.copy(),
            charged_energy_squared=proposed_charge,
            was_scaled=False,
        )

    log_scale = (
        0.5 * math.log(remaining_energy_squared)
        + math.log(sigma)
        - 0.5 * math.log(multiplier)
        - math.log(vector_norm)
        - _ENERGY_ROUNDOFF_MARGIN
    )
    scale = 0.0 if log_scale < math.log(_MIN_POSITIVE_FLOAT) else math.exp(log_scale)
    scale = min(scale, 1.0)

    def scaled_charge(candidate: Array) -> float:
        candidate_norm = math.hypot(*(abs(float(value)) for value in candidate))
        return _normalized_energy_squared_from_norm(
            candidate_norm,
            sigma,
            multiplier,
        )

    limited = vector * scale
    charge = scaled_charge(limited)
    for _ in range(16):
        if charge <= remaining_energy_squared or scale == 0.0:
            break
        correction = math.sqrt(remaining_energy_squared / charge)
        if not math.isfinite(correction) or correction <= 0.0:
            scale = 0.0
        else:
            scale *= min(correction, 1.0)
            scale = float(np.nextafter(scale, 0.0))
        limited = vector * scale
        charge = scaled_charge(limited)
    if charge > remaining_energy_squared:
        limited = np.zeros_like(vector)
        charge = 0.0
    return EnergyLimitedShift(
        shift=limited,
        charged_energy_squared=charge,
        was_scaled=True,
    )
