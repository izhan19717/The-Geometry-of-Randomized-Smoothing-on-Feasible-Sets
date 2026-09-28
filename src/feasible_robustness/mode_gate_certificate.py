"""Reference implementation of anchored mode-gate sampling and certification."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .convex_patch import anchored_tempered_weights
from .smoothing import simultaneous_clopper_pearson


Label = Hashable
Action = TypeVar("Action")
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AnchoredGateCertificate:
    """Confidence-adjusted certificate for one declared anchored family."""

    selected_label: Label
    runner_up: Label
    selected_count: int
    sample_count: int
    selected_lower: float
    runner_upper: float
    radius: float
    method: str
    confidence: float
    tau: float
    c_tau: float
    anchor_sampler_tv_error: float


def anchored_gate_weights(
    anchor_occupancy: ArrayLike,
    current_occupancy: ArrayLike | None,
    tau: float,
) -> FloatArray:
    """Return the fixed or anchored gate at one query center.

    ``anchor_occupancy`` is the normalized occupancy gate at the declared
    anchor. For ``tau > 0``, ``current_occupancy`` is the normalized occupancy
    gate at the query center. At the fixed-gate endpoint ``tau == 0``, callers
    may omit ``current_occupancy`` because the query occupancy is not used.
    """

    tau_value = float(tau)
    if not math.isfinite(tau_value) or not 0.0 <= tau_value <= 1.0:
        raise ValueError("tau must be finite and lie in [0, 1]")
    anchor = np.asarray(anchor_occupancy, dtype=np.float64)
    if tau_value == 0.0 and current_occupancy is None:
        current = anchor
    elif current_occupancy is None:
        raise ValueError("current_occupancy is required when tau is positive")
    else:
        current = np.asarray(current_occupancy, dtype=np.float64)
    return anchored_tempered_weights(anchor, current, tau_value)


def sample_anchored_mixture(
    anchor_occupancy: ArrayLike,
    current_occupancy: ArrayLike | None,
    tau: float,
    component_samplers: Sequence[Callable[[np.random.Generator], Action]],
    rng: np.random.Generator,
) -> tuple[int, Action]:
    """Draw one mode and one action from an anchored convex-mode mixture.

    Each callback must draw one exact IID sample from the component law at the
    current query center. The callbacks can close over the query center and
    component geometry. A proved approximate-sampler error must be debited in
    the certificate through ``anchor_sampler_tv_error`` below.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    gate = anchored_gate_weights(anchor_occupancy, current_occupancy, tau)
    if len(component_samplers) != gate.size:
        raise ValueError("component_samplers must contain one callback per mode")
    if any(not callable(sampler) for sampler in component_samplers):
        raise TypeError("every component sampler must be callable")
    mode = int(rng.choice(gate.size, p=gate))
    return mode, component_samplers[mode](rng)


def certify_anchored_gate_from_counts(
    counts: Mapping[Label, int],
    sigma: float,
    diameter_upper: float | None,
    tau: float,
    *,
    label_universe: Iterable[Label],
    delta: float = 0.001,
    anchor_sampler_tv_error: float = 0.0,
) -> AnchoredGateCertificate:
    """Certify a fixed-anchor mode-gate smoother from IID label counts.

    The complete finite label universe is required, including unseen labels.
    Bounds are simultaneous Clopper--Pearson intervals with family-wise
    failure probability ``delta``. The order of ``label_universe`` supplies
    the deterministic tie rule for equal counts and equal upper bounds.

    For ``tau == 0`` the function uses the Gaussian event-comparison radius.
    For ``tau > 0`` it uses the anchored KL--Pinsker radius and therefore
    requires a certified upper bound on the feasible-set diameter. If the IID
    anchor sampler targets a law within total variation ``epsilon`` of the
    ideal anchor law, pass that value as ``anchor_sampler_tv_error`` to obtain
    the error-adjusted ideal-family radius. Centerwise execution error away
    from the anchor requires the additional event corrections in Proposition
    G.2 of the paper.
    """

    sigma_value = float(sigma)
    tau_value = float(tau)
    delta_value = float(delta)
    epsilon = float(anchor_sampler_tv_error)
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(tau_value) or not 0.0 <= tau_value <= 1.0:
        raise ValueError("tau must be finite and lie in [0, 1]")
    if not math.isfinite(delta_value) or not 0.0 < delta_value < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("anchor_sampler_tv_error must lie in [0, 1]")

    labels = tuple(label_universe)
    bounds = simultaneous_clopper_pearson(counts, labels, delta_value)
    complete = {label: int(counts.get(label, 0)) for label in labels}
    selected = max(labels, key=complete.__getitem__)
    competitors = tuple(label for label in labels if label != selected)
    runner_up = max(competitors, key=lambda label: bounds[label][1])
    selected_lower = float(bounds[selected][0])
    runner_upper = float(bounds[runner_up][1])
    sample_count = sum(complete.values())

    if tau_value == 0.0:
        c_tau = 1.0
        adjusted_lower = max(0.0, selected_lower - epsilon)
        adjusted_upper = min(1.0, runner_upper + epsilon)
        if adjusted_lower <= adjusted_upper:
            radius = 0.0
        else:
            radius = float(
                0.5
                * sigma_value
                * (norm.ppf(adjusted_lower) - norm.ppf(adjusted_upper))
            )
        method = "fixed_gate_gaussian"
    else:
        if diameter_upper is None:
            raise ValueError("diameter_upper is required when tau is positive")
        diameter_value = float(diameter_upper)
        if not math.isfinite(diameter_value) or diameter_value < 0.0:
            raise ValueError("diameter_upper must be finite and nonnegative")
        c_tau = math.sqrt(
            1.0 + tau_value**2 * diameter_value**2 / (4.0 * sigma_value**2)
        )
        margin = max(0.0, selected_lower - runner_upper - 2.0 * epsilon)
        radius = float(sigma_value * margin / c_tau)
        method = "anchored_kl_pinsker"

    return AnchoredGateCertificate(
        selected_label=selected,
        runner_up=runner_up,
        selected_count=complete[selected],
        sample_count=sample_count,
        selected_lower=selected_lower,
        runner_upper=runner_upper,
        radius=radius,
        method=method,
        confidence=1.0 - delta_value,
        tau=tau_value,
        c_tau=c_tau,
        anchor_sampler_tv_error=epsilon,
    )
