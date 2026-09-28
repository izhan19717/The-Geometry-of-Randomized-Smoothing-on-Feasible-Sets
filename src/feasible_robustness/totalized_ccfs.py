"""Finite CCFS selection with an explicit abstention outcome.

This module totalizes the first-verified selector without pretending that a
finite fallback library is guaranteed to contain a feasible physical action.
If all Gaussian proposals and all predeclared fallback actions fail the same
verifier, the result carries ``action=None`` and ``abstained=True``.  A caller
must not step the physical system on that outcome.

For a certified comparison, the verifier, proposal order, and fallback
library construction must be the same measurable program at both compared
centers.  Abstention is part of the output law and must be charged by any
favorable trajectory event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from feasible_robustness.ccfs import sample_equicorrelated_proposals


Array = np.ndarray


@dataclass(frozen=True, eq=False)
class TotalizedCCFSSelection:
    """Result of a finite proposal and fallback search.

    ``selected_index`` is the zero-based Gaussian proposal index when a
    proposal succeeds.  It is ``None`` for both fallback and abstention.
    ``fallback_index`` is the zero-based fallback-library index when fallback
    succeeds and is otherwise ``None``.
    """

    action: Array | None
    selected_index: int | None
    fallback_index: int | None
    proposal_inspections: int
    fallback_inspections: int
    used_fallback: bool
    abstained: bool

    def __post_init__(self) -> None:
        if self.abstained != (self.action is None):
            raise ValueError("abstention and action presence are inconsistent")
        if self.abstained and self.used_fallback:
            raise ValueError("an abstention cannot be a successful fallback")


def _finite_action_matrix(
    value: Array | Sequence[Sequence[float]],
    *,
    name: str,
    action_dimension: int | None = None,
) -> Array:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (count, action_dimension)")
    if action_dimension is None:
        if matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(f"{name} must contain at least one finite action")
    elif matrix.shape[1:] != (action_dimension,):
        raise ValueError(f"{name} action dimension does not match proposals")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def select_first_verified_or_abstain(
    proposals: Array | Sequence[Sequence[float]],
    is_feasible: Callable[[Array], bool],
    fallback_library: Array | Sequence[Sequence[float]],
) -> TotalizedCCFSSelection:
    """Select the first verified action from two finite ordered collections.

    The fallback library is a concrete finite matrix rather than an arbitrary
    iterator.  This makes exhaustion observable and prevents an accidental
    unbounded rejection loop.  An empty fallback matrix with shape ``(0,d)``
    is allowed and causes immediate abstention after proposal exhaustion.
    """

    proposal_matrix = _finite_action_matrix(proposals, name="proposals")
    dimension = int(proposal_matrix.shape[1])
    fallback_matrix = _finite_action_matrix(
        fallback_library,
        name="fallback_library",
        action_dimension=dimension,
    )

    for index, proposal in enumerate(proposal_matrix):
        candidate = np.asarray(proposal, dtype=np.float64)
        if bool(is_feasible(candidate)):
            return TotalizedCCFSSelection(
                action=candidate.copy(),
                selected_index=index,
                fallback_index=None,
                proposal_inspections=index + 1,
                fallback_inspections=0,
                used_fallback=False,
                abstained=False,
            )

    for index, fallback_action in enumerate(fallback_matrix):
        candidate = np.asarray(fallback_action, dtype=np.float64)
        if bool(is_feasible(candidate)):
            return TotalizedCCFSSelection(
                action=candidate.copy(),
                selected_index=None,
                fallback_index=index,
                proposal_inspections=int(proposal_matrix.shape[0]),
                fallback_inspections=index + 1,
                used_fallback=True,
                abstained=False,
            )

    return TotalizedCCFSSelection(
        action=None,
        selected_index=None,
        fallback_index=None,
        proposal_inspections=int(proposal_matrix.shape[0]),
        fallback_inspections=int(fallback_matrix.shape[0]),
        used_fallback=False,
        abstained=True,
    )


def totalized_ccfs_step(
    center: Array | Sequence[float],
    sigma: float,
    cap: int,
    correlation: float,
    rng: np.random.Generator,
    is_feasible: Callable[[Array], bool],
    fallback_library: Array | Sequence[Sequence[float]],
) -> TotalizedCCFSSelection:
    """Sample one finite Gaussian bundle, then verify or abstain."""

    proposals = sample_equicorrelated_proposals(
        center=center,
        sigma=sigma,
        cap=cap,
        correlation=correlation,
        rng=rng,
    )
    return select_first_verified_or_abstain(
        proposals=proposals,
        is_feasible=is_feasible,
        fallback_library=fallback_library,
    )
