"""Develop a proof-carrying stable-direction bounded two-band filter.

This is an open CIFAR-10 *training-split* development study.  It never opens a
test split.  For each fixed binary AuditVotes ResNet task, the runner creates a
hash-seeded bank of unit Rademacher raw-pixel directions.  Only fit images are
used to rank those directions by clean-projection dispersion.  The eight most
stable directions are passed to a disjoint proposal-search stream, which
selects one direction and one bounded two-band filter

    alpha <= |<u, z> + b| <= beta,       0 < alpha < beta <= sigma.

The filter acts on the raw, unclipped Gaussian proposal.  Isotropic Gaussian
factorization leaves the coordinates orthogonal to ``u`` independent of the
event.  The retained ``u`` coordinate lies in an interval of total range
``2 beta``, so Popoviciu gives conditional variance at most ``beta**2``.
Thus ``Cov(Z | retained) <= sigma**2 I`` globally over Gaussian centers.

Selection fails closed unless at least 90% of development images retain an
adequate number of search proposals and mean search retention is at least
0.25.  A disjoint evaluation stream reports a covariance-radius lower bound,
a joint-mass lower bound, a population joint-mass radius upper bound, and an
unfiltered lower bound under one six-event per-image Bonferroni allocation.
All results remain development-only and have ``paper_eligibility = "none"``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "outputs/iclr2027_learned_stable_band_filter_development.json"
)
PAPER_ELIGIBILITY = "none"
EXPECTED_AUDITVOTES_COMMIT = "52b6a0db53947c815884ad62927d6af6dc584705"
EXPECTED_CHECKPOINT_SHA256 = (
    "420333fe0380cc437218c9b67c20bacf687932957da47340f900d4af5e05bd7c"
)
RAW_IMAGE_SHAPE = (3, 32, 32)
RAW_DIMENSION = int(np.prod(RAW_IMAGE_SHAPE))
MIN_DIRECTION_BANK_SIZE = 64
STABLE_DIRECTION_COUNT = 8
MIN_ADEQUATE_IMAGE_FRACTION = 0.90
MIN_MEAN_RETENTION = 0.25
BONFERRONI_TAIL_EVENTS = 6

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Fixed before running this development search.  Budget truncation preserves
# this order.
CONFUSABLE_CLASS_PAIRS: tuple[tuple[int, int], ...] = (
    (3, 5),
    (2, 4),
    (4, 7),
    (5, 7),
    (0, 8),
    (1, 9),
)


@dataclass(frozen=True)
class RemoteBudget:
    pair_count: int
    fit_per_class: int
    dev_per_class: int
    direction_bank_size: int
    stable_direction_count: int
    search_proposals: int
    evaluation_proposals: int
    batch_size: int
    offset_quantiles: int
    outer_fractions: tuple[float, ...]
    inner_fractions: tuple[float, ...]
    min_search_retained: int
    min_evaluation_retained: int


REMOTE_BUDGETS: dict[str, RemoteBudget] = {
    "smoke": RemoteBudget(
        pair_count=1,
        fit_per_class=16,
        dev_per_class=2,
        direction_bank_size=64,
        stable_direction_count=8,
        search_proposals=64,
        evaluation_proposals=128,
        batch_size=128,
        offset_quantiles=3,
        outer_fractions=(0.75, 1.0),
        inner_fractions=(0.025, 0.10),
        min_search_retained=8,
        min_evaluation_retained=8,
    ),
    "default": RemoteBudget(
        pair_count=len(CONFUSABLE_CLASS_PAIRS),
        fit_per_class=500,
        dev_per_class=24,
        direction_bank_size=64,
        stable_direction_count=8,
        search_proposals=512,
        evaluation_proposals=4096,
        batch_size=1024,
        offset_quantiles=7,
        outer_fractions=(0.50, 0.75, 1.0),
        inner_fractions=(0.025, 0.10, 0.25),
        min_search_retained=64,
        min_evaluation_retained=64,
    ),
}


@dataclass(frozen=True)
class StableBandCandidate:
    """A selected-bank direction and globally bounded two-band parameters."""

    direction_slot: int
    direction_bank_index: int
    offset: float
    inner: float
    outer: float


@dataclass(frozen=True)
class MultiDirectionProposalBatch:
    image_index: int
    true_binary_label: int
    task_labels: np.ndarray
    # Shape: (stable_direction_count, proposal_count).
    projections: np.ndarray


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return sha256(array.tobytes()).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big") % (2**63 - 1)


def deterministic_class_split(
    labels: Sequence[int],
    *,
    fit_per_class: int,
    dev_per_class: int,
    seed: int,
    classes: Iterable[int] = range(10),
) -> dict[int, dict[str, list[int]]]:
    """Hash-rank nonoverlapping fit/dev indices within each requested class."""

    if fit_per_class < 1 or dev_per_class < 1:
        raise ValueError("fit_per_class and dev_per_class must be positive")
    class_tuple = tuple(int(value) for value in classes)
    if len(class_tuple) != len(set(class_tuple)):
        raise ValueError("classes must be unique")
    buckets: dict[int, list[tuple[bytes, int]]] = {
        label: [] for label in class_tuple
    }
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label in buckets:
            rank = sha256(
                f"learned-stable-band-v2:{seed}:{label}:{index}".encode()
            ).digest()
            buckets[label].append((rank, index))
    result: dict[int, dict[str, list[int]]] = {}
    required = fit_per_class + dev_per_class
    for label in class_tuple:
        ordered = [index for _, index in sorted(buckets[label])]
        if len(ordered) < required:
            raise ValueError(
                f"class {label} has {len(ordered)} examples; need {required}"
            )
        result[label] = {
            "fit": ordered[:fit_per_class],
            "dev": ordered[fit_per_class:required],
        }
    return result


def rademacher_direction_bank(
    *,
    pair: tuple[int, int],
    seed: int,
    bank_size: int = MIN_DIRECTION_BANK_SIZE,
    dimension: int = RAW_DIMENSION,
) -> np.ndarray:
    """Create a version-stable hash-seeded bank of unit sign directions."""

    if bank_size < MIN_DIRECTION_BANK_SIZE:
        raise ValueError(f"direction bank must contain at least {MIN_DIRECTION_BANK_SIZE}")
    if dimension < 1 or len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("invalid dimension or class pair")
    byte_count = (dimension + 7) // 8
    directions = np.empty((bank_size, dimension), dtype=np.float64)
    scale = 1.0 / math.sqrt(dimension)
    for direction_index in range(bank_size):
        generated = bytearray()
        block_index = 0
        while len(generated) < byte_count:
            generated.extend(
                sha256(
                    (
                        f"stable-band-v2:rademacher:{seed}:{pair[0]}:{pair[1]}:"
                        f"{direction_index}:{block_index}"
                    ).encode()
                ).digest()
            )
            block_index += 1
        bits = np.unpackbits(
            np.frombuffer(bytes(generated[:byte_count]), dtype=np.uint8)
        )[:dimension]
        directions[direction_index] = (2.0 * bits.astype(np.float64) - 1.0) * scale
    return directions


def rank_stable_directions(
    bank: np.ndarray,
    fit_images: np.ndarray,
    *,
    retain: int = STABLE_DIRECTION_COUNT,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Rank directions by pooled clean fit-image projection dispersion."""

    directions = np.asarray(bank, dtype=np.float64)
    images = np.asarray(fit_images, dtype=np.float64)
    if directions.ndim != 2 or images.ndim < 2:
        raise ValueError("bank must be a matrix and fit_images must be batched")
    flattened = images.reshape(images.shape[0], -1)
    if flattened.shape[0] < 2 or flattened.shape[1] != directions.shape[1]:
        raise ValueError("fit image dimension does not match direction bank")
    if np.any(~np.isfinite(directions)) or np.any(~np.isfinite(flattened)):
        raise ValueError("directions and fit images must be finite")
    if not 1 <= retain <= directions.shape[0]:
        raise ValueError("invalid stable-direction retain count")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("every candidate direction must have unit norm")
    projections = flattened @ directions.T
    dispersions = np.std(projections, axis=0, ddof=0)
    means = np.mean(projections, axis=0)
    order = tuple(
        sorted(range(directions.shape[0]), key=lambda i: (dispersions[i], i))
    )
    rows = [
        {
            "stability_rank": rank,
            "bank_index": bank_index,
            "clean_projection_dispersion": float(dispersions[bank_index]),
            "clean_projection_mean": float(means[bank_index]),
            "direction_float64_sha256": _array_sha256(directions[bank_index]),
            "retained_for_search": rank < retain,
        }
        for rank, bank_index in enumerate(order)
    ]
    return order[:retain], rows


def validate_band_candidate(
    candidate: StableBandCandidate,
    sigma: float,
    *,
    stable_direction_count: int | None = None,
) -> None:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if candidate.direction_slot < 0 or candidate.direction_bank_index < 0:
        raise ValueError("direction indices must be nonnegative")
    if stable_direction_count is not None and not (
        candidate.direction_slot < stable_direction_count
    ):
        raise ValueError("direction slot is outside the retained stable bank")
    if not all(
        math.isfinite(value)
        for value in (candidate.offset, candidate.inner, candidate.outer)
    ):
        raise ValueError("band parameters must be finite")
    if not 0.0 < candidate.inner < candidate.outer <= sigma:
        raise ValueError("require 0 < inner < outer <= sigma")


def bounded_two_band_mask(
    projections: np.ndarray, candidate: StableBandCandidate, sigma: float
) -> np.ndarray:
    """Apply the exact bounded band to one direction's raw projections."""

    validate_band_candidate(candidate, sigma)
    values = np.asarray(projections, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("projections must be a finite vector")
    shifted = np.abs(values + candidate.offset)
    return (shifted >= candidate.inner) & (shifted <= candidate.outer)


def covariance_proof(
    direction: np.ndarray, candidate: StableBandCandidate, sigma: float
) -> dict[str, Any]:
    """Return the analytic center-uniform covariance proof payload."""

    validate_band_candidate(candidate, sigma)
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(vector)) or not math.isclose(
        float(np.linalg.norm(vector)), 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError("direction must be finite and unit norm")
    return {
        "filter": "alpha <= abs(dot(u, raw_proposal) + b) <= beta",
        "raw_proposal_before_clipping": True,
        "clipping_performed": False,
        "direction_norm": float(np.linalg.norm(vector)),
        "directional_support_after_offset": [
            [-candidate.outer, -candidate.inner],
            [candidate.inner, candidate.outer],
        ],
        "directional_variance_upper_by_popoviciu": candidate.outer**2,
        "orthogonal_covariance": "sigma^2 * (I - u u^T)",
        "cross_covariance": 0.0,
        "lambda_max_upper": sigma**2,
        "normalized_covariance_factor": 1.0,
        "global_over_all_gaussian_centers": True,
        "reason": (
            "Gaussian factorization makes the orthogonal component independent "
            "of retention; the retained scalar has range 2*beta, hence variance "
            "at most beta^2 <= sigma^2"
        ),
    }


def candidate_offsets(center_projections: np.ndarray, count: int) -> tuple[float, ...]:
    """Build deterministic offsets from search-stream per-image centers."""

    values = np.asarray(center_projections, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("center projections must be a finite vector of length >=2")
    if count < 2:
        raise ValueError("offset grid needs at least two quantiles")
    quantiles = np.linspace(0.1, 0.9, count)
    centers = np.quantile(values, quantiles, method="linear")
    return tuple(float(value) for value in np.unique(-centers))


def candidate_grid(
    *,
    stable_bank_indices: Sequence[int],
    offsets_by_slot: Sequence[Sequence[float]],
    sigma: float,
    outer_fractions: Sequence[float],
    inner_fractions: Sequence[float],
) -> tuple[StableBandCandidate, ...]:
    """Enumerate the full direction/offset/band development grid."""

    if len(stable_bank_indices) != len(offsets_by_slot) or not stable_bank_indices:
        raise ValueError("stable directions and offset grids must be nonempty and aligned")
    candidates: list[StableBandCandidate] = []
    for direction_slot, (bank_index, offsets) in enumerate(
        zip(stable_bank_indices, offsets_by_slot)
    ):
        for offset in offsets:
            for outer_fraction in outer_fractions:
                outer = sigma * float(outer_fraction)
                for inner_fraction in inner_fractions:
                    candidate = StableBandCandidate(
                        direction_slot=direction_slot,
                        direction_bank_index=int(bank_index),
                        offset=float(offset),
                        inner=outer * float(inner_fraction),
                        outer=outer,
                    )
                    validate_band_candidate(
                        candidate,
                        sigma,
                        stable_direction_count=len(stable_bank_indices),
                    )
                    candidates.append(candidate)
    if not candidates:
        raise ValueError("candidate grid is empty")
    return tuple(candidates)


def binary_tie_kl(top_probability: float) -> float:
    """Least binary categorical KL required to move p>1/2 to a tie."""

    probability = float(top_probability)
    if not math.isfinite(probability) or not 0.5 < probability <= 1.0:
        return 0.0
    complement = 1.0 - probability
    first = probability * math.log(2.0 * probability)
    second = (
        0.0
        if complement == 0.0
        else complement * math.log(2.0 * complement)
    )
    return first + second


def empirical_covariance_radius(
    selected_count: int, retained_count: int, sigma: float
) -> float:
    if retained_count < 1 or not 0 <= selected_count <= retained_count:
        return 0.0
    return float(
        sigma
        * math.sqrt(2.0 * binary_tie_kl(selected_count / retained_count))
    )


def empirical_joint_mass_radius(
    selected_count: int, competitor_count: int, proposals: int, sigma: float
) -> float:
    """Plug-in joint-mass radius, with mathematical infinite endpoints."""

    if (
        proposals < 1
        or min(selected_count, competitor_count) < 0
        or selected_count + competitor_count > proposals
    ):
        raise ValueError("invalid empirical joint counts")
    if selected_count <= competitor_count:
        return 0.0
    selected_mass = selected_count / proposals
    competitor_mass = competitor_count / proposals
    if selected_mass >= 1.0 or competitor_mass <= 0.0:
        return math.inf
    return float(
        0.5
        * sigma
        * (norm.ppf(selected_mass) - norm.ppf(competitor_mass))
    )


def one_sided_binomial_lower(successes: int, trials: int, error: float) -> float:
    if trials < 1 or not 0 <= successes <= trials or not 0.0 < error < 1.0:
        raise ValueError("invalid lower-bound inputs")
    if successes == 0:
        return 0.0
    return float(beta_distribution.ppf(error, successes, trials - successes + 1))


def one_sided_binomial_upper(successes: int, trials: int, error: float) -> float:
    if trials < 1 or not 0 <= successes <= trials or not 0.0 < error < 1.0:
        raise ValueError("invalid upper-bound inputs")
    if successes == trials:
        return 1.0
    return float(
        beta_distribution.ppf(1.0 - error, successes + 1, trials - successes)
    )


def certified_radii_from_counts(
    *,
    filtered_selected_label: int,
    unfiltered_selected_label: int,
    retained_counts: Sequence[int],
    unfiltered_counts: Sequence[int],
    proposal_count: int,
    sigma: float,
    delta: float,
    min_retained: int = 1,
) -> dict[str, Any]:
    """Return four radii under one six-tail per-image Bonferroni event.

    The selected labels and filter must come from an independent search stream.
    ``r_mass_U`` is ``None`` exactly when the Clopper--Pearson endpoints make
    the valid population-radius upper bound infinite.
    """

    retained = tuple(int(value) for value in retained_counts)
    unfiltered = tuple(int(value) for value in unfiltered_counts)
    if len(retained) != 2 or len(unfiltered) != 2:
        raise ValueError("binary counts must have length two")
    if filtered_selected_label not in (0, 1) or unfiltered_selected_label not in (0, 1):
        raise ValueError("selected labels must be binary")
    retained_total = sum(retained)
    if (
        proposal_count < 1
        or min_retained < 1
        or any(value < 0 for value in retained + unfiltered)
        or retained_total > proposal_count
        or sum(unfiltered) != proposal_count
        or not math.isfinite(sigma)
        or sigma <= 0
        or not 0.0 < delta < 1.0
    ):
        raise ValueError("invalid proposal counts, confidence, or scale")

    tail_error = delta / BONFERRONI_TAIL_EVENTS
    selected_retained = retained[filtered_selected_label]
    competitor_retained = retained[1 - filtered_selected_label]

    conditional_lower = 0.0
    r_cov_lower = 0.0
    if retained_total >= min_retained:
        conditional_lower = one_sided_binomial_lower(
            selected_retained, retained_total, tail_error
        )
        r_cov_lower = float(
            sigma * math.sqrt(2.0 * binary_tie_kl(conditional_lower))
        )

    selected_mass_lower = one_sided_binomial_lower(
        selected_retained, proposal_count, tail_error
    )
    competitor_mass_upper = one_sided_binomial_upper(
        competitor_retained, proposal_count, tail_error
    )
    r_mass_lower = 0.0
    if selected_mass_lower > competitor_mass_upper:
        r_mass_lower = float(
            0.5
            * sigma
            * (
                norm.ppf(selected_mass_lower)
                - norm.ppf(competitor_mass_upper)
            )
        )

    selected_mass_upper = one_sided_binomial_upper(
        selected_retained, proposal_count, tail_error
    )
    competitor_mass_lower = one_sided_binomial_lower(
        competitor_retained, proposal_count, tail_error
    )
    mass_upper_is_finite = not (
        selected_mass_upper >= 1.0 or competitor_mass_lower <= 0.0
    )
    r_mass_upper: float | None
    if not mass_upper_is_finite:
        r_mass_upper = None
    elif selected_mass_upper <= competitor_mass_lower:
        r_mass_upper = 0.0
    else:
        r_mass_upper = float(
            0.5
            * sigma
            * (
                norm.ppf(selected_mass_upper)
                - norm.ppf(competitor_mass_lower)
            )
        )

    unfiltered_lower = one_sided_binomial_lower(
        unfiltered[unfiltered_selected_label], proposal_count, tail_error
    )
    r_unfiltered_lower = (
        float(sigma * norm.ppf(unfiltered_lower))
        if unfiltered_lower > 0.5
        else 0.0
    )
    strong_separation = bool(
        r_mass_upper is not None and r_cov_lower > r_mass_upper
    )
    return {
        "bonferroni_tail_events": BONFERRONI_TAIL_EVENTS,
        "bonferroni_tail_error": tail_error,
        "bonferroni_total_error": tail_error * BONFERRONI_TAIL_EVENTS,
        "conditional_selected_probability_lower": conditional_lower,
        "joint_selected_mass_lower": selected_mass_lower,
        "joint_competitor_mass_upper": competitor_mass_upper,
        "joint_selected_mass_upper": selected_mass_upper,
        "joint_competitor_mass_lower": competitor_mass_lower,
        "unfiltered_selected_probability_lower": unfiltered_lower,
        "r_cov_L": r_cov_lower,
        "r_mass_L": r_mass_lower,
        "r_mass_U": r_mass_upper,
        "r_mass_U_is_finite": mass_upper_is_finite,
        "r_unfiltered_L": r_unfiltered_lower,
        "strong_separation": strong_separation,
    }


def _counts(labels: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int]:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("task labels must be a vector")
    if mask is not None:
        boolean_mask = np.asarray(mask, dtype=bool)
        if boolean_mask.shape != values.shape:
            raise ValueError("mask and labels must have the same shape")
        values = values[boolean_mask]
    if np.any((values < 0) | (values > 1)):
        raise ValueError("task labels must be binary")
    counts = np.bincount(values, minlength=2)
    return int(counts[0]), int(counts[1])


def _selected_label(counts: Sequence[int]) -> int:
    # Fixed tie rule: the left class in the declared pair wins.
    return int(int(counts[1]) > int(counts[0]))


def score_candidate(
    candidate: StableBandCandidate,
    batches: Sequence[MultiDirectionProposalBatch],
    *,
    sigma: float,
    min_retained: int,
    min_adequate_fraction: float = MIN_ADEQUATE_IMAGE_FRACTION,
    min_mean_retention: float = MIN_MEAN_RETENTION,
) -> tuple[dict[str, Any], dict[int, int]]:
    """Score feasibility and the two-component empirical search objective."""

    if not batches:
        raise ValueError("search batches are empty")
    if min_retained < 1 or not 0.0 < min_adequate_fraction <= 1.0:
        raise ValueError("invalid adequate-retention requirement")
    if not 0.0 < min_mean_retention <= 1.0:
        raise ValueError("invalid mean-retention requirement")
    validate_band_candidate(
        candidate,
        sigma,
        stable_direction_count=batches[0].projections.shape[0],
    )
    correct_covariance_radii: list[float] = []
    correct_positive_gaps: list[float] = []
    retention_rates: list[float] = []
    selected_by_image: dict[int, int] = {}
    adequate_count = 0
    nonzero_count = 0
    correct_count = 0
    for batch in batches:
        projections = np.asarray(batch.projections, dtype=np.float64)
        if projections.ndim != 2 or candidate.direction_slot >= projections.shape[0]:
            raise ValueError("search projection matrix is incompatible with candidate")
        labels = np.asarray(batch.task_labels, dtype=np.int64)
        if projections.shape[1] != labels.size or labels.size < 1:
            raise ValueError("proposal labels and projections are not aligned")
        mask = bounded_two_band_mask(
            projections[candidate.direction_slot], candidate, sigma
        )
        counts = _counts(labels, mask)
        selected = _selected_label(counts)
        selected_by_image[batch.image_index] = selected
        retained = sum(counts)
        retention_rates.append(retained / labels.size)
        nonzero_count += int(retained > 0)
        adequate_count += int(retained >= min_retained)
        covariance_radius = 0.0
        positive_gap = 0.0
        if retained >= min_retained and selected == batch.true_binary_label:
            correct_count += 1
            covariance_radius = empirical_covariance_radius(
                counts[selected], retained, sigma
            )
            mass_radius = empirical_joint_mass_radius(
                counts[selected], counts[1 - selected], labels.size, sigma
            )
            if math.isfinite(mass_radius):
                positive_gap = max(0.0, covariance_radius - mass_radius)
        correct_covariance_radii.append(covariance_radius)
        correct_positive_gaps.append(positive_gap)

    image_count = len(batches)
    adequate_fraction = adequate_count / image_count
    nonzero_fraction = nonzero_count / image_count
    mean_retention = float(np.mean(retention_rates))
    feasible = bool(
        adequate_fraction >= min_adequate_fraction
        and nonzero_fraction >= min_adequate_fraction
        and mean_retention >= min_mean_retention
    )
    mean_covariance = float(np.mean(correct_covariance_radii))
    mean_gap = float(np.mean(correct_positive_gaps))
    summary: dict[str, Any] = {
        "feasible": feasible,
        "required_adequate_image_fraction": min_adequate_fraction,
        "required_mean_retention": min_mean_retention,
        "adequate_retained_per_image": min_retained,
        "adequate_image_fraction": adequate_fraction,
        "nonzero_image_fraction": nonzero_fraction,
        "mean_retention": mean_retention,
        "min_retention": float(np.min(retention_rates)),
        "max_retention": float(np.max(retention_rates)),
        "correct_adequate_fraction": correct_count / image_count,
        "objective_mean_correct_empirical_covariance_radius": mean_covariance,
        "objective_mean_correct_positive_covariance_over_joint_gap": mean_gap,
        "selection_objective": (
            "lexicographically maximize mean correct empirical covariance "
            "radius, then mean correct positive covariance-minus-joint gap"
        ),
    }
    return summary, selected_by_image


def select_candidate(
    candidates: Sequence[StableBandCandidate],
    batches: Sequence[MultiDirectionProposalBatch],
    *,
    sigma: float,
    min_retained: int,
    min_adequate_fraction: float = MIN_ADEQUATE_IMAGE_FRACTION,
    min_mean_retention: float = MIN_MEAN_RETENTION,
) -> tuple[
    StableBandCandidate | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    """Select only among feasible grid rows; return ``None`` if none exist."""

    if not candidates:
        raise ValueError("candidate grid is empty")
    scored: list[
        tuple[tuple[float, ...], StableBandCandidate, dict[str, Any]]
    ] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        summary, _ = score_candidate(
            candidate,
            batches,
            sigma=sigma,
            min_retained=min_retained,
            min_adequate_fraction=min_adequate_fraction,
            min_mean_retention=min_mean_retention,
        )
        rows.append({**asdict(candidate), **summary})
        if summary["feasible"]:
            key = (
                summary["objective_mean_correct_empirical_covariance_radius"],
                summary[
                    "objective_mean_correct_positive_covariance_over_joint_gap"
                ],
                summary["correct_adequate_fraction"],
                summary["adequate_image_fraction"],
                summary["mean_retention"],
                -candidate.outer,
                -candidate.inner,
                -abs(candidate.offset),
                -candidate.direction_slot,
            )
            scored.append((key, candidate, summary))
    if not scored:
        return None, None, rows
    _, selected, selected_summary = max(scored, key=lambda row: row[0])
    return selected, selected_summary, rows


def summarize_evaluation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation rows are empty")
    radius_specs = (
        ("covariance_lower", "r_cov_L", "filtered_correct"),
        ("joint_mass_lower", "r_mass_L", "filtered_correct"),
        ("unfiltered_lower", "r_unfiltered_L", "unfiltered_correct"),
    )
    thresholds = (0.0, 0.05, 0.10, 0.15, 0.20)
    radius_summary: dict[str, Any] = {}
    for name, radius_key, correct_key in radius_specs:
        values = np.asarray([row[radius_key] for row in rows], dtype=float)
        correct = np.asarray([row[correct_key] for row in rows], dtype=bool)
        radius_summary[name] = {
            "positive_fraction": float(np.mean(values > 0.0)),
            "mean_radius": float(np.mean(values)),
            "median_radius": float(np.median(values)),
            "max_radius": float(np.max(values)),
            "correct_fraction_above_radius": {
                f"{threshold:.2f}": float(np.mean(correct & (values > threshold)))
                for threshold in thresholds
            },
        }
    finite_mass_upper = [
        float(row["r_mass_U"])
        for row in rows
        if row["r_mass_U"] is not None
    ]
    strong = np.asarray([row["strong_separation"] for row in rows], dtype=bool)
    correct = np.asarray([row["filtered_correct"] for row in rows], dtype=bool)
    return {
        "images": len(rows),
        "mean_retention": float(np.mean([row["retention_rate"] for row in rows])),
        "min_retention": float(np.min([row["retention_rate"] for row in rows])),
        "max_retention": float(np.max([row["retention_rate"] for row in rows])),
        "adequate_evaluation_fraction": float(
            np.mean([row["evaluation_retention_adequate"] for row in rows])
        ),
        "filtered_selection_accuracy": float(np.mean(correct)),
        "unfiltered_selection_accuracy": float(
            np.mean([row["unfiltered_correct"] for row in rows])
        ),
        "finite_r_mass_U_fraction": len(finite_mass_upper) / len(rows),
        "mean_finite_r_mass_U": (
            float(np.mean(finite_mass_upper)) if finite_mass_upper else None
        ),
        "strong_separation_definition": "r_cov_L > r_mass_U under one simultaneous event",
        "strong_separation_count": int(np.sum(strong)),
        "correct_strong_separation_count": int(np.sum(strong & correct)),
        "radii": radius_summary,
    }


def resolve_budget(args: argparse.Namespace) -> RemoteBudget:
    values = asdict(REMOTE_BUDGETS[args.budget])
    for name in (
        "pair_count",
        "fit_per_class",
        "dev_per_class",
        "direction_bank_size",
        "search_proposals",
        "evaluation_proposals",
        "batch_size",
        "offset_quantiles",
        "min_search_retained",
        "min_evaluation_retained",
    ):
        override = getattr(args, name, None)
        if override is not None:
            values[name] = override
    budget = RemoteBudget(**values)
    integer_fields = (
        budget.fit_per_class,
        budget.dev_per_class,
        budget.search_proposals,
        budget.evaluation_proposals,
        budget.batch_size,
        budget.offset_quantiles,
        budget.min_search_retained,
        budget.min_evaluation_retained,
    )
    if (
        not 1 <= budget.pair_count <= len(CONFUSABLE_CLASS_PAIRS)
        or min(integer_fields) < 1
        or budget.direction_bank_size < MIN_DIRECTION_BANK_SIZE
        or budget.stable_direction_count != STABLE_DIRECTION_COUNT
        or budget.offset_quantiles < 2
        or budget.min_search_retained > budget.search_proposals
        or budget.min_evaluation_retained > budget.evaluation_proposals
    ):
        raise ValueError("invalid resolved remote budget")
    return budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", choices=tuple(REMOTE_BUDGETS), default="default")
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2_609_190_2)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    for name in (
        "pair_count",
        "fit_per_class",
        "dev_per_class",
        "direction_bank_size",
        "search_proposals",
        "evaluation_proposals",
        "batch_size",
        "offset_quantiles",
        "min_search_retained",
        "min_evaluation_retained",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    return parser


def _dataset_image(dataset: Any, index: int) -> np.ndarray:
    image, _ = dataset[index]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    values = np.asarray(image, dtype=np.float64)
    if values.shape != RAW_IMAGE_SHAPE or np.any(~np.isfinite(values)):
        raise RuntimeError(
            f"expected a finite raw CIFAR-10 tensor with shape {RAW_IMAGE_SHAPE}"
        )
    return values


def _fit_stable_directions(
    dataset: Any,
    split: dict[int, dict[str, list[int]]],
    pair: tuple[int, int],
    *,
    bank_size: int,
    retain: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], list[dict[str, Any]]]:
    fit_indices = split[pair[0]]["fit"] + split[pair[1]]["fit"]
    fit_images = np.stack([_dataset_image(dataset, index) for index in fit_indices])
    bank = rademacher_direction_bank(
        pair=pair, seed=seed, bank_size=bank_size, dimension=RAW_DIMENSION
    )
    stable_indices, ranking = rank_stable_directions(
        bank, fit_images, retain=retain
    )
    stable = bank[np.asarray(stable_indices)].reshape(
        retain, *RAW_IMAGE_SHAPE
    )
    return bank, stable, stable_indices, ranking


def _sample_proposals(
    *,
    model: Any,
    torch: Any,
    image: Any,
    directions: np.ndarray,
    pair: tuple[int, int],
    image_index: int,
    true_binary_label: int,
    proposals: int,
    batch_size: int,
    sigma: float,
    seed: int,
    stream: str,
    device: Any,
) -> MultiDirectionProposalBatch:
    """Label raw Gaussian proposals and project before any clipping."""

    direction_values = np.asarray(directions, dtype=np.float64)
    if direction_values.ndim != 4 or direction_values.shape[1:] != RAW_IMAGE_SHAPE:
        raise ValueError("directions must have shape (count,3,32,32)")
    generator = torch.Generator(device=device)
    generator.manual_seed(
        stable_seed("stable-band-v2", seed, pair[0], pair[1], image_index, stream)
    )
    image_device = image.unsqueeze(0).to(device)
    direction_device = torch.as_tensor(
        direction_values.reshape(direction_values.shape[0], -1),
        dtype=torch.float64,
        device=device,
    )
    labels: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    completed = 0
    with torch.inference_mode():
        while completed < proposals:
            size = min(batch_size, proposals - completed)
            noise = torch.randn(
                (size, *RAW_IMAGE_SHAPE),
                generator=generator,
                device=device,
                dtype=image_device.dtype,
            )
            raw = image_device.expand(size, -1, -1, -1) + sigma * noise
            # No clipping or nonlinear preprocessing before this projection.
            projected = raw.to(torch.float64).flatten(1) @ direction_device.T
            logits = model(raw)
            binary = (logits[:, pair[1]] > logits[:, pair[0]]).to(torch.int64)
            projections.append(projected.T.cpu().numpy().astype(np.float64))
            labels.append(binary.cpu().numpy().astype(np.int64))
            completed += size
    return MultiDirectionProposalBatch(
        image_index=image_index,
        true_binary_label=true_binary_label,
        task_labels=np.concatenate(labels),
        projections=np.concatenate(projections, axis=1),
    )


def _pair_base_payload(
    *,
    pair: tuple[int, int],
    bank: np.ndarray,
    stable: np.ndarray,
    stable_indices: Sequence[int],
    ranking: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "class_pair": list(pair),
        "class_names": [CLASS_NAMES[pair[0]], CLASS_NAMES[pair[1]]],
        "binary_task": (
            "label 0 iff the released ResNet left-class logit is at least the "
            "right-class logit; otherwise label 1"
        ),
        "direction_learning": {
            "generator": "SHA-256-expanded unit Rademacher raw-pixel directions",
            "ranking_data": "fit images only",
            "ranking_metric": "pooled population standard deviation of clean projections",
            "bank_size": int(bank.shape[0]),
            "bank_shape": list(bank.shape),
            "bank_float64_sha256": _array_sha256(bank),
            "retained_count": len(stable_indices),
            "retained_bank_indices_in_stability_order": list(stable_indices),
            "retained_bank_float64_sha256": _array_sha256(stable),
            "full_ranking": list(ranking),
        },
    }


def _evaluate_pair(
    *,
    dataset: Any,
    model: Any,
    torch: Any,
    device: Any,
    split: dict[int, dict[str, list[int]]],
    pair: tuple[int, int],
    budget: RemoteBudget,
    sigma: float,
    delta: float,
    seed: int,
) -> dict[str, Any]:
    bank, stable, stable_indices, ranking = _fit_stable_directions(
        dataset,
        split,
        pair,
        bank_size=budget.direction_bank_size,
        retain=budget.stable_direction_count,
        seed=seed,
    )
    base = _pair_base_payload(
        pair=pair,
        bank=bank,
        stable=stable,
        stable_indices=stable_indices,
        ranking=ranking,
    )
    dev_entries = [
        (index, binary_label)
        for binary_label, cifar_label in enumerate(pair)
        for index in split[cifar_label]["dev"]
    ]
    search_batches: list[MultiDirectionProposalBatch] = []
    images: dict[int, Any] = {}
    for image_index, true_binary_label in dev_entries:
        image, observed_label = dataset[image_index]
        if int(observed_label) != pair[true_binary_label]:
            raise RuntimeError("deterministic split label changed")
        images[image_index] = image
        search_batches.append(
            _sample_proposals(
                model=model,
                torch=torch,
                image=image,
                directions=stable,
                pair=pair,
                image_index=image_index,
                true_binary_label=true_binary_label,
                proposals=budget.search_proposals,
                batch_size=budget.batch_size,
                sigma=sigma,
                seed=seed,
                stream="search",
                device=device,
            )
        )

    offsets_by_slot = [
        candidate_offsets(
            np.asarray(
                [batch.projections[slot].mean() for batch in search_batches]
            ),
            budget.offset_quantiles,
        )
        for slot in range(budget.stable_direction_count)
    ]
    candidates = candidate_grid(
        stable_bank_indices=stable_indices,
        offsets_by_slot=offsets_by_slot,
        sigma=sigma,
        outer_fractions=budget.outer_fractions,
        inner_fractions=budget.inner_fractions,
    )
    selected, search_summary, grid_rows = select_candidate(
        candidates,
        search_batches,
        sigma=sigma,
        min_retained=budget.min_search_retained,
    )
    grid_payload = [asdict(candidate) for candidate in candidates]
    common = {
        **base,
        "search_constraints": {
            "minimum_adequate_image_fraction": MIN_ADEQUATE_IMAGE_FRACTION,
            "minimum_mean_retention": MIN_MEAN_RETENTION,
            "adequate_retained_per_image": budget.min_search_retained,
        },
        "offset_grid_by_direction_slot": [list(values) for values in offsets_by_slot],
        "candidate_grid_sha256": _canonical_digest(grid_payload),
        "candidate_count": len(candidates),
        "candidate_scores": grid_rows,
        "search_stream": {
            "name": "search",
            "proposals_per_image": budget.search_proposals,
            "seed_derivation": (
                "SHA-256(stable-band-v2, seed, pair, image_index, stream)"
            ),
        },
        "evaluation_stream": {
            "name": "evaluation",
            "proposals_per_image": budget.evaluation_proposals,
            "disjoint_from_search_by_stream_tag": True,
        },
    }
    if selected is None or search_summary is None:
        return {
            **common,
            "status": "failed_no_feasible_candidate",
            "failure_reason": (
                "no direction/band grid row met both the 90% adequate-image "
                "constraint and the 0.25 mean-retention constraint"
            ),
            "selected_filter": None,
            "search_summary": None,
            "covariance_proof": None,
            "evaluation_summary": None,
            "evaluation_rows": [],
        }

    _, filtered_labels = score_candidate(
        selected,
        search_batches,
        sigma=sigma,
        min_retained=budget.min_search_retained,
    )
    unfiltered_labels = {
        batch.image_index: _selected_label(_counts(batch.task_labels))
        for batch in search_batches
    }
    selected_direction = stable[selected.direction_slot]
    evaluation_rows: list[dict[str, Any]] = []
    for image_index, true_binary_label in dev_entries:
        batch = _sample_proposals(
            model=model,
            torch=torch,
            image=images[image_index],
            directions=selected_direction[np.newaxis, ...],
            pair=pair,
            image_index=image_index,
            true_binary_label=true_binary_label,
            proposals=budget.evaluation_proposals,
            batch_size=budget.batch_size,
            sigma=sigma,
            seed=seed,
            stream="evaluation",
            device=device,
        )
        evaluation_candidate = StableBandCandidate(
            direction_slot=0,
            direction_bank_index=selected.direction_bank_index,
            offset=selected.offset,
            inner=selected.inner,
            outer=selected.outer,
        )
        mask = bounded_two_band_mask(
            batch.projections[0], evaluation_candidate, sigma
        )
        retained_counts = _counts(batch.task_labels, mask)
        unfiltered_counts = _counts(batch.task_labels)
        filtered_label = filtered_labels[image_index]
        unfiltered_label = unfiltered_labels[image_index]
        radii = certified_radii_from_counts(
            filtered_selected_label=filtered_label,
            unfiltered_selected_label=unfiltered_label,
            retained_counts=retained_counts,
            unfiltered_counts=unfiltered_counts,
            proposal_count=budget.evaluation_proposals,
            sigma=sigma,
            delta=delta,
            min_retained=budget.min_evaluation_retained,
        )
        retained = sum(retained_counts)
        evaluation_rows.append(
            {
                "image_index": image_index,
                "true_binary_label": true_binary_label,
                "true_cifar10_label": pair[true_binary_label],
                "filtered_binary_label": filtered_label,
                "unfiltered_binary_label": unfiltered_label,
                "filtered_correct": filtered_label == true_binary_label,
                "unfiltered_correct": unfiltered_label == true_binary_label,
                "retained_counts": list(retained_counts),
                "unfiltered_counts": list(unfiltered_counts),
                "retained": retained,
                "proposals": budget.evaluation_proposals,
                "retention_rate": retained / budget.evaluation_proposals,
                "evaluation_retention_adequate": (
                    retained >= budget.min_evaluation_retained
                ),
                **radii,
            }
        )

    return {
        **common,
        "status": "development_selection_and_evaluation_complete",
        "selected_filter": {
            "direction_slot_in_stable_bank": selected.direction_slot,
            "direction_bank_index": selected.direction_bank_index,
            "direction": selected_direction.reshape(-1).tolist(),
            "direction_float64_sha256": _array_sha256(selected_direction),
            "offset_b": selected.offset,
            "alpha": selected.inner,
            "beta": selected.outer,
        },
        "search_summary": search_summary,
        "covariance_proof": covariance_proof(
            selected_direction, selected, sigma
        ),
        "evaluation_summary": summarize_evaluation(evaluation_rows),
        "evaluation_rows": evaluation_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    budget = resolve_budget(args)
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(args.delta) or not 0.0 < args.delta < 1.0:
        raise ValueError("delta must lie in (0,1)")

    repo_root = args.repo_root.resolve()
    code_root = repo_root / "ImageClassify_Gaussian/ImageClassify_Conf/code"
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else repo_root
        / "ImageClassify_Gaussian/ImageClassify_Conf/models/cifar10/resnet110/"
        "noise_0.25/checkpoint.pth.tar"
    )
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else code_root / "dataset_cache"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    repository_commit = _git_head(repo_root)
    if repository_commit != EXPECTED_AUDITVOTES_COMMIT:
        raise RuntimeError(
            "AuditVotes repository is not at the pinned released commit: "
            f"{repository_commit}"
        )
    checkpoint_digest = _file_sha256(checkpoint)
    if checkpoint_digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint does not match the released ResNet-110 artifact")
    sys.path.insert(0, str(code_root))

    # Heavy dependencies stay lazy so pure geometry/statistics tests run on CPU.
    import torch
    import torchvision
    from torchvision import transforms

    if not torch.cuda.is_available():
        raise RuntimeError("the released AuditVotes architecture requires CUDA")
    from architectures import get_architecture  # type: ignore

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("arch") != "cifar_resnet110":
        raise RuntimeError("expected the released CIFAR-10 ResNet-110 architecture")
    model = get_architecture(payload["arch"], "cifar10")
    model.load_state_dict(payload["state_dict"])
    model.eval()

    # The only dataset constructor in this runner is the 50,000-image training split.
    dataset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=True,
        download=args.download,
        transform=transforms.ToTensor(),
    )
    if len(dataset) != 50_000 or len(dataset.targets) != 50_000:
        raise RuntimeError("expected the 50,000-image CIFAR-10 training split")
    pairs = CONFUSABLE_CLASS_PAIRS[: budget.pair_count]
    used_classes = sorted({label for pair in pairs for label in pair})
    split = deterministic_class_split(
        dataset.targets,
        fit_per_class=budget.fit_per_class,
        dev_per_class=budget.dev_per_class,
        seed=args.seed,
        classes=used_classes,
    )
    split_payload = {str(label): split[label] for label in used_classes}
    started = time.monotonic()
    pair_results = [
        _evaluate_pair(
            dataset=dataset,
            model=model,
            torch=torch,
            device=device,
            split=split,
            pair=pair,
            budget=budget,
            sigma=args.sigma,
            delta=args.delta,
            seed=args.seed,
        )
        for pair in pairs
    ]
    failed_pairs = [
        row["class_pair"]
        for row in pair_results
        if row["status"] == "failed_no_feasible_candidate"
    ]
    status = (
        "open_development_complete"
        if not failed_pairs
        else "open_development_failed_feasibility"
    )
    result = {
        "status": status,
        "paper_eligibility": PAPER_ELIGIBILITY,
        "study_role": "learned stable-direction bounded-band development only",
        "dataset": "CIFAR-10 training split only",
        "test_sets_accessed": [],
        "raw_proposals_before_clipping": True,
        "clipping_performed": False,
        "failed_feasibility_pairs": failed_pairs,
        "budget_name": args.budget,
        "budget": asdict(budget),
        "sigma": args.sigma,
        "per_image_delta": args.delta,
        "per_image_bonferroni": {
            "one_sided_tail_events": BONFERRONI_TAIL_EVENTS,
            "tail_error": args.delta / BONFERRONI_TAIL_EVENTS,
            "total_allocated_error": args.delta,
            "events": [
                "conditional selected-probability lower",
                "joint selected-mass lower",
                "joint competitor-mass upper",
                "joint selected-mass upper",
                "joint competitor-mass lower",
                "unfiltered selected-probability lower",
            ],
        },
        "confidence_scope": (
            "r_cov_L, r_mass_L, r_mass_U, and r_unfiltered_L are simultaneous "
            "within each fixed dev image; there is no familywise or population "
            "claim across development images or pairs"
        ),
        "seed": args.seed,
        "fixed_confusable_class_pairs": [list(pair) for pair in pairs],
        "split_strategy": (
            "within-class SHA-256 rank; fit prefix then nonoverlapping dev suffix"
        ),
        "split_sha256": _canonical_digest(split_payload),
        "split": split_payload,
        "assets": {
            "auditvotes_git_commit": repository_commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "core_py_sha256": _file_sha256(code_root / "core.py"),
            "architectures_py_sha256": _file_sha256(code_root / "architectures.py"),
            "datasets_py_sha256": _file_sha256(code_root / "datasets.py"),
            "cifar_resnet_py_sha256": _file_sha256(
                code_root / "archs/cifar_resnet.py"
            ),
            "source_sha256": _file_sha256(Path(__file__)),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "elapsed_seconds": time.monotonic() - started,
        "pairs": pair_results,
        "limitations": [
            "Every filter choice and every reported image belongs to development data.",
            "The result is ineligible for the paper without a separately frozen confirmation.",
            "The binary task compares only the two fixed released-ResNet logits per pair.",
            "Per-image confidence is not simultaneous across the development collection.",
            "SciPy evaluates exact-real Clopper--Pearson formulas numerically; endpoints are not interval-certified.",
            "The covariance proof applies to the fixed raw-proposal filter, not clipping or an input-dependent filter.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "status": status,
        "paper_eligibility": PAPER_ELIGIBILITY,
        "budget": args.budget,
        "elapsed_seconds": result["elapsed_seconds"],
        "failed_feasibility_pairs": failed_pairs,
        "pairs": [
            {
                "class_pair": row["class_pair"],
                "status": row["status"],
                "selected_filter": row["selected_filter"],
                "search_summary": row["search_summary"],
                "evaluation_summary": row["evaluation_summary"],
            }
            for row in pair_results
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
