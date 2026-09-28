"""Develop one proof-carrying global CIFAR-10 bounded two-band filter.

This study uses only the 50,000-image CIFAR-10 training split.  A fit cohort
ranks a SHA-256-seeded bank of unit Rademacher raw-pixel directions by clean
projection dispersion.  A nonoverlapping search cohort selects one direction
and one fixed filter

    alpha <= |<u, z> + b| <= beta,       0 < alpha < beta <= sigma.

The filter is applied to the raw, unclipped Gaussian proposal.  Gaussian
factorization and Popoviciu's inequality give ``Cov(Z | retained) <=
sigma**2 I`` for every center.  A third, nonoverlapping cohort evaluates the
fixed filter.  Within each evaluation image, an independent label-selection
stream fixes the predicted class and runner before a fresh estimation stream.

The evaluation reports multiclass conditional, joint-mass, and unfiltered
certificates under one explicit per-image Bonferroni family.  The result is an
open development study and is never eligible for the paper without a later,
separately frozen confirmation.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.iclr2027_learned_stable_band_filter_development import (  # noqa: E402
    StableBandCandidate,
    bounded_two_band_mask,
    candidate_grid,
    candidate_offsets,
    covariance_proof,
)


DEFAULT_OUTPUT = Path(
    "outputs/iclr2027_global_stable_band_filter_development.json"
)
PAPER_ELIGIBILITY = "none"
EXPECTED_AUDITVOTES_COMMIT = "52b6a0db53947c815884ad62927d6af6dc584705"
EXPECTED_CHECKPOINT_SHA256 = (
    "420333fe0380cc437218c9b67c20bacf687932957da47340f900d4af5e05bd7c"
)
RAW_IMAGE_SHAPE = (3, 32, 32)
RAW_DIMENSION = int(np.prod(RAW_IMAGE_SHAPE))
CLASS_COUNT = 10
MIN_DIRECTION_BANK_SIZE = 128
STABLE_DIRECTION_COUNT = 12
MIN_ADEQUATE_IMAGE_FRACTION = 0.90
MIN_MEAN_RETENTION = 0.25
DEVELOPMENT_TARGET_RADIUS = 0.20
REPORT_RADII = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
MAX_DEVELOPMENT_MEAN_RETENTION = 0.60
MAX_DEVELOPMENT_ACCURACY_DROP = 0.01
MIN_DEVELOPMENT_COVARIANCE_ADVANTAGE = 0.05
MIN_STRONG_MARGIN = 0.01
MIN_STRONG_MARGIN_FRACTION = 0.05

# Conditional selected lower and nine competitor uppers, joint selected lower
# and nine competitor uppers, joint selected upper and selected runner lower,
# then unfiltered selected lower and nine competitor uppers.
BONFERRONI_TAIL_EVENTS = 32

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


@dataclass(frozen=True)
class DevelopmentBudget:
    fit_per_class: int
    search_per_class: int
    evaluation_per_class: int
    direction_bank_size: int
    stable_direction_count: int
    search_proposals: int
    label_selection_proposals: int
    evaluation_proposals: int
    batch_size: int
    offset_quantiles: int
    outer_fractions: tuple[float, ...]
    inner_fractions: tuple[float, ...]
    min_search_retained: int
    min_label_selection_retained: int
    min_evaluation_retained: int


DEVELOPMENT_BUDGETS: dict[str, DevelopmentBudget] = {
    "smoke": DevelopmentBudget(
        fit_per_class=8,
        search_per_class=1,
        evaluation_per_class=1,
        direction_bank_size=128,
        stable_direction_count=12,
        search_proposals=64,
        label_selection_proposals=64,
        evaluation_proposals=128,
        batch_size=128,
        offset_quantiles=3,
        outer_fractions=(0.75, 1.0),
        inner_fractions=(0.025, 0.10),
        min_search_retained=8,
        min_label_selection_retained=8,
        min_evaluation_retained=8,
    ),
    "default": DevelopmentBudget(
        fit_per_class=300,
        search_per_class=20,
        evaluation_per_class=20,
        direction_bank_size=128,
        stable_direction_count=12,
        search_proposals=512,
        label_selection_proposals=512,
        evaluation_proposals=4096,
        batch_size=1024,
        offset_quantiles=7,
        outer_fractions=(0.50, 0.75, 1.0),
        inner_fractions=(0.025, 0.10, 0.25),
        min_search_retained=64,
        min_label_selection_retained=64,
        min_evaluation_retained=64,
    ),
}


@dataclass(frozen=True)
class GlobalProposalBatch:
    image_index: int
    true_label: int
    labels: np.ndarray
    # Shape is (number of retained stable directions, proposal count).
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


def deterministic_three_way_split(
    labels: Sequence[int],
    *,
    fit_per_class: int,
    search_per_class: int,
    evaluation_per_class: int,
    seed: int,
    classes: Iterable[int] = range(CLASS_COUNT),
) -> dict[int, dict[str, list[int]]]:
    """Hash-rank disjoint fit, filter-search, and evaluation cohorts."""

    sizes = (fit_per_class, search_per_class, evaluation_per_class)
    if min(sizes) < 1:
        raise ValueError("all per-class split sizes must be positive")
    class_tuple = tuple(int(value) for value in classes)
    if len(class_tuple) != len(set(class_tuple)):
        raise ValueError("classes must be unique")
    if any(label < 0 or label >= CLASS_COUNT for label in class_tuple):
        raise ValueError("class index is outside CIFAR-10")
    buckets: dict[int, list[tuple[bytes, int]]] = {
        label: [] for label in class_tuple
    }
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label in buckets:
            rank = sha256(
                f"global-stable-band-v1:{seed}:{label}:{index}".encode()
            ).digest()
            buckets[label].append((rank, index))
    result: dict[int, dict[str, list[int]]] = {}
    required = sum(sizes)
    for label in class_tuple:
        ordered = [index for _, index in sorted(buckets[label])]
        if len(ordered) < required:
            raise ValueError(
                f"class {label} has {len(ordered)} examples; need {required}"
            )
        fit_end = fit_per_class
        search_end = fit_end + search_per_class
        result[label] = {
            "fit": ordered[:fit_end],
            "filter_search": ordered[fit_end:search_end],
            "evaluation": ordered[search_end:required],
        }
    return result


def global_rademacher_direction_bank(
    *,
    seed: int,
    bank_size: int = MIN_DIRECTION_BANK_SIZE,
    dimension: int = RAW_DIMENSION,
) -> np.ndarray:
    """Create a version-stable global bank of unit sign directions."""

    if bank_size < MIN_DIRECTION_BANK_SIZE:
        raise ValueError(f"direction bank must contain at least {MIN_DIRECTION_BANK_SIZE}")
    if dimension < 1:
        raise ValueError("dimension must be positive")
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
                        f"global-stable-band-v1:rademacher:{seed}:"
                        f"{direction_index}:{block_index}"
                    ).encode()
                ).digest()
            )
            block_index += 1
        bits = np.unpackbits(
            np.frombuffer(bytes(generated[:byte_count]), dtype=np.uint8)
        )[:dimension]
        directions[direction_index] = (
            2.0 * bits.astype(np.float64) - 1.0
        ) * scale
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
        raise ValueError("bank must be a matrix and fit images must be batched")
    flattened = images.reshape(images.shape[0], -1)
    if flattened.shape[0] < 2 or flattened.shape[1] != directions.shape[1]:
        raise ValueError("fit image dimension does not match direction bank")
    if np.any(~np.isfinite(directions)) or np.any(~np.isfinite(flattened)):
        raise ValueError("directions and fit images must be finite")
    if not 1 <= retain <= directions.shape[0]:
        raise ValueError("invalid stable-direction retain count")
    if not np.allclose(
        np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
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


def _counts(labels: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("labels must be a vector")
    if mask is not None:
        boolean_mask = np.asarray(mask, dtype=bool)
        if boolean_mask.shape != values.shape:
            raise ValueError("mask and labels must have the same shape")
        values = values[boolean_mask]
    if np.any((values < 0) | (values >= CLASS_COUNT)):
        raise ValueError("labels must be CIFAR-10 class indices")
    return np.bincount(values, minlength=CLASS_COUNT).astype(np.int64)


def selected_and_runner(counts: Sequence[int]) -> tuple[int, int]:
    """Return the top two classes with the lower-index deterministic tie rule."""

    values = tuple(int(value) for value in counts)
    if len(values) != CLASS_COUNT or any(value < 0 for value in values):
        raise ValueError("counts must contain ten nonnegative entries")
    order = sorted(range(CLASS_COUNT), key=lambda label: (-values[label], label))
    return order[0], order[1]


def multiclass_tie_kl(selected_probability: float, competitor_probability: float) -> float:
    """Least categorical KL required to tie one selected/competitor pair."""

    p_a = float(selected_probability)
    p_b = float(competitor_probability)
    if (
        not math.isfinite(p_a)
        or not math.isfinite(p_b)
        or p_a <= p_b
        or p_a <= 0.0
        or p_a > 1.0
        or p_b < 0.0
        or p_b > 1.0
    ):
        return 0.0
    pair_mass = p_a + p_b
    first = p_a * math.log(2.0 * p_a / pair_mass)
    second = 0.0 if p_b == 0.0 else p_b * math.log(2.0 * p_b / pair_mass)
    return max(0.0, first + second)


def empirical_conditional_radius(counts: Sequence[int], sigma: float) -> float:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape != (CLASS_COUNT,) or np.any(values < 0):
        raise ValueError("counts must contain ten nonnegative entries")
    total = int(np.sum(values))
    if total < 1 or not math.isfinite(sigma) or sigma <= 0.0:
        return 0.0
    selected, runner = selected_and_runner(values)
    divergence = multiclass_tie_kl(
        values[selected] / total, values[runner] / total
    )
    return float(sigma * math.sqrt(2.0 * divergence))


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
    return float(beta_distribution.ppf(1.0 - error, successes + 1, trials - successes))


def _gaussian_radius(lower_selected: float, upper_competitor: float, sigma: float) -> float:
    if lower_selected <= upper_competitor:
        return 0.0
    return float(
        0.5
        * sigma
        * (norm.ppf(lower_selected) - norm.ppf(upper_competitor))
    )


def multiclass_certified_radii_from_counts(
    *,
    filtered_selected_label: int,
    filtered_runner_label: int,
    unfiltered_selected_label: int,
    retained_counts: Sequence[int],
    unfiltered_counts: Sequence[int],
    proposal_count: int,
    sigma: float,
    delta: float,
    min_retained: int = 1,
) -> dict[str, Any]:
    """Compute simultaneous multiclass radii under one 32-tail family."""

    retained = np.asarray(retained_counts, dtype=np.int64)
    unfiltered = np.asarray(unfiltered_counts, dtype=np.int64)
    labels = (filtered_selected_label, filtered_runner_label, unfiltered_selected_label)
    retained_total = int(np.sum(retained))
    if (
        retained.shape != (CLASS_COUNT,)
        or unfiltered.shape != (CLASS_COUNT,)
        or np.any(retained < 0)
        or np.any(unfiltered < 0)
        or retained_total > proposal_count
        or int(np.sum(unfiltered)) != proposal_count
        or any(label < 0 or label >= CLASS_COUNT for label in labels)
        or filtered_selected_label == filtered_runner_label
        or proposal_count < 1
        or min_retained < 1
        or not math.isfinite(sigma)
        or sigma <= 0.0
        or not 0.0 < delta < 1.0
    ):
        raise ValueError("invalid multiclass counts, labels, confidence, or scale")

    tail_error = delta / BONFERRONI_TAIL_EVENTS
    conditional_selected_lower = 0.0
    conditional_competitor_uppers = [1.0] * CLASS_COUNT
    r_cov_lower = 0.0
    if retained_total >= min_retained:
        conditional_selected_lower = one_sided_binomial_lower(
            int(retained[filtered_selected_label]), retained_total, tail_error
        )
        conditional_competitor_uppers = [
            (
                0.0
                if label == filtered_selected_label
                else one_sided_binomial_upper(
                    int(retained[label]), retained_total, tail_error
                )
            )
            for label in range(CLASS_COUNT)
        ]
        largest_conditional_competitor = max(conditional_competitor_uppers)
        divergence = multiclass_tie_kl(
            conditional_selected_lower, largest_conditional_competitor
        )
        r_cov_lower = float(sigma * math.sqrt(2.0 * divergence))
    else:
        largest_conditional_competitor = 1.0

    joint_selected_lower = one_sided_binomial_lower(
        int(retained[filtered_selected_label]), proposal_count, tail_error
    )
    joint_competitor_uppers = [
        (
            0.0
            if label == filtered_selected_label
            else one_sided_binomial_upper(
                int(retained[label]), proposal_count, tail_error
            )
        )
        for label in range(CLASS_COUNT)
    ]
    largest_joint_competitor = max(joint_competitor_uppers)
    r_mass_lower = _gaussian_radius(
        joint_selected_lower, largest_joint_competitor, sigma
    )

    joint_selected_upper = one_sided_binomial_upper(
        int(retained[filtered_selected_label]), proposal_count, tail_error
    )
    joint_runner_lower = one_sided_binomial_lower(
        int(retained[filtered_runner_label]), proposal_count, tail_error
    )
    mass_upper_is_finite = not (
        joint_selected_upper >= 1.0 or joint_runner_lower <= 0.0
    )
    r_mass_upper: float | None
    if not mass_upper_is_finite:
        r_mass_upper = None
    else:
        r_mass_upper = max(
            0.0,
            float(
                0.5
                * sigma
                * (norm.ppf(joint_selected_upper) - norm.ppf(joint_runner_lower))
            ),
        )

    unfiltered_selected_lower = one_sided_binomial_lower(
        int(unfiltered[unfiltered_selected_label]), proposal_count, tail_error
    )
    unfiltered_competitor_uppers = [
        (
            0.0
            if label == unfiltered_selected_label
            else one_sided_binomial_upper(
                int(unfiltered[label]), proposal_count, tail_error
            )
        )
        for label in range(CLASS_COUNT)
    ]
    largest_unfiltered_competitor = max(unfiltered_competitor_uppers)
    r_unfiltered_lower = _gaussian_radius(
        unfiltered_selected_lower, largest_unfiltered_competitor, sigma
    )
    return {
        "bonferroni_tail_events": BONFERRONI_TAIL_EVENTS,
        "bonferroni_tail_error": tail_error,
        "bonferroni_total_error": tail_error * BONFERRONI_TAIL_EVENTS,
        "conditional_selected_probability_lower": conditional_selected_lower,
        "conditional_competitor_probability_uppers": conditional_competitor_uppers,
        "conditional_largest_competitor_upper": largest_conditional_competitor,
        "joint_selected_mass_lower": joint_selected_lower,
        "joint_competitor_mass_uppers": joint_competitor_uppers,
        "joint_largest_competitor_upper": largest_joint_competitor,
        "joint_selected_mass_upper": joint_selected_upper,
        "joint_selected_runner_mass_lower": joint_runner_lower,
        "unfiltered_selected_probability_lower": unfiltered_selected_lower,
        "unfiltered_competitor_probability_uppers": unfiltered_competitor_uppers,
        "unfiltered_largest_competitor_upper": largest_unfiltered_competitor,
        "r_cov_L": r_cov_lower,
        "r_mass_L": r_mass_lower,
        "r_mass_U": r_mass_upper,
        "r_mass_U_is_finite": mass_upper_is_finite,
        "r_unfiltered_L": r_unfiltered_lower,
        "strong_separation": bool(
            r_mass_upper is not None and r_cov_lower > r_mass_upper
        ),
    }


def score_candidate(
    candidate: StableBandCandidate,
    batches: Sequence[GlobalProposalBatch],
    *,
    sigma: float,
    min_retained: int,
    target_radius: float = DEVELOPMENT_TARGET_RADIUS,
    min_adequate_fraction: float = MIN_ADEQUATE_IMAGE_FRACTION,
    min_mean_retention: float = MIN_MEAN_RETENTION,
) -> dict[str, Any]:
    """Score a fixed candidate on the filter-search cohort."""

    if not batches or min_retained < 1:
        raise ValueError("search batches and retained requirement must be nonempty")
    if not 0.0 < min_adequate_fraction <= 1.0:
        raise ValueError("invalid adequate-image fraction")
    if not 0.0 < min_mean_retention <= 1.0:
        raise ValueError("invalid mean retention")
    if not math.isfinite(target_radius) or target_radius < 0.0:
        raise ValueError("target radius must be finite and nonnegative")

    correct_at_target: list[bool] = []
    correct_radii: list[float] = []
    retention_rates: list[float] = []
    adequate: list[bool] = []
    correct: list[bool] = []
    for batch in batches:
        projections = np.asarray(batch.projections, dtype=np.float64)
        labels = np.asarray(batch.labels, dtype=np.int64)
        if (
            projections.ndim != 2
            or candidate.direction_slot >= projections.shape[0]
            or projections.shape[1] != labels.size
            or labels.size < 1
        ):
            raise ValueError("search labels and projections are not aligned")
        mask = bounded_two_band_mask(
            projections[candidate.direction_slot], candidate, sigma
        )
        counts = _counts(labels, mask)
        retained = int(np.sum(counts))
        selected, _ = selected_and_runner(counts)
        is_adequate = retained >= min_retained
        is_correct = is_adequate and selected == batch.true_label
        radius = empirical_conditional_radius(counts, sigma) if is_correct else 0.0
        adequate.append(is_adequate)
        correct.append(is_correct)
        correct_radii.append(radius)
        correct_at_target.append(is_correct and radius >= target_radius)
        retention_rates.append(retained / labels.size)

    adequate_fraction = float(np.mean(adequate))
    mean_retention = float(np.mean(retention_rates))
    feasible = bool(
        adequate_fraction >= min_adequate_fraction
        and mean_retention >= min_mean_retention
    )
    return {
        "feasible": feasible,
        "required_adequate_image_fraction": min_adequate_fraction,
        "required_mean_retention": min_mean_retention,
        "adequate_retained_per_image": min_retained,
        "adequate_image_fraction": adequate_fraction,
        "mean_retention": mean_retention,
        "min_retention": float(np.min(retention_rates)),
        "max_retention": float(np.max(retention_rates)),
        "correct_adequate_fraction": float(np.mean(correct)),
        "objective_radius": target_radius,
        "objective_correct_empirical_certificate_accuracy": float(
            np.mean(correct_at_target)
        ),
        "objective_mean_correct_empirical_covariance_radius": float(
            np.mean(correct_radii)
        ),
        "selection_objective": (
            "lexicographically maximize correct empirical conditional-"
            "certificate accuracy at radius 0.20, then mean correct empirical "
            "conditional radius, then mean retention"
        ),
    }


def select_candidate(
    candidates: Sequence[StableBandCandidate],
    batches: Sequence[GlobalProposalBatch],
    *,
    sigma: float,
    min_retained: int,
) -> tuple[StableBandCandidate | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Select one feasible global candidate using the frozen objective."""

    if not candidates:
        raise ValueError("candidate grid is empty")
    rows: list[dict[str, Any]] = []
    scored: list[tuple[tuple[float, ...], StableBandCandidate, dict[str, Any]]] = []
    for candidate in candidates:
        summary = score_candidate(
            candidate, batches, sigma=sigma, min_retained=min_retained
        )
        rows.append({**asdict(candidate), **summary})
        if summary["feasible"]:
            key = (
                summary["objective_correct_empirical_certificate_accuracy"],
                summary["objective_mean_correct_empirical_covariance_radius"],
                summary["mean_retention"],
                summary["correct_adequate_fraction"],
                summary["adequate_image_fraction"],
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
        ("conditional_covariance_lower", "r_cov_L", "filtered_correct"),
        ("joint_mass_lower", "r_mass_L", "filtered_correct"),
        ("unfiltered_lower", "r_unfiltered_L", "unfiltered_correct"),
    )
    radius_summary: dict[str, Any] = {}
    for name, radius_key, correct_key in radius_specs:
        values = np.asarray([row[radius_key] for row in rows], dtype=float)
        correct = np.asarray([row[correct_key] for row in rows], dtype=bool)
        radius_summary[name] = {
            "positive_fraction": float(np.mean(values > 0.0)),
            "mean_radius": float(np.mean(values)),
            "median_radius": float(np.median(values)),
            "max_radius": float(np.max(values)),
            "correct_certified_accuracy": {
                f"{radius:.2f}": float(np.mean(correct & (values >= radius)))
                for radius in REPORT_RADII
            },
        }
    retention = np.asarray([row["retention_rate"] for row in rows], dtype=float)
    selection_retention = np.asarray(
        [row["label_selection_retention_rate"] for row in rows], dtype=float
    )
    adequate = np.asarray(
        [row["evaluation_retention_adequate"] for row in rows], dtype=bool
    )
    selection_adequate = np.asarray(
        [row["label_selection_retention_adequate"] for row in rows], dtype=bool
    )
    filtered_correct = np.asarray(
        [row["filtered_correct"] for row in rows], dtype=bool
    )
    strong = np.asarray([row["strong_separation"] for row in rows], dtype=bool)
    strong_margin = np.asarray(
        [
            (
                row["r_cov_L"] - row["r_mass_U"]
                if row["r_mass_U"] is not None
                else -math.inf
            )
            for row in rows
        ],
        dtype=float,
    )
    finite_mass_upper = [
        float(row["r_mass_U"]) for row in rows if row["r_mass_U"] is not None
    ]
    return {
        "images": len(rows),
        "mean_retention": float(np.mean(retention)),
        "min_retention": float(np.min(retention)),
        "max_retention": float(np.max(retention)),
        "model_evaluation_fraction": float(np.mean(retention)),
        "model_evaluation_fraction_interpretation": (
            "fraction of a fixed raw-proposal budget requiring model evaluation "
            "when the fixed filter is applied before the classifier; this is "
            "not a saving under a fixed accepted-sample budget"
        ),
        "adequate_evaluation_fraction": float(np.mean(adequate)),
        "mean_label_selection_retention": float(np.mean(selection_retention)),
        "adequate_label_selection_fraction": float(np.mean(selection_adequate)),
        "filtered_selection_accuracy": float(np.mean(filtered_correct)),
        "unfiltered_selection_accuracy": float(
            np.mean([row["unfiltered_correct"] for row in rows])
        ),
        "finite_r_mass_U_fraction": len(finite_mass_upper) / len(rows),
        "mean_finite_r_mass_U": (
            float(np.mean(finite_mass_upper)) if finite_mass_upper else None
        ),
        "strong_separation_definition": (
            "r_cov_L > r_mass_U under one simultaneous per-image event"
        ),
        "strong_separation_count": int(np.sum(strong)),
        "correct_strong_separation_count": int(
            np.sum(strong & filtered_correct)
        ),
        "correct_strong_margin_threshold": MIN_STRONG_MARGIN,
        "correct_strong_margin_count": int(
            np.sum(filtered_correct & (strong_margin >= MIN_STRONG_MARGIN))
        ),
        "correct_strong_margin_fraction": float(
            np.mean(filtered_correct & (strong_margin >= MIN_STRONG_MARGIN))
        ),
        "radii": radius_summary,
    }


def evaluate_development_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen go or no-go rule on the disjoint evaluation cohort."""

    covariance_at_target = summary["radii"]["conditional_covariance_lower"][
        "correct_certified_accuracy"
    ][f"{DEVELOPMENT_TARGET_RADIUS:.2f}"]
    joint_at_target = summary["radii"]["joint_mass_lower"][
        "correct_certified_accuracy"
    ][f"{DEVELOPMENT_TARGET_RADIUS:.2f}"]
    checks = {
        "evaluation_retention_adequate": (
            summary["adequate_evaluation_fraction"]
            >= MIN_ADEQUATE_IMAGE_FRACTION
        ),
        "selection_retention_adequate": (
            summary["adequate_label_selection_fraction"]
            >= MIN_ADEQUATE_IMAGE_FRACTION
        ),
        "mean_retention_at_least_minimum": (
            summary["mean_retention"] >= MIN_MEAN_RETENTION
        ),
        "mean_label_selection_retention_at_least_minimum": (
            summary["mean_label_selection_retention"] >= MIN_MEAN_RETENTION
        ),
        "mean_retention_at_most_maximum": (
            summary["mean_retention"] <= MAX_DEVELOPMENT_MEAN_RETENTION
        ),
        "filtered_accuracy_within_one_point_of_unfiltered": (
            summary["filtered_selection_accuracy"]
            >= summary["unfiltered_selection_accuracy"]
            - MAX_DEVELOPMENT_ACCURACY_DROP
        ),
        "covariance_accuracy_exceeds_joint_by_five_points_at_radius_0_20": (
            covariance_at_target
            >= joint_at_target + MIN_DEVELOPMENT_COVARIANCE_ADVANTAGE
        ),
        "five_percent_have_correct_strong_margin_at_least_0_01": (
            summary["correct_strong_margin_fraction"]
            >= MIN_STRONG_MARGIN_FRACTION
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "target_radius": DEVELOPMENT_TARGET_RADIUS,
            "minimum_adequate_fraction": MIN_ADEQUATE_IMAGE_FRACTION,
            "minimum_mean_retention": MIN_MEAN_RETENTION,
            "maximum_mean_retention": MAX_DEVELOPMENT_MEAN_RETENTION,
            "maximum_filtered_accuracy_drop": MAX_DEVELOPMENT_ACCURACY_DROP,
            "minimum_covariance_over_joint_accuracy": (
                MIN_DEVELOPMENT_COVARIANCE_ADVANTAGE
            ),
            "minimum_correct_strong_margin": MIN_STRONG_MARGIN,
            "minimum_correct_strong_margin_fraction": (
                MIN_STRONG_MARGIN_FRACTION
            ),
        },
        "observed": {
            "mean_retention": summary["mean_retention"],
            "filtered_selection_accuracy": summary[
                "filtered_selection_accuracy"
            ],
            "unfiltered_selection_accuracy": summary[
                "unfiltered_selection_accuracy"
            ],
            "covariance_correct_certified_accuracy_at_radius_0_20": (
                covariance_at_target
            ),
            "joint_correct_certified_accuracy_at_radius_0_20": joint_at_target,
            "correct_strong_margin_fraction": summary[
                "correct_strong_margin_fraction"
            ],
        },
    }


def resolve_budget(args: argparse.Namespace) -> DevelopmentBudget:
    values = asdict(DEVELOPMENT_BUDGETS[args.budget])
    integer_names = (
        "fit_per_class",
        "search_per_class",
        "evaluation_per_class",
        "direction_bank_size",
        "search_proposals",
        "label_selection_proposals",
        "evaluation_proposals",
        "batch_size",
        "offset_quantiles",
        "min_search_retained",
        "min_label_selection_retained",
        "min_evaluation_retained",
    )
    for name in integer_names:
        override = getattr(args, name, None)
        if override is not None:
            values[name] = override
    budget = DevelopmentBudget(**values)
    checked = tuple(getattr(budget, name) for name in integer_names)
    if (
        min(checked) < 1
        or budget.direction_bank_size < MIN_DIRECTION_BANK_SIZE
        or budget.stable_direction_count != STABLE_DIRECTION_COUNT
        or budget.offset_quantiles < 2
        or budget.min_search_retained > budget.search_proposals
        or budget.min_label_selection_retained > budget.label_selection_proposals
        or budget.min_evaluation_retained > budget.evaluation_proposals
    ):
        raise ValueError("invalid resolved development budget")
    return budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--budget", choices=tuple(DEVELOPMENT_BUDGETS), default="default"
    )
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2_609_200_1)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    for name in (
        "fit_per_class",
        "search_per_class",
        "evaluation_per_class",
        "direction_bank_size",
        "search_proposals",
        "label_selection_proposals",
        "evaluation_proposals",
        "batch_size",
        "offset_quantiles",
        "min_search_retained",
        "min_label_selection_retained",
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


def _sample_proposals(
    *,
    model: Any,
    torch: Any,
    image: Any,
    directions: np.ndarray,
    image_index: int,
    true_label: int,
    proposals: int,
    batch_size: int,
    sigma: float,
    seed: int,
    stream: str,
    device: Any,
) -> GlobalProposalBatch:
    """Classify and project raw Gaussian proposals before any clipping."""

    direction_values = np.asarray(directions, dtype=np.float64)
    if direction_values.ndim != 4 or direction_values.shape[1:] != RAW_IMAGE_SHAPE:
        raise ValueError("directions must have shape (count,3,32,32)")
    generator = torch.Generator(device=device)
    generator.manual_seed(
        stable_seed("global-stable-band-v1", seed, image_index, stream)
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
            projected = raw.to(torch.float64).flatten(1) @ direction_device.T
            predicted = torch.argmax(model(raw), dim=1)
            projections.append(projected.T.cpu().numpy().astype(np.float64))
            labels.append(predicted.cpu().numpy().astype(np.int64))
            completed += size
    return GlobalProposalBatch(
        image_index=image_index,
        true_label=true_label,
        labels=np.concatenate(labels),
        projections=np.concatenate(projections, axis=1),
    )


def _cohort_entries(
    split: dict[int, dict[str, list[int]]], cohort: str
) -> list[tuple[int, int]]:
    return [
        (index, label)
        for label in range(CLASS_COUNT)
        for index in split[label][cohort]
    ]


def _top_score_rows(rows: Sequence[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    feasible = [row for row in rows if row["feasible"]]
    return sorted(
        feasible,
        key=lambda row: (
            row["objective_correct_empirical_certificate_accuracy"],
            row["objective_mean_correct_empirical_covariance_radius"],
            row["mean_retention"],
        ),
        reverse=True,
    )[:limit]


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    budget = resolve_budget(args)
    if not math.isfinite(args.sigma) or args.sigma <= 0.0:
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

    # Heavy dependencies remain lazy so all mathematical tests are CPU-only.
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

    # This is the only dataset constructor.  No test or CIFAR-10.2 object exists.
    dataset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=True,
        download=args.download,
        transform=transforms.ToTensor(),
    )
    if len(dataset) != 50_000 or len(dataset.targets) != 50_000:
        raise RuntimeError("expected the 50,000-image CIFAR-10 training split")
    split = deterministic_three_way_split(
        dataset.targets,
        fit_per_class=budget.fit_per_class,
        search_per_class=budget.search_per_class,
        evaluation_per_class=budget.evaluation_per_class,
        seed=args.seed,
    )
    split_payload = {str(label): split[label] for label in range(CLASS_COUNT)}
    started = time.monotonic()

    fit_indices = [
        index for label in range(CLASS_COUNT) for index in split[label]["fit"]
    ]
    fit_images = np.stack([_dataset_image(dataset, index) for index in fit_indices])
    bank = global_rademacher_direction_bank(
        seed=args.seed,
        bank_size=budget.direction_bank_size,
        dimension=RAW_DIMENSION,
    )
    stable_indices, ranking = rank_stable_directions(
        bank, fit_images, retain=budget.stable_direction_count
    )
    del fit_images
    stable = bank[np.asarray(stable_indices)].reshape(
        budget.stable_direction_count, *RAW_IMAGE_SHAPE
    )

    search_batches: list[GlobalProposalBatch] = []
    for image_index, true_label in _cohort_entries(split, "filter_search"):
        image, observed_label = dataset[image_index]
        if int(observed_label) != true_label:
            raise RuntimeError("deterministic split label changed")
        search_batches.append(
            _sample_proposals(
                model=model,
                torch=torch,
                image=image,
                directions=stable,
                image_index=image_index,
                true_label=true_label,
                proposals=budget.search_proposals,
                batch_size=budget.batch_size,
                sigma=args.sigma,
                seed=args.seed,
                stream="filter_search",
                device=device,
            )
        )
    offsets_by_slot = [
        candidate_offsets(
            np.asarray([batch.projections[slot].mean() for batch in search_batches]),
            budget.offset_quantiles,
        )
        for slot in range(budget.stable_direction_count)
    ]
    candidates = candidate_grid(
        stable_bank_indices=stable_indices,
        offsets_by_slot=offsets_by_slot,
        sigma=args.sigma,
        outer_fractions=budget.outer_fractions,
        inner_fractions=budget.inner_fractions,
    )
    selected, search_summary, score_rows = select_candidate(
        candidates,
        search_batches,
        sigma=args.sigma,
        min_retained=budget.min_search_retained,
    )
    candidate_grid_payload = [asdict(candidate) for candidate in candidates]
    common = {
        "paper_eligibility": PAPER_ELIGIBILITY,
        "study_role": "global ten-class stable-band development only",
        "dataset": "CIFAR-10 training split only",
        "test_sets_accessed": [],
        "raw_proposals_before_clipping": True,
        "clipping_performed": False,
        "budget_name": args.budget,
        "budget": asdict(budget),
        "sigma": args.sigma,
        "per_image_delta": args.delta,
        "seed": args.seed,
        "split_strategy": (
            "within-class SHA-256 rank into disjoint fit, filter-search, and "
            "evaluation cohorts"
        ),
        "split_sha256": _canonical_digest(split_payload),
        "split": split_payload,
        "direction_learning": {
            "generator": "SHA-256-expanded unit Rademacher raw-pixel directions",
            "ranking_data": "fit cohort only",
            "ranking_metric": "pooled population standard deviation of clean projections",
            "bank_size": int(bank.shape[0]),
            "bank_float64_sha256": _array_sha256(bank),
            "retained_count": len(stable_indices),
            "retained_bank_indices_in_stability_order": list(stable_indices),
            "retained_bank_float64_sha256": _array_sha256(stable),
            "full_ranking": ranking,
        },
        "candidate_count": len(candidates),
        "candidate_grid_sha256": _canonical_digest(candidate_grid_payload),
        "candidate_scores_sha256": _canonical_digest(score_rows),
        "top_feasible_candidate_scores": _top_score_rows(score_rows),
        "filter_search_stream": {
            "cohort": "filter_search",
            "proposals_per_image": budget.search_proposals,
            "images_disjoint_from_fit_and_evaluation": True,
        },
        "evaluation_protocol": {
            "cohort": "evaluation",
            "images_disjoint_from_fit_and_filter_search": True,
            "label_selection_proposals_per_image": budget.label_selection_proposals,
            "estimation_proposals_per_image": budget.evaluation_proposals,
            "streams_disjoint_by_hash_tag": True,
        },
        "per_image_bonferroni": {
            "one_sided_tail_events": BONFERRONI_TAIL_EVENTS,
            "tail_error": args.delta / BONFERRONI_TAIL_EVENTS,
            "total_allocated_error": args.delta,
            "events": {
                "conditional": "one selected lower and nine competitor uppers",
                "joint_lower": "one selected lower and nine competitor uppers",
                "joint_upper": "one selected upper and one chosen-runner lower",
                "unfiltered": "one selected lower and nine competitor uppers",
            },
        },
        "confidence_scope": (
            "all reported lower and upper radii are simultaneous within each "
            "fixed evaluation image; no collection-level confidence claim is made"
        ),
        "assets": {
            "auditvotes_git_commit": repository_commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "core_py_sha256": _file_sha256(code_root / "core.py"),
            "architectures_py_sha256": _file_sha256(code_root / "architectures.py"),
            "datasets_py_sha256": _file_sha256(code_root / "datasets.py"),
            "cifar_resnet_py_sha256": _file_sha256(code_root / "archs/cifar_resnet.py"),
            "source_sha256": _file_sha256(Path(__file__)),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }

    evaluation_rows: list[dict[str, Any]] = []
    if selected is None or search_summary is None:
        status = "open_development_failed_filter_search_feasibility"
        selected_filter = None
        proof = None
        evaluation_summary = None
        development_gate = None
    else:
        selected_direction = stable[selected.direction_slot]
        selected_filter = {
            "direction_slot_in_stable_bank": selected.direction_slot,
            "direction_bank_index": selected.direction_bank_index,
            "direction_float64_sha256": _array_sha256(selected_direction),
            "offset_b": selected.offset,
            "alpha": selected.inner,
            "beta": selected.outer,
        }
        proof = covariance_proof(selected_direction, selected, args.sigma)
        evaluation_candidate = StableBandCandidate(
            direction_slot=0,
            direction_bank_index=selected.direction_bank_index,
            offset=selected.offset,
            inner=selected.inner,
            outer=selected.outer,
        )
        for image_index, true_label in _cohort_entries(split, "evaluation"):
            image, observed_label = dataset[image_index]
            if int(observed_label) != true_label:
                raise RuntimeError("deterministic split label changed")
            selection_batch = _sample_proposals(
                model=model,
                torch=torch,
                image=image,
                directions=selected_direction[np.newaxis, ...],
                image_index=image_index,
                true_label=true_label,
                proposals=budget.label_selection_proposals,
                batch_size=budget.batch_size,
                sigma=args.sigma,
                seed=args.seed,
                stream="evaluation_label_selection",
                device=device,
            )
            selection_mask = bounded_two_band_mask(
                selection_batch.projections[0], evaluation_candidate, args.sigma
            )
            selection_retained_counts = _counts(
                selection_batch.labels, selection_mask
            )
            selection_unfiltered_counts = _counts(selection_batch.labels)
            filtered_selected, filtered_runner = selected_and_runner(
                selection_retained_counts
            )
            unfiltered_selected, _ = selected_and_runner(
                selection_unfiltered_counts
            )

            evaluation_batch = _sample_proposals(
                model=model,
                torch=torch,
                image=image,
                directions=selected_direction[np.newaxis, ...],
                image_index=image_index,
                true_label=true_label,
                proposals=budget.evaluation_proposals,
                batch_size=budget.batch_size,
                sigma=args.sigma,
                seed=args.seed,
                stream="evaluation_estimation",
                device=device,
            )
            evaluation_mask = bounded_two_band_mask(
                evaluation_batch.projections[0], evaluation_candidate, args.sigma
            )
            retained_counts = _counts(evaluation_batch.labels, evaluation_mask)
            unfiltered_counts = _counts(evaluation_batch.labels)
            radii = multiclass_certified_radii_from_counts(
                filtered_selected_label=filtered_selected,
                filtered_runner_label=filtered_runner,
                unfiltered_selected_label=unfiltered_selected,
                retained_counts=retained_counts,
                unfiltered_counts=unfiltered_counts,
                proposal_count=budget.evaluation_proposals,
                sigma=args.sigma,
                delta=args.delta,
                min_retained=budget.min_evaluation_retained,
            )
            retained = int(np.sum(retained_counts))
            selection_retained = int(np.sum(selection_retained_counts))
            evaluation_rows.append(
                {
                    "image_index": image_index,
                    "true_label": true_label,
                    "true_class_name": CLASS_NAMES[true_label],
                    "filtered_selected_label": filtered_selected,
                    "filtered_runner_label": filtered_runner,
                    "unfiltered_selected_label": unfiltered_selected,
                    "filtered_correct": filtered_selected == true_label,
                    "unfiltered_correct": unfiltered_selected == true_label,
                    "label_selection_retained_counts": selection_retained_counts.tolist(),
                    "label_selection_retained": selection_retained,
                    "label_selection_retention_rate": (
                        selection_retained / budget.label_selection_proposals
                    ),
                    "label_selection_retention_adequate": (
                        selection_retained >= budget.min_label_selection_retained
                    ),
                    "retained_counts": retained_counts.tolist(),
                    "unfiltered_counts": unfiltered_counts.tolist(),
                    "retained": retained,
                    "proposals": budget.evaluation_proposals,
                    "retention_rate": retained / budget.evaluation_proposals,
                    "evaluation_retention_adequate": (
                        retained >= budget.min_evaluation_retained
                    ),
                    **radii,
                }
            )
        evaluation_summary = summarize_evaluation(evaluation_rows)
        development_gate = evaluate_development_gate(evaluation_summary)
        status = (
            "open_development_passed_predeclared_gate"
            if development_gate["passed"]
            else "open_development_failed_predeclared_gate"
        )

    result = {
        **common,
        "status": status,
        "selected_filter": selected_filter,
        "search_summary": search_summary,
        "covariance_proof": proof,
        "evaluation_summary": evaluation_summary,
        "development_gate": development_gate,
        "evaluation_rows": evaluation_rows,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Every image belongs to the CIFAR-10 training split.",
            "The selected filter and all reported images are development data.",
            "The result is ineligible for the paper without a separately frozen confirmation.",
            "Per-image confidence is not simultaneous across the evaluation collection.",
            "SciPy evaluates Clopper-Pearson endpoints numerically rather than by interval arithmetic.",
            "The covariance proof applies to the fixed raw-proposal filter and not to clipping or an input-dependent filter.",
            "The audit evaluates every proposal to retain an unfiltered comparator; a deployed filter-first implementation evaluates only retained proposals.",
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
        "selected_filter": selected_filter,
        "search_summary": search_summary,
        "evaluation_summary": evaluation_summary,
        "output": str(args.output),
    }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
