"""Precision replication of one frozen global CIFAR-10 two-band filter.

This development-only study uses the CIFAR-10 training split.  It reconstructs
and verifies the filter selected in
``iclr2027_global_stable_band_filter_development_v1.json``, excludes the union
of every image used by the three preceding learned-filter studies, and
hash-ranks 20 fresh images per class.

For every image, filtered and unfiltered label-selection and estimation
streams are independent.  The filtered streams draw a fixed raw-proposal
budget and apply the frozen filter before the classifier.  Rejected proposals
never enter the network and are not replaced.  The estimation uses one
collection-wide family-wise error budget across all 200 images and all 32
one-sided tails per image.

The replication has a frozen success rule, runs every record without interim
stopping, and remains ineligible for the paper because all images come from
the CIFAR-10 training split.
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
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.iclr2027_global_stable_band_filter_development import (  # noqa: E402
    BONFERRONI_TAIL_EVENTS,
    CLASS_COUNT,
    CLASS_NAMES,
    EXPECTED_AUDITVOTES_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    RAW_DIMENSION,
    RAW_IMAGE_SHAPE,
    _array_sha256,
    _canonical_digest,
    _file_sha256,
    _git_head,
    global_rademacher_direction_bank,
    multiclass_certified_radii_from_counts,
    selected_and_runner,
    stable_seed,
)


PAPER_ELIGIBILITY = "none"
STUDY_ROLE = "frozen-filter CIFAR-10 training precision replication"
DEFAULT_SOURCE_RESULT = Path(
    "outputs/iclr2027_global_stable_band_filter_development_v1.json"
)
DEFAULT_LEARNED_BAND_RESULT = Path(
    "outputs/iclr2027_learned_band_filter_development_v1.json"
)
DEFAULT_STABLE_BAND_RESULT = Path(
    "outputs/iclr2027_learned_stable_band_filter_development_v2.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/iclr2027_global_band_precision_replication_v1.json"
)

EXPECTED_SOURCE_RESULT_SHA256 = (
    "c4bc41b212015990c71f182006020df099e7622e415bfc1e031167855ad2e5ed"
)
EXPECTED_LEARNED_BAND_RESULT_SHA256 = (
    "c7097556463ab12fa617795375281b29b184327b88f0654378a6c2eefd1f37c6"
)
EXPECTED_STABLE_BAND_RESULT_SHA256 = (
    "6b1c6b09addec09bb89e4b892d380e5e97a8b9c539324eb53f474c3d40eaf1fa"
)
EXPECTED_SOURCE_SPLIT_SHA256 = (
    "56572fd20403ff13b11ee3387999b3c85816ca93a208669e9a017f5d6fbf0282"
)
EXPECTED_LEARNED_BAND_SPLIT_SHA256 = (
    "5140eaf60c6bf2877d6cd7fd522f5a02d462a7a41e78ed37ecd76f5a6644781f"
)
EXPECTED_STABLE_BAND_SPLIT_SHA256 = (
    "6f9701a311d3b601bdc1f145add5214f68219ff3f9a606e374b73fcf1ecf2152"
)
EXPECTED_SOURCE_CANDIDATE_GRID_SHA256 = (
    "96996feeedcdfacfbdc367fe92312b8c4ba106d5959ad908c06816444e0f9515"
)
EXPECTED_SOURCE_CANDIDATE_SCORES_SHA256 = (
    "afbe310cc6c63c53b950ab7ea212a4a2232f40b000f9e9494418f7678c2304ae"
)
EXPECTED_EXCLUSION_UNION_COUNT = 11_697

FROZEN_BANK_SEED = 26_092_001
FROZEN_BANK_SIZE = 128
FROZEN_DIRECTION_INDEX = 110
FROZEN_DIRECTION_SHA256 = (
    "a641e5803b4c61e6f84794ff8f716ed4f984e7b89fa7778dc00684d6c61c6342"
)
FROZEN_OFFSET = 0.10721304813989381
FROZEN_ALPHA = 0.0625
FROZEN_BETA = 0.25
SIGMA = 0.25

COHORT_SEED = 26_092_002
COHORT_PER_CLASS = 20
TOTAL_IMAGES = CLASS_COUNT * COHORT_PER_CLASS
LABEL_SELECTION_PROPOSALS = 10_000
ESTIMATION_PROPOSALS = 100_000
COLLECTION_FWER = 0.001
PER_IMAGE_DELTA = COLLECTION_FWER / TOTAL_IMAGES
TAIL_ERROR = COLLECTION_FWER / (TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS)
TARGET_RADIUS = 0.20
MIN_MEAN_RETENTION = 0.25
MAX_MEAN_RETENTION = 0.60
MAX_ACCURACY_DROP = 0.01
MIN_CERTIFIED_ACCURACY_ADVANTAGE = 0.05
MIN_STRONG_MARGIN = 0.01
MIN_CORRECT_STRONG_IMAGES = 10
MAX_MCNEMAR_P = 0.01


@dataclass(frozen=True)
class FrozenFilter:
    bank_seed: int = FROZEN_BANK_SEED
    bank_size: int = FROZEN_BANK_SIZE
    direction_index: int = FROZEN_DIRECTION_INDEX
    direction_float64_sha256: str = FROZEN_DIRECTION_SHA256
    offset_b: float = FROZEN_OFFSET
    alpha: float = FROZEN_ALPHA
    beta: float = FROZEN_BETA
    sigma: float = SIGMA


@dataclass(frozen=True)
class StreamCounts:
    counts: tuple[int, ...]
    raw_proposals: int
    model_calls: int


def _frozen_selected_filter_payload() -> dict[str, Any]:
    return {
        "direction_slot_in_stable_bank": 0,
        "direction_bank_index": FROZEN_DIRECTION_INDEX,
        "direction_float64_sha256": FROZEN_DIRECTION_SHA256,
        "offset_b": FROZEN_OFFSET,
        "alpha": FROZEN_ALPHA,
        "beta": FROZEN_BETA,
    }


def validate_source_payload(payload: Mapping[str, Any]) -> dict[int, set[int]]:
    """Validate the frozen V1 result and return its per-class exclusions."""

    required_equalities = {
        "seed": FROZEN_BANK_SEED,
        "sigma": SIGMA,
        "paper_eligibility": PAPER_ELIGIBILITY,
        "dataset": "CIFAR-10 training split only",
        "test_sets_accessed": [],
        "split_sha256": EXPECTED_SOURCE_SPLIT_SHA256,
        "candidate_grid_sha256": EXPECTED_SOURCE_CANDIDATE_GRID_SHA256,
        "candidate_scores_sha256": EXPECTED_SOURCE_CANDIDATE_SCORES_SHA256,
        "selected_filter": _frozen_selected_filter_payload(),
    }
    for key, expected in required_equalities.items():
        if payload.get(key) != expected:
            raise ValueError(f"frozen source field {key!r} does not match")
    split = payload.get("split")
    if not isinstance(split, Mapping) or set(split) != {
        str(label) for label in range(CLASS_COUNT)
    }:
        raise ValueError("frozen source split does not contain all ten classes")
    exclusions: dict[int, set[int]] = {}
    expected_sizes = {"fit": 300, "filter_search": 20, "evaluation": 20}
    all_indices: set[int] = set()
    for label in range(CLASS_COUNT):
        cohorts = split[str(label)]
        if not isinstance(cohorts, Mapping) or set(cohorts) != set(expected_sizes):
            raise ValueError(f"invalid frozen split cohorts for class {label}")
        class_indices: set[int] = set()
        for name, expected_size in expected_sizes.items():
            values = cohorts[name]
            if (
                not isinstance(values, list)
                or len(values) != expected_size
                or any(not isinstance(value, int) for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"invalid frozen {name} cohort for class {label}")
            if class_indices.intersection(values):
                raise ValueError(f"overlapping frozen cohorts for class {label}")
            class_indices.update(values)
        if all_indices.intersection(class_indices):
            raise ValueError("frozen source reuses an image across classes")
        all_indices.update(class_indices)
        exclusions[label] = class_indices
    return exclusions


def load_frozen_source(path: Path) -> tuple[dict[str, Any], dict[int, set[int]]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = _file_sha256(resolved)
    if digest != EXPECTED_SOURCE_RESULT_SHA256:
        raise RuntimeError(
            "source development result does not match the frozen V1 artifact"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source development result must be a JSON object")
    return payload, validate_source_payload(payload)


def _split_indices(payload: Mapping[str, Any]) -> dict[int, set[int]]:
    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("learned-filter result has no split object")
    result = {label: set() for label in range(CLASS_COUNT)}
    for raw_label, cohorts in split.items():
        label = int(raw_label)
        if label < 0 or label >= CLASS_COUNT or not isinstance(cohorts, Mapping):
            raise ValueError("learned-filter split is malformed")
        class_values: set[int] = set()
        for values in cohorts.values():
            if (
                not isinstance(values, list)
                or any(not isinstance(value, int) for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError("learned-filter cohort is malformed")
            if class_values.intersection(values):
                raise ValueError("learned-filter cohorts overlap within a class")
            class_values.update(values)
        result[label] = class_values
    return result


def _load_pinned_exclusion_result(
    path: Path, *, expected_sha256: str, expected_split_sha256: str
) -> tuple[dict[str, Any], dict[int, set[int]], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = _file_sha256(resolved)
    if digest != expected_sha256:
        raise RuntimeError(f"exclusion result does not match pinned hash: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("dataset") != "CIFAR-10 training split only"
        or payload.get("test_sets_accessed") != []
        or payload.get("paper_eligibility") != PAPER_ELIGIBILITY
        or payload.get("split_sha256") != expected_split_sha256
    ):
        raise ValueError(f"pinned exclusion metadata does not match: {resolved}")
    indices = _split_indices(payload)
    flattened = sorted(index for values in indices.values() for index in values)
    metadata = {
        "path": str(resolved),
        "sha256": digest,
        "split_sha256": payload["split_sha256"],
        "status": payload.get("status"),
        "count": len(flattened),
        "indices_sha256": _canonical_digest(flattened),
    }
    return payload, indices, metadata


def load_frozen_sources(
    *,
    global_result_path: Path,
    learned_band_result_path: Path,
    stable_band_result_path: Path,
) -> tuple[dict[str, Any], dict[int, set[int]], list[dict[str, Any]]]:
    """Load all pinned searches and form their exact per-class index union."""

    global_payload, global_indices = load_frozen_source(global_result_path)
    global_flattened = sorted(
        index for values in global_indices.values() for index in values
    )
    source_metadata = [
        {
            "role": "filter source and exclusion source",
            "path": str(global_result_path.resolve()),
            "sha256": EXPECTED_SOURCE_RESULT_SHA256,
            "split_sha256": global_payload["split_sha256"],
            "status": global_payload.get("status"),
            "count": len(global_flattened),
            "indices_sha256": _canonical_digest(global_flattened),
        }
    ]
    sources = [global_indices]
    for role, path, digest, split_digest in (
        (
            "exclusion source",
            learned_band_result_path,
            EXPECTED_LEARNED_BAND_RESULT_SHA256,
            EXPECTED_LEARNED_BAND_SPLIT_SHA256,
        ),
        (
            "exclusion source",
            stable_band_result_path,
            EXPECTED_STABLE_BAND_RESULT_SHA256,
            EXPECTED_STABLE_BAND_SPLIT_SHA256,
        ),
    ):
        _, indices, metadata = _load_pinned_exclusion_result(
            path, expected_sha256=digest, expected_split_sha256=split_digest
        )
        sources.append(indices)
        source_metadata.append({"role": role, **metadata})
    union = {label: set() for label in range(CLASS_COUNT)}
    for source in sources:
        for label in range(CLASS_COUNT):
            union[label].update(source[label])
    union_count = sum(len(values) for values in union.values())
    if union_count != EXPECTED_EXCLUSION_UNION_COUNT:
        raise RuntimeError(
            "pinned learned-filter exclusion union changed: "
            f"expected {EXPECTED_EXCLUSION_UNION_COUNT}, observed {union_count}"
        )
    return global_payload, union, source_metadata


def reconstruct_frozen_direction() -> np.ndarray:
    bank = global_rademacher_direction_bank(
        seed=FROZEN_BANK_SEED,
        bank_size=FROZEN_BANK_SIZE,
        dimension=RAW_DIMENSION,
    )
    direction = np.asarray(bank[FROZEN_DIRECTION_INDEX], dtype=np.float64)
    if direction.shape != (RAW_DIMENSION,):
        raise RuntimeError("reconstructed direction has the wrong dimension")
    if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("reconstructed direction is not unit norm")
    if _array_sha256(direction) != FROZEN_DIRECTION_SHA256:
        raise RuntimeError("reconstructed direction does not match the frozen hash")
    return direction.reshape(RAW_IMAGE_SHAPE)


def hash_rank_fresh_indices(
    labels: Sequence[int],
    exclusions: Mapping[int, set[int]],
    *,
    per_class: int = COHORT_PER_CLASS,
    seed: int = COHORT_SEED,
) -> dict[int, list[int]]:
    """Hash-rank fresh within-class indices after all V1 exclusions."""

    if per_class < 1:
        raise ValueError("per-class cohort size must be positive")
    if set(exclusions) != set(range(CLASS_COUNT)):
        raise ValueError("exclusions must cover all ten classes")
    buckets: dict[int, list[tuple[bytes, int]]] = {
        label: [] for label in range(CLASS_COUNT)
    }
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label < 0 or label >= CLASS_COUNT:
            raise ValueError("dataset label is outside CIFAR-10")
        if index in exclusions[label]:
            continue
        rank = sha256(
            f"global-band-precision-v1:{seed}:{label}:{index}".encode()
        ).digest()
        buckets[label].append((rank, index))
    result: dict[int, list[int]] = {}
    for label in range(CLASS_COUNT):
        ordered = [index for _, index in sorted(buckets[label])]
        if len(ordered) < per_class:
            raise ValueError(f"class {label} has too few fresh images")
        result[label] = ordered[:per_class]
        if set(result[label]).intersection(exclusions[label]):
            raise RuntimeError("fresh cohort intersects the frozen source split")
    flattened = [index for label in range(CLASS_COUNT) for index in result[label]]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("precision cohort reuses an image")
    return result


def exact_one_sided_mcnemar(
    covariance_success: Sequence[bool], joint_success: Sequence[bool]
) -> dict[str, Any]:
    """Exact paired test for covariance success exceeding joint success."""

    covariance = np.asarray(covariance_success, dtype=bool)
    joint = np.asarray(joint_success, dtype=bool)
    if covariance.ndim != 1 or covariance.shape != joint.shape or covariance.size < 1:
        raise ValueError("paired success vectors must be nonempty and aligned")
    covariance_only = int(np.sum(covariance & ~joint))
    joint_only = int(np.sum(~covariance & joint))
    discordant = covariance_only + joint_only
    p_value = (
        1.0
        if discordant == 0
        else float(binom.sf(covariance_only - 1, discordant, 0.5))
    )
    return {
        "alternative": "covariance correct-certified accuracy is greater",
        "covariance_only": covariance_only,
        "joint_only": joint_only,
        "discordant": discordant,
        "exact_one_sided_p_value": p_value,
    }


def summarize_replication_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_records: int = TOTAL_IMAGES
) -> dict[str, Any]:
    if len(rows) != expected_records or expected_records < 1:
        raise ValueError("replication must contain the complete frozen cohort")
    complete = np.asarray([bool(row["record_complete"]) for row in rows])
    filtered_correct = np.asarray([bool(row["filtered_correct"]) for row in rows])
    unfiltered_correct = np.asarray([bool(row["unfiltered_correct"]) for row in rows])
    r_cov = np.asarray([float(row["r_cov_L"]) for row in rows])
    r_mass = np.asarray([float(row["r_mass_L"]) for row in rows])
    r_unfiltered = np.asarray([float(row["r_unfiltered_L"]) for row in rows])
    retention = np.asarray([float(row["estimation_retention_rate"]) for row in rows])
    selection_retention = np.asarray(
        [float(row["label_selection_retention_rate"]) for row in rows]
    )
    covariance_success = filtered_correct & (r_cov >= TARGET_RADIUS)
    joint_success = filtered_correct & (r_mass >= TARGET_RADIUS)
    unfiltered_success = unfiltered_correct & (r_unfiltered >= TARGET_RADIUS)
    strong_margin = np.asarray(
        [
            (
                float(row["r_cov_L"]) - float(row["r_mass_U"])
                if row["r_mass_U"] is not None
                else -math.inf
            )
            for row in rows
        ]
    )
    correct_strong = filtered_correct & (strong_margin >= MIN_STRONG_MARGIN)
    filtered_calls = int(
        sum(
            int(row["filtered_label_selection_model_calls"])
            + int(row["filtered_estimation_model_calls"])
            for row in rows
        )
    )
    filtered_raw = int(
        sum(
            int(row["filtered_label_selection_raw_proposals"])
            + int(row["filtered_estimation_raw_proposals"])
            for row in rows
        )
    )
    unfiltered_calls = int(
        sum(
            int(row["unfiltered_label_selection_model_calls"])
            + int(row["unfiltered_estimation_model_calls"])
            for row in rows
        )
    )
    unfiltered_raw = int(
        sum(
            int(row["unfiltered_label_selection_raw_proposals"])
            + int(row["unfiltered_estimation_raw_proposals"])
            for row in rows
        )
    )
    mcnemar = exact_one_sided_mcnemar(covariance_success, joint_success)
    observed = {
        "records": len(rows),
        "all_records_complete": bool(np.all(complete)),
        "mean_estimation_retention": float(np.mean(retention)),
        "mean_label_selection_retention": float(np.mean(selection_retention)),
        "min_estimation_retention": float(np.min(retention)),
        "max_estimation_retention": float(np.max(retention)),
        "filtered_selection_accuracy": float(np.mean(filtered_correct)),
        "unfiltered_selection_accuracy": float(np.mean(unfiltered_correct)),
        "covariance_correct_certified_accuracy_at_radius_0_20": float(
            np.mean(covariance_success)
        ),
        "joint_correct_certified_accuracy_at_radius_0_20": float(
            np.mean(joint_success)
        ),
        "unfiltered_correct_certified_accuracy_at_radius_0_20": float(
            np.mean(unfiltered_success)
        ),
        "correct_strong_margin_count": int(np.sum(correct_strong)),
        "filtered_raw_proposals": filtered_raw,
        "filtered_model_calls": filtered_calls,
        "filtered_model_call_fraction": filtered_calls / filtered_raw,
        "unfiltered_raw_proposals": unfiltered_raw,
        "unfiltered_model_calls": unfiltered_calls,
        "unfiltered_model_call_fraction": unfiltered_calls / unfiltered_raw,
        "total_model_calls": filtered_calls + unfiltered_calls,
        "mcnemar": mcnemar,
    }
    checks = {
        "all_200_records_complete": observed["all_records_complete"],
        "mean_retention_at_least_0_25": (
            observed["mean_estimation_retention"] >= MIN_MEAN_RETENTION
        ),
        "mean_retention_at_most_0_60": (
            observed["mean_estimation_retention"] <= MAX_MEAN_RETENTION
        ),
        "filtered_accuracy_no_more_than_0_01_below_unfiltered": (
            observed["filtered_selection_accuracy"]
            >= observed["unfiltered_selection_accuracy"] - MAX_ACCURACY_DROP
        ),
        "covariance_accuracy_at_0_20_at_least_0_05_above_joint": (
            observed["covariance_correct_certified_accuracy_at_radius_0_20"]
            >= observed["joint_correct_certified_accuracy_at_radius_0_20"]
            + MIN_CERTIFIED_ACCURACY_ADVANTAGE
        ),
        "at_least_10_correct_images_have_covariance_lower_above_mass_upper_by_0_01": (
            observed["correct_strong_margin_count"] >= MIN_CORRECT_STRONG_IMAGES
        ),
        "one_sided_exact_mcnemar_p_at_most_0_01": (
            mcnemar["exact_one_sided_p_value"] <= MAX_MCNEMAR_P
        ),
    }
    return {
        "passed_predeclared_success_rule": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "target_radius": TARGET_RADIUS,
            "minimum_mean_retention": MIN_MEAN_RETENTION,
            "maximum_mean_retention": MAX_MEAN_RETENTION,
            "maximum_filtered_accuracy_drop": MAX_ACCURACY_DROP,
            "minimum_covariance_over_joint_certified_accuracy": (
                MIN_CERTIFIED_ACCURACY_ADVANTAGE
            ),
            "minimum_correct_strong_margin": MIN_STRONG_MARGIN,
            "minimum_correct_strong_images": MIN_CORRECT_STRONG_IMAGES,
            "maximum_exact_one_sided_mcnemar_p": MAX_MCNEMAR_P,
        },
        "observed": observed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument(
        "--learned-band-result", type=Path, default=DEFAULT_LEARNED_BAND_RESULT
    )
    parser.add_argument(
        "--stable-band-result", type=Path, default=DEFAULT_STABLE_BAND_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _run_filtered_stream(
    *,
    model: Any,
    torch: Any,
    image: Any,
    direction_device: Any,
    image_index: int,
    proposals: int,
    batch_size: int,
    stream: str,
    device: Any,
) -> StreamCounts:
    """Apply the frozen raw-proposal filter before every classifier call."""

    generator = torch.Generator(device=device)
    generator.manual_seed(
        stable_seed("global-band-precision-v1", COHORT_SEED, image_index, stream)
    )
    image_device = image.unsqueeze(0).to(device)
    counts = np.zeros(CLASS_COUNT, dtype=np.int64)
    model_calls = 0
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
            raw = image_device.expand(size, -1, -1, -1) + SIGMA * noise
            projection = raw.to(torch.float64).flatten(1) @ direction_device
            absolute = torch.abs(projection + FROZEN_OFFSET)
            retained = (absolute >= FROZEN_ALPHA) & (absolute <= FROZEN_BETA)
            accepted = raw[retained]
            accepted_count = int(accepted.shape[0])
            if accepted_count:
                predictions = torch.argmax(model(accepted), dim=1)
                counts += (
                    torch.bincount(predictions, minlength=CLASS_COUNT)
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
            model_calls += accepted_count
            completed += size
    if int(np.sum(counts)) != model_calls or completed != proposals:
        raise RuntimeError("filtered stream accounting failed")
    return StreamCounts(tuple(int(value) for value in counts), completed, model_calls)


def _run_unfiltered_stream(
    *,
    model: Any,
    torch: Any,
    image: Any,
    image_index: int,
    proposals: int,
    batch_size: int,
    stream: str,
    device: Any,
) -> StreamCounts:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        stable_seed("global-band-precision-v1", COHORT_SEED, image_index, stream)
    )
    image_device = image.unsqueeze(0).to(device)
    counts = np.zeros(CLASS_COUNT, dtype=np.int64)
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
            raw = image_device.expand(size, -1, -1, -1) + SIGMA * noise
            predictions = torch.argmax(model(raw), dim=1)
            counts += (
                torch.bincount(predictions, minlength=CLASS_COUNT)
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            completed += size
    if int(np.sum(counts)) != completed or completed != proposals:
        raise RuntimeError("unfiltered stream accounting failed")
    return StreamCounts(tuple(int(value) for value in counts), completed, completed)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    source_result, exclusions, source_results_metadata = load_frozen_sources(
        global_result_path=args.source_result,
        learned_band_result_path=args.learned_band_result,
        stable_band_result_path=args.stable_band_result,
    )
    direction = reconstruct_frozen_direction()

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
        raise RuntimeError("AuditVotes repository is not at the pinned commit")
    checkpoint_digest = _file_sha256(checkpoint)
    if checkpoint_digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint does not match the released artifact")
    sys.path.insert(0, str(code_root))

    # Heavy imports are lazy so unit tests never need CUDA or dataset access.
    import torch
    import torchvision
    from torchvision import transforms

    if not torch.cuda.is_available():
        raise RuntimeError("the released AuditVotes architecture requires CUDA")
    from architectures import get_architecture  # type: ignore

    torch.manual_seed(COHORT_SEED)
    torch.cuda.manual_seed_all(COHORT_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("arch") != "cifar_resnet110":
        raise RuntimeError("expected the released CIFAR-10 ResNet-110")
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
    for label, excluded in exclusions.items():
        if any(int(dataset.targets[index]) != label for index in excluded):
            raise RuntimeError("source exclusion label does not match the dataset")
    cohort = hash_rank_fresh_indices(dataset.targets, exclusions)
    cohort_payload = {str(label): cohort[label] for label in range(CLASS_COUNT)}
    ordered_cohort = [
        (index, label)
        for label in range(CLASS_COUNT)
        for index in cohort[label]
    ]
    direction_device = torch.as_tensor(
        direction.reshape(-1), dtype=torch.float64, device=device
    )

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for ordinal, (image_index, true_label) in enumerate(ordered_cohort, start=1):
        image, observed_label = dataset[image_index]
        if int(observed_label) != true_label:
            raise RuntimeError("precision cohort label changed")

        filtered_selection = _run_filtered_stream(
            model=model,
            torch=torch,
            image=image,
            direction_device=direction_device,
            image_index=image_index,
            proposals=LABEL_SELECTION_PROPOSALS,
            batch_size=args.batch_size,
            stream="filtered_label_selection",
            device=device,
        )
        unfiltered_selection = _run_unfiltered_stream(
            model=model,
            torch=torch,
            image=image,
            image_index=image_index,
            proposals=LABEL_SELECTION_PROPOSALS,
            batch_size=args.batch_size,
            stream="unfiltered_label_selection",
            device=device,
        )
        filtered_selected, filtered_runner = selected_and_runner(
            filtered_selection.counts
        )
        unfiltered_selected, _ = selected_and_runner(unfiltered_selection.counts)

        filtered_estimation = _run_filtered_stream(
            model=model,
            torch=torch,
            image=image,
            direction_device=direction_device,
            image_index=image_index,
            proposals=ESTIMATION_PROPOSALS,
            batch_size=args.batch_size,
            stream="filtered_estimation",
            device=device,
        )
        unfiltered_estimation = _run_unfiltered_stream(
            model=model,
            torch=torch,
            image=image,
            image_index=image_index,
            proposals=ESTIMATION_PROPOSALS,
            batch_size=args.batch_size,
            stream="unfiltered_estimation",
            device=device,
        )
        radii = multiclass_certified_radii_from_counts(
            filtered_selected_label=filtered_selected,
            filtered_runner_label=filtered_runner,
            unfiltered_selected_label=unfiltered_selected,
            retained_counts=filtered_estimation.counts,
            unfiltered_counts=unfiltered_estimation.counts,
            proposal_count=ESTIMATION_PROPOSALS,
            sigma=SIGMA,
            delta=PER_IMAGE_DELTA,
            min_retained=1,
        )
        complete = bool(
            filtered_selection.raw_proposals == LABEL_SELECTION_PROPOSALS
            and unfiltered_selection.raw_proposals == LABEL_SELECTION_PROPOSALS
            and filtered_estimation.raw_proposals == ESTIMATION_PROPOSALS
            and unfiltered_estimation.raw_proposals == ESTIMATION_PROPOSALS
            and filtered_selection.model_calls == sum(filtered_selection.counts)
            and filtered_estimation.model_calls == sum(filtered_estimation.counts)
            and unfiltered_selection.model_calls == LABEL_SELECTION_PROPOSALS
            and unfiltered_estimation.model_calls == ESTIMATION_PROPOSALS
        )
        row = {
            "ordinal": ordinal,
            "image_index": image_index,
            "true_label": true_label,
            "true_class_name": CLASS_NAMES[true_label],
            "record_complete": complete,
            "filtered_selected_label": filtered_selected,
            "filtered_runner_label": filtered_runner,
            "unfiltered_selected_label": unfiltered_selected,
            "filtered_correct": filtered_selected == true_label,
            "unfiltered_correct": unfiltered_selected == true_label,
            "filtered_label_selection_counts": list(filtered_selection.counts),
            "unfiltered_label_selection_counts": list(unfiltered_selection.counts),
            "filtered_estimation_counts": list(filtered_estimation.counts),
            "unfiltered_estimation_counts": list(unfiltered_estimation.counts),
            "filtered_label_selection_raw_proposals": (
                filtered_selection.raw_proposals
            ),
            "filtered_label_selection_model_calls": filtered_selection.model_calls,
            "unfiltered_label_selection_raw_proposals": (
                unfiltered_selection.raw_proposals
            ),
            "unfiltered_label_selection_model_calls": (
                unfiltered_selection.model_calls
            ),
            "filtered_estimation_raw_proposals": filtered_estimation.raw_proposals,
            "filtered_estimation_model_calls": filtered_estimation.model_calls,
            "unfiltered_estimation_raw_proposals": (
                unfiltered_estimation.raw_proposals
            ),
            "unfiltered_estimation_model_calls": unfiltered_estimation.model_calls,
            "label_selection_retention_rate": (
                filtered_selection.model_calls / LABEL_SELECTION_PROPOSALS
            ),
            "estimation_retention_rate": (
                filtered_estimation.model_calls / ESTIMATION_PROPOSALS
            ),
            "no_replacement_proposals_drawn": True,
            **radii,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": ordinal,
                    "total": TOTAL_IMAGES,
                    "image_index": image_index,
                    "retention": row["estimation_retention_rate"],
                    "r_cov_L": row["r_cov_L"],
                    "r_mass_L": row["r_mass_L"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    elapsed = time.monotonic() - started
    summary = summarize_replication_rows(rows)
    passed = summary["passed_predeclared_success_rule"]
    result = {
        "status": (
            "development_precision_replication_passed_predeclared_success_rule"
            if passed
            else "development_precision_replication_failed_predeclared_success_rule"
        ),
        "paper_eligibility": PAPER_ELIGIBILITY,
        "study_role": STUDY_ROLE,
        "dataset": "CIFAR-10 training split only",
        "test_sets_accessed": [],
        "all_records_run_without_interim_stopping": True,
        "raw_proposals_before_clipping": True,
        "clipping_performed": False,
        "replacement_proposals_drawn": False,
        "filtered_stream_classifier_order": (
            "fixed raw-proposal filter first, classifier only on retained proposals"
        ),
        "frozen_filter": asdict(FrozenFilter()),
        "source_result": {
            "path": str(args.source_result.resolve()),
            "sha256": EXPECTED_SOURCE_RESULT_SHA256,
            "source_status": source_result["status"],
            "source_selected_filter": source_result["selected_filter"],
            "source_split_sha256": source_result["split_sha256"],
            "source_candidate_grid_sha256": source_result[
                "candidate_grid_sha256"
            ],
            "source_candidate_scores_sha256": source_result[
                "candidate_scores_sha256"
            ],
        },
        "all_pinned_learned_filter_results": source_results_metadata,
        "exclusion": {
            "rule": (
                "union every fit, dev, filter_search, and evaluation index in "
                "the three hash-pinned learned-filter results"
            ),
            "excluded_per_class": {
                str(label): sorted(exclusions[label])
                for label in range(CLASS_COUNT)
            },
            "excluded_count_per_class": {
                str(label): len(exclusions[label]) for label in range(CLASS_COUNT)
            },
            "total_excluded": sum(len(values) for values in exclusions.values()),
            "expected_total_excluded": EXPECTED_EXCLUSION_UNION_COUNT,
            "excluded_indices_sha256": _canonical_digest(
                {
                    str(label): sorted(exclusions[label])
                    for label in range(CLASS_COUNT)
                }
            ),
            "fresh_cohort_intersection_count": 0,
        },
        "cohort": {
            "selection": "within-class SHA-256 rank after frozen exclusions",
            "rank_prefix": "global-band-precision-v1",
            "seed": COHORT_SEED,
            "per_class": COHORT_PER_CLASS,
            "images": TOTAL_IMAGES,
            "indices": cohort_payload,
            "indices_sha256": _canonical_digest(cohort_payload),
        },
        "proposal_protocol": {
            "filtered_label_selection_raw_proposals_per_image": (
                LABEL_SELECTION_PROPOSALS
            ),
            "filtered_estimation_raw_proposals_per_image": ESTIMATION_PROPOSALS,
            "unfiltered_label_selection_proposals_per_image": (
                LABEL_SELECTION_PROPOSALS
            ),
            "unfiltered_estimation_proposals_per_image": ESTIMATION_PROPOSALS,
            "filtered_and_unfiltered_streams_independent": True,
            "selection_and_estimation_streams_independent": True,
            "stream_seeds": (
                "SHA-256 stable_seed over study tag, cohort seed, image index, "
                "and stream name"
            ),
        },
        "confidence": {
            "collection_fwer": COLLECTION_FWER,
            "images": TOTAL_IMAGES,
            "one_sided_tails_per_image": BONFERRONI_TAIL_EVENTS,
            "per_image_delta": PER_IMAGE_DELTA,
            "one_sided_tail_error": TAIL_ERROR,
            "allocated_error": (
                TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS * TAIL_ERROR
            ),
            "scope": "all 32 one-sided tails on every one of the 200 images",
        },
        "predeclared_success_rule": summary,
        "evaluation_rows_sha256": _canonical_digest(rows),
        "evaluation_rows": rows,
        "actual_wall_time_seconds": elapsed,
        "assets": {
            "auditvotes_git_commit": repository_commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "core_py_sha256": _file_sha256(code_root / "core.py"),
            "architectures_py_sha256": _file_sha256(
                code_root / "architectures.py"
            ),
            "datasets_py_sha256": _file_sha256(code_root / "datasets.py"),
            "cifar_resnet_py_sha256": _file_sha256(
                code_root / "archs/cifar_resnet.py"
            ),
            "source_code_sha256": _file_sha256(Path(__file__)),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "limitations": [
            "All 200 images belong to the CIFAR-10 training split.",
            "The filter was selected on earlier CIFAR-10 training cohorts.",
            "Fresh indices are disjoint from filter development but are not a model holdout because the checkpoint was trained on CIFAR-10 training images.",
            "The result is development evidence and is ineligible for the paper.",
            "Filtered and unfiltered predictions use independent Monte Carlo streams.",
            "SciPy evaluates Clopper-Pearson endpoints numerically.",
            "The covariance proof applies to the raw-proposal filter without clipping.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "paper_eligibility": PAPER_ELIGIBILITY,
                "observed": summary["observed"],
                "actual_wall_time_seconds": elapsed,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
