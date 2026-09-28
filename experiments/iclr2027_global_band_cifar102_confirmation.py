#!/usr/bin/env python3
"""Paper-eligible external confirmation of the frozen global two-band filter.

This runner is deliberately fail-closed.  It accepts only the official
``cifar102_test.npz`` artifact from the pinned CIFAR-10.2 repository revision,
in the stored order of all 2,000 images.  Before a model is loaded, it verifies
an independently frozen protocol, this source file and its scientific helper
modules, a successful hash-pinned development precision replication, the
AuditVotes checkout and checkpoint, and the dataset repository and NPZ file.

Each image uses four independent deterministic Gaussian streams: filtered and
unfiltered label selection, followed by filtered and unfiltered estimation.
The filtered streams apply the frozen raw two-band rule before the classifier.
There is no clipping and no replacement or top-up after rejection.  A single
collection FWER of 0.001 covers all 2,000 * 32 one-sided confidence tails.

Progress is an append-only JSONL exact prefix of dataset indices 0..1999.  An
immutable manifest is byte-hashed into every row, so ``--resume`` cannot mix
protocols, code, assets, checkpoints, or datasets.  The runner always attempts
the full remaining prefix and has no outcome-based early stopping.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.iclr2027_global_band_precision_replication import (  # noqa: E402
    EXPECTED_AUDITVOTES_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_LEARNED_BAND_RESULT_SHA256,
    ESTIMATION_PROPOSALS,
    FROZEN_ALPHA,
    FROZEN_BETA,
    FROZEN_OFFSET,
    LABEL_SELECTION_PROPOSALS,
    EXPECTED_SOURCE_RESULT_SHA256,
    EXPECTED_STABLE_BAND_RESULT_SHA256,
    SIGMA,
    FrozenFilter,
    exact_one_sided_mcnemar,
    reconstruct_frozen_direction,
)
from experiments.iclr2027_global_stable_band_filter_development import (  # noqa: E402
    BONFERRONI_TAIL_EVENTS,
    CLASS_COUNT,
    CLASS_NAMES,
    RAW_IMAGE_SHAPE,
    _canonical_digest,
    _file_sha256,
    multiclass_certified_radii_from_counts,
    selected_and_runner,
    stable_seed,
)


PAPER_ELIGIBILITY = "paper-eligible external confirmation"
STUDY_ROLE = "frozen-filter CIFAR-10.2 full-test external confirmation"
PROTOCOL_STATUS = "frozen_before_cifar102_confirmation_execution"
DEVELOPMENT_PASS_STATUS = (
    "development_precision_replication_passed_predeclared_success_rule"
)

EXPECTED_CIFAR102_REPOSITORY_COMMIT = (
    "521467a52ad494b78ff9c70051099c0d9ba74d40"
)
EXPECTED_CIFAR102_FILENAME = "cifar102_test.npz"
EXPECTED_CIFAR102_BYTES = 6_393_664
EXPECTED_CIFAR102_GIT_BLOB_SHA1 = "7a3829681be56af70ea80ec99cf3822e2f5edff3"
EXPECTED_CIFAR102_SHA256 = (
    "e4fd6462bd5141ed293acd52ea570952f65d1bfd4ec9b068f87742f75a7a632e"
)

TOTAL_IMAGES = 2_000
IMAGES_PER_CLASS = 200
COLLECTION_FWER = 0.001
PER_IMAGE_DELTA = COLLECTION_FWER / TOTAL_IMAGES
TAIL_ERROR = COLLECTION_FWER / (TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS)
TAIL_DECOMPOSITION = {
    "conditional_selected_lower_and_nine_competitor_uppers": 10,
    "joint_selected_lower_and_nine_competitor_uppers": 10,
    "joint_selected_upper_and_independently_selected_runner_lower": 2,
    "unfiltered_selected_lower_and_nine_competitor_uppers": 10,
}
TARGET_RADIUS = 0.20
MIN_MEAN_RETENTION = 0.25
MAX_MEAN_RETENTION = 0.60
MAX_ACCURACY_DROP = 0.01
MIN_CERTIFIED_ACCURACY_ADVANTAGE = 0.05
MIN_STRONG_MARGIN = 0.01
MIN_CORRECT_STRONG_IMAGES = 100
MAX_MCNEMAR_P = 0.001

CONFIRMATION_SEED = 26_092_003
STREAM_NAMESPACE = "global-band-cifar102-confirmation-v1"
STREAM_NAMES = (
    "filtered_label_selection",
    "filtered_estimation",
    "unfiltered_label_selection",
    "unfiltered_estimation",
)
INFERENCE_BATCH_SIZE = 1_024

PREFLIGHT_INSTRUMENTATION_HISTORY = [
    {
        "status": "failed_before_model_loading_and_before_any_image_evaluation",
        "prior_protocol_sha256": (
            "d0104e203ac3b3891d39a5b2654fd7a453143784edfffa9c22ae323c06225e6b"
        ),
        "prior_runner_source_sha256": (
            "643ad9c1d427475d8a08931f01734cc24f4569811b4f0347d607fd3323170f02"
        ),
        "failure_artifact_sha256": (
            "13324f52a2381115122194daa28935884b63bc810d51ce314068befcac2e1577"
        ),
        "model_loaded": False,
        "image_outcomes_observed": 0,
        "correction": (
            "validate the five arrays in the hash-pinned official NPZ rather "
            "than requiring an images-and-labels-only archive"
        ),
    }
]

DEFAULT_DEVELOPMENT_RESULT = Path(
    "outputs/iclr2027_global_band_precision_replication_v1.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "outputs/iclr2027_global_band_cifar102_confirmation_v1"
)
PRECISION_SOURCE = PROJECT_ROOT / (
    "experiments/iclr2027_global_band_precision_replication.py"
)
CERTIFICATE_SOURCE = PROJECT_ROOT / (
    "experiments/iclr2027_global_stable_band_filter_development.py"
)
AUDITVOTES_CODE_FILES = {
    "core_py_sha256": Path("ImageClassify_Gaussian/ImageClassify_Conf/code/core.py"),
    "architectures_py_sha256": Path(
        "ImageClassify_Gaussian/ImageClassify_Conf/code/architectures.py"
    ),
    "datasets_py_sha256": Path(
        "ImageClassify_Gaussian/ImageClassify_Conf/code/datasets.py"
    ),
    "cifar_resnet_py_sha256": Path(
        "ImageClassify_Gaussian/ImageClassify_Conf/code/archs/cifar_resnet.py"
    ),
}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Cifar102Dataset:
    """Validated in-memory CIFAR-10.2 data in official stored order."""

    images: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        # Torch stays lazy so pure unit tests do not need a GPU or model stack.
        import torch

        if not 0 <= index < len(self):
            raise IndexError(index)
        chw = np.array(self.images[index].transpose(2, 0, 1), copy=True)
        image = torch.from_numpy(chw).to(dtype=torch.float32).div_(255.0)
        return image, int(self.labels[index])


@dataclass(frozen=True)
class StreamCounts:
    counts: tuple[int, ...]
    raw_proposals: int
    model_calls: int


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_blob_sha1(path: Path) -> str:
    """Return Git's SHA-1 object id for a file without invoking Git."""

    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(
    mapping: Mapping[str, Any], key: str, expected: Any, *, context: str
) -> None:
    if key not in mapping or mapping[key] != expected:
        raise ValueError(f"{context} field {key!r} does not match the frozen value")


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise ValueError(f"{context} must not be a placeholder digest")
    return value


def load_cifar102_npz(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_CIFAR102_SHA256,
    expected_git_blob_sha1: str | None = EXPECTED_CIFAR102_GIT_BLOB_SHA1,
    expected_bytes: int = EXPECTED_CIFAR102_BYTES,
    expected_count: int = TOTAL_IMAGES,
    expected_per_class: int = IMAGES_PER_CLASS,
    require_official_filename: bool = True,
) -> Cifar102Dataset:
    """Load a hash-pinned CIFAR-10.2 NPZ and validate its complete schema."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if require_official_filename and resolved.name != EXPECTED_CIFAR102_FILENAME:
        raise ValueError(
            f"official dataset file must be named {EXPECTED_CIFAR102_FILENAME}"
        )
    if resolved.stat().st_size != int(expected_bytes):
        raise ValueError("CIFAR-10.2 NPZ byte size does not match the frozen artifact")
    if _file_sha256(resolved) != expected_sha256:
        raise ValueError("CIFAR-10.2 NPZ SHA-256 does not match the frozen artifact")
    if (
        expected_git_blob_sha1 is not None
        and _git_blob_sha1(resolved) != expected_git_blob_sha1
    ):
        raise ValueError(
            "CIFAR-10.2 NPZ Git blob SHA-1 does not match the frozen artifact"
        )
    if expected_count < 1 or expected_per_class < 1:
        raise ValueError("expected dataset and class counts must be positive")
    if expected_count != CLASS_COUNT * expected_per_class:
        raise ValueError("expected total must equal ten times the per-class count")

    # The archive is opened only after its byte size and digest are verified.
    with np.load(resolved, allow_pickle=False) as archive:
        expected_keys = {
            "images",
            "labels",
            "ti_indices",
            "keywords",
            "label_names",
        }
        if set(archive.files) != expected_keys:
            raise ValueError("CIFAR-10.2 NPZ schema does not match the official file")
        images = np.array(archive["images"], copy=True)
        labels = np.array(archive["labels"], copy=True)
        ti_indices = np.array(archive["ti_indices"], copy=True)
        keywords = np.array(archive["keywords"], copy=True)
        label_names = np.array(archive["label_names"], copy=True)

    expected_shape = (expected_count, 32, 32, 3)
    if images.shape != expected_shape:
        raise ValueError(f"CIFAR-10.2 images must have shape {expected_shape}")
    if images.dtype != np.uint8:
        raise ValueError("CIFAR-10.2 images must have uint8 dtype")
    if labels.shape != (expected_count,):
        raise ValueError(f"CIFAR-10.2 labels must have shape ({expected_count},)")
    if labels.dtype.kind not in "iu":
        raise ValueError("CIFAR-10.2 labels must have integer dtype")
    labels64 = labels.astype(np.int64, copy=False)
    if np.any(labels64 < 0) or np.any(labels64 >= CLASS_COUNT):
        raise ValueError("CIFAR-10.2 labels must be integers in 0..9")
    histogram = np.bincount(labels64, minlength=CLASS_COUNT)
    if histogram.tolist() != [expected_per_class] * CLASS_COUNT:
        raise ValueError(
            "CIFAR-10.2 must contain exactly "
            f"{expected_per_class} images per class"
        )
    if ti_indices.shape != (expected_count,) or ti_indices.dtype.kind not in "iu":
        raise ValueError("CIFAR-10.2 ti_indices must be an integer vector")
    if np.unique(ti_indices).size != expected_count:
        raise ValueError("CIFAR-10.2 ti_indices must be unique")
    if keywords.shape != (expected_count,) or keywords.dtype.kind != "U":
        raise ValueError("CIFAR-10.2 keywords must be a Unicode vector")
    if np.any(keywords == ""):
        raise ValueError("CIFAR-10.2 keywords must be nonempty")
    expected_label_names = np.asarray(
        [
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
        ]
    )
    if not np.array_equal(label_names, expected_label_names):
        raise ValueError("CIFAR-10.2 label_names do not match the official classes")
    return Cifar102Dataset(images=images, labels=labels64)


def validate_development_precision_result(payload: Mapping[str, Any]) -> None:
    """Fail unless the pinned development replication passed in full."""

    required = {
        "status": DEVELOPMENT_PASS_STATUS,
        "paper_eligibility": "none",
        "dataset": "CIFAR-10 training split only",
        "test_sets_accessed": [],
        "all_records_run_without_interim_stopping": True,
        "raw_proposals_before_clipping": True,
        "clipping_performed": False,
        "replacement_proposals_drawn": False,
        "frozen_filter": asdict(FrozenFilter()),
    }
    for key, expected in required.items():
        _require_equal(payload, key, expected, context="development result")

    success = payload.get("predeclared_success_rule")
    if not isinstance(success, Mapping):
        raise ValueError("development result has no predeclared success rule")
    if success.get("passed_predeclared_success_rule") is not True:
        raise ValueError("development precision replication did not pass")
    checks = success.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("development precision replication checks did not all pass")
    observed = success.get("observed")
    if not isinstance(observed, Mapping) or observed.get("records") != 200:
        raise ValueError("development precision replication is not the full 200 images")
    if observed.get("all_records_complete") is not True:
        raise ValueError("development precision replication has incomplete records")

    proposal = payload.get("proposal_protocol")
    if not isinstance(proposal, Mapping):
        raise ValueError("development result has no proposal protocol")
    proposal_required = {
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
    }
    for key, expected in proposal_required.items():
        _require_equal(proposal, key, expected, context="development proposal")

    confidence = payload.get("confidence")
    if not isinstance(confidence, Mapping):
        raise ValueError("development result has no confidence accounting")
    confidence_required = {
        "collection_fwer": 0.001,
        "images": 200,
        "one_sided_tails_per_image": BONFERRONI_TAIL_EVENTS,
    }
    for key, expected in confidence_required.items():
        _require_equal(confidence, key, expected, context="development confidence")

    assets = payload.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("development result has no pinned model assets")
    _require_equal(
        assets,
        "auditvotes_git_commit",
        EXPECTED_AUDITVOTES_COMMIT,
        context="development assets",
    )
    for key in AUDITVOTES_CODE_FILES:
        _require_sha256(assets.get(key), context=f"development assets {key}")
    _require_equal(
        assets,
        "checkpoint_sha256",
        EXPECTED_CHECKPOINT_SHA256,
        context="development assets",
    )

    source_result = payload.get("source_result")
    if not isinstance(source_result, Mapping):
        raise ValueError("development result omits the failed global V1 source")
    _require_equal(
        source_result,
        "sha256",
        EXPECTED_SOURCE_RESULT_SHA256,
        context="development source result",
    )
    _require_equal(
        source_result,
        "source_status",
        "open_development_failed_predeclared_gate",
        context="development source result",
    )
    history = payload.get("all_pinned_learned_filter_results")
    if not isinstance(history, list) or len(history) != 3:
        raise ValueError("development result omits the full adaptive artifact history")
    history_digests = {
        row.get("sha256") for row in history if isinstance(row, Mapping)
    }
    expected_history_digests = {
        EXPECTED_SOURCE_RESULT_SHA256,
        EXPECTED_LEARNED_BAND_RESULT_SHA256,
        EXPECTED_STABLE_BAND_RESULT_SHA256,
    }
    if history_digests != expected_history_digests:
        raise ValueError("development adaptive artifact history hashes changed")

    rows = payload.get("evaluation_rows")
    rows_digest = payload.get("evaluation_rows_sha256")
    if rows is not None or rows_digest is not None:
        if not isinstance(rows, list) or len(rows) != 200:
            raise ValueError("development result evaluation rows are incomplete")
        if rows_digest != _canonical_digest(rows):
            raise ValueError("development result evaluation-row digest changed")


def _validate_protocol_contract(
    protocol: Mapping[str, Any], *, source_code_path: Path
) -> str:
    _require_equal(protocol, "status", PROTOCOL_STATUS, context="protocol")
    _require_equal(
        protocol, "paper_eligibility", PAPER_ELIGIBILITY, context="protocol"
    )
    _require_equal(protocol, "study_role", STUDY_ROLE, context="protocol")
    _require_equal(
        protocol,
        "pre_outcome_instrumentation_history",
        PREFLIGHT_INSTRUMENTATION_HISTORY,
        context="protocol",
    )
    source_digest = _require_sha256(
        protocol.get("source_code_sha256"), context="protocol source_code_sha256"
    )
    if source_digest != _file_sha256(source_code_path.resolve()):
        raise ValueError("confirmation source-code hash does not match the protocol")

    dependencies = protocol.get("dependency_source_sha256")
    if not isinstance(dependencies, Mapping):
        raise ValueError("protocol has no dependency source hashes")
    expected_dependencies = {
        "precision_replication": _file_sha256(PRECISION_SOURCE),
        "certificate_implementation": _file_sha256(CERTIFICATE_SOURCE),
    }
    if dict(dependencies) != expected_dependencies:
        raise ValueError("scientific dependency source hashes changed")

    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("protocol has no dataset declaration")
    dataset_required = {
        "name": "CIFAR-10.2 official test NPZ",
        "filename": EXPECTED_CIFAR102_FILENAME,
        "repository_commit": EXPECTED_CIFAR102_REPOSITORY_COMMIT,
        "byte_size": EXPECTED_CIFAR102_BYTES,
        "git_blob_sha1": EXPECTED_CIFAR102_GIT_BLOB_SHA1,
        "sha256": EXPECTED_CIFAR102_SHA256,
        "images_key": "images",
        "labels_key": "labels",
        "archive_keys": [
            "images",
            "labels",
            "ti_indices",
            "keywords",
            "label_names",
        ],
        "images": TOTAL_IMAGES,
        "images_per_class": IMAGES_PER_CLASS,
        "order": "stored NPZ order, indices 0 through 1999",
    }
    for key, expected in dataset_required.items():
        _require_equal(dataset, key, expected, context="protocol dataset")

    model = protocol.get("auditvotes")
    if not isinstance(model, Mapping):
        raise ValueError("protocol has no AuditVotes declaration")
    model_required = {
        "git_commit": EXPECTED_AUDITVOTES_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "architecture": "cifar_resnet110",
    }
    for key, expected in model_required.items():
        _require_equal(model, key, expected, context="protocol AuditVotes")

    _require_equal(
        protocol,
        "frozen_filter",
        asdict(FrozenFilter()),
        context="protocol",
    )

    sampling = protocol.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("protocol has no sampling declaration")
    sampling_required = {
        "seed": CONFIRMATION_SEED,
        "stream_namespace": STREAM_NAMESPACE,
        "stream_names": list(STREAM_NAMES),
        "filtered_selection_raw_proposals_per_image": LABEL_SELECTION_PROPOSALS,
        "filtered_estimation_raw_proposals_per_image": ESTIMATION_PROPOSALS,
        "unfiltered_selection_raw_proposals_per_image": LABEL_SELECTION_PROPOSALS,
        "unfiltered_estimation_raw_proposals_per_image": ESTIMATION_PROPOSALS,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "four_independent_streams_per_image": True,
        "filter_before_classifier": True,
        "clipping": False,
        "top_up_after_rejection": False,
    }
    for key, expected in sampling_required.items():
        _require_equal(sampling, key, expected, context="protocol sampling")

    confidence = protocol.get("confidence")
    if not isinstance(confidence, Mapping):
        raise ValueError("protocol has no confidence declaration")
    confidence_required = {
        "collection_fwer": COLLECTION_FWER,
        "images": TOTAL_IMAGES,
        "one_sided_tails_per_image": BONFERRONI_TAIL_EVENTS,
        "total_one_sided_tails": TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS,
        "per_image_delta": PER_IMAGE_DELTA,
        "one_sided_tail_error": TAIL_ERROR,
        "tail_decomposition": TAIL_DECOMPOSITION,
        "validity_conditional_on_independently_selected_labels": True,
    }
    for key, expected in confidence_required.items():
        _require_equal(confidence, key, expected, context="protocol confidence")

    success = protocol.get("success_rule")
    if not isinstance(success, Mapping):
        raise ValueError("protocol has no success rule")
    success_required = {
        "all_2000_records_complete": True,
        "mean_retention_interval": [MIN_MEAN_RETENTION, MAX_MEAN_RETENTION],
        "maximum_filtered_accuracy_drop_from_unfiltered": MAX_ACCURACY_DROP,
        "target_radius": TARGET_RADIUS,
        "minimum_covariance_accuracy_advantage_over_joint": (
            MIN_CERTIFIED_ACCURACY_ADVANTAGE
        ),
        "minimum_correct_strong_margin": MIN_STRONG_MARGIN,
        "minimum_correct_strong_images": MIN_CORRECT_STRONG_IMAGES,
        "maximum_one_sided_exact_mcnemar_p": MAX_MCNEMAR_P,
        "all_images_remain_in_denominator": True,
        "strong_endpoint_compares_population_covariance_certificate_radius_to_population_joint_mass_gaussian_radius": True,
        "joint_mass_upper_uses_independently_selected_runner": True,
        "population_largest_competitor_mass_is_at_least_runner_mass": True,
        "strong_margin_is_not_a_lower_bound_on_true_robust_radius_gap": True,
    }
    for key, expected in success_required.items():
        _require_equal(success, key, expected, context="protocol success rule")

    _require_equal(
        protocol,
        "stopping_rule",
        "run all 2000 images without outcome-based early stopping",
        context="protocol",
    )
    proof = protocol.get("fixed_filter_proof_contract")
    if not isinstance(proof, Mapping):
        raise ValueError("protocol has no fixed-filter proof contract")
    proof_required = {
        "global_image_independent_event": True,
        "direction_is_unit_norm": True,
        "direction_float64_sha256": FrozenFilter().direction_float64_sha256,
        "band_rule": "alpha <= abs(<u,z> + b) <= beta",
        "parameter_constraint": "0 < alpha < beta <= sigma",
        "raw_unclipped_proposals": True,
        "filter_applied_before_classifier": True,
        "no_replacement_or_top_up": True,
    }
    for key, expected in proof_required.items():
        _require_equal(proof, key, expected, context="protocol proof contract")

    reporting = protocol.get("reporting_and_interpretation")
    if not isinstance(reporting, Mapping):
        raise ValueError("protocol has no reporting and interpretation contract")
    reporting_required = {
        "report_all_opened_outcomes_even_if_success_rule_fails": True,
        "no_post_opening_filter_threshold_code_or_seed_retuning": True,
        "finite_balanced_benchmark_census": True,
        "macro_average_over_200_images_per_class": True,
        "mcnemar_is_secondary_superpopulation_style_evidence": True,
        "mcnemar_is_not_sampling_uncertainty_for_the_finite_census": True,
        "mcnemar_does_not_establish_generalization_across_systems": True,
        "class_stratified_descriptive_summaries_required": True,
        "full_training_side_adaptive_history_must_be_disclosed": True,
        "failed_global_v1_decision_must_be_disclosed": True,
        "passed_precision_result_and_hash_must_be_disclosed": True,
        "pre_outcome_instrumentation_failure_must_be_disclosed": True,
        "external_holdout_clean_only_if_frozen_before_opening": True,
        "resume_tail_recovery": (
            "discard at most one unterminated final byte fragment; never "
            "discard a newline-terminated record"
        ),
    }
    for key, expected in reporting_required.items():
        _require_equal(
            reporting, key, expected, context="protocol reporting contract"
        )
    development = protocol.get("development_precision_result")
    if not isinstance(development, Mapping):
        raise ValueError("protocol has no development precision result pin")
    _require_equal(
        development,
        "required_status",
        DEVELOPMENT_PASS_STATUS,
        context="protocol development result",
    )
    _require_equal(
        development,
        "required_passed_predeclared_success_rule",
        True,
        context="protocol development result",
    )
    return _require_sha256(
        development.get("sha256"),
        context="protocol development precision result SHA-256",
    )


def load_frozen_protocol(
    protocol_path: Path,
    development_result_path: Path,
    *,
    source_code_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the protocol and its successful, hash-pinned prerequisite."""

    source = Path(__file__) if source_code_path is None else source_code_path
    protocol_resolved = protocol_path.resolve()
    development_resolved = development_result_path.resolve()
    if not protocol_resolved.is_file():
        raise FileNotFoundError(protocol_resolved)
    protocol = _read_json_object(protocol_resolved)
    expected_development_digest = _validate_protocol_contract(
        protocol, source_code_path=source
    )
    if not development_resolved.is_file():
        raise FileNotFoundError(development_resolved)
    if _file_sha256(development_resolved) != expected_development_digest:
        raise ValueError("development precision result does not match protocol hash")
    development = _read_json_object(development_resolved)
    validate_development_precision_result(development)
    return protocol, development


def confirmation_stream_seed(image_index: int, stream: str) -> int:
    if not 0 <= int(image_index) < TOTAL_IMAGES:
        raise ValueError("confirmation image index must be in 0..1999")
    if stream not in STREAM_NAMES:
        raise ValueError("unknown confirmation stream")
    return stable_seed(STREAM_NAMESPACE, CONFIRMATION_SEED, int(image_index), stream)


def verify_auditvotes_assets(
    repo_root: Path, checkpoint: Path, development: Mapping[str, Any]
) -> dict[str, str]:
    """Verify the checkout, checkpoint, and model-relevant source files."""

    if _git_head(repo_root) != EXPECTED_AUDITVOTES_COMMIT:
        raise RuntimeError("AuditVotes repository is not at the pinned commit")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if _file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("AuditVotes checkpoint does not match the released artifact")
    assets = development["assets"]
    observed: dict[str, str] = {}
    for key, relative_path in AUDITVOTES_CODE_FILES.items():
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _file_sha256(path)
        if digest != assets[key]:
            raise RuntimeError(f"AuditVotes source asset changed: {relative_path}")
        observed[key] = digest
    return observed


def _run_filtered_stream(
    *,
    model: Any,
    torch: Any,
    image: Any,
    direction_device: Any,
    image_index: int,
    proposals: int,
    device: Any,
    stream: str,
) -> StreamCounts:
    generator = torch.Generator(device=device)
    generator.manual_seed(confirmation_stream_seed(image_index, stream))
    image_device = image.unsqueeze(0).to(device)
    counts = np.zeros(CLASS_COUNT, dtype=np.int64)
    model_calls = 0
    completed = 0
    with torch.inference_mode():
        while completed < proposals:
            size = min(INFERENCE_BATCH_SIZE, proposals - completed)
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
    if completed != proposals or int(np.sum(counts)) != model_calls:
        raise RuntimeError("filtered stream accounting failed")
    return StreamCounts(tuple(int(value) for value in counts), completed, model_calls)


def _run_unfiltered_stream(
    *,
    model: Any,
    torch: Any,
    image: Any,
    image_index: int,
    proposals: int,
    device: Any,
    stream: str,
) -> StreamCounts:
    generator = torch.Generator(device=device)
    generator.manual_seed(confirmation_stream_seed(image_index, stream))
    image_device = image.unsqueeze(0).to(device)
    counts = np.zeros(CLASS_COUNT, dtype=np.int64)
    completed = 0
    with torch.inference_mode():
        while completed < proposals:
            size = min(INFERENCE_BATCH_SIZE, proposals - completed)
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
    if completed != proposals or int(np.sum(counts)) != completed:
        raise RuntimeError("unfiltered stream accounting failed")
    return StreamCounts(tuple(int(value) for value in counts), completed, completed)


def _load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"unterminated JSONL row {line_number} in {path}")
            if not line.strip():
                raise ValueError(f"blank JSONL row {line_number} in {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL row {line_number} in {path}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {line_number} in {path}")
            rows.append(row)
    return rows


def recover_unterminated_jsonl_tail(path: Path) -> int:
    """Discard only an uncommitted final fragment and return its byte count.

    A row is committed only once its terminating newline has been written and
    fsynced.  Therefore an interrupted final fragment is never an outcome row.
    No newline-terminated content is changed, and all surviving rows still pass
    the manifest, ordering, and accounting checks before execution resumes.
    """

    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return 0
    last_newline = payload.rfind(b"\n")
    committed_length = last_newline + 1
    discarded = len(payload) - committed_length
    with path.open("r+b") as handle:
        handle.truncate(committed_length)
        handle.flush()
        os.fsync(handle.fileno())
    return discarded


def _valid_count_vector(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == CLASS_COUNT
        and all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in value)
    )


def validate_resume_prefix(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    *,
    manifest_sha256: str,
) -> None:
    """Validate exact stored-order prefix and all completed-stream accounting."""

    if len(rows) > len(labels):
        raise ValueError("resume contains more rows than the frozen dataset")
    for position, row in enumerate(rows):
        if row.get("ordinal") != position + 1 or row.get("image_index") != position:
            raise ValueError("resume rows are not an exact prefix of indices 0..1999")
        if row.get("true_label") != int(labels[position]):
            raise ValueError("resume true label differs from the frozen dataset")
        if row.get("manifest_sha256") != manifest_sha256:
            raise ValueError("resume row belongs to another immutable manifest")
        if row.get("record_complete") is not True:
            raise ValueError("only fully completed records may enter the resume log")
        for field in (
            "filtered_label_selection_counts",
            "unfiltered_label_selection_counts",
            "filtered_estimation_counts",
            "unfiltered_estimation_counts",
        ):
            if not _valid_count_vector(row.get(field)):
                raise ValueError(f"resume row has invalid {field}")
        accounting = (
            (
                "filtered_label_selection",
                LABEL_SELECTION_PROPOSALS,
                sum(row["filtered_label_selection_counts"]),
            ),
            (
                "unfiltered_label_selection",
                LABEL_SELECTION_PROPOSALS,
                LABEL_SELECTION_PROPOSALS,
            ),
            (
                "filtered_estimation",
                ESTIMATION_PROPOSALS,
                sum(row["filtered_estimation_counts"]),
            ),
            (
                "unfiltered_estimation",
                ESTIMATION_PROPOSALS,
                ESTIMATION_PROPOSALS,
            ),
        )
        for prefix, raw_expected, calls_expected in accounting:
            if row.get(f"{prefix}_raw_proposals") != raw_expected:
                raise ValueError(f"resume row changed {prefix} raw proposal budget")
            if row.get(f"{prefix}_model_calls") != calls_expected:
                raise ValueError(f"resume row has inconsistent {prefix} model calls")
        if sum(row["unfiltered_label_selection_counts"]) != LABEL_SELECTION_PROPOSALS:
            raise ValueError("resume unfiltered selection counts are incomplete")
        if sum(row["unfiltered_estimation_counts"]) != ESTIMATION_PROPOSALS:
            raise ValueError("resume unfiltered estimation counts are incomplete")


def build_run_manifest(
    *,
    protocol_path: Path,
    development_result_path: Path,
    dataset_path: Path,
    cifar102_repository_root: Path,
    auditvotes_repository_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "status": "frozen_cifar102_confirmation_run_manifest",
        "protocol_sha256": _file_sha256(protocol_path.resolve()),
        "source_code_sha256": _file_sha256(Path(__file__).resolve()),
        "precision_replication_source_sha256": _file_sha256(PRECISION_SOURCE),
        "certificate_implementation_source_sha256": _file_sha256(
            CERTIFICATE_SOURCE
        ),
        "development_precision_result_sha256": _file_sha256(
            development_result_path.resolve()
        ),
        "cifar102_repository_commit": _git_head(cifar102_repository_root.resolve()),
        "cifar102_npz_sha256": _file_sha256(dataset_path.resolve()),
        "cifar102_npz_git_blob_sha1": _git_blob_sha1(dataset_path.resolve()),
        "cifar102_npz_bytes": dataset_path.resolve().stat().st_size,
        "auditvotes_git_commit": _git_head(auditvotes_repository_root.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path.resolve()),
        "auditvotes_code_sha256": {
            key: _file_sha256(auditvotes_repository_root.resolve() / relative_path)
            for key, relative_path in AUDITVOTES_CODE_FILES.items()
        },
        "dataset_order": "all stored NPZ indices 0 through 1999",
        "total_images": TOTAL_IMAGES,
        "sampling_seed": CONFIRMATION_SEED,
        "stream_namespace": STREAM_NAMESPACE,
        "stream_names": list(STREAM_NAMES),
        "label_selection_raw_proposals_per_stream_per_image": (
            LABEL_SELECTION_PROPOSALS
        ),
        "estimation_raw_proposals_per_stream_per_image": ESTIMATION_PROPOSALS,
        "collection_fwer": COLLECTION_FWER,
        "one_sided_tails": TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS,
        "no_outcome_based_early_stopping": True,
        "mandatory_reporting_after_any_external_opening": True,
        "resume_tail_recovery": (
            "discard at most one unterminated final byte fragment; never "
            "discard a newline-terminated record"
        ),
    }
    return {**identity, "run_identity_sha256": _canonical_digest(identity)}


def prepare_run_manifest(
    output_directory: Path,
    expected_manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[Path, str]:
    """Create or verify the immutable manifest and return its file digest."""

    output = output_directory.resolve()
    manifest_path = output / "manifest.json"
    rows_path = output / "per_image.jsonl"
    summary_path = output / "summary.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("resume requires the immutable manifest")
        observed = _read_json_object(manifest_path)
        if observed != dict(expected_manifest):
            raise ValueError("resume manifest does not match current run identity")
    else:
        if manifest_path.exists() or rows_path.exists() or summary_path.exists():
            raise FileExistsError("confirmation output exists; pass --resume")
        output.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, expected_manifest, exclusive=True)
    return manifest_path, _file_sha256(manifest_path)


def summarize_confirmation_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_records: int = TOTAL_IMAGES
) -> dict[str, Any]:
    """Apply the frozen success rule with every dataset image in denominator."""

    if expected_records < 1 or len(rows) != expected_records:
        raise ValueError("confirmation must contain the complete frozen dataset")
    indices = [row.get("image_index") for row in rows]
    if indices != list(range(expected_records)):
        raise ValueError("confirmation rows must be indices 0..N-1 in fixed order")

    complete = np.asarray([row.get("record_complete") is True for row in rows])
    true_labels = np.asarray([int(row["true_label"]) for row in rows])
    if np.any((true_labels < 0) | (true_labels >= CLASS_COUNT)):
        raise ValueError("confirmation rows contain a label outside 0..9")
    if np.bincount(true_labels, minlength=CLASS_COUNT).tolist() != (
        [expected_records // CLASS_COUNT] * CLASS_COUNT
    ):
        raise ValueError("confirmation rows are not the balanced ten-class census")
    filtered_correct = np.asarray([bool(row["filtered_correct"]) for row in rows])
    unfiltered_correct = np.asarray(
        [bool(row["unfiltered_correct"]) for row in rows]
    )
    r_cov = np.asarray([float(row["r_cov_L"]) for row in rows])
    r_mass = np.asarray([float(row["r_mass_L"]) for row in rows])
    r_unfiltered = np.asarray([float(row["r_unfiltered_L"]) for row in rows])
    retention = np.asarray(
        [float(row["estimation_retention_rate"]) for row in rows]
    )
    selection_retention = np.asarray(
        [float(row["label_selection_retention_rate"]) for row in rows]
    )
    if not np.all(np.isfinite(retention)) or np.any((retention < 0) | (retention > 1)):
        raise ValueError("estimation retention rates must be finite probabilities")
    if not np.all(np.isfinite(selection_retention)) or np.any(
        (selection_retention < 0) | (selection_retention > 1)
    ):
        raise ValueError("selection retention rates must be finite probabilities")

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

    filtered_raw = int(
        sum(
            int(row["filtered_label_selection_raw_proposals"])
            + int(row["filtered_estimation_raw_proposals"])
            for row in rows
        )
    )
    filtered_calls = int(
        sum(
            int(row["filtered_label_selection_model_calls"])
            + int(row["filtered_estimation_model_calls"])
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
    unfiltered_calls = int(
        sum(
            int(row["unfiltered_label_selection_model_calls"])
            + int(row["unfiltered_estimation_model_calls"])
            for row in rows
        )
    )
    mcnemar = exact_one_sided_mcnemar(covariance_success, joint_success)
    class_stratified = {}
    for label in range(CLASS_COUNT):
        mask = true_labels == label
        class_stratified[str(label)] = {
            "class_name": CLASS_NAMES[label],
            "images": int(np.sum(mask)),
            "mean_estimation_retention": float(np.mean(retention[mask])),
            "filtered_selection_accuracy": float(np.mean(filtered_correct[mask])),
            "unfiltered_selection_accuracy": float(
                np.mean(unfiltered_correct[mask])
            ),
            "covariance_correct_certified_fraction_at_radius_0_20": float(
                np.mean(covariance_success[mask])
            ),
            "joint_correct_certified_fraction_at_radius_0_20": float(
                np.mean(joint_success[mask])
            ),
            "correct_strong_margin_count": int(np.sum(correct_strong[mask])),
        }
    observed = {
        "records": len(rows),
        "denominator": len(rows),
        "all_records_complete": bool(np.all(complete)),
        "mean_estimation_retention": float(np.mean(retention)),
        "mean_label_selection_retention": float(np.mean(selection_retention)),
        "min_estimation_retention": float(np.min(retention)),
        "max_estimation_retention": float(np.max(retention)),
        "filtered_selection_correct": int(np.sum(filtered_correct)),
        "filtered_selection_accuracy": float(np.mean(filtered_correct)),
        "unfiltered_selection_correct": int(np.sum(unfiltered_correct)),
        "unfiltered_selection_accuracy": float(np.mean(unfiltered_correct)),
        "covariance_correct_certified_count_at_radius_0_20": int(
            np.sum(covariance_success)
        ),
        "covariance_correct_certified_fraction_at_radius_0_20": float(
            np.mean(covariance_success)
        ),
        "joint_correct_certified_count_at_radius_0_20": int(np.sum(joint_success)),
        "joint_correct_certified_fraction_at_radius_0_20": float(
            np.mean(joint_success)
        ),
        "unfiltered_correct_certified_count_at_radius_0_20": int(
            np.sum(unfiltered_success)
        ),
        "unfiltered_correct_certified_fraction_at_radius_0_20": float(
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
        "class_stratified_descriptive_summaries": class_stratified,
    }
    checks = {
        "all_2000_records_complete": observed["all_records_complete"],
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
        "covariance_correct_certified_fraction_at_least_0_05_above_joint": (
            observed["covariance_correct_certified_fraction_at_radius_0_20"]
            >= observed["joint_correct_certified_fraction_at_radius_0_20"]
            + MIN_CERTIFIED_ACCURACY_ADVANTAGE
        ),
        "at_least_100_correct_images_have_covariance_lower_above_mass_upper_by_0_01": (
            observed["correct_strong_margin_count"] >= MIN_CORRECT_STRONG_IMAGES
        ),
        "one_sided_exact_mcnemar_p_at_most_0_001": (
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
            "minimum_covariance_over_joint_certified_fraction": (
                MIN_CERTIFIED_ACCURACY_ADVANTAGE
            ),
            "minimum_correct_strong_margin": MIN_STRONG_MARGIN,
            "minimum_correct_strong_images": MIN_CORRECT_STRONG_IMAGES,
            "maximum_exact_one_sided_mcnemar_p": MAX_MCNEMAR_P,
            "denominator": expected_records,
        },
        "observed": observed,
        "interpretation": {
            "estimand": (
                "finite balanced CIFAR-10.2 benchmark census; the overall "
                "fraction is also the ten-class macro-average because each "
                "class contributes exactly 200 images"
            ),
            "mcnemar_role": "secondary superpopulation-style paired evidence",
            "mcnemar_is_sampling_uncertainty_for_finite_census": False,
            "generalizes_across_models_filters_or_systems": False,
            "strong_margin_endpoint": (
                "population covariance-certificate radius versus the standard "
                "population joint-mass Gaussian radius for the independently "
                "selected filtered label and runner"
            ),
            "strong_margin_is_true_robust_radius_gap_lower_bound": False,
            "runner_upper_bound_reason": (
                "the largest population competitor mass is at least the mass "
                "of the independently selected runner, so the runner lower "
                "confidence bound yields a conservative r_mass_U"
            ),
        },
    }


def _load_model(repo_root: Path, checkpoint: Path):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the released AuditVotes architecture requires CUDA")
    code_root = repo_root / "ImageClassify_Gaussian/ImageClassify_Conf/code"
    sys.path.insert(0, str(code_root))
    from architectures import get_architecture  # type: ignore

    torch.manual_seed(CONFIRMATION_SEED)
    torch.cuda.manual_seed_all(CONFIRMATION_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("arch") != "cifar_resnet110":
        raise RuntimeError("expected the released CIFAR-10 ResNet-110")
    model = get_architecture(payload["arch"], "cifar10")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, device, torch


def _evaluate_one(
    *,
    model: Any,
    torch: Any,
    image: Any,
    true_label: int,
    image_index: int,
    direction_device: Any,
    device: Any,
    manifest_sha256: str,
) -> dict[str, Any]:
    filtered_selection = _run_filtered_stream(
        model=model,
        torch=torch,
        image=image,
        direction_device=direction_device,
        image_index=image_index,
        proposals=LABEL_SELECTION_PROPOSALS,
        device=device,
        stream="filtered_label_selection",
    )
    unfiltered_selection = _run_unfiltered_stream(
        model=model,
        torch=torch,
        image=image,
        image_index=image_index,
        proposals=LABEL_SELECTION_PROPOSALS,
        device=device,
        stream="unfiltered_label_selection",
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
        device=device,
        stream="filtered_estimation",
    )
    unfiltered_estimation = _run_unfiltered_stream(
        model=model,
        torch=torch,
        image=image,
        image_index=image_index,
        proposals=ESTIMATION_PROPOSALS,
        device=device,
        stream="unfiltered_estimation",
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
    if not complete:
        raise RuntimeError("refusing to append an incomplete confirmation record")
    return {
        "ordinal": image_index + 1,
        "image_index": image_index,
        "true_label": int(true_label),
        "true_class_name": CLASS_NAMES[int(true_label)],
        "manifest_sha256": manifest_sha256,
        "record_complete": True,
        "stream_seeds": {
            name: confirmation_stream_seed(image_index, name)
            for name in STREAM_NAMES
        },
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
        "unfiltered_label_selection_model_calls": unfiltered_selection.model_calls,
        "filtered_estimation_raw_proposals": filtered_estimation.raw_proposals,
        "filtered_estimation_model_calls": filtered_estimation.model_calls,
        "unfiltered_estimation_raw_proposals": unfiltered_estimation.raw_proposals,
        "unfiltered_estimation_model_calls": unfiltered_estimation.model_calls,
        "label_selection_retention_rate": (
            filtered_selection.model_calls / LABEL_SELECTION_PROPOSALS
        ),
        "estimation_retention_rate": (
            filtered_estimation.model_calls / ESTIMATION_PROPOSALS
        ),
        "raw_filter_applied_before_classifier": True,
        "clipping_performed": False,
        "top_up_proposals_drawn": False,
        **radii,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--development-result", type=Path, default=DEFAULT_DEVELOPMENT_RESULT
    )
    parser.add_argument("--cifar102-repo-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--auditvotes-repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    development_path = args.development_result.resolve()
    cifar102_repo = args.cifar102_repo_root.resolve()
    dataset_path = args.dataset.resolve()
    auditvotes_repo = args.auditvotes_repo_root.resolve()
    checkpoint = args.checkpoint.resolve()

    # Every protocol and provenance gate occurs before model loading/execution.
    protocol, development = load_frozen_protocol(
        protocol_path, development_path
    )
    if _git_head(cifar102_repo) != EXPECTED_CIFAR102_REPOSITORY_COMMIT:
        raise RuntimeError("CIFAR-10.2 repository is not at the pinned commit")
    verify_auditvotes_assets(auditvotes_repo, checkpoint, development)
    direction = reconstruct_frozen_direction()
    dataset = load_cifar102_npz(dataset_path)

    manifest = build_run_manifest(
        protocol_path=protocol_path,
        development_result_path=development_path,
        dataset_path=dataset_path,
        cifar102_repository_root=cifar102_repo,
        auditvotes_repository_root=auditvotes_repo,
        checkpoint_path=checkpoint,
    )
    manifest_path, manifest_digest = prepare_run_manifest(
        args.output, manifest, resume=args.resume
    )
    rows_path = args.output.resolve() / "per_image.jsonl"
    existing: list[dict[str, Any]] = []
    discarded_tail_bytes = 0
    if args.resume and rows_path.exists():
        discarded_tail_bytes = recover_unterminated_jsonl_tail(rows_path)
        if discarded_tail_bytes:
            print(
                json.dumps(
                    {
                        "resume_recovery": "discarded_unterminated_final_fragment",
                        "discarded_bytes": discarded_tail_bytes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        existing = _load_jsonl_strict(rows_path)
        validate_resume_prefix(
            existing, dataset.labels, manifest_sha256=manifest_digest
        )
    elif args.resume and not rows_path.exists():
        existing = []

    model, device, torch = _load_model(auditvotes_repo, checkpoint)
    direction_device = torch.as_tensor(
        direction.reshape(-1), dtype=torch.float64, device=device
    )
    started = time.monotonic()
    mode = "a" if rows_path.exists() else "x"
    with rows_path.open(mode, encoding="utf-8") as handle:
        for image_index in range(len(existing), TOTAL_IMAGES):
            image, true_label = dataset[image_index]
            row = _evaluate_one(
                model=model,
                torch=torch,
                image=image,
                true_label=true_label,
                image_index=image_index,
                direction_device=direction_device,
                device=device,
                manifest_sha256=manifest_digest,
            )
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                json.dumps(
                    {
                        "completed": image_index + 1,
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

    rows = _load_jsonl_strict(rows_path)
    validate_resume_prefix(rows, dataset.labels, manifest_sha256=manifest_digest)
    summary = summarize_confirmation_rows(rows)
    result = {
        "status": (
            "external_confirmation_passed_predeclared_success_rule"
            if summary["passed_predeclared_success_rule"]
            else "external_confirmation_failed_predeclared_success_rule"
        ),
        "paper_eligibility": PAPER_ELIGIBILITY,
        "study_role": STUDY_ROLE,
        "protocol_sha256": _file_sha256(protocol_path),
        "development_precision_result_sha256": _file_sha256(development_path),
        "development_precision_status": development["status"],
        "development_history_disclosure": {
            "failed_global_v1_source": development["source_result"],
            "all_pinned_training_side_filter_artifacts": development[
                "all_pinned_learned_filter_results"
            ],
            "passed_precision_result": {
                "status": development["status"],
                "sha256": _file_sha256(development_path),
            },
            "full_adaptive_train_side_history_must_be_reported": True,
        },
        "pre_outcome_instrumentation_history": protocol[
            "pre_outcome_instrumentation_history"
        ],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "per_image_jsonl_sha256": _file_sha256(rows_path),
        "all_images_in_official_fixed_order": True,
        "all_images_remain_in_denominator": True,
        "all_records_run_without_outcome_based_early_stopping": True,
        "raw_filter_applied_before_classifier": True,
        "clipping_performed": False,
        "top_up_proposals_drawn": False,
        "fixed_filter_proof_contract": protocol["fixed_filter_proof_contract"],
        "confidence": {
            "collection_fwer": COLLECTION_FWER,
            "images": TOTAL_IMAGES,
            "one_sided_tails_per_image": BONFERRONI_TAIL_EVENTS,
            "total_one_sided_tails": (
                TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS
            ),
            "per_image_delta": PER_IMAGE_DELTA,
            "one_sided_tail_error": TAIL_ERROR,
            "tail_decomposition": TAIL_DECOMPOSITION,
            "validity_conditional_on_independently_selected_labels": True,
            "allocated_error": (
                TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS * TAIL_ERROR
            ),
        },
        "predeclared_success_rule": summary,
        "reporting_and_interpretation": protocol[
            "reporting_and_interpretation"
        ],
        "paper_facing_disclosures": [
            "Report all 2,000 opened outcomes even when the success rule fails.",
            "This is a finite balanced benchmark census, not a random sample from CIFAR-10.2.",
            "McNemar is secondary superpopulation-style paired evidence, not sampling uncertainty for this finite census or evidence across systems.",
            "Disclose the full training-side adaptive search history, the failed global V1 decision, and the passed hash-pinned precision result.",
            "External-holdout cleanliness relies on freezing filter, thresholds, code, and seeds before opening CIFAR-10.2.",
            "The strong margin is not a lower confidence bound on a true robust-radius gap.",
        ],
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "resume_unterminated_tail_bytes_discarded_this_invocation": (
            discarded_tail_bytes
        ),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
    }
    summary_path = args.output.resolve() / "summary.json"
    _write_json(summary_path, result, exclusive=not summary_path.exists())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
