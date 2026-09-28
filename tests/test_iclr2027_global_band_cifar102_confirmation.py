"""Pure tests for the frozen CIFAR-10.2 external-confirmation runner."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.iclr2027_global_band_cifar102_confirmation import (
    BONFERRONI_TAIL_EVENTS,
    CERTIFICATE_SOURCE,
    COLLECTION_FWER,
    CONFIRMATION_SEED,
    DEVELOPMENT_PASS_STATUS,
    ESTIMATION_PROPOSALS,
    EXPECTED_AUDITVOTES_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_LEARNED_BAND_RESULT_SHA256,
    EXPECTED_CIFAR102_BYTES,
    EXPECTED_CIFAR102_GIT_BLOB_SHA1,
    EXPECTED_CIFAR102_REPOSITORY_COMMIT,
    EXPECTED_CIFAR102_SHA256,
    EXPECTED_SOURCE_RESULT_SHA256,
    EXPECTED_STABLE_BAND_RESULT_SHA256,
    FrozenFilter,
    IMAGES_PER_CLASS,
    LABEL_SELECTION_PROPOSALS,
    MAX_MCNEMAR_P,
    MIN_CORRECT_STRONG_IMAGES,
    PAPER_ELIGIBILITY,
    PER_IMAGE_DELTA,
    PREFLIGHT_INSTRUMENTATION_HISTORY,
    PRECISION_SOURCE,
    PROTOCOL_STATUS,
    STREAM_NAMESPACE,
    STREAM_NAMES,
    STUDY_ROLE,
    TAIL_ERROR,
    TOTAL_IMAGES,
    _file_sha256,
    _git_blob_sha1,
    confirmation_stream_seed,
    load_cifar102_npz,
    load_frozen_protocol,
    prepare_run_manifest,
    recover_unterminated_jsonl_tail,
    summarize_confirmation_rows,
    validate_resume_prefix,
)
from experiments.iclr2027_global_stable_band_filter_development import stable_seed


RUNNER_SOURCE = Path("experiments/iclr2027_global_band_cifar102_confirmation.py")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _synthetic_npz(
    path: Path,
    *,
    images: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    ti_indices: np.ndarray | None = None,
    keywords: np.ndarray | None = None,
    label_names: np.ndarray | None = None,
    extra: bool = False,
) -> tuple[int, str, str]:
    if images is None:
        images = np.zeros((20, 32, 32, 3), dtype=np.uint8)
        images[:, 0, 0, 0] = np.arange(20, dtype=np.uint8)
    if labels is None:
        labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    if ti_indices is None:
        ti_indices = np.arange(images.shape[0], dtype=np.int64)
    if keywords is None:
        keywords = np.asarray(
            [f"keyword_{index}" for index in range(images.shape[0])]
        )
    if label_names is None:
        label_names = np.asarray(
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
    if extra:
        np.savez(
            path,
            images=images,
            labels=labels,
            ti_indices=ti_indices,
            keywords=keywords,
            label_names=label_names,
            metadata=np.asarray([1]),
        )
    else:
        np.savez(
            path,
            images=images,
            labels=labels,
            ti_indices=ti_indices,
            keywords=keywords,
            label_names=label_names,
        )
    return path.stat().st_size, _file_sha256(path), _git_blob_sha1(path)


def _development_payload() -> dict[str, object]:
    return {
        "status": DEVELOPMENT_PASS_STATUS,
        "paper_eligibility": "none",
        "dataset": "CIFAR-10 training split only",
        "test_sets_accessed": [],
        "all_records_run_without_interim_stopping": True,
        "raw_proposals_before_clipping": True,
        "clipping_performed": False,
        "replacement_proposals_drawn": False,
        "frozen_filter": asdict(FrozenFilter()),
        "predeclared_success_rule": {
            "passed_predeclared_success_rule": True,
            "checks": {"all_200_records_complete": True, "gate": True},
            "observed": {"records": 200, "all_records_complete": True},
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
        },
        "confidence": {
            "collection_fwer": 0.001,
            "images": 200,
            "one_sided_tails_per_image": BONFERRONI_TAIL_EVENTS,
        },
        "assets": {
            "auditvotes_git_commit": EXPECTED_AUDITVOTES_COMMIT,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "core_py_sha256": "a" * 64,
            "architectures_py_sha256": "b" * 64,
            "datasets_py_sha256": "c" * 64,
            "cifar_resnet_py_sha256": "d" * 64,
        },
        "source_result": {
            "sha256": EXPECTED_SOURCE_RESULT_SHA256,
            "source_status": "open_development_failed_predeclared_gate",
        },
        "all_pinned_learned_filter_results": [
            {"sha256": EXPECTED_SOURCE_RESULT_SHA256},
            {"sha256": EXPECTED_LEARNED_BAND_RESULT_SHA256},
            {"sha256": EXPECTED_STABLE_BAND_RESULT_SHA256},
        ],
    }


def _protocol_payload(development_sha256: str) -> dict[str, object]:
    return {
        "status": PROTOCOL_STATUS,
        "paper_eligibility": PAPER_ELIGIBILITY,
        "study_role": STUDY_ROLE,
        "pre_outcome_instrumentation_history": PREFLIGHT_INSTRUMENTATION_HISTORY,
        "source_code_sha256": _file_sha256(RUNNER_SOURCE),
        "dependency_source_sha256": {
            "precision_replication": _file_sha256(PRECISION_SOURCE),
            "certificate_implementation": _file_sha256(CERTIFICATE_SOURCE),
        },
        "development_precision_result": {
            "sha256": development_sha256,
            "required_status": DEVELOPMENT_PASS_STATUS,
            "required_passed_predeclared_success_rule": True,
        },
        "dataset": {
            "name": "CIFAR-10.2 official test NPZ",
            "filename": "cifar102_test.npz",
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
        },
        "auditvotes": {
            "git_commit": EXPECTED_AUDITVOTES_COMMIT,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "architecture": "cifar_resnet110",
        },
        "frozen_filter": asdict(FrozenFilter()),
        "sampling": {
            "seed": CONFIRMATION_SEED,
            "stream_namespace": STREAM_NAMESPACE,
            "stream_names": list(STREAM_NAMES),
            "filtered_selection_raw_proposals_per_image": (
                LABEL_SELECTION_PROPOSALS
            ),
            "filtered_estimation_raw_proposals_per_image": ESTIMATION_PROPOSALS,
            "unfiltered_selection_raw_proposals_per_image": (
                LABEL_SELECTION_PROPOSALS
            ),
            "unfiltered_estimation_raw_proposals_per_image": ESTIMATION_PROPOSALS,
            "inference_batch_size": 1024,
            "four_independent_streams_per_image": True,
            "filter_before_classifier": True,
            "clipping": False,
            "top_up_after_rejection": False,
        },
        "confidence": {
            "collection_fwer": COLLECTION_FWER,
            "images": TOTAL_IMAGES,
            "one_sided_tails_per_image": BONFERRONI_TAIL_EVENTS,
            "total_one_sided_tails": TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS,
            "per_image_delta": PER_IMAGE_DELTA,
            "one_sided_tail_error": TAIL_ERROR,
            "tail_decomposition": {
                "conditional_selected_lower_and_nine_competitor_uppers": 10,
                "joint_selected_lower_and_nine_competitor_uppers": 10,
                "joint_selected_upper_and_independently_selected_runner_lower": 2,
                "unfiltered_selected_lower_and_nine_competitor_uppers": 10,
            },
            "validity_conditional_on_independently_selected_labels": True,
        },
        "success_rule": {
            "all_2000_records_complete": True,
            "mean_retention_interval": [0.25, 0.60],
            "maximum_filtered_accuracy_drop_from_unfiltered": 0.01,
            "target_radius": 0.20,
            "minimum_covariance_accuracy_advantage_over_joint": 0.05,
            "minimum_correct_strong_margin": 0.01,
            "minimum_correct_strong_images": MIN_CORRECT_STRONG_IMAGES,
            "maximum_one_sided_exact_mcnemar_p": MAX_MCNEMAR_P,
            "all_images_remain_in_denominator": True,
            "strong_endpoint_compares_population_covariance_certificate_radius_to_population_joint_mass_gaussian_radius": True,
            "joint_mass_upper_uses_independently_selected_runner": True,
            "population_largest_competitor_mass_is_at_least_runner_mass": True,
            "strong_margin_is_not_a_lower_bound_on_true_robust_radius_gap": True,
        },
        "fixed_filter_proof_contract": {
            "global_image_independent_event": True,
            "direction_is_unit_norm": True,
            "direction_float64_sha256": FrozenFilter().direction_float64_sha256,
            "band_rule": "alpha <= abs(<u,z> + b) <= beta",
            "parameter_constraint": "0 < alpha < beta <= sigma",
            "raw_unclipped_proposals": True,
            "filter_applied_before_classifier": True,
            "no_replacement_or_top_up": True,
        },
        "reporting_and_interpretation": {
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
        },
        "stopping_rule": (
            "run all 2000 images without outcome-based early stopping"
        ),
    }


def _passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(TOTAL_IMAGES):
        covariance_success = index < 120
        joint_success = index < 10
        rows.append(
            {
                "image_index": index,
                "true_label": index % 10,
                "record_complete": True,
                "filtered_correct": True,
                "unfiltered_correct": True,
                "r_cov_L": 0.21 if covariance_success else 0.19,
                "r_mass_L": 0.21 if joint_success else 0.10,
                "r_mass_U": 0.19,
                "r_unfiltered_L": 0.21 if index < 100 else 0.19,
                "estimation_retention_rate": 0.40,
                "label_selection_retention_rate": 0.40,
                "filtered_label_selection_model_calls": 4_000,
                "filtered_estimation_model_calls": 40_000,
                "unfiltered_label_selection_model_calls": (
                    LABEL_SELECTION_PROPOSALS
                ),
                "unfiltered_estimation_model_calls": ESTIMATION_PROPOSALS,
                "filtered_label_selection_raw_proposals": (
                    LABEL_SELECTION_PROPOSALS
                ),
                "filtered_estimation_raw_proposals": ESTIMATION_PROPOSALS,
                "unfiltered_label_selection_raw_proposals": (
                    LABEL_SELECTION_PROPOSALS
                ),
                "unfiltered_estimation_raw_proposals": ESTIMATION_PROPOSALS,
            }
        )
    return rows


def _resume_row(index: int, label: int, manifest: str) -> dict[str, object]:
    zeroes = [0] * 9
    return {
        "ordinal": index + 1,
        "image_index": index,
        "true_label": label,
        "manifest_sha256": manifest,
        "record_complete": True,
        "filtered_label_selection_counts": [4_000, *zeroes],
        "unfiltered_label_selection_counts": [10_000, *zeroes],
        "filtered_estimation_counts": [40_000, *zeroes],
        "unfiltered_estimation_counts": [100_000, *zeroes],
        "filtered_label_selection_raw_proposals": LABEL_SELECTION_PROPOSALS,
        "filtered_label_selection_model_calls": 4_000,
        "unfiltered_label_selection_raw_proposals": LABEL_SELECTION_PROPOSALS,
        "unfiltered_label_selection_model_calls": LABEL_SELECTION_PROPOSALS,
        "filtered_estimation_raw_proposals": ESTIMATION_PROPOSALS,
        "filtered_estimation_model_calls": 40_000,
        "unfiltered_estimation_raw_proposals": ESTIMATION_PROPOSALS,
        "unfiltered_estimation_model_calls": ESTIMATION_PROPOSALS,
    }


def test_synthetic_npz_loader_checks_schema_order_dtype_labels_and_balance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic.npz"
    byte_size, digest, blob_digest = _synthetic_npz(path)
    dataset = load_cifar102_npz(
        path,
        expected_sha256=digest,
        expected_git_blob_sha1=blob_digest,
        expected_bytes=byte_size,
        expected_count=20,
        expected_per_class=2,
        require_official_filename=False,
    )
    assert dataset.images.shape == (20, 32, 32, 3)
    assert dataset.images.dtype == np.uint8
    assert dataset.labels.tolist() == np.repeat(np.arange(10), 2).tolist()
    assert dataset.images[:, 0, 0, 0].tolist() == list(range(20))

    with pytest.raises(ValueError, match="SHA-256"):
        load_cifar102_npz(
            path,
            expected_sha256="f" * 64,
            expected_git_blob_sha1=blob_digest,
            expected_bytes=byte_size,
            expected_count=20,
            expected_per_class=2,
            require_official_filename=False,
        )
    with pytest.raises(ValueError, match="Git blob SHA-1"):
        load_cifar102_npz(
            path,
            expected_sha256=digest,
            expected_git_blob_sha1="f" * 40,
            expected_bytes=byte_size,
            expected_count=20,
            expected_per_class=2,
            require_official_filename=False,
        )
    with pytest.raises(ValueError, match="byte size"):
        load_cifar102_npz(
            path,
            expected_sha256=digest,
            expected_git_blob_sha1=blob_digest,
            expected_bytes=byte_size + 1,
            expected_count=20,
            expected_per_class=2,
            require_official_filename=False,
        )

    extra_path = tmp_path / "extra.npz"
    extra_size, extra_digest, extra_blob = _synthetic_npz(extra_path, extra=True)
    with pytest.raises(ValueError, match="schema"):
        load_cifar102_npz(
            extra_path,
            expected_sha256=extra_digest,
            expected_git_blob_sha1=extra_blob,
            expected_bytes=extra_size,
            expected_count=20,
            expected_per_class=2,
            require_official_filename=False,
        )

    bad_labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    bad_labels[0] = 1
    bad_path = tmp_path / "imbalanced.npz"
    bad_size, bad_digest, bad_blob = _synthetic_npz(bad_path, labels=bad_labels)
    with pytest.raises(ValueError, match="per class"):
        load_cifar102_npz(
            bad_path,
            expected_sha256=bad_digest,
            expected_git_blob_sha1=bad_blob,
            expected_bytes=bad_size,
            expected_count=20,
            expected_per_class=2,
            require_official_filename=False,
        )

    bad_metadata_path = tmp_path / "bad_metadata.npz"
    bad_names = np.asarray(
        [
            "plane",
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
    metadata_size, metadata_digest, metadata_blob = _synthetic_npz(
        bad_metadata_path, label_names=bad_names
    )
    with pytest.raises(ValueError, match="label_names"):
        load_cifar102_npz(
            bad_metadata_path,
            expected_sha256=metadata_digest,
            expected_git_blob_sha1=metadata_blob,
            expected_bytes=metadata_size,
            expected_count=20,
            expected_per_class=2,
            require_official_filename=False,
        )


@pytest.mark.parametrize(
    ("images", "labels", "message"),
    [
        (
            np.zeros((20, 32, 32, 3), dtype=np.float32),
            np.repeat(np.arange(10, dtype=np.int64), 2),
            "uint8",
        ),
        (
            np.zeros((20, 32, 32, 1), dtype=np.uint8),
            np.repeat(np.arange(10, dtype=np.int64), 2),
            "shape",
        ),
        (
            np.zeros((20, 32, 32, 3), dtype=np.uint8),
            np.repeat(np.arange(10, dtype=np.float64), 2),
            "integer dtype",
        ),
        (
            np.zeros((20, 32, 32, 3), dtype=np.uint8),
            np.asarray([*np.repeat(np.arange(9), 2), -1, 9], dtype=np.int64),
            "0..9",
        ),
    ],
)
def test_synthetic_npz_loader_rejects_invalid_arrays(
    tmp_path: Path, images: np.ndarray, labels: np.ndarray, message: str
) -> None:
    path = tmp_path / "invalid.npz"
    byte_size, digest, blob_digest = _synthetic_npz(
        path, images=images, labels=labels
    )
    with pytest.raises(ValueError, match=message):
        load_cifar102_npz(
            path,
            expected_sha256=digest,
            expected_git_blob_sha1=blob_digest,
            expected_bytes=byte_size,
            expected_count=20,
            expected_per_class=2,
            require_official_filename=False,
        )

def test_protocol_requires_source_hash_and_passed_hash_pinned_development(
    tmp_path: Path,
) -> None:
    development_path = tmp_path / "development.json"
    _write_json(development_path, _development_payload())
    protocol_path = tmp_path / "protocol.json"
    protocol = _protocol_payload(_file_sha256(development_path))
    _write_json(protocol_path, protocol)
    loaded, development = load_frozen_protocol(protocol_path, development_path)
    assert loaded["status"] == PROTOCOL_STATUS
    assert development["status"] == DEVELOPMENT_PASS_STATUS

    protocol["source_code_sha256"] = "f" * 64
    _write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="source-code hash"):
        load_frozen_protocol(protocol_path, development_path)

    failed = _development_payload()
    failed["status"] = "development_precision_replication_failed"
    _write_json(development_path, failed)
    protocol = _protocol_payload(_file_sha256(development_path))
    _write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="development result field 'status'"):
        load_frozen_protocol(protocol_path, development_path)


def test_four_confirmation_streams_are_distinct_and_fresh() -> None:
    seeds = {
        confirmation_stream_seed(index, stream)
        for index in range(5)
        for stream in STREAM_NAMES
    }
    assert len(seeds) == 5 * len(STREAM_NAMES)
    assert confirmation_stream_seed(3, STREAM_NAMES[0]) == (
        confirmation_stream_seed(3, STREAM_NAMES[0])
    )
    old = stable_seed(
        "global-band-precision-v1", 26_092_002, 3, "filtered_label_selection"
    )
    assert confirmation_stream_seed(3, "filtered_label_selection") != old
    with pytest.raises(ValueError, match="unknown"):
        confirmation_stream_seed(3, "not-a-stream")


def test_collection_confidence_allocation_is_exact() -> None:
    assert TOTAL_IMAGES == 2_000
    assert BONFERRONI_TAIL_EVENTS == 32
    assert PER_IMAGE_DELTA == pytest.approx(COLLECTION_FWER / TOTAL_IMAGES)
    assert TAIL_ERROR == pytest.approx(
        COLLECTION_FWER / (TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS)
    )
    assert TOTAL_IMAGES * BONFERRONI_TAIL_EVENTS * TAIL_ERROR == pytest.approx(
        COLLECTION_FWER
    )


def test_success_rule_uses_all_2000_images_and_exact_mcnemar() -> None:
    rows = _passing_rows()
    summary = summarize_confirmation_rows(rows)
    assert summary["passed_predeclared_success_rule"] is True
    assert all(summary["checks"].values())
    observed = summary["observed"]
    assert observed["denominator"] == TOTAL_IMAGES
    assert observed["mean_estimation_retention"] == pytest.approx(0.40)
    assert observed["filtered_model_call_fraction"] == pytest.approx(0.40)
    assert observed["unfiltered_model_call_fraction"] == 1.0
    assert observed["correct_strong_margin_count"] == 120
    assert len(observed["class_stratified_descriptive_summaries"]) == 10
    assert all(
        class_row["images"] == 200
        for class_row in observed[
            "class_stratified_descriptive_summaries"
        ].values()
    )
    assert observed["mcnemar"]["covariance_only"] == 110
    assert observed["mcnemar"]["joint_only"] == 0
    assert observed["mcnemar"]["exact_one_sided_p_value"] <= 0.001

    rows[0] = {**rows[0], "record_complete": False}
    failed = summarize_confirmation_rows(rows)
    assert failed["passed_predeclared_success_rule"] is False
    assert failed["checks"]["all_2000_records_complete"] is False


def test_summary_rejects_missing_duplicate_or_reordered_dataset_indices() -> None:
    rows = _passing_rows()
    with pytest.raises(ValueError, match="complete frozen dataset"):
        summarize_confirmation_rows(rows[:-1])
    rows[4] = {**rows[4], "image_index": 3}
    with pytest.raises(ValueError, match="fixed order"):
        summarize_confirmation_rows(rows)


def test_resume_requires_exact_prefix_manifest_and_stream_accounting() -> None:
    manifest = "a" * 64
    labels = [2, 7, 1]
    rows = [_resume_row(0, 2, manifest), _resume_row(1, 7, manifest)]
    validate_resume_prefix(rows, labels, manifest_sha256=manifest)

    wrong_order = [rows[1]]
    with pytest.raises(ValueError, match="exact prefix"):
        validate_resume_prefix(wrong_order, labels, manifest_sha256=manifest)
    wrong_manifest = [{**rows[0], "manifest_sha256": "b" * 64}]
    with pytest.raises(ValueError, match="another immutable manifest"):
        validate_resume_prefix(wrong_manifest, labels, manifest_sha256=manifest)
    wrong_calls = [{**rows[0], "filtered_estimation_model_calls": 39_999}]
    with pytest.raises(ValueError, match="inconsistent"):
        validate_resume_prefix(wrong_calls, labels, manifest_sha256=manifest)


def test_immutable_manifest_must_match_exactly_on_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest = {"status": "frozen", "identity": "abc"}
    manifest_path, digest = prepare_run_manifest(output, manifest, resume=False)
    assert manifest_path.is_file()
    assert digest == _file_sha256(manifest_path)
    _, resumed_digest = prepare_run_manifest(output, manifest, resume=True)
    assert resumed_digest == digest
    with pytest.raises(ValueError, match="run identity"):
        prepare_run_manifest(
            output, {"status": "frozen", "identity": "changed"}, resume=True
        )


def test_resume_recovers_only_an_unterminated_final_fragment(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    committed = b'{"image_index": 0}\n'
    fragment = b'{"image_index": 1'
    rows_path.write_bytes(committed + fragment)
    assert recover_unterminated_jsonl_tail(rows_path) == len(fragment)
    assert rows_path.read_bytes() == committed
    assert recover_unterminated_jsonl_tail(rows_path) == 0


def test_runner_has_no_download_path_and_filters_raw_samples_before_model() -> None:
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "requests." not in source
    assert "urlopen(" not in source
    assert "torchvision.datasets" not in source
    assert 'accepted = raw[retained]' in source
    assert 'model(accepted)' in source
    assert ".clamp(" not in source
    assert "while completed < proposals" in source
    assert "for image_index in range(len(existing), TOTAL_IMAGES)" in source
