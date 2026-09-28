import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.iclr2027_auditvotes_cifar101_fixed_cohort import (
    analyze_confirmation_rows,
    certificates_from_shared_counts,
    choose_attack_alternative,
    cohort_newline_sha256,
    family_allocation,
    hash_ranked_indices,
    load_cifar101_npy,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cifar101_loader_checks_bytes_digest_shape_and_scaling(tmp_path: Path) -> None:
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    images[1, 0, 0] = np.asarray([0, 127, 255], dtype=np.uint8)
    labels = np.asarray([2, 7], dtype=np.int64)
    images_path = tmp_path / "images.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(images_path, images, allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)

    dataset = load_cifar101_npy(
        images_path,
        labels_path,
        expected_images_sha256=_sha256(images_path),
        expected_labels_sha256=_sha256(labels_path),
        expected_images_bytes=images_path.stat().st_size,
        expected_labels_bytes=labels_path.stat().st_size,
        expected_count=2,
    )
    assert dataset.images.shape == (2, 32, 32, 3)
    assert dataset.labels.tolist() == [2, 7]
    assert (dataset.images[1, 0, 0] / 255).tolist() == pytest.approx(
        [0.0, 127 / 255, 1.0]
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_cifar101_npy(
            images_path,
            labels_path,
            expected_images_sha256="0" * 64,
            expected_labels_sha256=_sha256(labels_path),
            expected_images_bytes=images_path.stat().st_size,
            expected_labels_bytes=labels_path.stat().st_size,
            expected_count=2,
        )


def test_hash_ranked_cohort_is_deterministic_and_digestible() -> None:
    first = hash_ranked_indices(40, 12, "fixed-salt")
    second = hash_ranked_indices(40, 12, "fixed-salt")
    assert first == second
    assert len(first) == len(set(first)) == 12
    assert first != hash_ranked_indices(40, 12, "another-salt")
    expected = hashlib.sha256(
        "".join(f"{index}\n" for index in first).encode()
    ).hexdigest()
    assert cohort_newline_sha256(first) == expected


def test_alternative_label_rule_uses_unfiltered_then_retained_runner() -> None:
    assert choose_attack_alternative(0, 2, [10, 9, 1]) == (
        2,
        "unfiltered_selected_label",
    )
    assert choose_attack_alternative(0, 0, [10, 9, 9]) == (
        1,
        "largest_retained_competitor",
    )


def test_shared_counts_produce_all_declared_certificate_variants() -> None:
    result = certificates_from_shared_counts(
        selection_all=[60, 40],
        selection_retained=[55, 5],
        estimation_all=[600, 400],
        estimation_retained=[450, 50],
        proposals=1000,
        sigma=0.25,
        alpha=0.001,
    )
    assert result["conditional_label"] == 0
    assert result["gaussian_label"] == 0
    assert result["attack_alternative_label"] == 1
    assert result["estimation_retained"] == 500
    assert result["released"]["radius"] > 0.0
    assert result["explicit_joint"]["radius"] > 0.0
    assert result["rejection_complement"]["radius"] > 0.0
    assert result["algorithm_one_hybrid"]["radius"] >= max(
        result["explicit_joint"]["radius"],
        result["rejection_complement"]["radius"],
    ) - 0.02
    assert result["unfiltered"]["radius"] > 0.0


def test_family_allocation_reserves_every_possible_inference() -> None:
    allocation = family_allocation(128, 10, 0.001)
    assert allocation["inferences_per_attempt"] == 19
    assert allocation["family_size"] == 2432
    assert allocation["per_inference_alpha"] == pytest.approx(0.001 / 2432)


def test_confirmation_summary_keeps_every_fixed_cohort_outcome() -> None:
    rows = [
        {
            "index": 4,
            "population_violation_verified": True,
            "conditional_correct": True,
            "released_positive": True,
            "diagnostic_disagreement_eligible": True,
            "search_status": "positive_screen",
            "screen_positive": True,
            "confirmation_performed": True,
        },
        {
            "index": 1,
            "population_violation_verified": False,
            "conditional_correct": True,
            "released_positive": True,
            "diagnostic_disagreement_eligible": False,
            "search_status": "no_positive_screen",
            "screen_positive": False,
            "confirmation_performed": False,
        },
        {
            "index": 9,
            "population_violation_verified": False,
            "conditional_correct": False,
            "released_positive": False,
            "diagnostic_disagreement_eligible": False,
            "search_status": "nonpositive_released_radius",
            "screen_positive": False,
            "confirmation_performed": False,
        },
    ]
    summary = analyze_confirmation_rows(rows, [4, 1, 9])
    assert summary["primary_fixed_cohort_yield"]["successes"] == 1
    assert summary["primary_fixed_cohort_yield"]["trials"] == 3
    assert summary["primary_fixed_cohort_yield"]["fraction"] == pytest.approx(1 / 3)
    assert summary["secondary_correct_positive_radius_yield"]["trials"] == 2
    assert summary["diagnostic_filtered_unfiltered_disagreement_yield"][
        "trials"
    ] == 1
    assert summary["search_status_counts"] == {
        "no_positive_screen": 1,
        "nonpositive_released_radius": 1,
        "positive_screen": 1,
    }
