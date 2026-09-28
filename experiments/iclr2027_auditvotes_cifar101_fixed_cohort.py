#!/usr/bin/env python3
"""Run the frozen AuditVotes study on a fixed CIFAR-10.1 cohort.

The three stages are deliberately separate.

``evaluate`` evaluates every CIFAR-10.1 image with independent label-selection
and probability-estimation proposal streams.  All certificate variants use the
same stored estimation counts.

``search`` attacks the protocol's fixed, hash-ranked cohort.  The cohort is a
function of the dataset size and a salt, never of model output.  Every cohort
member is retained in the output.  A nonpositive released radius, an
unsuccessful screen, and a successful screen therefore remain distinguishable.

``confirm`` uses fresh Gaussian proposals for every positive screen.  One
Bonferroni family covers both endpoint-label tests and the nominal substituted
radius lower bound for every possible cohort attempt.  The primary attack
yield always uses the entire fixed cohort as its denominator.

The script expects a protocol frozen after this file and the imported boundary
search implementation are finalized.  It verifies the protocol digest chain,
the upstream AuditVotes checkout and checkpoint, and both CIFAR-10.1 NPY files
before model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import beta, norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.iclr2027_auditvotes_boundary_search import (  # noqa: E402
    _hard_counts,
    _optimize_candidate,
    derived_seed,
    pairwise_winner_pvalues,
)
from feasible_robustness.filtered_certificate import (  # noqa: E402
    certify_filtered_hybrid_from_counts,
    certify_filtered_joint_mass_from_counts,
    gaussian_joint_mass_radius,
    one_sided_binomial_lower,
)


PROTOCOL_STATUS = "frozen_before_cifar101_fixed_cohort_execution"
BOUNDARY_SCRIPT = PROJECT_ROOT / "experiments/iclr2027_auditvotes_boundary_search.py"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON row on line {line_number}")
            rows.append(row)
    return rows


@dataclass(frozen=True)
class Cifar101Dataset:
    """In-memory CIFAR-10.1 NPY dataset with torchvision-compatible samples."""

    images: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        import torch

        if not 0 <= index < len(self):
            raise IndexError(index)
        # Copy after transposition so torch never aliases a read-only NPY view.
        chw = np.array(self.images[index].transpose(2, 0, 1), copy=True)
        image = torch.from_numpy(chw).to(dtype=torch.float32).div_(255.0)
        return image, int(self.labels[index])


def load_cifar101_npy(
    images_path: Path,
    labels_path: Path,
    *,
    expected_images_sha256: str,
    expected_labels_sha256: str,
    expected_images_bytes: int,
    expected_labels_bytes: int,
    expected_count: int,
    num_classes: int = 10,
) -> Cifar101Dataset:
    """Load and validate official CIFAR-10.1 uint8 NHWC NPY files."""

    for path, expected_bytes, expected_digest in (
        (images_path, expected_images_bytes, expected_images_sha256),
        (labels_path, expected_labels_bytes, expected_labels_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected_bytes):
            raise ValueError(f"byte size changed for {path}")
        if sha256_path(path) != str(expected_digest):
            raise ValueError(f"SHA-256 changed for {path}")

    images = np.load(images_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    if images.shape != (expected_count, 32, 32, 3):
        raise ValueError(
            "CIFAR-10.1 images must have shape "
            f"({expected_count}, 32, 32, 3)"
        )
    if images.dtype != np.uint8:
        raise ValueError("CIFAR-10.1 images must have uint8 dtype")
    if labels.shape != (expected_count,):
        raise ValueError(f"CIFAR-10.1 labels must have shape ({expected_count},)")
    if labels.dtype.kind not in "iu":
        raise ValueError("CIFAR-10.1 labels must have integer dtype")
    labels64 = labels.astype(np.int64, copy=False)
    if np.any(labels64 < 0) or np.any(labels64 >= num_classes):
        raise ValueError("CIFAR-10.1 labels fall outside the declared classes")
    return Cifar101Dataset(images=images, labels=labels64)


def hash_ranked_indices(total: int, count: int, salt: str) -> list[int]:
    """Return a deterministic output-independent cohort in hash-rank order."""

    if total < 1 or not 1 <= count <= total or not salt:
        raise ValueError("require nonempty salt and 1 <= count <= total")
    return sorted(
        range(total),
        key=lambda index: (
            hashlib.sha256(f"{salt}:{index}".encode()).digest(),
            index,
        ),
    )[:count]


def cohort_newline_sha256(indices: Sequence[int]) -> str:
    payload = "".join(f"{int(index)}\n" for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("protocol is not frozen for this study")
    dataset = protocol["dataset"]
    total = int(dataset["count"])
    attack = protocol["attack"]
    declared = [int(value) for value in attack["cohort_indices"]]
    expected = hash_ranked_indices(
        total, int(attack["cohort_size"]), str(attack["cohort_salt"])
    )
    if declared != expected:
        raise ValueError("declared cohort is not the required hash-ranked cohort")
    if cohort_newline_sha256(declared) != str(attack["cohort_newline_sha256"]):
        raise ValueError("declared cohort digest changed")
    if int(protocol["model"]["num_classes"]) < 2:
        raise ValueError("num_classes must be at least two")
    return protocol


def _verify_provenance(
    protocol: dict[str, Any],
    repo_root: Path,
    checkpoint: Path,
    images_path: Path,
    labels_path: Path,
) -> Cifar101Dataset:
    provenance = protocol["provenance"]
    if sha256_path(Path(__file__).resolve()) != provenance["study_script_sha256"]:
        raise ValueError("study script digest does not match the frozen protocol")
    if sha256_path(BOUNDARY_SCRIPT) != provenance["boundary_script_sha256"]:
        raise ValueError("boundary-search script digest changed")
    if _git_head(repo_root) != provenance["auditvotes_git_commit"]:
        raise ValueError("AuditVotes commit does not match the frozen protocol")
    if sha256_path(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("AuditVotes checkpoint digest changed")
    core_path = repo_root / "ImageClassify_Gaussian/ImageClassify_Conf/code/core.py"
    if sha256_path(core_path) != provenance["core_py_sha256"]:
        raise ValueError("AuditVotes core.py digest changed")

    dataset = protocol["dataset"]
    return load_cifar101_npy(
        images_path,
        labels_path,
        expected_images_sha256=str(dataset["images_sha256"]),
        expected_labels_sha256=str(dataset["labels_sha256"]),
        expected_images_bytes=int(dataset["images_bytes"]),
        expected_labels_bytes=int(dataset["labels_bytes"]),
        expected_count=int(dataset["count"]),
        num_classes=int(protocol["model"]["num_classes"]),
    )


def _load_model(repo_root: Path, checkpoint: Path, protocol: dict[str, Any]):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the frozen implementation requires CUDA")
    code_root = repo_root / "ImageClassify_Gaussian/ImageClassify_Conf/code"
    sys.path.insert(0, str(code_root))
    from architectures import get_architecture  # type: ignore

    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = str(payload["arch"])
    expected_architecture = protocol["model"].get("architecture")
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError("checkpoint architecture differs from the protocol")
    model = get_architecture(architecture, "cifar10")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, device


def choose_label(counts: Iterable[int], *, exclude: int | None = None) -> int:
    values = np.asarray(list(counts), dtype=np.int64)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 0):
        raise ValueError("counts must be a nonnegative one-dimensional vector")
    allowed = np.ones(len(values), dtype=bool)
    if exclude is not None:
        if not 0 <= exclude < len(values):
            raise ValueError("excluded label is outside the count vector")
        allowed[exclude] = False
    maximum = values[allowed].max()
    return int(np.flatnonzero(allowed & (values == maximum))[0])


def choose_attack_alternative(
    filtered_label: int,
    unfiltered_label: int,
    selection_retained_counts: Sequence[int],
) -> tuple[int, str]:
    """Apply the frozen alternative-label rule using selection counts only."""

    if filtered_label != unfiltered_label:
        return int(unfiltered_label), "unfiltered_selected_label"
    return (
        choose_label(selection_retained_counts, exclude=filtered_label),
        "largest_retained_competitor",
    )


def disagreement_eligibility(
    true_label: int,
    filtered_label: int,
    unfiltered_label: int,
    released_radius: float,
) -> bool:
    """Return the predeclared diagnostic disagreement eligibility flag."""

    return bool(
        filtered_label == true_label
        and filtered_label != unfiltered_label
        and released_radius > 0.0
    )


def _binary_radius(sigma: float, lower: float) -> float:
    return 0.0 if lower <= 0.5 else float(sigma * norm.ppf(lower))


def certificates_from_shared_counts(
    *,
    selection_all: Sequence[int],
    selection_retained: Sequence[int],
    estimation_all: Sequence[int],
    estimation_retained: Sequence[int],
    proposals: int,
    sigma: float,
    alpha: float,
) -> dict[str, Any]:
    """Compute all declared certificates from one pair of proposal batches."""

    arrays = [
        np.asarray(values, dtype=np.int64)
        for values in (
            selection_all,
            selection_retained,
            estimation_all,
            estimation_retained,
        )
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("every count vector must be one-dimensional")
    num_classes = len(arrays[0])
    if num_classes < 2 or any(len(array) != num_classes for array in arrays):
        raise ValueError("count vectors must share at least two classes")
    if any(np.any(array < 0) for array in arrays):
        raise ValueError("counts must be nonnegative")
    if int(arrays[2].sum()) != proposals or int(arrays[3].sum()) > proposals:
        raise ValueError("estimation counts disagree with proposal count")

    conditional_label = choose_label(arrays[1])
    gaussian_label = choose_label(arrays[0])
    alternative_label, alternative_source = choose_attack_alternative(
        conditional_label, gaussian_label, arrays[1]
    )
    retained = int(arrays[3].sum())

    conditional_count = int(arrays[3][conditional_label])
    conditional_lower = (
        one_sided_binomial_lower(conditional_count, retained, alpha)
        if retained > 0
        else 0.0
    )
    released_radius = _binary_radius(sigma, conditional_lower)

    gaussian_count = int(arrays[2][gaussian_label])
    gaussian_lower = one_sided_binomial_lower(gaussian_count, proposals, alpha)
    gaussian_radius = _binary_radius(sigma, gaussian_lower)

    labels = tuple(range(num_classes))
    retained_mapping = {
        label: int(arrays[3][label]) for label in labels
    }
    explicit = certify_filtered_joint_mass_from_counts(
        retained_mapping,
        proposals,
        sigma,
        label_universe=labels,
        delta=alpha,
        selected_label=conditional_label,
        selection_independent=True,
    )

    explicit_lower = one_sided_binomial_lower(
        conditional_count, proposals, alpha / 2.0
    )
    competing_accepted = retained - conditional_count
    selected_or_rejected_lower = one_sided_binomial_lower(
        proposals - competing_accepted, proposals, alpha / 2.0
    )
    complement_upper = 1.0 - selected_or_rejected_lower
    complement_radius = gaussian_joint_mass_radius(
        explicit_lower, complement_upper, sigma
    )

    hybrid = certify_filtered_hybrid_from_counts(
        retained_mapping,
        proposals,
        sigma,
        label_universe=labels,
        selected_label=conditional_label,
        delta=alpha,
        selection_independent=True,
    )
    return {
        "conditional_label": conditional_label,
        "gaussian_label": gaussian_label,
        "attack_alternative_label": alternative_label,
        "attack_alternative_source": alternative_source,
        "estimation_retained": retained,
        "acceptance_rate": retained / proposals,
        "released": {
            "top_count": conditional_count,
            "conditional_probability_lower": conditional_lower,
            "radius": released_radius,
        },
        "explicit_joint": {
            "top_count": conditional_count,
            "top_mass_lower": explicit.selected_lower,
            "runner_label": int(explicit.runner_up),
            "runner_mass_upper": explicit.runner_upper,
            "radius": explicit.radius,
        },
        "rejection_complement": {
            "top_mass_lower": explicit_lower,
            "all_competitors_mass_upper": complement_upper,
            "radius": complement_radius,
        },
        "algorithm_one_hybrid": {
            "top_mass_lower": hybrid.selected_lower,
            "explicit_runner_mass_upper": hybrid.explicit_runner_upper,
            "complement_runner_mass_upper": hybrid.complement_runner_upper,
            "runner_mass_upper": hybrid.runner_upper,
            "radius": hybrid.radius,
        },
        "unfiltered": {
            "top_count": gaussian_count,
            "probability_lower": gaussian_lower,
            "radius": gaussian_radius,
        },
    }


def _draw_counts(
    model,
    image,
    proposals: int,
    batch_size: int,
    sigma: float,
    threshold: float,
    num_classes: int,
    seed: int,
    device,
) -> tuple[list[int], list[int]]:
    import torch

    all_counts = torch.zeros(num_classes, dtype=torch.int64, device="cpu")
    retained_counts = torch.zeros(num_classes, dtype=torch.int64, device="cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    completed = 0
    with torch.inference_mode():
        while completed < proposals:
            size = min(batch_size, proposals - completed)
            noise = torch.randn(
                (size, *image.shape[1:]),
                generator=generator,
                device=device,
                dtype=image.dtype,
            )
            logits = model(image.expand(size, -1, -1, -1) + sigma * noise)
            probabilities = torch.softmax(logits, dim=1)
            confidence, labels = probabilities.max(dim=1)
            all_counts += torch.bincount(labels.cpu(), minlength=num_classes)
            retained = labels[confidence > threshold]
            if retained.numel():
                retained_counts += torch.bincount(
                    retained.cpu(), minlength=num_classes
                )
            completed += size
    return (
        [int(value) for value in all_counts.tolist()],
        [int(value) for value in retained_counts.tolist()],
    )


def _evaluate_one(
    model,
    image,
    true_label: int,
    index: int,
    protocol: dict[str, Any],
    device,
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    model_spec = protocol["model"]
    base_seed = int(evaluation["base_seed"])
    selection_all, selection_retained = _draw_counts(
        model,
        image,
        int(evaluation["selection_proposals"]),
        int(evaluation["inference_batch_size"]),
        float(model_spec["sigma"]),
        float(model_spec["confidence_threshold"]),
        int(model_spec["num_classes"]),
        derived_seed(base_seed, index, "selection"),
        device,
    )
    estimation_all, estimation_retained = _draw_counts(
        model,
        image,
        int(evaluation["estimation_proposals"]),
        int(evaluation["inference_batch_size"]),
        float(model_spec["sigma"]),
        float(model_spec["confidence_threshold"]),
        int(model_spec["num_classes"]),
        derived_seed(base_seed, index, "estimation"),
        device,
    )
    result = certificates_from_shared_counts(
        selection_all=selection_all,
        selection_retained=selection_retained,
        estimation_all=estimation_all,
        estimation_retained=estimation_retained,
        proposals=int(evaluation["estimation_proposals"]),
        sigma=float(model_spec["sigma"]),
        alpha=float(evaluation["alpha"]),
    )
    conditional_label = int(result["conditional_label"])
    gaussian_label = int(result["gaussian_label"])
    released_radius = float(result["released"]["radius"])
    return {
        "index": index,
        "true_label": int(true_label),
        "selection_all_counts": selection_all,
        "selection_retained_counts": selection_retained,
        "estimation_all_counts": estimation_all,
        "estimation_retained_counts": estimation_retained,
        **result,
        "conditional_correct": conditional_label == int(true_label),
        "gaussian_correct": gaussian_label == int(true_label),
        "released_positive": released_radius > 0.0,
        "diagnostic_disagreement_eligible": disagreement_eligibility(
            int(true_label), conditional_label, gaussian_label, released_radius
        ),
    }


def _certified_accuracy(
    rows: Sequence[dict[str, Any]],
    label_field: str,
    method: str,
    radii: Sequence[float],
) -> dict[str, float]:
    return {
        f"{radius:.2f}": float(
            np.mean(
                [
                    bool(row[label_field])
                    and float(row[method]["radius"]) > radius
                    for row in rows
                ]
            )
        )
        for radius in radii
    }


def summarize_evaluation(
    rows: Sequence[dict[str, Any]], radii: Sequence[float]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    methods = (
        "released",
        "explicit_joint",
        "rejection_complement",
        "algorithm_one_hybrid",
        "unfiltered",
    )
    summary: dict[str, Any] = {
        "images": len(rows),
        "mean_acceptance_rate": float(
            np.mean([float(row["acceptance_rate"]) for row in rows])
        ),
        "conditional_accuracy": float(
            np.mean([bool(row["conditional_correct"]) for row in rows])
        ),
        "gaussian_accuracy": float(
            np.mean([bool(row["gaussian_correct"]) for row in rows])
        ),
        "diagnostic_disagreement_eligible_count": int(
            sum(bool(row["diagnostic_disagreement_eligible"]) for row in rows)
        ),
        "methods": {},
    }
    for method in methods:
        label_field = "gaussian_correct" if method == "unfiltered" else "conditional_correct"
        values = np.asarray([float(row[method]["radius"]) for row in rows])
        summary["methods"][method] = {
            "positive_fraction": float(np.mean(values > 0.0)),
            "median_radius": float(np.median(values)),
            "certified_accuracy": _certified_accuracy(
                rows, label_field, method, radii
            ),
        }
    return summary


def _common_manifest(
    protocol_path: Path,
    protocol: dict[str, Any],
    checkpoint: Path,
    images_path: Path,
    labels_path: Path,
) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256_path(protocol_path),
        "study_script_sha256": sha256_path(Path(__file__).resolve()),
        "boundary_script_sha256": sha256_path(BOUNDARY_SCRIPT),
        "checkpoint_sha256": sha256_path(checkpoint),
        "images_sha256": sha256_path(images_path),
        "labels_sha256": sha256_path(labels_path),
        "auditvotes_git_commit": protocol["provenance"]["auditvotes_git_commit"],
    }


def _validate_resume_prefix(
    rows: Sequence[dict[str, Any]], expected_indices: Sequence[int], field: str = "index"
) -> None:
    indices = [int(row[field]) for row in rows]
    if indices != list(expected_indices[: len(indices)]):
        raise ValueError("resume rows are not an exact prefix of the frozen order")


def run_evaluate(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    repo_root = args.repo_root.resolve()
    checkpoint = args.checkpoint.resolve()
    images_path = args.images.resolve()
    labels_path = args.labels.resolve()
    dataset = _verify_provenance(
        protocol, repo_root, checkpoint, images_path, labels_path
    )
    model, device = _load_model(repo_root, checkpoint, protocol)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "per_image.jsonl"
    manifest_path = output / "manifest.json"
    common = _common_manifest(
        protocol_path, protocol, checkpoint, images_path, labels_path
    )
    existing: list[dict[str, Any]] = []
    if args.resume and rows_path.exists():
        existing = _load_jsonl(rows_path)
        _validate_resume_prefix(existing, list(range(len(dataset))))
        prior = _read_json(manifest_path)
        for key, value in common.items():
            if prior.get(key) != value:
                raise ValueError(f"resume manifest changed at {key}")
    elif rows_path.exists():
        raise FileExistsError("evaluation output exists; pass --resume to continue")
    _write_json(
        manifest_path,
        {**common, "status": "evaluation_in_progress", "completed": len(existing)},
    )

    started = time.monotonic()
    with rows_path.open("a" if existing else "w") as handle:
        for index in range(len(existing), len(dataset)):
            image, label = dataset[index]
            row = _evaluate_one(
                model, image.unsqueeze(0).to(device), label, index, protocol, device
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index + 1}/{len(dataset)} index={index} "
                f"released={row['released']['radius']:.6f} "
                f"hybrid={row['algorithm_one_hybrid']['radius']:.6f}",
                flush=True,
            )

    rows = _load_jsonl(rows_path)
    _validate_resume_prefix(rows, list(range(len(dataset))))
    summary = summarize_evaluation(
        rows, [float(value) for value in protocol["evaluation"]["radii"]]
    )
    _write_json(output / "summary.json", summary)
    import torch

    manifest = {
        **common,
        "status": "evaluation_complete",
        "completed": len(rows),
        "per_image_sha256": sha256_path(rows_path),
        "summary_sha256": sha256_path(output / "summary.json"),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _verify_evaluation(
    directory: Path,
    protocol_path: Path,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("status") != "evaluation_complete":
        raise ValueError("evaluation is not complete")
    if manifest.get("protocol_sha256") != sha256_path(protocol_path):
        raise ValueError("evaluation uses another protocol")
    rows_path = directory / "per_image.jsonl"
    if manifest.get("per_image_sha256") != sha256_path(rows_path):
        raise ValueError("evaluation rows changed")
    rows = _load_jsonl(rows_path)
    if len(rows) != expected_count:
        raise ValueError("evaluation does not cover the full dataset")
    _validate_resume_prefix(rows, list(range(expected_count)))
    return rows, manifest


def _screen_candidate(
    model,
    original,
    row: dict[str, Any],
    protocol: dict[str, Any],
    device,
) -> dict[str, Any]:
    import torch

    attack = protocol["attack"]
    model_spec = protocol["model"]
    candidate = {
        "index": int(row["index"]),
        "original_label": int(row["conditional_label"]),
        "alternative_label": int(row["attack_alternative_label"]),
        "released_radius": float(row["released"]["radius"]),
    }
    radius, proposals = _optimize_candidate(
        model,
        original,
        candidate,
        attack,
        float(model_spec["sigma"]),
        float(model_spec["confidence_threshold"]),
        int(protocol["seeds"]["search"]),
        device,
    )
    screened: list[dict[str, Any]] = []
    tensors: list[Any] = []
    for proposal_number, (surrogate, delta, source) in enumerate(proposals):
        perturbed = (original + delta).detach()
        counts = _hard_counts(
            model,
            perturbed,
            int(attack["screen_proposals"]),
            int(attack["inference_batch_size"]),
            float(model_spec["sigma"]),
            float(model_spec["confidence_threshold"]),
            int(model_spec["num_classes"]),
            derived_seed(
                int(protocol["seeds"]["screen"]),
                int(row["index"]),
                proposal_number,
            ),
            device,
        )
        alternative = int(row["attack_alternative_label"])
        competitors = np.delete(counts, alternative)
        difference = int(counts[alternative]) - int(competitors.max())
        screened.append(
            {
                "proposal_number": proposal_number,
                "source": source,
                "surrogate_objective": float(surrogate),
                "counts": [int(value) for value in counts],
                "alternative_minus_largest_competitor": difference,
                "alternative_is_unique_winner": difference > 0,
                "l2_distance": float(
                    torch.linalg.vector_norm(perturbed - original).double().cpu()
                ),
            }
        )
        tensors.append(perturbed.cpu().numpy().astype(np.float32))
    if not screened:
        raise RuntimeError("optimizer produced no proposals")
    best_number = max(
        range(len(screened)),
        key=lambda number: (
            int(screened[number]["alternative_minus_largest_competitor"]),
            float(screened[number]["surrogate_objective"]),
            -number,
        ),
    )
    return {
        "radius_limit": float(radius),
        "screened_proposals": screened,
        "chosen_proposal_number": best_number,
        "chosen": screened[best_number],
        "screen_positive": bool(screened[best_number]["alternative_is_unique_winner"]),
        "tensor": tensors[best_number],
    }


def run_search(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    repo_root = args.repo_root.resolve()
    checkpoint = args.checkpoint.resolve()
    images_path = args.images.resolve()
    labels_path = args.labels.resolve()
    dataset = _verify_provenance(
        protocol, repo_root, checkpoint, images_path, labels_path
    )
    evaluation_dir = args.evaluation.resolve()
    evaluation_rows, evaluation_manifest = _verify_evaluation(
        evaluation_dir, protocol_path, len(dataset)
    )
    model, device = _load_model(repo_root, checkpoint, protocol)

    cohort = [int(value) for value in protocol["attack"]["cohort_indices"]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempts_path = output / "attempts.jsonl"
    manifest_path = output / "manifest.json"
    common = {
        **_common_manifest(
            protocol_path, protocol, checkpoint, images_path, labels_path
        ),
        "evaluation_manifest_sha256": sha256_path(evaluation_dir / "manifest.json"),
        "evaluation_rows_sha256": evaluation_manifest["per_image_sha256"],
    }
    existing: list[dict[str, Any]] = []
    if args.resume and attempts_path.exists():
        existing = _load_jsonl(attempts_path)
        _validate_resume_prefix(existing, cohort)
        prior = _read_json(manifest_path)
        for key, value in common.items():
            if prior.get(key) != value:
                raise ValueError(f"resume manifest changed at {key}")
    elif attempts_path.exists():
        raise FileExistsError("search output exists; pass --resume to continue")
    _write_json(
        manifest_path,
        {**common, "status": "search_in_progress", "completed": len(existing)},
    )

    started = time.monotonic()
    with attempts_path.open("a" if existing else "w") as handle:
        for position in range(len(existing), len(cohort)):
            index = cohort[position]
            evaluation = evaluation_rows[index]
            base = {
                "cohort_position": position,
                "index": index,
                "true_label": int(evaluation["true_label"]),
                "conditional_label": int(evaluation["conditional_label"]),
                "gaussian_label": int(evaluation["gaussian_label"]),
                "attack_alternative_label": int(
                    evaluation["attack_alternative_label"]
                ),
                "attack_alternative_source": evaluation[
                    "attack_alternative_source"
                ],
                "conditional_correct": bool(evaluation["conditional_correct"]),
                "released_radius": float(evaluation["released"]["radius"]),
                "released_positive": bool(evaluation["released_positive"]),
                "diagnostic_disagreement_eligible": bool(
                    evaluation["diagnostic_disagreement_eligible"]
                ),
            }
            if not base["released_positive"]:
                attempt = {
                    **base,
                    "search_status": "nonpositive_released_radius",
                    "screen_positive": False,
                }
            else:
                image, _ = dataset[index]
                original = image.unsqueeze(0).to(device)
                screen = _screen_candidate(
                    model, original, evaluation, protocol, device
                )
                asset = output / f"cohort_{position:03d}_index_{index}.npz"
                np.savez_compressed(asset, image=screen.pop("tensor"))
                attempt = {
                    **base,
                    **screen,
                    "search_status": (
                        "positive_screen"
                        if bool(screen["screen_positive"])
                        else "no_positive_screen"
                    ),
                    "asset": asset.name,
                    "asset_sha256": sha256_path(asset),
                }
            handle.write(json.dumps(attempt, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{position + 1}/{len(cohort)} index={index} "
                f"status={attempt['search_status']}",
                flush=True,
            )

    attempts = _load_jsonl(attempts_path)
    _validate_resume_prefix(attempts, cohort)
    summary = {
        "status": "search_complete_confirmation_unopened",
        **common,
        "attempts_sha256": sha256_path(attempts_path),
        "cohort_size": len(cohort),
        "positive_radius_count": sum(
            bool(row["released_positive"]) for row in attempts
        ),
        "positive_screen_count": sum(
            bool(row["screen_positive"]) for row in attempts
        ),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "attempts": attempts,
    }
    _write_json(output / "search_summary.json", summary)
    _write_json(
        manifest_path,
        {
            **common,
            "status": "search_complete_confirmation_unopened",
            "completed": len(attempts),
            "attempts_sha256": sha256_path(attempts_path),
            "search_summary_sha256": sha256_path(output / "search_summary.json"),
        },
    )
    print(json.dumps({key: summary[key] for key in ("cohort_size", "positive_radius_count", "positive_screen_count")}, indent=2))


def family_allocation(
    cohort_size: int, num_classes: int, familywise_alpha: float
) -> dict[str, float | int]:
    """Allocate one error budget across every possible cohort inference."""

    if cohort_size < 1 or num_classes < 2:
        raise ValueError("cohort and class counts must be positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    inferences_per_attempt = 2 * (num_classes - 1) + 1
    family_size = cohort_size * inferences_per_attempt
    return {
        "cohort_size": cohort_size,
        "inferences_per_attempt": inferences_per_attempt,
        "family_size": family_size,
        "familywise_alpha": familywise_alpha,
        "per_inference_alpha": familywise_alpha / family_size,
    }


def _exact_binomial_interval(
    successes: int, trials: int, error: float = 0.05
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(error / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(
            beta.ppf(1.0 - error / 2.0, successes + 1, trials - successes)
        )
    )
    return lower, upper


def analyze_confirmation_rows(
    rows: Sequence[dict[str, Any]], cohort_indices: Sequence[int]
) -> dict[str, Any]:
    """Summarize attack yield while retaining every fixed-cohort failure."""

    if len(rows) != len(cohort_indices):
        raise ValueError("confirmation rows do not cover the fixed cohort")
    if [int(row["index"]) for row in rows] != [int(x) for x in cohort_indices]:
        raise ValueError("confirmation row order differs from the fixed cohort")
    primary_successes = sum(bool(row["population_violation_verified"]) for row in rows)

    secondary = [
        row
        for row in rows
        if bool(row["conditional_correct"]) and bool(row["released_positive"])
    ]
    disagreement = [
        row for row in rows if bool(row["diagnostic_disagreement_eligible"])
    ]

    def rate_payload(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        trials = len(selected)
        successes = sum(
            bool(row["population_violation_verified"]) for row in selected
        )
        if trials == 0:
            return {
                "successes": 0,
                "trials": 0,
                "fraction": None,
                "exact_95_percent_interval": None,
            }
        lower, upper = _exact_binomial_interval(successes, trials)
        return {
            "successes": successes,
            "trials": trials,
            "fraction": successes / trials,
            "exact_95_percent_interval": [lower, upper],
        }

    return {
        "primary_fixed_cohort_yield": rate_payload(rows),
        "secondary_correct_positive_radius_yield": rate_payload(secondary),
        "diagnostic_filtered_unfiltered_disagreement_yield": rate_payload(
            disagreement
        ),
        "search_status_counts": dict(
            sorted(Counter(str(row["search_status"]) for row in rows).items())
        ),
        "positive_screen_count": sum(bool(row["screen_positive"]) for row in rows),
        "confirmed_screen_count": sum(
            bool(row.get("confirmation_performed", False)) for row in rows
        ),
        "verified_count": primary_successes,
    }


def _confirm_positive_attempt(
    model,
    dataset: Cifar101Dataset,
    attempt: dict[str, Any],
    position: int,
    search_dir: Path,
    protocol: dict[str, Any],
    per_inference_alpha: float,
    device,
) -> dict[str, Any]:
    import torch

    asset = search_dir / str(attempt["asset"])
    if sha256_path(asset) != attempt["asset_sha256"]:
        raise ValueError(f"candidate asset changed for index {attempt['index']}")
    stored = np.load(asset, allow_pickle=False)
    if set(stored.files) != {"image"}:
        raise ValueError("candidate asset has unexpected arrays")
    perturbed_array = stored["image"]
    index = int(attempt["index"])
    original_image, _ = dataset[index]
    original = original_image.unsqueeze(0).to(device)
    if perturbed_array.shape != tuple(original.shape):
        raise ValueError("candidate tensor shape changed")
    perturbed = torch.as_tensor(
        perturbed_array, dtype=original.dtype, device=device
    )
    original_label = int(attempt["conditional_label"])
    alternative_label = int(attempt["attack_alternative_label"])
    confirmation = protocol["confirmation"]
    model_spec = protocol["model"]

    endpoints: list[dict[str, Any]] = []
    for endpoint_name, image, winner in (
        ("original", original, original_label),
        ("perturbed", perturbed, alternative_label),
    ):
        counts = _hard_counts(
            model,
            image,
            int(confirmation["proposals_per_endpoint"]),
            int(confirmation["inference_batch_size"]),
            float(model_spec["sigma"]),
            float(model_spec["confidence_threshold"]),
            int(model_spec["num_classes"]),
            derived_seed(
                int(protocol["seeds"]["confirmation"]),
                position,
                index,
                endpoint_name,
            ),
            device,
        )
        pvalues = pairwise_winner_pvalues(counts, winner)
        endpoints.append(
            {
                "endpoint": endpoint_name,
                "winner": winner,
                "counts": [int(value) for value in counts],
                "retained": int(counts.sum()),
                "pairwise_pvalues": pvalues,
                "maximum_pairwise_pvalue": max(pvalues),
                "winner_verified": max(pvalues) < per_inference_alpha,
            }
        )

    nominal = endpoints[0]
    retained = int(nominal["retained"])
    conditional_lower = (
        one_sided_binomial_lower(
            int(nominal["counts"][original_label]),
            retained,
            per_inference_alpha,
        )
        if retained > 0
        else 0.0
    )
    radius_lower = _binary_radius(float(model_spec["sigma"]), conditional_lower)
    distance = float(
        np.linalg.norm(
            perturbed_array.astype(np.float64)
            - original_image.unsqueeze(0).numpy().astype(np.float64)
        )
    )
    verified = bool(
        original_label != alternative_label
        and endpoints[0]["winner_verified"]
        and endpoints[1]["winner_verified"]
        and distance < radius_lower
    )
    return {
        **{
            key: value
            for key, value in attempt.items()
            if key not in {"screened_proposals"}
        },
        "confirmation_performed": True,
        "endpoints": endpoints,
        "fresh_conditional_probability_lower": conditional_lower,
        "fresh_substituted_radius_lower": radius_lower,
        "l2_distance_recomputed": distance,
        "inside_fresh_substituted_radius_lower": distance < radius_lower,
        "population_violation_verified": verified,
    }


def run_confirm(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    repo_root = args.repo_root.resolve()
    checkpoint = args.checkpoint.resolve()
    images_path = args.images.resolve()
    labels_path = args.labels.resolve()
    dataset = _verify_provenance(
        protocol, repo_root, checkpoint, images_path, labels_path
    )
    search_summary_path = args.search_summary.resolve()
    search = _read_json(search_summary_path)
    if search.get("status") != "search_complete_confirmation_unopened":
        raise ValueError("search is not frozen before confirmation")
    if search.get("protocol_sha256") != sha256_path(protocol_path):
        raise ValueError("search uses another protocol")
    attempts = search["attempts"]
    cohort = [int(value) for value in protocol["attack"]["cohort_indices"]]
    _validate_resume_prefix(attempts, cohort)
    if len(attempts) != len(cohort):
        raise ValueError("search does not cover the full fixed cohort")
    search_dir = search_summary_path.parent
    if search.get("attempts_sha256") != sha256_path(search_dir / "attempts.jsonl"):
        raise ValueError("search attempts changed")

    allocation = family_allocation(
        len(cohort),
        int(protocol["model"]["num_classes"]),
        float(protocol["confirmation"]["familywise_alpha"]),
    )
    model, device = _load_model(repo_root, checkpoint, protocol)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".jsonl")
    existing: list[dict[str, Any]] = []
    if args.resume and partial.exists():
        existing = _load_jsonl(partial)
        _validate_resume_prefix(existing, cohort)
    elif partial.exists():
        raise FileExistsError("confirmation output exists; pass --resume to continue")

    started = time.monotonic()
    with partial.open("a" if existing else "w") as handle:
        for position in range(len(existing), len(cohort)):
            attempt = attempts[position]
            if bool(attempt["screen_positive"]):
                row = _confirm_positive_attempt(
                    model,
                    dataset,
                    attempt,
                    position,
                    search_dir,
                    protocol,
                    float(allocation["per_inference_alpha"]),
                    device,
                )
            else:
                row = {
                    **{
                        key: value
                        for key, value in attempt.items()
                        if key not in {"screened_proposals"}
                    },
                    "confirmation_performed": False,
                    "population_violation_verified": False,
                }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{position + 1}/{len(cohort)} index={row['index']} "
                f"verified={row['population_violation_verified']}",
                flush=True,
            )

    rows = _load_jsonl(partial)
    _validate_resume_prefix(rows, cohort)
    analysis = analyze_confirmation_rows(rows, cohort)
    result = {
        "status": "confirmation_complete",
        "protocol_sha256": sha256_path(protocol_path),
        "search_summary_sha256": sha256_path(search_summary_path),
        "confirmation_rows_sha256": sha256_path(partial),
        "family_allocation": allocation,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "analysis": analysis,
        "rows": rows,
    }
    _write_json(output, result)
    print(json.dumps(analysis, indent=2, sort_keys=True))


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    _add_shared_arguments(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.set_defaults(function=run_evaluate)

    search = subparsers.add_parser("search")
    _add_shared_arguments(search)
    search.add_argument("--evaluation", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--resume", action="store_true")
    search.set_defaults(function=run_search)

    confirm = subparsers.add_parser("confirm")
    _add_shared_arguments(confirm)
    confirm.add_argument("--search-summary", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    confirm.add_argument("--resume", action="store_true")
    confirm.set_defaults(function=run_confirm)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
