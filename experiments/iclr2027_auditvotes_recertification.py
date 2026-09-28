"""Recertify the released AuditVotes Gaussian confidence filter.

The script keeps the released predictor, confidence filter, Gaussian proposal
law, sample budgets, and selected inputs fixed.  It reports three certificates
from the same proposal batches.

1. The released conditional-probability calculation.
2. Ordinary Gaussian smoothing without confidence filtering.
3. A sound joint filtered-label-mass calculation.

The third calculation counts every Gaussian proposal in its denominator.
Filtered proposals contribute to no class count.  The conditional classifier
and the joint-mass classifier have the same class ordering because every class
shares the same acceptance probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src/feasible_robustness"))
from filtered_certificate import (  # type: ignore  # noqa: E402
    gaussian_joint_mass_radius,
    one_sided_binomial_lower,
    one_sided_binomial_upper,
)


EXPECTED_AUDITVOTES_COMMIT = "52b6a0db53947c815884ad62927d6af6dc584705"
EXPECTED_CHECKPOINT_SHA256 = (
    "420333fe0380cc437218c9b67c20bacf687932957da47340f900d4af5e05bd7c"
)


def _binary_radius(sigma: float, lower_top: float) -> float:
    if not lower_top > 0.5:
        return 0.0
    return float(sigma * norm.ppf(lower_top))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _seed_for(base_seed: int, image_index: int, stream: int) -> int:
    payload = f"{base_seed}:{image_index}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


@dataclass(frozen=True)
class CountBatch:
    all_labels: list[int]
    retained_labels: list[int]
    proposals: int


@dataclass(frozen=True)
class ImageResult:
    index: int
    true_label: int
    conditional_label: int
    gaussian_label: int
    selection_retained: int
    estimation_retained: int
    estimation_proposals: int
    acceptance_rate: float
    conditional_top_count: int
    conditional_probability_lower: float
    released_radius: float
    gaussian_top_count: int
    gaussian_probability_lower: float
    gaussian_radius: float
    joint_top_count: int
    joint_probability_lower: float
    joint_runner_up_upper: float
    joint_radius: float
    conditional_correct: bool
    gaussian_correct: bool
    released_positive: bool
    gaussian_positive: bool
    joint_positive: bool
    elapsed_seconds: float


def _draw_counts(
    model: torch.nn.Module,
    image: torch.Tensor,
    proposals: int,
    batch_size: int,
    sigma: float,
    confidence_threshold: float,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> CountBatch:
    all_counts = torch.zeros(num_classes, dtype=torch.int64, device="cpu")
    retained_counts = torch.zeros(num_classes, dtype=torch.int64, device="cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    completed = 0
    with torch.inference_mode():
        while completed < proposals:
            size = min(batch_size, proposals - completed)
            batch = image.expand(size, -1, -1, -1)
            noise = torch.randn(
                batch.shape, generator=generator, device=device, dtype=batch.dtype
            )
            logits = model(batch + sigma * noise)
            probabilities = torch.softmax(logits, dim=1)
            confidences, labels = probabilities.max(dim=1)
            all_counts += torch.bincount(
                labels.cpu(), minlength=num_classes
            )
            retained = labels[confidences > confidence_threshold]
            if retained.numel():
                retained_counts += torch.bincount(
                    retained.cpu(), minlength=num_classes
                )
            completed += size
    return CountBatch(
        all_labels=[int(x) for x in all_counts.tolist()],
        retained_labels=[int(x) for x in retained_counts.tolist()],
        proposals=proposals,
    )


def _choose_label(counts: Iterable[int]) -> int:
    array = np.asarray(list(counts), dtype=np.int64)
    return int(np.flatnonzero(array == array.max())[0])


def _evaluate_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    true_label: int,
    index: int,
    n0: int,
    n: int,
    batch_size: int,
    sigma: float,
    confidence_threshold: float,
    alpha: float,
    num_classes: int,
    base_seed: int,
    device: torch.device,
) -> ImageResult:
    started = time.monotonic()
    selection = _draw_counts(
        model,
        image,
        n0,
        batch_size,
        sigma,
        confidence_threshold,
        num_classes,
        _seed_for(base_seed, index, 0),
        device,
    )
    estimation = _draw_counts(
        model,
        image,
        n,
        batch_size,
        sigma,
        confidence_threshold,
        num_classes,
        _seed_for(base_seed, index, 1),
        device,
    )

    conditional_label = _choose_label(selection.retained_labels)
    gaussian_label = _choose_label(selection.all_labels)
    retained_total = int(sum(estimation.retained_labels))

    conditional_top = int(estimation.retained_labels[conditional_label])
    conditional_lower = (
        one_sided_binomial_lower(conditional_top, retained_total, alpha)
        if retained_total > 0
        else 0.0
    )
    released_radius = _binary_radius(sigma, conditional_lower)

    gaussian_top = int(estimation.all_labels[gaussian_label])
    gaussian_lower = one_sided_binomial_lower(gaussian_top, n, alpha)
    gaussian_radius = _binary_radius(sigma, gaussian_lower)

    # Independent selection makes conditional_label fixed for this estimation
    # batch.  Split alpha between its lower bound and simultaneous upper bounds
    # for all competing labels.
    joint_top = int(estimation.retained_labels[conditional_label])
    joint_lower = one_sided_binomial_lower(joint_top, n, alpha / 2.0)
    competitor_error = alpha / (2.0 * (num_classes - 1))
    joint_runner_up_upper = max(
        one_sided_binomial_upper(int(count), n, competitor_error)
        for label, count in enumerate(estimation.retained_labels)
        if label != conditional_label
    )
    joint_radius = gaussian_joint_mass_radius(
        joint_lower, joint_runner_up_upper, sigma
    )

    return ImageResult(
        index=index,
        true_label=int(true_label),
        conditional_label=conditional_label,
        gaussian_label=gaussian_label,
        selection_retained=int(sum(selection.retained_labels)),
        estimation_retained=retained_total,
        estimation_proposals=n,
        acceptance_rate=retained_total / n,
        conditional_top_count=conditional_top,
        conditional_probability_lower=conditional_lower,
        released_radius=released_radius,
        gaussian_top_count=gaussian_top,
        gaussian_probability_lower=gaussian_lower,
        gaussian_radius=gaussian_radius,
        joint_top_count=joint_top,
        joint_probability_lower=joint_lower,
        joint_runner_up_upper=joint_runner_up_upper,
        joint_radius=joint_radius,
        conditional_correct=conditional_label == int(true_label),
        gaussian_correct=gaussian_label == int(true_label),
        released_positive=released_radius > 0.0,
        gaussian_positive=gaussian_radius > 0.0,
        joint_positive=joint_radius > 0.0,
        elapsed_seconds=time.monotonic() - started,
    )


def _summary(results: list[ImageResult], radii: list[float]) -> dict[str, object]:
    def certified_accuracy(label_field: str, radius_field: str) -> dict[str, float]:
        values = {}
        for threshold in radii:
            successes = sum(
                bool(getattr(row, label_field))
                and float(getattr(row, radius_field)) > threshold
                for row in results
            )
            values[str(threshold)] = successes / len(results)
        return values

    released = np.asarray([row.released_radius for row in results])
    joint = np.asarray([row.joint_radius for row in results])
    positive = released > 0
    retention = np.divide(
        joint,
        released,
        out=np.zeros_like(joint),
        where=positive,
    )
    return {
        "images": len(results),
        "mean_acceptance_rate": float(
            np.mean([row.acceptance_rate for row in results])
        ),
        "conditional_accuracy": float(
            np.mean([row.conditional_correct for row in results])
        ),
        "gaussian_accuracy": float(
            np.mean([row.gaussian_correct for row in results])
        ),
        "released_positive_fraction": float(np.mean(released > 0)),
        "joint_positive_fraction": float(np.mean(joint > 0)),
        "median_released_radius": float(np.median(released)),
        "median_joint_radius": float(np.median(joint)),
        "median_joint_to_released_ratio_for_released_positive": float(
            np.median(retention[positive]) if positive.any() else 0.0
        ),
        "released_certified_accuracy": certified_accuracy(
            "conditional_correct", "released_radius"
        ),
        "joint_certified_accuracy": certified_accuracy(
            "conditional_correct", "joint_radius"
        ),
        "gaussian_certified_accuracy": certified_accuracy(
            "gaussian_correct", "gaussian_radius"
        ),
        "total_elapsed_seconds": float(sum(row.elapsed_seconds for row in results)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2_609_050_1)
    parser.add_argument(
        "--radii", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7]
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    checkpoint = args.checkpoint.resolve()
    repository_commit = _git_head(repo_root)
    if repository_commit != EXPECTED_AUDITVOTES_COMMIT:
        raise RuntimeError("AuditVotes repository is not at the pinned commit")
    checkpoint_digest = _sha256(checkpoint)
    if checkpoint_digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint does not match the released artifact")
    code_root = repo_root / "ImageClassify_Gaussian/ImageClassify_Conf/code"
    sys.path.insert(0, str(code_root))
    from architectures import get_architecture  # type: ignore
    from datasets import get_dataset, get_num_classes  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("This pinned implementation requires CUDA")
    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = get_architecture(payload["arch"], "cifar10")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    dataset = get_dataset("cifar10", "test")
    num_classes = get_num_classes("cifar10")

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "per_image.jsonl"
    metadata = {
        "auditvotes_git_commit": repository_commit,
        "checkpoint_sha256": checkpoint_digest,
        "core_py_sha256": _sha256(code_root / "core.py"),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"repo_root", "checkpoint", "output"}
        },
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "device": torch.cuda.get_device_name(0),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    results: list[ImageResult] = []
    with rows_path.open("w") as handle:
        for offset in range(args.max_images):
            index = offset * args.skip
            image, true_label = dataset[index]
            row = _evaluate_image(
                model,
                image.unsqueeze(0).to(device),
                int(true_label),
                index,
                args.n0,
                args.n,
                args.batch_size,
                args.sigma,
                args.confidence_threshold,
                args.alpha,
                num_classes,
                args.seed,
                device,
            )
            results.append(row)
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{offset + 1}/{args.max_images} index={index} "
                f"accept={row.acceptance_rate:.4f} "
                f"released={row.released_radius:.4f} "
                f"joint={row.joint_radius:.4f} "
                f"seconds={row.elapsed_seconds:.2f}",
                flush=True,
            )

    summary = _summary(results, args.radii)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
