"""Select and confirm a sound confidence threshold for AuditVotes.

The released CIFAR-10 predictor, Gaussian proposal law, and checkpoint remain
fixed.  Every Gaussian proposal is evaluated once and reused for all declared
confidence thresholds.  Rejected proposals remain in the denominator of each
joint retained-label mass.

The first 1,000 CIFAR-10 test images select one global threshold under the
frozen protocol.  The selected threshold is then reported on the remaining
9,000 images together with the unfiltered and released-threshold controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
class ThresholdResult:
    threshold: float
    selected_label: int
    selection_retained: int
    estimation_retained: int
    acceptance_rate: float
    selected_joint_count: int
    selected_joint_lower: float
    runner_joint_upper: float
    joint_radius: float
    correct: bool


@dataclass(frozen=True)
class ImageResult:
    index: int
    true_label: int
    elapsed_seconds: float
    thresholds: tuple[ThresholdResult, ...]


def _choose_label(counts: np.ndarray) -> int:
    return int(np.flatnonzero(counts == counts.max())[0])


def _draw_threshold_counts(
    model: Any,
    image: Any,
    proposals: int,
    batch_size: int,
    sigma: float,
    thresholds: tuple[float, ...],
    num_classes: int,
    seed: int,
    device: Any,
) -> np.ndarray:
    import torch

    counts = np.zeros((len(thresholds), num_classes), dtype=np.int64)
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
            confidence_array = confidences.cpu().numpy()
            label_array = labels.cpu().numpy()
            for threshold_index, threshold in enumerate(thresholds):
                retained = label_array[confidence_array > threshold]
                counts[threshold_index] += np.bincount(
                    retained, minlength=num_classes
                )
            completed += size
    return counts


def _evaluate_image(
    model: Any,
    image: Any,
    true_label: int,
    index: int,
    n0: int,
    n: int,
    batch_size: int,
    sigma: float,
    thresholds: tuple[float, ...],
    alpha: float,
    num_classes: int,
    base_seed: int,
    device: Any,
) -> ImageResult:
    started = time.monotonic()
    selection = _draw_threshold_counts(
        model,
        image,
        n0,
        batch_size,
        sigma,
        thresholds,
        num_classes,
        _seed_for(base_seed, index, 0),
        device,
    )
    estimation = _draw_threshold_counts(
        model,
        image,
        n,
        batch_size,
        sigma,
        thresholds,
        num_classes,
        _seed_for(base_seed, index, 1),
        device,
    )
    rows: list[ThresholdResult] = []
    for threshold_index, threshold in enumerate(thresholds):
        selected_label = _choose_label(selection[threshold_index])
        selected_count = int(estimation[threshold_index, selected_label])
        selected_lower = one_sided_binomial_lower(
            selected_count, n, alpha / 2.0
        )
        competitor_error = alpha / (2.0 * (num_classes - 1))
        runner_upper = max(
            one_sided_binomial_upper(int(count), n, competitor_error)
            for label, count in enumerate(estimation[threshold_index])
            if label != selected_label
        )
        radius = gaussian_joint_mass_radius(
            selected_lower, runner_upper, sigma
        )
        retained = int(estimation[threshold_index].sum())
        rows.append(
            ThresholdResult(
                threshold=threshold,
                selected_label=selected_label,
                selection_retained=int(selection[threshold_index].sum()),
                estimation_retained=retained,
                acceptance_rate=retained / n,
                selected_joint_count=selected_count,
                selected_joint_lower=selected_lower,
                runner_joint_upper=runner_upper,
                joint_radius=radius,
                correct=selected_label == int(true_label),
            )
        )
    return ImageResult(
        index=index,
        true_label=int(true_label),
        elapsed_seconds=time.monotonic() - started,
        thresholds=tuple(rows),
    )


def _threshold_summary(
    rows: list[ImageResult], thresholds: tuple[float, ...], radii: tuple[float, ...]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for threshold_index, threshold in enumerate(thresholds):
        selected = [row.thresholds[threshold_index] for row in rows]
        radius_array = np.asarray([item.joint_radius for item in selected])
        output[str(threshold)] = {
            "images": len(selected),
            "selected_label_accuracy": float(
                np.mean([item.correct for item in selected])
            ),
            "mean_acceptance_rate": float(
                np.mean([item.acceptance_rate for item in selected])
            ),
            "positive_certificate_fraction": float(np.mean(radius_array > 0.0)),
            "median_radius": float(np.median(radius_array)),
            "certified_accuracy": {
                str(radius): float(
                    np.mean(
                        [
                            item.correct and item.joint_radius > radius
                            for item in selected
                        ]
                    )
                )
                for radius in radii
            },
        }
    return output


def _select_threshold(
    development_summary: dict[str, Any],
    thresholds: tuple[float, ...],
    objective_radii: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    scores = {
        str(threshold): float(
            np.mean(
                [
                    development_summary[str(threshold)]["certified_accuracy"][
                        str(radius)
                    ]
                    for radius in objective_radii
                ]
            )
        )
        for threshold in thresholds
    }
    best_score = max(scores.values())
    chosen = min(
        threshold
        for threshold in thresholds
        if math.isclose(scores[str(threshold)], best_score, abs_tol=1e-15)
    )
    return chosen, scores


def _load_completed(path: Path) -> dict[int, ImageResult]:
    if not path.exists():
        return {}
    completed: dict[int, ImageResult] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            threshold_rows = tuple(
                ThresholdResult(**item) for item in payload.pop("thresholds")
            )
            row = ImageResult(thresholds=threshold_rows, **payload)
            completed[row.index] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    thresholds = tuple(float(value) for value in protocol["thresholds"])
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("protocol thresholds must be unique and increasing")
    proposal = protocol["proposal_batches"]
    n0 = int(proposal["selection_samples"])
    n = int(proposal["estimation_samples"])
    alpha = float(proposal["error_probability"])
    base_seed = int(proposal["base_seed"])
    development = range(
        int(protocol["development_indices"]["start"]),
        int(protocol["development_indices"]["stop"]),
    )
    confirmation = range(
        int(protocol["confirmation_indices"]["start"]),
        int(protocol["confirmation_indices"]["stop"]),
    )
    indices = tuple(development) + tuple(confirmation)

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
        "protocol_sha256": _sha256(protocol_path),
        "auditvotes_git_commit": repository_commit,
        "checkpoint_sha256": checkpoint_digest,
        "core_py_sha256": _sha256(code_root / "core.py"),
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "device": torch.cuda.get_device_name(0),
    }
    metadata_path = args.output / "metadata.json"
    if args.resume and metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise RuntimeError("resume metadata does not match the current run")
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    completed = _load_completed(rows_path) if args.resume else {}
    mode = "a" if args.resume else "w"
    with rows_path.open(mode) as handle:
        for position, index in enumerate(indices, start=1):
            if index in completed:
                continue
            image, true_label = dataset[index]
            row = _evaluate_image(
                model,
                image.unsqueeze(0).to(device),
                int(true_label),
                index,
                n0,
                n,
                args.batch_size,
                0.25,
                thresholds,
                alpha,
                num_classes,
                base_seed,
                device,
            )
            completed[index] = row
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{position}/{len(indices)} index={index} "
                f"seconds={row.elapsed_seconds:.2f}",
                flush=True,
            )

    development_rows = [completed[index] for index in development]
    confirmation_rows = [completed[index] for index in confirmation]
    report_radii = tuple(
        float(value) for value in np.arange(0.0, 0.751, 0.05)
    )
    development_summary = _threshold_summary(
        development_rows, thresholds, report_radii
    )
    objective_radii = tuple(
        float(value) for value in protocol["selection_objective"]["radii"]
    )
    chosen, scores = _select_threshold(
        development_summary, thresholds, objective_radii
    )
    confirmation_summary = _threshold_summary(
        confirmation_rows, thresholds, report_radii
    )
    summary = {
        "selected_threshold": chosen,
        "development_objective_scores": scores,
        "objective_radii": objective_radii,
        "development": development_summary,
        "confirmation": confirmation_summary,
        "primary_confirmation": {
            "selected": confirmation_summary[str(chosen)],
            "unfiltered": confirmation_summary["0.0"],
            "released_threshold": confirmation_summary["0.9"],
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary["primary_confirmation"], indent=2))


if __name__ == "__main__":
    main()
