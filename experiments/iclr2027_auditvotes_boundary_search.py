"""Search and independently confirm AuditVotes conditional-label changes.

The search stage uses a differentiable expectation-over-noise objective only to
propose nearby inputs.  The confirmation stage uses fresh Gaussian proposals,
hard released confidence events, and familywise exact pairwise binomial tests.
No search sample is reused for confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def confidence_log_margin_numpy(logits: np.ndarray, label: int, threshold: float) -> np.ndarray:
    """Return a margin that is positive exactly when class ``label`` exceeds threshold."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 1 or not 0 <= label < values.shape[-1]:
        raise ValueError("invalid logits or label")
    if not 0.5 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between one half and one")
    competitors = np.delete(values, label, axis=-1)
    maximum = np.max(competitors, axis=-1, keepdims=True)
    logsumexp = np.squeeze(
        maximum + np.log(np.sum(np.exp(competitors - maximum), axis=-1, keepdims=True)),
        axis=-1,
    )
    return values[..., label] - logsumexp - math.log(threshold / (1.0 - threshold))


def project_l2_box_numpy(original: np.ndarray, proposed: np.ndarray, radius: float) -> np.ndarray:
    """Project a proposed image to the box and then to an L2 ball around original."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    anchor = np.asarray(original, dtype=np.float64)
    candidate = np.clip(np.asarray(proposed, dtype=np.float64), 0.0, 1.0)
    displacement = candidate - anchor
    norm = float(np.linalg.norm(displacement.reshape(-1), ord=2))
    if norm > radius and norm > 0.0:
        displacement *= radius / norm
    return anchor + displacement


def pairwise_winner_pvalues(counts: Iterable[int], winner: int) -> list[float]:
    """Exact one-sided tests of one retained-label mass against every competitor."""
    array = np.asarray(list(counts), dtype=np.int64)
    if array.ndim != 1 or not 0 <= winner < len(array) or np.any(array < 0):
        raise ValueError("invalid counts or winner")
    output: list[float] = []
    for competitor, count in enumerate(array):
        if competitor == winner:
            continue
        total = int(array[winner] + count)
        output.append(
            1.0
            if total == 0
            else float(
                binomtest(
                    int(array[winner]),
                    total,
                    p=0.5,
                    alternative="greater",
                ).pvalue
            )
        )
    return output


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "frozen_before_boundary_search":
        raise ValueError("protocol is not frozen for boundary search")
    return payload


def _load_reference_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def _validate_candidates(
    protocol: dict[str, Any], reference_rows: dict[int, dict[str, Any]]
) -> None:
    for candidate in protocol["candidates"]:
        index = int(candidate["index"])
        if index not in reference_rows:
            raise ValueError(f"missing reference row {index}")
        row = reference_rows[index]
        exact_fields = {
            "original_label": "conditional_label",
            "alternative_label": "gaussian_label",
            "true_label": "true_label",
        }
        for protocol_field, row_field in exact_fields.items():
            if int(candidate[protocol_field]) != int(row[row_field]):
                raise ValueError(f"candidate {index} disagrees on {protocol_field}")
        for protocol_field, row_field in (
            ("released_radius", "released_radius"),
            ("joint_radius", "joint_radius"),
        ):
            if not math.isclose(
                float(candidate[protocol_field]),
                float(row[row_field]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise ValueError(f"candidate {index} disagrees on {protocol_field}")


def _load_model_and_dataset(repo_root: Path, checkpoint: Path):
    import torch

    code_root = repo_root / "ImageClassify_Gaussian/ImageClassify_Conf/code"
    sys.path.insert(0, str(code_root))
    from architectures import get_architecture  # type: ignore
    from datasets import get_dataset, get_num_classes  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen study")
    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = get_architecture(payload["arch"], "cifar10")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, get_dataset("cifar10", "test"), get_num_classes("cifar10"), device


def _torch_confidence_margin(logits, label: int, threshold: float):
    import torch

    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[label] = False
    return (
        logits[:, label]
        - torch.logsumexp(logits[:, mask], dim=1)
        - math.log(threshold / (1.0 - threshold))
    )


def _torch_project(original, proposed, radius: float):
    import torch

    candidate = proposed.clamp(0.0, 1.0)
    displacement = candidate - original
    norm = torch.linalg.vector_norm(displacement)
    scale = torch.clamp(torch.as_tensor(radius, device=norm.device) / norm.clamp_min(1e-30), max=1.0)
    return original + displacement * scale


def _hard_counts(
    model,
    image,
    proposals: int,
    batch_size: int,
    sigma: float,
    threshold: float,
    num_classes: int,
    seed: int,
    device,
) -> np.ndarray:
    import torch

    counts = torch.zeros(num_classes, dtype=torch.int64, device="cpu")
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
            retained = labels[confidence > threshold]
            if retained.numel():
                counts += torch.bincount(retained.cpu(), minlength=num_classes)
            completed += size
    return counts.numpy()


def _optimize_candidate(
    model,
    original,
    candidate: dict[str, Any],
    search: dict[str, Any],
    sigma: float,
    threshold: float,
    base_seed: int,
    device,
):
    import torch

    original_label = int(candidate["original_label"])
    alternative_label = int(candidate["alternative_label"])
    radius = float(candidate["released_radius"]) * float(search["radius_factor"])
    restarts = int(search["restarts"])
    steps = int(search["steps"])
    eot = int(search["eot_proposals"])
    microbatch = int(search["gradient_microbatch"])
    temperatures = [float(value) for value in search["temperatures"]]
    if eot % 2 or eot % microbatch:
        raise ValueError("EOT proposals must be even and divisible by the microbatch")

    saved: list[tuple[float, Any, str]] = []
    for restart in range(restarts):
        generator = torch.Generator(device=device)
        generator.manual_seed(derived_seed(base_seed, candidate["index"], restart, "init"))
        if restart == 0:
            delta = torch.zeros_like(original)
        else:
            delta = torch.randn(original.shape, generator=generator, device=device, dtype=original.dtype)
            delta = delta / torch.linalg.vector_norm(delta).clamp_min(1e-30)
            delta = delta * radius * (restart / restarts)
            delta = _torch_project(original, original + delta, radius) - original
        delta = delta.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=radius * float(search["step_fraction"]))
        best_value = -math.inf
        best_delta = delta.detach().clone()
        final_value = -math.inf
        for step in range(steps):
            temperature = temperatures[min(len(temperatures) - 1, step * len(temperatures) // steps)]
            noise_generator = torch.Generator(device=device)
            noise_generator.manual_seed(
                derived_seed(base_seed, candidate["index"], restart, step, "eot")
            )
            half = torch.randn(
                (eot // 2, *original.shape[1:]),
                generator=noise_generator,
                device=device,
                dtype=original.dtype,
            )
            noises = torch.cat([half, -half], dim=0)
            optimizer.zero_grad(set_to_none=True)
            objective_value = 0.0
            for start in range(0, eot, microbatch):
                noise = noises[start : start + microbatch]
                perturbed = original + delta + sigma * noise
                logits = model(perturbed)
                original_margin = _torch_confidence_margin(logits, original_label, threshold)
                alternative_margin = _torch_confidence_margin(logits, alternative_label, threshold)
                objective = (
                    torch.sigmoid(alternative_margin / temperature)
                    - torch.sigmoid(original_margin / temperature)
                ).mean()
                (-objective * (microbatch / eot)).backward()
                objective_value += float(objective.detach()) * (microbatch / eot)
            optimizer.step()
            with torch.no_grad():
                projected = _torch_project(original, original + delta, radius)
                delta.copy_(projected - original)
            final_value = objective_value
            if objective_value > best_value:
                best_value = objective_value
                best_delta = delta.detach().clone()
        saved.append((best_value, best_delta, f"restart_{restart}_best"))
        saved.append((final_value, delta.detach().clone(), f"restart_{restart}_final"))
    return radius, saved


def run_search(args: argparse.Namespace) -> None:
    import torch

    protocol = _load_protocol(args.protocol)
    reference_rows = _load_reference_rows(args.reference_rows)
    _validate_candidates(protocol, reference_rows)
    provenance = protocol["provenance"]
    if sha256_path(Path(__file__).resolve()) != provenance["search_script_sha256"]:
        raise ValueError("search script digest does not match protocol")
    if _git_head(args.repo_root) != provenance["auditvotes_git_commit"]:
        raise ValueError("AuditVotes commit does not match protocol")
    if sha256_path(args.checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint digest does not match protocol")
    core_path = args.repo_root / "ImageClassify_Gaussian/ImageClassify_Conf/code/core.py"
    if sha256_path(core_path) != provenance["core_py_sha256"]:
        raise ValueError("core.py digest does not match protocol")

    model, dataset, num_classes, device = _load_model_and_dataset(args.repo_root, args.checkpoint)
    sigma = float(protocol["model"]["sigma"])
    threshold = float(protocol["model"]["confidence_threshold"])
    search = protocol["search"]
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    screened: list[dict[str, Any]] = []
    started = time.monotonic()
    for candidate in protocol["candidates"]:
        index = int(candidate["index"])
        image, _ = dataset[index]
        original = image.unsqueeze(0).to(device)
        radius, proposed = _optimize_candidate(
            model,
            original,
            candidate,
            search,
            sigma,
            threshold,
            int(protocol["seeds"]["search"]),
            device,
        )
        best: dict[str, Any] | None = None
        for proposal_number, (surrogate, delta, source) in enumerate(proposed):
            perturbed = (original + delta).detach()
            counts = _hard_counts(
                model,
                perturbed,
                int(search["screen_proposals"]),
                int(search["inference_batch_size"]),
                sigma,
                threshold,
                num_classes,
                derived_seed(
                    int(protocol["seeds"]["screen"]), index, proposal_number
                ),
                device,
            )
            original_count = int(counts[int(candidate["original_label"])])
            alternative_count = int(counts[int(candidate["alternative_label"])])
            row = {
                "index": index,
                "source": source,
                "surrogate_objective": surrogate,
                "counts": [int(value) for value in counts],
                "original_count": original_count,
                "alternative_count": alternative_count,
                "screen_difference": alternative_count - original_count,
                "radius_limit": radius,
                "l2_distance": float(
                    torch.linalg.vector_norm(perturbed - original).double().cpu()
                ),
                "tensor": perturbed.cpu().numpy().astype(np.float32),
            }
            if best is None or (row["screen_difference"], row["surrogate_objective"]) > (
                best["screen_difference"],
                best["surrogate_objective"],
            ):
                best = row
        if best is None:
            raise RuntimeError("optimizer produced no proposal")
        asset = output_dir / f"candidate_{index}.npz"
        np.savez_compressed(asset, image=best.pop("tensor"))
        best["asset"] = asset.name
        best["asset_sha256"] = sha256_path(asset)
        best["original_label"] = int(candidate["original_label"])
        best["alternative_label"] = int(candidate["alternative_label"])
        best["released_radius"] = float(candidate["released_radius"])
        screened.append(best)
        print(
            f"index={index} screen_difference={best['screen_difference']} "
            f"distance={best['l2_distance']:.8f}",
            flush=True,
        )

    eligible = [row for row in screened if int(row["screen_difference"]) > 0]
    candidate_order = {int(row["index"]): position for position, row in enumerate(protocol["candidates"])}
    eligible.sort(
        key=lambda row: (
            -int(row["screen_difference"]),
            candidate_order[int(row["index"])],
        )
    )
    selected = eligible[: int(search["maximum_confirmed_pairs"])]
    summary = {
        "status": "search_complete_confirmation_unopened",
        "protocol_sha256": sha256_path(args.protocol),
        "reference_rows_sha256": sha256_path(args.reference_rows),
        "elapsed_seconds": time.monotonic() - started,
        "screened": screened,
        "selected": selected,
    }
    path = output_dir / "search_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "summary": str(path)}, indent=2))


def run_confirm(args: argparse.Namespace) -> None:
    protocol = _load_protocol(args.protocol)
    search_summary = json.loads(args.search_summary.read_text())
    if search_summary["protocol_sha256"] != sha256_path(args.protocol):
        raise ValueError("search summary uses another protocol")
    selected = search_summary["selected"]
    maximum_pairs = int(protocol["search"]["maximum_confirmed_pairs"])
    if len(selected) > maximum_pairs:
        raise ValueError("too many selected pairs")
    provenance = protocol["provenance"]
    if sha256_path(Path(__file__).resolve()) != provenance["search_script_sha256"]:
        raise ValueError("search script digest does not match protocol")
    if _git_head(args.repo_root) != provenance["auditvotes_git_commit"]:
        raise ValueError("AuditVotes commit does not match protocol")
    if sha256_path(args.checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint digest does not match protocol")

    model, dataset, num_classes, device = _load_model_and_dataset(args.repo_root, args.checkpoint)
    sigma = float(protocol["model"]["sigma"])
    threshold = float(protocol["model"]["confidence_threshold"])
    confirmation = protocol["confirmation"]
    per_comparison_alpha = float(confirmation["familywise_alpha"]) / (
        maximum_pairs * 2 * (num_classes - 1)
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for pair_position, selected_row in enumerate(selected):
        index = int(selected_row["index"])
        asset = args.search_summary.parent / selected_row["asset"]
        if sha256_path(asset) != selected_row["asset_sha256"]:
            raise ValueError(f"candidate asset digest changed for {index}")
        perturbed_array = np.load(asset, allow_pickle=False)["image"]
        original_image, _ = dataset[index]
        original = original_image.unsqueeze(0).to(device)
        import torch

        perturbed = torch.as_tensor(perturbed_array, device=device, dtype=original.dtype)
        original_label = int(selected_row["original_label"])
        alternative_label = int(selected_row["alternative_label"])
        endpoint_rows = []
        for endpoint_name, image, winner in (
            ("original", original, original_label),
            ("perturbed", perturbed, alternative_label),
        ):
            counts = _hard_counts(
                model,
                image,
                int(confirmation["proposals_per_endpoint"]),
                int(confirmation["inference_batch_size"]),
                sigma,
                threshold,
                num_classes,
                derived_seed(
                    int(protocol["seeds"]["confirmation"]),
                    pair_position,
                    index,
                    endpoint_name,
                ),
                device,
            )
            pvalues = pairwise_winner_pvalues(counts, winner)
            endpoint_rows.append(
                {
                    "endpoint": endpoint_name,
                    "winner": winner,
                    "counts": [int(value) for value in counts],
                    "retained": int(counts.sum()),
                    "pairwise_pvalues": pvalues,
                    "winner_verified": bool(max(pvalues) < per_comparison_alpha),
                }
            )
        distance = float(
            np.linalg.norm(
                perturbed_array.astype(np.float64)
                - original_image.unsqueeze(0).numpy().astype(np.float64)
            )
        )
        inside = distance < float(selected_row["released_radius"])
        verified = bool(
            inside
            and endpoint_rows[0]["winner_verified"]
            and endpoint_rows[1]["winner_verified"]
            and original_label != alternative_label
        )
        rows.append(
            {
                "index": index,
                "original_label": original_label,
                "alternative_label": alternative_label,
                "released_radius": float(selected_row["released_radius"]),
                "l2_distance": distance,
                "inside_released_radius": inside,
                "endpoints": endpoint_rows,
                "label_change_verified": verified,
            }
        )
        print(f"index={index} verified={verified} distance={distance:.8f}", flush=True)

    output = {
        "status": "confirmation_complete",
        "protocol_sha256": sha256_path(args.protocol),
        "search_summary_sha256": sha256_path(args.search_summary),
        "familywise_alpha": float(confirmation["familywise_alpha"]),
        "per_comparison_alpha": per_comparison_alpha,
        "selected_pair_count": len(selected),
        "verified_pair_count": sum(bool(row["label_change_verified"]) for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--protocol", type=Path, required=True)
    shared.add_argument("--repo-root", type=Path, required=True)
    shared.add_argument("--checkpoint", type=Path, required=True)

    search = subparsers.add_parser("search", parents=[shared])
    search.add_argument("--reference-rows", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.set_defaults(function=run_search)

    confirm = subparsers.add_parser("confirm", parents=[shared])
    confirm.add_argument("--search-summary", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    confirm.set_defaults(function=run_confirm)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
