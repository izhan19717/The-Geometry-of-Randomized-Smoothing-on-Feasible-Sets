"""Engineering-only integration gate for all frozen Goal2 controllers.

The executable is frozen only after all three terminal actor/normalizer pairs
exist.  It reuses one already-open SafetyPointGoal2 environment seed and one
already-open activation-noise seed.  It checks controller loading, immutable
inference, exact restored-state query/execute, and the explicit totalized
abstention branch.  It opens no benchmark split and produces no paper result.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import sys
import types
from typing import Any, Callable

import numpy as np

from experiments.iclr2027_safety_gym_engineering_smoke import (
    _set_agent_position,
    execute_verified,
    file_sha256,
    load_actor_bundle,
    oracle_query,
    ordered_fallback_grid,
    snapshot_simulator,
)


def _load_isolated_totalized_core() -> tuple[type[Any], Callable[..., Any]]:
    """Load the two frozen core modules without executing package-wide imports."""

    source_dir = Path(__file__).resolve().parents[1] / "src" / "feasible_robustness"
    module_names = (
        "feasible_robustness",
        "feasible_robustness.ccfs",
        "feasible_robustness.totalized_ccfs",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        package = types.ModuleType("feasible_robustness")
        package.__path__ = [str(source_dir)]
        sys.modules["feasible_robustness"] = package
        loaded: dict[str, Any] = {}
        for short_name in ("ccfs", "totalized_ccfs"):
            full_name = f"feasible_robustness.{short_name}"
            spec = importlib.util.spec_from_file_location(
                full_name, source_dir / f"{short_name}.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load frozen runtime source: {short_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            loaded[short_name] = module
        totalized = loaded["totalized_ccfs"]
        return (
            totalized.TotalizedCCFSSelection,
            totalized.select_first_verified_or_abstain,
        )
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


TotalizedCCFSSelection, select_first_verified_or_abstain = (
    _load_isolated_totalized_core()
)


ENGINEERING_SEED = 1570860784
ACTIVATION_NOISE_SEED = 1368584074
CONTROLLER_SEEDS = (1120275774, 1346831933, 658990255)
DEFAULT_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "SAFETY_GYM_ALL_CONTROLLER_INTEGRATION_PROTOCOL_V1_20260901.json"
)


def _self_sha256() -> str:
    digest = sha256()
    with Path(__file__).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _terminal_row(progress_path: Path) -> dict[str, str]:
    with progress_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 500:
        raise RuntimeError("controller progress table does not contain 500 epochs")
    final = rows[-1]
    if final.get("Train/Epoch") != "500":
        raise RuntimeError("controller terminal epoch is not 500")
    if final.get("Train/TotalSteps") != "10000000":
        raise RuntimeError("controller terminal step is not 10000000")
    return final


def _coarse_actions() -> np.ndarray:
    values = (-1.0, 0.0, 1.0)
    return np.asarray(
        [[first, second] for first in values for second in values],
        dtype=np.float64,
    )


def _negative_selection(
    failed_actions: np.ndarray,
    is_feasible: Callable[[np.ndarray], bool],
) -> TotalizedCCFSSelection:
    matrix = np.asarray(failed_actions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1:] != (2,):
        raise ValueError("failed_actions must contain at least one two-vector")
    return select_first_verified_or_abstain(
        proposals=matrix[:1],
        is_feasible=is_feasible,
        fallback_library=matrix[1:],
    )


def _selection_json(selection: TotalizedCCFSSelection) -> dict[str, Any]:
    return {
        "action": selection.action.tolist() if selection.action is not None else None,
        "selected_index": selection.selected_index,
        "fallback_index": selection.fallback_index,
        "proposal_inspections": selection.proposal_inspections,
        "fallback_inspections": selection.fallback_inspections,
        "used_fallback": selection.used_fallback,
        "abstained": selection.abstained,
    }


def verify_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_all_controller_engineering_gate":
        raise RuntimeError("all-controller integration protocol is not frozen")
    if file_sha256(Path(__file__)) != protocol["source_sha256"]:
        raise RuntimeError("all-controller integration source differs from protocol")
    if protocol["environment"] != args.environment:
        raise RuntimeError("environment differs from all-controller protocol")
    if protocol["engineering_seed"] != args.engineering_seed:
        raise RuntimeError("engineering seed differs from all-controller protocol")
    if protocol["activation_noise_seed"] != args.activation_noise_seed:
        raise RuntimeError("activation-noise seed differs from protocol")
    if protocol["boundary_offset"] != args.boundary_offset:
        raise RuntimeError("boundary offset differs from protocol")
    if tuple(protocol["controller_seeds"]) != CONTROLLER_SEEDS:
        raise RuntimeError("controller seed order differs from protocol")
    for relative, expected in protocol["runtime_source_sha256"].items():
        if file_sha256(Path(relative)) != expected:
            raise RuntimeError(f"runtime source differs from protocol: {relative}")
    return protocol


def _artifact_paths(controller_root: Path, seed: int) -> dict[str, Path]:
    directory = controller_root / f"seed{seed}"
    return {
        "model499.pt": directory / "model499.pt",
        "state499.pkl": directory / "state499.pkl",
        "config.json": directory / "config.json",
        "progress.csv": directory / "progress.csv",
    }


def verify_controller_artifacts(
    paths: dict[str, Path], expected: dict[str, str]
) -> dict[str, str]:
    if set(paths) != set(expected):
        raise RuntimeError("controller artifact set differs from protocol")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    for name, expected_sha256 in expected.items():
        if actual[name] != expected_sha256:
            raise RuntimeError(f"controller artifact SHA-256 mismatch: {name}")
    return actual


def run_controller(
    *,
    env: Any,
    mujoco: Any,
    safety_gymnasium: Any,
    protocol: dict[str, Any],
    controller_root: Path,
    seed: int,
    activation_noise_seed: int,
    engineering_seed: int,
    boundary_offset: float,
) -> dict[str, Any]:
    del safety_gymnasium  # version is recorded at the outer level
    artifact_paths = _artifact_paths(controller_root, seed)
    expected = protocol["controller_artifact_sha256"][str(seed)]
    artifact_hashes = verify_controller_artifacts(artifact_paths, expected)
    final = _terminal_row(artifact_paths["progress.csv"])
    bundle, torch = load_actor_bundle(
        artifact_paths["model499.pt"],
        artifact_paths["state499.pkl"],
        expected_checkpoint_sha256=expected["model499.pt"],
        expected_normalizer_sha256=expected["state499.pkl"],
    )
    normalizer_before = bundle.normalizer.digest

    observation, _ = env.reset(seed=engineering_seed)
    observation = np.asarray(observation, dtype=np.float64)
    center_first, scale_first = bundle.infer(observation, torch)
    center_second, scale_second = bundle.infer(observation, torch)
    inference_exact = bool(
        np.array_equal(center_first, center_second)
        and np.array_equal(scale_first, scale_second)
    )
    rng = np.random.default_rng(activation_noise_seed)
    noisy = np.clip(
        center_first + scale_first * rng.standard_normal(2), -1.0, 1.0
    )
    candidates = [noisy, np.clip(center_first, -1.0, 1.0)]
    candidates.extend(ordered_fallback_grid(center_first, resolution=3))
    live_snapshot = snapshot_simulator(env)
    queried = None
    candidate_index = None
    for index, candidate in enumerate(candidates):
        record = oracle_query(env, live_snapshot, candidate, mujoco)
        if record.feasible:
            queried = record
            candidate_index = index
            break
    if queried is None:
        raise RuntimeError("controller engineering fixture found no executable action")
    actual, query_execute_exact = execute_verified(env, live_snapshot, queried)

    env.reset(seed=engineering_seed)
    task = env.unwrapped.task
    hazard_names = sorted(
        name for name in task.world.geoms if name.startswith("hazard")
    )
    if not hazard_names:
        raise RuntimeError("declared Goal2 fixture contains no hazard")
    hazard = task.world.geoms[hazard_names[0]]
    hazard_center = np.asarray(hazard["pos"][:2], dtype=np.float64)
    hazard_radius = float(np.asarray(hazard["size"], dtype=np.float64)[0])
    target = hazard_center + np.array(
        [hazard_radius + boundary_offset, 0.0], dtype=np.float64
    )
    _set_agent_position(env, target, mujoco)
    abstention_snapshot = snapshot_simulator(env)
    failed_actions = np.asarray(
        [
            action
            for action in _coarse_actions()
            if not oracle_query(env, abstention_snapshot, action, mujoco).feasible
        ],
        dtype=np.float64,
    )
    if failed_actions.size == 0:
        raise RuntimeError("negative fixture contains no failed action")
    selection = _negative_selection(
        failed_actions,
        lambda action: oracle_query(
            env, abstention_snapshot, action, mujoco
        ).feasible,
    )
    post = snapshot_simulator(env)
    state_unchanged = post.digest == abstention_snapshot.digest
    counters_unchanged = bool(
        post.elapsed_steps == abstention_snapshot.elapsed_steps
        and post.builder_steps == abstention_snapshot.builder_steps
    )
    gates = {
        "artifact_hashes_match": True,
        "terminal_row_matches": True,
        "actor_inference_repeats_bit_exactly": inference_exact,
        "actor_parameters_are_frozen": all(
            not parameter.requires_grad for parameter in bundle.actor.parameters()
        ),
        "normalizer_remained_immutable": (
            normalizer_before == bundle.normalizer.digest
        ),
        "queried_transition_was_feasible": bool(queried.feasible),
        "query_execute_serialization_is_exact": bool(query_execute_exact),
        "negative_fixture_emitted_explicit_abstention": bool(
            selection.abstained and selection.action is None
        ),
        "negative_fixture_was_fully_inspected": bool(
            selection.proposal_inspections + selection.fallback_inspections
            == failed_actions.shape[0]
        ),
        "abstention_left_state_unchanged": bool(state_unchanged),
        "abstention_did_not_advance_step_counters": counters_unchanged,
    }
    return {
        "controller_seed": seed,
        "passes": all(gates.values()),
        "artifact_sha256": artifact_hashes,
        "terminal_training_row": {
            "epoch": int(final["Train/Epoch"]),
            "total_steps": int(final["Train/TotalSteps"]),
            "episode_cost": float(final["Metrics/EpCost"]),
            "episode_return": float(final["Metrics/EpRet"]),
        },
        "query_execute": {
            "candidate_index": candidate_index,
            "action": queried.action.tolist(),
            "queried_cost": queried.cost,
            "executed_cost": actual.cost,
            "queried_digest": queried.digest,
            "executed_digest": actual.digest,
            "exact": bool(query_execute_exact),
        },
        "abstention": {
            "failed_fixture_actions": failed_actions.tolist(),
            "selection": _selection_json(selection),
            "state_unchanged": bool(state_unchanged),
            "step_counters_unchanged": counters_unchanged,
        },
        "gates": gates,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco
    import safety_gymnasium
    import torch

    protocol = verify_protocol(args)
    env = safety_gymnasium.make(args.environment)
    try:
        controllers = [
            run_controller(
                env=env,
                mujoco=mujoco,
                safety_gymnasium=safety_gymnasium,
                protocol=protocol,
                controller_root=args.controller_root,
                seed=seed,
                activation_noise_seed=args.activation_noise_seed,
                engineering_seed=args.engineering_seed,
                boundary_offset=args.boundary_offset,
            )
            for seed in CONTROLLER_SEEDS
        ]
    finally:
        env.close()
    passes = all(controller["passes"] for controller in controllers)
    return {
        "status": (
            "all_controller_engineering_integration_passed"
            if passes
            else "all_controller_engineering_integration_failed"
        ),
        "passes": passes,
        "paper_eligibility": "none; engineering mechanics only",
        "protocol_path": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "source_sha256": _self_sha256(),
        "environment": args.environment,
        "engineering_seed": args.engineering_seed,
        "activation_noise_seed": args.activation_noise_seed,
        "controllers": controllers,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "safety_gymnasium": safety_gymnasium.__version__,
            "mujoco": mujoco.__version__,
        },
        "claim_boundary": [
            "No new environment or activation-noise seed is opened.",
            "The finite negative fixture is not the benchmark fallback library.",
            "The run checks mechanics, not policy quality or population behavior.",
            "Abstention is not a physical or safe action.",
            "No development, validation, confirmation, or attack claim is made.",
        ],
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# All-controller SafetyPointGoal2 engineering integration",
        "",
        f"Status: `{result['status']}`.",
        "",
        "This output has no paper-result eligibility.",
        "",
    ]
    for controller in result["controllers"]:
        lines.extend(
            [
                f"## Controller seed {controller['controller_seed']}",
                "",
                *[
                    f"- {'PASS' if passed else 'FAIL'}: `{name}`"
                    for name, passed in controller["gates"].items()
                ],
                "",
            ]
        )
    lines.extend(["## Claim boundary", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment", default="SafetyPointGoal2-v0")
    parser.add_argument("--engineering-seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument(
        "--activation-noise-seed", type=int, default=ACTIVATION_NOISE_SEED
    )
    parser.add_argument("--boundary-offset", type=float, default=2.5e-4)
    parser.add_argument(
        "--controller-root",
        type=Path,
        default=Path("output/iclr2027_safety_gym_safe_controllers_v2"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "output/iclr2027_safety_gym_all_controller_integration/result.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "output/iclr2027_safety_gym_all_controller_integration/result.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "passes": result["passes"]}, indent=2))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
