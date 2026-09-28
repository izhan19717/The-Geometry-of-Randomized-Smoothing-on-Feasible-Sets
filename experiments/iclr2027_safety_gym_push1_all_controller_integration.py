"""Engineering-only terminal-controller integration for SafetyPointPush1-v0.

The program may run only after the three frozen PPO-Lagrangian training jobs
finish.  It reuses one previously opened engineering environment seed and one
previously opened activation-noise seed.  It verifies terminal artifacts,
immutable actor inference, exact restored-state query and execution, and the
explicit abstention branch.  It opens no research split and creates no paper
result.
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
    ActorBundle,
    FrozenNormalizer,
    _set_agent_position,
    file_sha256,
    ordered_fallback_grid,
)
from experiments.iclr2027_safety_gym_push1_engineering_smoke import (
    execute_verified_push1,
    oracle_query_push1,
    snapshot_push1,
)


ENGINEERING_SEED = 1570860784
ACTIVATION_NOISE_SEED = 1368584074
CONTROLLER_SEEDS = (167901043, 603015692, 334024361)
ENVIRONMENT = "SafetyPointPush1-v0"
OBSERVATION_DIMENSION = 76
ACTION_DIMENSION = 2
HIDDEN_SIZES = (64, 64)
DEFAULT_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "SAFETY_GYM_PUSH1_ALL_CONTROLLER_INTEGRATION_PROTOCOL_V1_20260903.json"
)
DEFAULT_CONTROLLER_ROOT = Path("safe_controller_push1_runs_v1_20260901")
DEFAULT_OUTPUT_JSON = Path(
    "output/iclr2027_safety_gym_push1_all_controller_integration_v1/result.json"
)
DEFAULT_OUTPUT_MD = Path(
    "output/iclr2027_safety_gym_push1_all_controller_integration_v1/result.md"
)


def _load_isolated_totalized_core() -> tuple[type[Any], Callable[..., Any]]:
    """Load the two runtime modules without executing package-wide imports."""

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


def _controller_directory(controller_root: Path, seed: int) -> Path:
    parent = (
        controller_root
        / f"ccfs_safe_push1_ppo_lag_v1_seed{seed}"
        / ENVIRONMENT
        / "ppo_lag"
    )
    matches = sorted(parent.glob(f"seed-{seed}-*"))
    matches = [path for path in matches if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one terminal controller directory for seed {seed}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _artifact_paths(controller_root: Path, seed: int) -> dict[str, Path]:
    directory = _controller_directory(controller_root, seed)
    return {
        "model499.pt": directory / "torch_save" / "model499.pt",
        "state499.pkl": directory / "state499.pkl",
        "config.json": directory / "config.json",
        "progress.csv": directory / "progress.csv",
        f"seed{seed}_terminal.log": directory / f"seed{seed}_terminal.log",
        f"seed{seed}_error.log": directory / f"seed{seed}_error.log",
    }


def verify_controller_artifacts(
    paths: dict[str, Path], expected: dict[str, str]
) -> dict[str, str]:
    if set(paths) != set(expected):
        raise RuntimeError("controller artifact set differs from the protocol")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    for name, expected_sha256 in expected.items():
        if actual[name] != expected_sha256:
            raise RuntimeError(f"controller artifact SHA-256 mismatch: {name}")
    return actual


def load_push1_actor_bundle(
    checkpoint_path: Path,
    normalizer_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_normalizer_sha256: str,
) -> tuple[ActorBundle, Any]:
    """Load one 76-input terminal actor and an immutable normalizer copy."""

    if file_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError("actor checkpoint SHA-256 differs from the protocol")
    if file_sha256(normalizer_path) != expected_normalizer_sha256:
        raise RuntimeError("normalizer SHA-256 differs from the protocol")

    import joblib
    import torch
    from safepo.common.model import Actor

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    actor = Actor(
        obs_dim=OBSERVATION_DIMENSION,
        act_dim=ACTION_DIMENSION,
        hidden_sizes=list(HIDDEN_SIZES),
    )
    actor.load_state_dict(state_dict, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    saved_state = joblib.load(normalizer_path)
    if set(saved_state) != {"Normalizer"}:
        raise RuntimeError("unexpected keys in the normalizer checkpoint")
    running = saved_state["Normalizer"]
    normalizer = FrozenNormalizer(
        mean=np.array(running.mean, dtype=np.float64, copy=True),
        variance=np.array(running.var, dtype=np.float64, copy=True),
        count=float(running.count),
    )
    expected_shape = (OBSERVATION_DIMENSION,)
    if normalizer.mean.shape != expected_shape:
        raise RuntimeError("normalizer mean is not the expected 76-vector")
    if normalizer.variance.shape != expected_shape:
        raise RuntimeError("normalizer variance is not the expected 76-vector")
    standard_deviation = (
        torch.exp(actor.log_std).detach().cpu().numpy().astype(np.float64)
    )
    if standard_deviation.shape != (ACTION_DIMENSION,):
        raise RuntimeError("actor scale is not the expected two-vector")
    if not np.all(np.isfinite(standard_deviation)) or np.any(
        standard_deviation <= 0.0
    ):
        raise RuntimeError("actor scale is not finite and positive")
    return ActorBundle(actor, normalizer, standard_deviation), torch


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
    if protocol.get("status") != "frozen_before_push1_controller_integration_v1":
        raise RuntimeError("Push1 controller integration protocol is not frozen")
    if file_sha256(Path(__file__)) != protocol["source_sha256"]:
        raise RuntimeError("Push1 controller integration source differs from protocol")
    if protocol["environment"] != args.environment or args.environment != ENVIRONMENT:
        raise RuntimeError("environment differs from the Push1 protocol")
    if protocol["engineering_seed"] != args.engineering_seed:
        raise RuntimeError("engineering seed differs from the Push1 protocol")
    if protocol["activation_noise_seed"] != args.activation_noise_seed:
        raise RuntimeError("activation-noise seed differs from the Push1 protocol")
    if protocol["boundary_offset"] != args.boundary_offset:
        raise RuntimeError("boundary offset differs from the Push1 protocol")
    if tuple(protocol["controller_seeds"]) != CONTROLLER_SEEDS:
        raise RuntimeError("controller seed order differs from the Push1 protocol")
    if protocol["actor_shape"] != {
        "observation_dimension": OBSERVATION_DIMENSION,
        "action_dimension": ACTION_DIMENSION,
        "hidden_sizes": list(HIDDEN_SIZES),
    }:
        raise RuntimeError("actor shape differs from the Push1 protocol")
    for relative, expected in protocol["runtime_source_sha256"].items():
        if file_sha256(Path(relative)) != expected:
            raise RuntimeError(f"runtime source differs from protocol: {relative}")
    for record in protocol["bound_inputs"]:
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"bound input differs from protocol: {path}")
    return protocol


def run_controller(
    *,
    env: Any,
    mujoco: Any,
    protocol: dict[str, Any],
    controller_root: Path,
    seed: int,
    activation_noise_seed: int,
    engineering_seed: int,
    boundary_offset: float,
) -> dict[str, Any]:
    artifact_paths = _artifact_paths(controller_root, seed)
    expected = protocol["controller_artifact_sha256"][str(seed)]
    artifact_hashes = verify_controller_artifacts(artifact_paths, expected)
    final = _terminal_row(artifact_paths["progress.csv"])
    bundle, torch = load_push1_actor_bundle(
        artifact_paths["model499.pt"],
        artifact_paths["state499.pkl"],
        expected_checkpoint_sha256=expected["model499.pt"],
        expected_normalizer_sha256=expected["state499.pkl"],
    )
    normalizer_before = bundle.normalizer.digest

    observation, _ = env.reset(seed=engineering_seed)
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (OBSERVATION_DIMENSION,):
        raise RuntimeError("Push1 observation is not the expected 76-vector")
    center_first, scale_first = bundle.infer(observation, torch)
    center_second, scale_second = bundle.infer(observation, torch)
    inference_exact = bool(
        np.array_equal(center_first, center_second)
        and np.array_equal(scale_first, scale_second)
    )
    rng = np.random.default_rng(activation_noise_seed)
    noisy = np.clip(
        center_first + scale_first * rng.standard_normal(ACTION_DIMENSION),
        -1.0,
        1.0,
    )
    candidates = [noisy, np.clip(center_first, -1.0, 1.0)]
    candidates.extend(ordered_fallback_grid(center_first, resolution=3))
    live_snapshot = snapshot_push1(env)
    queried = None
    candidate_index = None
    for index, candidate in enumerate(candidates):
        record = oracle_query_push1(env, live_snapshot, candidate, mujoco)
        if record.feasible:
            queried = record
            candidate_index = index
            break
    if queried is None:
        raise RuntimeError("Push1 engineering fixture found no executable action")
    actual, query_execute_exact = execute_verified_push1(env, live_snapshot, queried)

    env.reset(seed=engineering_seed)
    task = env.unwrapped.task
    hazard_names = sorted(
        name for name in task.world.geoms if name.startswith("hazard")
    )
    if not hazard_names:
        raise RuntimeError("declared Push1 fixture contains no hazard")
    hazard = task.world.geoms[hazard_names[0]]
    hazard_center = np.asarray(hazard["pos"][:2], dtype=np.float64)
    hazard_radius = float(np.asarray(hazard["size"], dtype=np.float64)[0])
    target = hazard_center + np.array(
        [hazard_radius + boundary_offset, 0.0], dtype=np.float64
    )
    _set_agent_position(env, target, mujoco)
    task.last_dist_box = float(task.dist_box())
    task.last_box_goal = float(task.dist_box_goal())
    abstention_snapshot = snapshot_push1(env)
    failed_actions = np.asarray(
        [
            action
            for action in _coarse_actions()
            if not oracle_query_push1(
                env, abstention_snapshot, action, mujoco
            ).feasible
        ],
        dtype=np.float64,
    )
    if failed_actions.size == 0:
        raise RuntimeError("Push1 negative fixture contains no failed action")
    selection = _negative_selection(
        failed_actions,
        lambda action: oracle_query_push1(
            env, abstention_snapshot, action, mujoco
        ).feasible,
    )
    post = snapshot_push1(env)
    state_unchanged = post.digest == abstention_snapshot.digest
    counters_unchanged = bool(
        post.core.elapsed_steps == abstention_snapshot.core.elapsed_steps
        and post.core.builder_steps == abstention_snapshot.core.builder_steps
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
            "push1_controller_engineering_integration_passed"
            if passes
            else "push1_controller_engineering_integration_failed"
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
            "The finite negative fixture is not a benchmark fallback library.",
            "The run checks mechanics, not policy quality or population behavior.",
            "Abstention is not a physical or safe action.",
            "No development, validation, confirmation, or attack claim is made.",
        ],
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SafetyPointPush1 terminal-controller engineering integration",
        "",
        f"Status is `{result['status']}`.",
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
                    f"- {'PASS' if passed else 'FAIL'} `{name}`"
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
    parser.add_argument("--environment", default=ENVIRONMENT)
    parser.add_argument("--engineering-seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument(
        "--activation-noise-seed", type=int, default=ACTIVATION_NOISE_SEED
    )
    parser.add_argument("--boundary-offset", type=float, default=2.5e-4)
    parser.add_argument(
        "--controller-root", type=Path, default=DEFAULT_CONTROLLER_ROOT
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
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
