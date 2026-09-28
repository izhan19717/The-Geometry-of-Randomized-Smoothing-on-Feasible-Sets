"""Frozen development evaluation of feasible trajectory smoothing on Push1.

The evaluator loads all three predeclared terminal PPO-Lagrangian controllers.
It compares a finite correlated first-verified selector with single-proposal,
IID-bundle, and capped moving-rejection controls.  Every filtered action is
queried from an exact restored simulator snapshot before execution.  Exhausted
finite fallback libraries emit abstention and do not step the environment.
The primary event reaches the first native goal before any positive declared
cost, mismatch, termination, truncation, abstention, or the 1,000-step horizon.

Only the development split is implemented.  Its result cannot enter the paper
as held-out evidence and cannot authorize validation or confirmation without a
new frozen protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import beta

from experiments.iclr2027_safety_gym_engineering_smoke import (
    file_sha256,
    ordered_fallback_grid,
)
from experiments.iclr2027_safety_gym_push1_all_controller_integration import (
    CONTROLLER_SEEDS,
    ENVIRONMENT,
    _artifact_paths,
    load_push1_actor_bundle,
    verify_controller_artifacts,
)
from experiments.iclr2027_safety_gym_push1_engineering_smoke import (
    execute_verified_push1,
    oracle_query_push1,
    snapshot_push1,
)
from feasible_robustness.ccfs import comparison_multiplier


SOURCE_PATH = Path("experiments/iclr2027_safety_gym_push1_trajectory_development.py")
TEST_PATH = Path("tests/test_iclr2027_safety_gym_push1_trajectory_development.py")
LAUNCHER_PATH = Path(
    "scripts/launch_iclr2027_safety_gym_push1_trajectory_development_vm_v1.sh"
)
DEFAULT_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "SAFETY_GYM_PUSH1_TRAJECTORY_DEVELOPMENT_PROTOCOL_V1_20260903.json"
)
DEFAULT_CONTROLLER_ROOT = Path("safe_controller_push1_runs_v1_20260901")
DEFAULT_OUTPUT_ROOT = Path(
    "output/iclr2027_safety_gym_push1_trajectory_development_v1"
)
PRIMARY_METHOD = "ccfs_lambda1p05_k4"
MATCHED_SINGLE_METHOD = "verified_k1_matched_lambda1p05"
SAME_SCALE_SINGLE_METHOD = "verified_k1_same_scale"
LAMBDA_TARGET = 1.05


def stable_seed(label: str) -> tuple[str, int]:
    digest = sha256(label.encode("utf-8")).hexdigest()
    return digest, int(digest[:8], 16) % (2**31)


def method_configurations() -> list[dict[str, Any]]:
    rho = 59.0 / 63.0
    matched_iid_factor = math.sqrt(4.0 / LAMBDA_TARGET)
    matched_single_factor = 1.0 / math.sqrt(LAMBDA_TARGET)
    rows = [
        {
            "method": "gaussian_k1_unfiltered",
            "family": "unfiltered_single",
            "K": 1,
            "rho": 0.0,
            "sigma_factor": 1.0,
            "certificate_multiplier": 1.0,
            "certificate_valid": True,
        },
        {
            "method": SAME_SCALE_SINGLE_METHOD,
            "family": "verified_bundle",
            "K": 1,
            "rho": 0.0,
            "sigma_factor": 1.0,
            "certificate_multiplier": 1.0,
            "certificate_valid": True,
        },
        {
            "method": MATCHED_SINGLE_METHOD,
            "family": "verified_bundle",
            "K": 1,
            "rho": 0.0,
            "sigma_factor": matched_single_factor,
            "certificate_multiplier": LAMBDA_TARGET,
            "certificate_valid": True,
        },
        {
            "method": PRIMARY_METHOD,
            "family": "verified_bundle",
            "K": 4,
            "rho": rho,
            "sigma_factor": 1.0,
            "certificate_multiplier": LAMBDA_TARGET,
            "certificate_valid": True,
        },
        {
            "method": "iid_k4_same_marginal",
            "family": "verified_bundle",
            "K": 4,
            "rho": 0.0,
            "sigma_factor": 1.0,
            "certificate_multiplier": 4.0,
            "certificate_valid": True,
        },
        {
            "method": "iid_k4_matched_lambda1p05",
            "family": "verified_bundle",
            "K": 4,
            "rho": 0.0,
            "sigma_factor": matched_iid_factor,
            "certificate_multiplier": LAMBDA_TARGET,
            "certificate_valid": True,
        },
        {
            "method": "moving_rejection_cap256",
            "family": "moving_rejection",
            "K": 256,
            "rho": 0.0,
            "sigma_factor": 1.0,
            "certificate_multiplier": None,
            "certificate_valid": False,
        },
    ]
    names = [row["method"] for row in rows]
    if len(names) != len(set(names)):
        raise AssertionError("method names must be unique")
    return rows


def reconstructed_multiplier(config: dict[str, Any]) -> float | None:
    if not config["certificate_valid"]:
        return None
    return float(
        comparison_multiplier(int(config["K"]), float(config["rho"]))
        / float(config["sigma_factor"]) ** 2
    )


def one_sided_clopper_pearson_lower(
    successes: int, trials: int, alpha: float
) -> float:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def stopped_reach_outcome(
    *,
    exact: bool,
    cost: float,
    goal_met: bool,
    terminated: bool,
    truncated: bool,
) -> tuple[str | None, bool]:
    """Classify one executed transition for the stopped reach event.

    Failure takes precedence when a transition reports more than one terminal
    condition.  A successful trace reaches the first native goal before any
    positive declared cost, query-to-execution mismatch, environment
    termination, truncation, abstention, or the declared horizon.
    """

    if not exact:
        return "query_execute_mismatch", False
    if not math.isfinite(cost):
        return "nonfinite_cost", False
    if cost != 0.0:
        return "positive_cost", False
    if terminated:
        return "terminated", False
    if truncated:
        return "truncated", False
    if goal_met:
        return "first_goal", True
    return None, False


def protocol_method_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    expected = method_configurations()
    if protocol["methods"] != expected:
        raise RuntimeError("method configurations differ from the frozen protocol")
    for row in expected:
        reconstructed = reconstructed_multiplier(row)
        declared = row["certificate_multiplier"]
        if declared is None:
            if reconstructed is not None:
                raise RuntimeError("invalid row received a certificate multiplier")
        elif not math.isclose(
            float(declared), float(reconstructed), rel_tol=0.0, abs_tol=2e-14
        ):
            raise RuntimeError("certificate multiplier is inconsistent")
    return expected


def verify_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_push1_trajectory_development_v1":
        raise RuntimeError("Push1 trajectory development protocol is not frozen")
    if protocol.get("paper_eligibility") != "development_only":
        raise RuntimeError("protocol paper eligibility is incorrect")
    if protocol["environment"] != ENVIRONMENT:
        raise RuntimeError("environment differs from protocol")
    if file_sha256(SOURCE_PATH) != protocol["source_sha256"]:
        raise RuntimeError("source differs from protocol")
    if file_sha256(TEST_PATH) != protocol["test_sha256"]:
        raise RuntimeError("test differs from protocol")
    if file_sha256(LAUNCHER_PATH) != protocol["launcher_sha256"]:
        raise RuntimeError("launcher differs from protocol")
    for path, expected in protocol["runtime_source_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"runtime source differs from protocol at {path}")
    for record in protocol["bound_inputs"]:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"bound input differs from protocol at {record['path']}")
    if tuple(protocol["controller_seeds"]) != CONTROLLER_SEEDS:
        raise RuntimeError("controller seed order differs from protocol")
    for record in protocol["development_environment_seeds"]:
        digest, seed = stable_seed(record["label"])
        if digest != record["label_sha256"] or seed != record["seed"]:
            raise RuntimeError("development seed derivation differs from protocol")
    protocol_method_rows(protocol)
    return protocol


def raw_innovations(
    base_seed: int,
    controller_seed: int,
    environment_seed: int,
    scale_index: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequence = np.random.SeedSequence(
        [base_seed, controller_seed, environment_seed, scale_index, step]
    )
    rng = np.random.default_rng(sequence)
    return rng.standard_normal(2), rng.standard_normal((4, 2))


def proposal_block(
    center: np.ndarray,
    base_sigma: float,
    config: dict[str, Any],
    shared: np.ndarray,
    individual: np.ndarray,
) -> np.ndarray:
    cap = int(config["K"])
    if config["family"] == "moving_rejection":
        raise ValueError("moving rejection uses its own finite stream")
    if cap > individual.shape[0]:
        raise ValueError("innovation matrix is smaller than the proposal cap")
    rho = float(config["rho"])
    sigma = float(base_sigma * float(config["sigma_factor"]))
    noise = math.sqrt(rho) * shared[None, :]
    noise = noise + math.sqrt(1.0 - rho) * individual[:cap]
    return np.clip(center[None, :] + sigma * noise, -1.0, 1.0)


def moving_rng(
    base_seed: int,
    controller_seed: int,
    environment_seed: int,
    scale_index: int,
    step: int,
) -> np.random.Generator:
    code, _ = stable_seed("moving-rejection-cap256")
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                base_seed,
                controller_seed,
                environment_seed,
                scale_index,
                step,
                int(code[:8], 16),
            ]
        )
    )


def _verified_selection(
    env: Any,
    mujoco: Any,
    snapshot: Any,
    proposals: np.ndarray,
    center: np.ndarray,
    fallback_resolution: int,
) -> dict[str, Any]:
    proposal_queries = 0
    for index, action in enumerate(proposals):
        proposal_queries += 1
        queried = oracle_query_push1(env, snapshot, action, mujoco)
        if queried.feasible:
            actual, exact = execute_verified_push1(env, snapshot, queried)
            return {
                "actual": actual,
                "exact": bool(exact),
                "proposal_queries": proposal_queries,
                "fallback_queries": 0,
                "selected_index": index,
                "used_fallback": False,
                "abstained": False,
            }
    fallback_queries = 0
    for fallback_index, action in enumerate(
        ordered_fallback_grid(center, resolution=fallback_resolution)
    ):
        fallback_queries += 1
        queried = oracle_query_push1(env, snapshot, action, mujoco)
        if queried.feasible:
            actual, exact = execute_verified_push1(env, snapshot, queried)
            return {
                "actual": actual,
                "exact": bool(exact),
                "proposal_queries": proposal_queries,
                "fallback_queries": fallback_queries,
                "selected_index": None,
                "fallback_index": fallback_index,
                "used_fallback": True,
                "abstained": False,
            }
    return {
        "actual": None,
        "exact": True,
        "proposal_queries": proposal_queries,
        "fallback_queries": fallback_queries,
        "selected_index": None,
        "used_fallback": False,
        "abstained": True,
    }


def _moving_selection(
    env: Any,
    mujoco: Any,
    snapshot: Any,
    center: np.ndarray,
    sigma: float,
    cap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    for index in range(cap):
        action = np.clip(center + sigma * rng.standard_normal(2), -1.0, 1.0)
        queried = oracle_query_push1(env, snapshot, action, mujoco)
        if queried.feasible:
            actual, exact = execute_verified_push1(env, snapshot, queried)
            return {
                "actual": actual,
                "exact": bool(exact),
                "proposal_queries": index + 1,
                "fallback_queries": 0,
                "selected_index": index,
                "used_fallback": False,
                "abstained": False,
            }
    return {
        "actual": None,
        "exact": True,
        "proposal_queries": cap,
        "fallback_queries": 0,
        "selected_index": None,
        "used_fallback": False,
        "abstained": True,
    }


def run_episode(
    env: Any,
    mujoco: Any,
    torch: Any,
    bundle: Any,
    controller_seed: int,
    environment_seed: int,
    scale_index: int,
    base_sigma: float,
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    observation, _ = env.reset(seed=environment_seed)
    observation = np.asarray(observation, dtype=np.float64)
    total_reward = 0.0
    cumulative_cost = 0.0
    executed_steps = 0
    proposal_queries = 0
    fallback_queries = 0
    rejected_proposals = 0
    fallback_steps = 0
    query_execute_mismatches = 0
    positive_cost_steps = 0
    goal_met = False
    abstained = False
    event = False
    stop_reason = "horizon"
    started = time.perf_counter()
    for step in range(int(protocol["horizon"])):
        raw_center, _ = bundle.infer(observation, torch)
        center = np.clip(raw_center, -1.0, 1.0)
        shared, individual = raw_innovations(
            int(protocol["activation_base_seed"]),
            controller_seed,
            environment_seed,
            scale_index,
            step,
        )
        family = config["family"]
        if family == "unfiltered_single":
            action = proposal_block(
                center, base_sigma, config, shared, individual
            )[0]
            next_observation, reward, cost, terminated, truncated, info = env.step(
                action
            )
            actual = None
            exact = True
            proposal_queries += 0
        else:
            snapshot = snapshot_push1(env)
            if family == "verified_bundle":
                proposals = proposal_block(
                    center, base_sigma, config, shared, individual
                )
                selection = _verified_selection(
                    env,
                    mujoco,
                    snapshot,
                    proposals,
                    center,
                    int(protocol["fallback_resolution"]),
                )
            elif family == "moving_rejection":
                selection = _moving_selection(
                    env,
                    mujoco,
                    snapshot,
                    center,
                    base_sigma,
                    int(config["K"]),
                    moving_rng(
                        int(protocol["activation_base_seed"]),
                        controller_seed,
                        environment_seed,
                        scale_index,
                        step,
                    ),
                )
            else:
                raise RuntimeError(f"unknown method family {family}")
            proposal_queries += int(selection["proposal_queries"])
            fallback_queries += int(selection["fallback_queries"])
            rejected_proposals += int(selection["proposal_queries"])
            if selection["selected_index"] is not None:
                rejected_proposals -= 1
            fallback_steps += int(selection["used_fallback"])
            if selection["abstained"]:
                abstained = True
                stop_reason = "abstention"
                if snapshot_push1(env).digest != snapshot.digest:
                    raise RuntimeError("abstention changed the simulator state")
                break
            actual = selection["actual"]
            exact = bool(selection["exact"])
            if actual is None:
                raise RuntimeError("non-abstaining selection has no transition")
            next_observation = actual.observation
            reward = actual.reward
            cost = actual.cost
            terminated = actual.terminated
            truncated = actual.truncated
            info = actual.info
        if next_observation is None or reward is None or cost is None:
            raise RuntimeError("executed transition is incomplete")
        total_reward += float(reward)
        cumulative_cost += float(cost)
        positive_cost_steps += int(float(cost) != 0.0)
        query_execute_mismatches += int(not exact)
        executed_steps += 1
        observation = np.asarray(next_observation, dtype=np.float64)
        current_goal = bool(info.get("goal_met", False))
        goal_met = goal_met or current_goal
        transition_reason, transition_event = stopped_reach_outcome(
            exact=exact,
            cost=float(cost),
            goal_met=current_goal,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if transition_reason is not None:
            stop_reason = transition_reason
            event = transition_event
            break
    runtime_ms = 1000.0 * (time.perf_counter() - started)
    if event != (stop_reason == "first_goal"):
        raise RuntimeError("stopped reach-event state is inconsistent")
    return {
        "controller_seed": controller_seed,
        "environment_seed": environment_seed,
        "noise_scale": base_sigma,
        "noise_scale_index": scale_index,
        "method": config["method"],
        "family": family,
        "trajectory_event": int(event),
        "goal_met": int(goal_met),
        "abstained": int(abstained),
        "stop_reason": stop_reason,
        "zero_cumulative_declared_cost": int(positive_cost_steps == 0),
        "cumulative_declared_cost": cumulative_cost,
        "episode_return": total_reward,
        "executed_steps": executed_steps,
        "proposal_queries": proposal_queries,
        "fallback_queries": fallback_queries,
        "rejected_proposals": rejected_proposals,
        "fallback_steps": fallback_steps,
        "query_execute_mismatches": query_execute_mismatches,
        "runtime_ms": runtime_ms,
        "certificate_valid": bool(config["certificate_valid"]),
        "certificate_multiplier": config["certificate_multiplier"],
    }


def evaluate_shard(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if args.shard_count != int(protocol["shard_count"]):
        raise RuntimeError("shard count differs from protocol")
    import mujoco
    import safety_gymnasium

    records: list[dict[str, Any]] = []
    selected_seeds = [
        int(record["seed"])
        for index, record in enumerate(protocol["development_environment_seeds"])
        if index % args.shard_count == args.shard_index
    ]
    configurations = protocol_method_rows(protocol)
    env = safety_gymnasium.make(ENVIRONMENT)
    try:
        for controller_seed in CONTROLLER_SEEDS:
            paths = _artifact_paths(args.controller_root, controller_seed)
            expected = protocol["controller_artifact_sha256"][str(controller_seed)]
            verify_controller_artifacts(paths, expected)
            bundle, torch = load_push1_actor_bundle(
                paths["model499.pt"],
                paths["state499.pkl"],
                expected_checkpoint_sha256=expected["model499.pt"],
                expected_normalizer_sha256=expected["state499.pkl"],
            )
            normalizer_digest = bundle.normalizer.digest
            for scale_index, base_sigma in enumerate(protocol["noise_scales"]):
                for environment_seed in selected_seeds:
                    for config in configurations:
                        record = run_episode(
                            env,
                            mujoco,
                            torch,
                            bundle,
                            controller_seed,
                            environment_seed,
                            scale_index,
                            float(base_sigma),
                            config,
                            protocol,
                        )
                        records.append(record)
            if bundle.normalizer.digest != normalizer_digest:
                raise RuntimeError("frozen normalizer changed during evaluation")
    finally:
        env.close()
    return {
        "status": "push1_trajectory_development_shard_complete",
        "paper_eligibility": "development_only",
        "protocol_sha256": file_sha256(args.protocol),
        "source_sha256": file_sha256(SOURCE_PATH),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "environment_seeds": selected_seeds,
        "records": records,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "safety_gymnasium": safety_gymnasium.__version__,
            "mujoco": mujoco.__version__,
        },
    }


def _summary_rows(
    records: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            int(record["controller_seed"]),
            float(record["noise_scale"]),
            str(record["method"]),
        )
        groups[key].append(record)
    comparison_count = len(groups)
    alpha = float(protocol["familywise_alpha"]) / comparison_count
    rows = []
    for (controller_seed, noise_scale, method), values in sorted(groups.items()):
        count = len(values)
        successes = sum(int(value["trajectory_event"]) for value in values)
        lower = one_sided_clopper_pearson_lower(successes, count, alpha)
        multiplier = values[0]["certificate_multiplier"]
        radius = None
        if multiplier is not None and lower > float(protocol["target_event_probability"]):
            radius = float(
                noise_scale
                / math.sqrt(float(multiplier))
                * (
                    ndtri(lower)
                    - ndtri(float(protocol["target_event_probability"]))
                )
            )
        rows.append(
            {
                "controller_seed": controller_seed,
                "noise_scale": noise_scale,
                "method": method,
                "trajectory_count": count,
                "trajectory_event_count": successes,
                "trajectory_event_probability": successes / count,
                "simultaneous_cp_lower": lower,
                "certified_adaptive_center_radius": radius,
                "goal_probability": float(np.mean([v["goal_met"] for v in values])),
                "abstention_probability": float(
                    np.mean([v["abstained"] for v in values])
                ),
                "zero_cost_probability": float(
                    np.mean([v["zero_cumulative_declared_cost"] for v in values])
                ),
                "mean_return": float(np.mean([v["episode_return"] for v in values])),
                "mean_executed_steps": float(
                    np.mean([v["executed_steps"] for v in values])
                ),
                "mean_proposal_queries": float(
                    np.mean([v["proposal_queries"] for v in values])
                ),
                "mean_fallback_queries": float(
                    np.mean([v["fallback_queries"] for v in values])
                ),
                "mean_rejected_proposals": float(
                    np.mean([v["rejected_proposals"] for v in values])
                ),
                "trajectory_fallback_probability": float(
                    np.mean([v["fallback_steps"] > 0 for v in values])
                ),
                "mean_runtime_ms": float(np.mean([v["runtime_ms"] for v in values])),
                "query_execute_mismatches": int(
                    sum(v["query_execute_mismatches"] for v in values)
                ),
                "certificate_valid": bool(values[0]["certificate_valid"]),
                "certificate_multiplier": multiplier,
            }
        )
    return rows


def _screening(summary: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (row["controller_seed"], row["noise_scale"], row["method"]): row
        for row in summary
    }
    evaluations = []
    for scale in protocol["noise_scales"]:
        controller_checks = []
        for seed in CONTROLLER_SEEDS:
            primary = by_key[(seed, float(scale), PRIMARY_METHOD)]
            matched = by_key[(seed, float(scale), MATCHED_SINGLE_METHOD)]
            same_scale = by_key[(seed, float(scale), SAME_SCALE_SINGLE_METHOD)]
            radius_ratio = None
            if (
                primary["certified_adaptive_center_radius"] is not None
                and matched["certified_adaptive_center_radius"] not in (None, 0.0)
            ):
                radius_ratio = (
                    primary["certified_adaptive_center_radius"]
                    / matched["certified_adaptive_center_radius"]
                )
            oracle_ratio = primary["mean_proposal_queries"] / max(
                matched["mean_proposal_queries"], 1e-12
            )
            controller_checks.append(
                {
                    "controller_seed": seed,
                    "event_difference_to_matched_single": (
                        primary["trajectory_event_probability"]
                        - matched["trajectory_event_probability"]
                    ),
                    "radius_ratio_to_matched_single": radius_ratio,
                    "proposal_query_ratio_to_matched_single": oracle_ratio,
                    "primary_fallback_probability": primary[
                        "trajectory_fallback_probability"
                    ],
                    "same_scale_single_fallback_probability": same_scale[
                        "trajectory_fallback_probability"
                    ],
                    "passes": bool(
                        primary["query_execute_mismatches"] == 0
                        and primary["trajectory_event_probability"]
                        >= matched["trajectory_event_probability"]
                        - float(protocol["screening_thresholds"]["event_tolerance"])
                        and radius_ratio is not None
                        and radius_ratio
                        >= float(
                            protocol["screening_thresholds"]["minimum_radius_ratio"]
                        )
                        and oracle_ratio
                        <= float(
                            protocol["screening_thresholds"][
                                "maximum_proposal_query_ratio"
                            ]
                        )
                        and primary["mean_rejected_proposals"] > 0.0
                    ),
                }
            )
        evaluations.append(
            {
                "noise_scale": float(scale),
                "controller_checks": controller_checks,
                "passes_all_controllers": all(
                    check["passes"] for check in controller_checks
                ),
            }
        )
    eligible = [row for row in evaluations if row["passes_all_controllers"]]
    return {
        "evaluations": evaluations,
        "eligible_noise_scales": [row["noise_scale"] for row in eligible],
        "promotion_condition_met": bool(eligible),
        "selected_noise_scale": min(
            (row["noise_scale"] for row in eligible), default=None
        ),
        "validation_status": "closed",
        "confirmation_status": "closed",
        "attack_status": "closed",
    }


def merge_shards(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    shard_paths = sorted(args.shard_directory.glob("shard_*.json"))
    expected_count = int(protocol["shard_count"])
    if len(shard_paths) != expected_count:
        raise RuntimeError("shard file count differs from protocol")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    if {int(shard["shard_index"]) for shard in shards} != set(range(expected_count)):
        raise RuntimeError("shard index set is incomplete")
    records = [record for shard in shards for record in shard["records"]]
    keys = [
        (
            record["controller_seed"],
            record["environment_seed"],
            record["noise_scale_index"],
            record["method"],
        )
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("merged records contain duplicates")
    expected_records = (
        len(CONTROLLER_SEEDS)
        * len(protocol["development_environment_seeds"])
        * len(protocol["noise_scales"])
        * len(protocol["methods"])
    )
    if len(records) != expected_records:
        raise RuntimeError("merged record count differs from protocol")
    summary = _summary_rows(records, protocol)
    screening = _screening(summary, protocol)
    deterministic_records = [
        {key: value for key, value in record.items() if key != "runtime_ms"}
        for record in records
    ]
    deterministic_sha256 = sha256(
        json.dumps(
            deterministic_records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "push1_trajectory_development_complete",
        "paper_eligibility": "development_only",
        "protocol_path": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "source_sha256": file_sha256(SOURCE_PATH),
        "record_count": len(records),
        "deterministic_record_sha256": deterministic_sha256,
        "summary": summary,
        "development_screening": screening,
        "claim_boundary": [
            "Development results are exploratory and cannot enter the paper as held-out evidence.",
            "The certificate concerns residual action-center shifts under the frozen causal program.",
            "The result is not an observation-space or forward-invariance certificate.",
            "The one-step verifier uses privileged exact simulator state.",
            "Validation, confirmation, and attack splits remain closed.",
        ],
    }


def markdown_result(result: dict[str, Any]) -> str:
    lines = [
        "# Push1 feasible trajectory development result",
        "",
        f"Status is `{result['status']}`.",
        "",
        "This artifact is development-only.",
        "",
        f"Complete trajectory records are `{result['record_count']}`.",
        f"Promotion condition is `{result['development_screening']['promotion_condition_met']}`.",
        f"Selected noise scale is `{result['development_screening']['selected_noise_scale']}`.",
        "",
        "## Screening by noise scale",
        "",
    ]
    for row in result["development_screening"]["evaluations"]:
        lines.append(
            f"- scale `{row['noise_scale']}` passes all controllers `"
            f"{row['passes_all_controllers']}`"
        )
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run-shard", "merge"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--controller-root", type=Path, default=DEFAULT_CONTROLLER_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--shard-directory", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = verify_protocol(args)
    if args.mode == "run-shard":
        payload = evaluate_shard(args, protocol)
        destination = args.output_json or (
            args.shard_directory / f"shard_{args.shard_index}.json"
        )
        write_json(destination, payload)
    else:
        payload = merge_shards(args, protocol)
        destination = args.output_json or args.shard_directory / "result.json"
        markdown = args.output_md or args.shard_directory / "result.md"
        write_json(destination, payload)
        markdown.write_text(markdown_result(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
