"""Frozen Goal2 safety-event confirmation with three learned controllers.

The primary event is no positive native cost before first goal, environment
stop, or the 500-step limit. Environment stop without a goal is a failure.
Goal-before-cost is retained as a separate descriptive utility outcome.
The target initial law is uniform over the frozen reset-seed cohort.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

from scipy.special import ndtr, ndtri

from experiments.iclr2027_safety_gym_all_controller_integration import (
    CONTROLLER_SEEDS,
    _artifact_paths,
    verify_controller_artifacts,
)
from experiments.iclr2027_safety_gym_engineering_smoke import (
    file_sha256,
    load_actor_bundle,
)
from experiments.iclr2027_safety_gym_push1_trajectory_development import stable_seed
from experiments.iclr2027_push1_observable_trajectory_pilot import (
    KAPPA,
    METHODS,
    NOISE_SEED,
    RHO_FOUR,
    run_episode,
)


PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "GOAL2_SAFE_TRAJECTORY_CONFIRMATION_V1_20260919.json"
)
CONTROLLER_ROOT = Path("output/iclr2027_safety_gym_safe_controllers_v2")


def one_sided_hoeffding_lower(successes: int, count: int, alpha: float) -> float:
    if count < 1 or not 0 <= successes <= count or not 0 < alpha < 1:
        raise ValueError("invalid count or error allocation")
    return max(0.0, successes / count - math.sqrt(math.log(1 / alpha) / (2 * count)))


def safety_event(record: dict[str, Any]) -> int:
    if record["native_cost_hit"]:
        return 0
    return int(record["stop_reason"] in ("first_goal_without_cost", "horizon"))


def validate_protocol(protocol: dict[str, Any]) -> list[int]:
    if protocol["status"] != "frozen_before_confirmation_v1":
        raise RuntimeError("protocol is not frozen")
    if protocol["task"] != "SafetyPointGoal2-v0":
        raise RuntimeError("task changed")
    if protocol["controller_seeds"] != list(CONTROLLER_SEEDS):
        raise RuntimeError("controller list changed")
    if protocol["methods"] != list(METHODS):
        raise RuntimeError("proposal methods changed")
    if protocol["noise_seed"] != NOISE_SEED or protocol["rho_four"] != RHO_FOUR:
        raise RuntimeError("proposal noise law changed")
    if protocol["kappa"] != KAPPA:
        raise RuntimeError("energy multiplier changed")
    if file_sha256(Path(__file__)) != protocol["confirmation_source_sha256"]:
        raise RuntimeError("confirmation source changed")
    for relative, digest in protocol["runtime_source_sha256"].items():
        if file_sha256(Path(relative)) != digest:
            raise RuntimeError(f"runtime source changed at {relative}")
    count = int(protocol["trajectory_count"])
    if count != 256 or protocol["horizon"] != 500:
        raise RuntimeError("sample count or horizon changed")
    seeds = [
        stable_seed(f"{protocol['seed_prefix']}{index}")[1]
        for index in range(count)
    ]
    digest = sha256(json.dumps(seeds, separators=(",", ":")).encode()).hexdigest()
    if digest != protocol["ordered_seed_sha256"] or len(set(seeds)) != count:
        raise RuntimeError("seed list changed or collided")
    return seeds


def summarize(
    records: list[dict[str, Any]], protocol: dict[str, Any], seeds: list[int]
) -> dict[str, Any]:
    count = int(protocol["trajectory_count"])
    sigma = float(protocol["sigma"])
    B = float(protocol["attack_budget"])
    q = float(protocol["target_safety_probability"])
    delta = float(protocol["familywise_error"])
    alpha = delta / (len(CONTROLLER_SEEDS) * len(METHODS))
    if not 0 < q < 1 or not 0 < delta < 1 or sigma <= 0 or B <= 0:
        raise ValueError("invalid target, error, noise, or attack parameter")
    expected = {
        (seed, controller_seed, method, budget)
        for seed in seeds
        for controller_seed in CONTROLLER_SEEDS
        for method in METHODS
        for budget in (0.0, B)
    }
    keys = [
        (r["environment_seed"], r["controller_seed"], r["method"], r["attack_budget"])
        for r in records
    ]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise RuntimeError("missing, extra, or duplicate trajectory arm")
    rows = {}
    for controller_seed in CONTROLLER_SEEDS:
        for method in METHODS:
            nominal = [
                r for r in records
                if r["controller_seed"] == controller_seed
                and r["method"] == method and r["attack_budget"] == 0.0
            ]
            attacked = [
                r for r in records
                if r["controller_seed"] == controller_seed
                and r["method"] == method and r["attack_budget"] == B
            ]
            if len(nominal) != count or len(attacked) != count:
                raise RuntimeError("incomplete method arm")
            for record in nominal + attacked:
                if record["sigma"] != sigma or record["horizon"] != protocol["horizon"]:
                    raise RuntimeError("noise scale or horizon changed")
                if record["event"] not in (0, 1):
                    raise RuntimeError("invalid goal event")
                if record["event"] and (
                    record["native_cost_hit"]
                    or record["stop_reason"] != "first_goal_without_cost"
                ):
                    raise RuntimeError("goal event inconsistent with trace")
                limit = KAPPA * float(record["attack_budget"]) ** 2 / sigma**2
                if not math.isclose(record["pathwise_energy_bound"], limit, abs_tol=1e-12):
                    raise RuntimeError("declared energy changed")
                if record["total_energy"] > limit + 1e-10:
                    raise RuntimeError("pathwise energy violation")
            successes = sum(safety_event(r) for r in nominal)
            lower = one_sided_hoeffding_lower(successes, count, alpha)
            radius = (
                sigma * (ndtri(lower) - ndtri(q)) / math.sqrt(KAPPA)
                if lower > q else None
            )
            tested_lower = (
                float(ndtr(ndtri(lower) - math.sqrt(KAPPA) * B / sigma))
                if lower > 0 else 0.0
            )
            rows[f"seed={controller_seed}/{method}"] = {
                "nominal_safety_count": successes,
                "nominal_safety_fraction": successes / count,
                "simultaneous_hoeffding_lower": lower,
                "certified_adaptive_center_radius": radius,
                "lower_safety_probability_at_tested_budget": tested_lower,
                "attacked_safety_count": sum(safety_event(r) for r in attacked),
                "nominal_goal_before_cost_count": sum(r["event"] for r in nominal),
                "attacked_goal_before_cost_count": sum(r["event"] for r in attacked),
                "nominal_cost_count": sum(r["native_cost_hit"] for r in nominal),
                "nominal_mean_executed_steps": sum(r["executed_steps"] for r in nominal) / count,
                "nominal_mean_fallback_steps": sum(r["fallback_steps"] for r in nominal) / count,
                "max_attacked_energy": max(r["total_energy"] for r in attacked),
                "pathwise_energy_limit": KAPPA * B**2 / sigma**2,
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    seeds = validate_protocol(protocol)
    import safety_gymnasium

    if safety_gymnasium.__version__ != protocol["safety_gymnasium_version"]:
        raise RuntimeError("simulator version changed")
    env = safety_gymnasium.make(protocol["task"])
    records: list[dict[str, Any]] = []
    started = time.time()
    try:
        if tuple(env.unwrapped.obs_space_dict.spaces) != (
            "accelerometer", "velocimeter", "gyro", "magnetometer",
            "goal_lidar", "hazards_lidar", "vases_lidar",
        ):
            raise RuntimeError("observation key order changed")
        integration = json.loads(Path(protocol["integration_protocol"]).read_text())
        for controller_seed in CONTROLLER_SEEDS:
            paths = _artifact_paths(CONTROLLER_ROOT, controller_seed)
            expected_hashes = protocol["controller_artifact_sha256"][str(controller_seed)]
            if expected_hashes != integration["controller_artifact_sha256"][str(controller_seed)]:
                raise RuntimeError("controller hashes differ from integration freeze")
            verify_controller_artifacts(paths, expected_hashes)
            actor, torch = load_actor_bundle(
                paths["model499.pt"], paths["state499.pkl"],
                expected_checkpoint_sha256=expected_hashes["model499.pt"],
                expected_normalizer_sha256=expected_hashes["state499.pkl"],
            )
            for environment_seed in seeds:
                for method in METHODS:
                    for budget in (0.0, float(protocol["attack_budget"])):
                        records.append(
                            run_episode(
                                env, actor, torch, controller_seed,
                                environment_seed, float(protocol["sigma"]),
                                method, budget, int(protocol["horizon"]),
                            )
                        )
    finally:
        env.close()
    result = {
        "status": "held_out_confirmation_complete",
        "protocol_sha256": sha256(protocol_bytes).hexdigest(),
        "source_sha256": file_sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
        "versions": {
            "python": platform.python_version(),
            "safety_gymnasium": safety_gymnasium.__version__,
            "torch": torch.__version__,
        },
        "ordered_seed_sha256": protocol["ordered_seed_sha256"],
        "summary": summarize(records, protocol, seeds),
        "records": records,
    }
    output = args.output or Path(protocol["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
