"""Opened development comparison of three frozen Goal2 safety controllers.

This pilot reuses the observable action verifier from the held-out study, but
uses new development reset labels. Its outcomes may guide a later protocol;
they are not confirmation evidence and must not be inserted into the paper.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time

from experiments.iclr2027_safety_gym_all_controller_integration import (
    CONTROLLER_SEEDS,
    _artifact_paths,
    verify_controller_artifacts,
)
from experiments.iclr2027_safety_gym_engineering_smoke import load_actor_bundle
from experiments.iclr2027_safety_gym_push1_trajectory_development import stable_seed
from experiments.iclr2027_push1_observable_trajectory_pilot import (
    METHODS,
    run_episode,
)


INTEGRATION_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "SAFETY_GYM_ALL_CONTROLLER_INTEGRATION_PROTOCOL_V1_20260901.json"
)
CONTROLLER_ROOT = Path("output/iclr2027_safety_gym_safe_controllers_v2")
DEFAULT_OUTPUT = Path("outputs/iclr2027_goal2_safe_controller_open_pilot_v1.json")
SEED_PREFIX = "goal2-safe-controller-open-pilot-20260919-v1-"
SIGMA = 0.2
HORIZON = 500


def development_seeds(count: int) -> list[int]:
    if not 1 <= count <= 32:
        raise ValueError("opened pilot count outside declared range")
    seeds = [stable_seed(f"{SEED_PREFIX}{index}")[1] for index in range(count)]
    if len(set(seeds)) != count:
        raise RuntimeError("development seed collision")
    return seeds


def summarize(records: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["controller_seed"], record["method"])].append(record)
    return {
        f"seed={controller_seed}/{method}": {
            "trajectories": len(group),
            "event_count": sum(row["event"] for row in group),
            "cost_count": sum(row["native_cost_hit"] for row in group),
            "mean_fallback_steps": sum(row["fallback_steps"] for row in group) / len(group),
            "mean_executed_steps": sum(row["executed_steps"] for row in group) / len(group),
        }
        for (controller_seed, method), group in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-seeds", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    seeds = development_seeds(args.pilot_seeds)
    protocol_bytes = INTEGRATION_PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    if tuple(protocol["controller_seeds"]) != CONTROLLER_SEEDS:
        raise RuntimeError("controller seed list changed")

    import safety_gymnasium

    if safety_gymnasium.__version__ != "1.0.0":
        raise RuntimeError("simulator version changed")
    env = safety_gymnasium.make("SafetyPointGoal2-v0")
    records: list[dict] = []
    started = time.time()
    try:
        if tuple(env.unwrapped.obs_space_dict.spaces) != (
            "accelerometer", "velocimeter", "gyro", "magnetometer",
            "goal_lidar", "hazards_lidar", "vases_lidar",
        ):
            raise RuntimeError("observation key order changed")
        for controller_seed in CONTROLLER_SEEDS:
            paths = _artifact_paths(CONTROLLER_ROOT, controller_seed)
            expected = protocol["controller_artifact_sha256"][str(controller_seed)]
            verify_controller_artifacts(paths, expected)
            actor, torch = load_actor_bundle(
                paths["model499.pt"], paths["state499.pkl"],
                expected_checkpoint_sha256=expected["model499.pt"],
                expected_normalizer_sha256=expected["state499.pkl"],
            )
            for environment_seed in seeds:
                for method in METHODS:
                    records.append(
                        run_episode(
                            env, actor, torch, controller_seed, environment_seed,
                            SIGMA, method, 0.0, HORIZON,
                        )
                    )
    finally:
        env.close()
    output = {
        "status": "open_development_only",
        "paper_eligibility": "none",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "integration_protocol_sha256": sha256(protocol_bytes).hexdigest(),
        "seed_prefix": SEED_PREFIX,
        "development_seeds": seeds,
        "sigma": SIGMA,
        "horizon": HORIZON,
        "elapsed_seconds": time.time() - started,
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps(output["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
