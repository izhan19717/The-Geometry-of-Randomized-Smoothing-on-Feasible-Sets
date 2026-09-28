"""Frozen held-out Goal2 evaluation of adaptive feasible-trajectory smoothing.

The controller, verifier, fallback, event, horizon, noise law, and predictable
attack are common across nominal and shifted laws. The run produces nominal
event counts for an exact finite-sample trajectory certificate and separate
shifted trajectories as an empirical check. It certifies residual action-center
shifts, not arbitrary changes to observations, dynamics, or constraints.
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

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import beta

from experiments.iclr2027_safety_gym_engineering_smoke import (
    CHECKPOINT_SHA256,
    NORMALIZER_SHA256,
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
    "GOAL2_OBSERVABLE_TRAJECTORY_CONFIRMATION_V1_20260918.json"
)


def one_sided_cp_lower(successes: int, trials: int, alpha: float) -> float:
    if not 0 <= successes <= trials or trials < 1 or not 0 < alpha < 1:
        raise ValueError("invalid binomial count or confidence allocation")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def validate_protocol(protocol: dict[str, Any]) -> list[int]:
    if protocol["status"] != "frozen_before_confirmation_v1":
        raise RuntimeError("protocol is not frozen")
    if protocol["task"] != "SafetyPointGoal2-v0":
        raise RuntimeError("task changed")
    if protocol["methods"] != list(METHODS):
        raise RuntimeError("method set changed")
    if protocol["controller_seed"] != 1637115473:
        raise RuntimeError("controller changed")
    if protocol["controller_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint digest changed")
    if protocol["normalizer_sha256"] != NORMALIZER_SHA256:
        raise RuntimeError("normalizer digest changed")
    if protocol["noise_seed"] != NOISE_SEED or protocol["rho_four"] != RHO_FOUR:
        raise RuntimeError("proposal law changed")
    if protocol["kappa"] != KAPPA:
        raise RuntimeError("energy multiplier changed")
    source = Path("experiments/iclr2027_push1_observable_trajectory_pilot.py")
    if file_sha256(source) != protocol["runtime_source_sha256"]:
        raise RuntimeError("observable rollout source changed")
    if file_sha256(Path(__file__)) != protocol["confirmation_source_sha256"]:
        raise RuntimeError("confirmation source changed")
    for relative_path, expected_digest in protocol["dependency_source_sha256"].items():
        if file_sha256(Path(relative_path)) != expected_digest:
            raise RuntimeError(f"dependency source changed at {relative_path}")
    count = int(protocol["trajectory_count"])
    if count < 1 or int(protocol["horizon"]) < 1:
        raise RuntimeError("invalid sample count or horizon")
    labels = [f"{protocol['seed_prefix']}{i}" for i in range(count)]
    seeds = [stable_seed(label)[1] for label in labels]
    digest = sha256(json.dumps(seeds, separators=(",", ":")).encode()).hexdigest()
    if digest != protocol["ordered_seed_sha256"] or len(set(seeds)) != count:
        raise RuntimeError("held-out seed list changed or collided")
    return seeds


def summarize(records: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    count = int(protocol["trajectory_count"])
    sigma = float(protocol["sigma"])
    target = float(protocol["target_event_probability"])
    attack_budget = float(protocol["attack_budget"])
    delta = float(protocol["familywise_error"])
    if not 0 < target < 1 or not 0 < delta < 1 or sigma <= 0 or attack_budget < 0:
        raise ValueError("invalid statistical or attack parameter")
    rows: dict[str, Any] = {}
    for method in METHODS:
        nominal = [row for row in records if row["method"] == method and row["attack_budget"] == 0]
        attacked = [row for row in records if row["method"] == method and row["attack_budget"] > 0]
        if len(nominal) != count or len(attacked) != count:
            raise RuntimeError("missing or duplicate trajectory arm")
        nominal_successes = sum(row["event"] for row in nominal)
        attacked_successes = sum(row["event"] for row in attacked)
        lower = one_sided_cp_lower(nominal_successes, count, delta / len(METHODS))
        mu = max(0.0, float(ndtri(lower) - ndtri(target))) if lower > target else 0.0
        radius = sigma * mu / math.sqrt(KAPPA)
        tested_mu = math.sqrt(KAPPA) * attack_budget / sigma
        event_lower_at_tested_budget = (
            float(ndtr(ndtri(lower) - tested_mu)) if lower > 0 else 0.0
        )
        rows[method] = {
            "nominal_event_count": nominal_successes,
            "nominal_event_fraction": nominal_successes / count,
            "simultaneous_nominal_cp_lower": lower,
            "target_event_probability": target,
            "certified_adaptive_l2_energy_radius": radius if lower > target else None,
            "tested_attack_budget": attack_budget,
            "tested_budget_within_certificate": bool(radius >= attack_budget),
            "lower_event_probability_at_tested_budget": event_lower_at_tested_budget,
            "attacked_event_count": attacked_successes,
            "attacked_event_fraction": attacked_successes / count,
            "nominal_mean_rejected_proposals": float(
                np.mean([row["rejected_proposals"] for row in nominal])
            ),
            "nominal_mean_fallback_steps": float(
                np.mean([row["fallback_steps"] for row in nominal])
            ),
            "nominal_mean_executed_steps": float(
                np.mean([row["executed_steps"] for row in nominal])
            ),
            "nominal_cost_count": sum(row["native_cost_hit"] for row in nominal),
            "attacked_cost_count": sum(row["native_cost_hit"] for row in attacked),
            "maximum_attacked_energy": max(row["total_energy"] for row in attacked),
            "declared_pathwise_energy_bound": KAPPA * attack_budget**2 / sigma**2,
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
    controller_dir = Path(protocol["controller_directory"])
    actor, torch = load_actor_bundle(
        controller_dir / "model149.pt",
        controller_dir / "state149.pkl",
    )
    env = safety_gymnasium.make(protocol["task"])
    records: list[dict[str, Any]] = []
    started = time.time()
    try:
        if tuple(env.unwrapped.obs_space_dict.spaces) != (
            "accelerometer", "velocimeter", "gyro", "magnetometer",
            "goal_lidar", "hazards_lidar", "vases_lidar",
        ):
            raise RuntimeError("observation key order changed")
        for environment_seed in seeds:
            for method in METHODS:
                for budget in (0.0, float(protocol["attack_budget"])):
                    records.append(
                        run_episode(
                            env, actor, torch,
                            int(protocol["controller_seed"]), environment_seed,
                            float(protocol["sigma"]), method, budget,
                            int(protocol["horizon"]),
                        )
                    )
    finally:
        env.close()
    observed_keys = [
        (row["environment_seed"], row["method"], row["attack_budget"])
        for row in records
    ]
    if len(set(observed_keys)) != len(observed_keys):
        raise RuntimeError("duplicate arm record")
    result = {
        "status": "held_out_confirmation_complete",
        "protocol_sha256": sha256(protocol_bytes).hexdigest(),
        "source_sha256": file_sha256(Path(__file__)),
        "runtime_source_sha256": protocol["runtime_source_sha256"],
        "ordered_seed_sha256": protocol["ordered_seed_sha256"],
        "trajectory_count_per_arm": len(seeds),
        "elapsed_seconds": time.time() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "safety_gymnasium": safety_gymnasium.__version__,
            "torch": torch.__version__,
        },
        "summary": summarize(records, protocol),
        "records": records,
        "claim_boundary": [
            "The certificate covers predictable residual action-center shifts under one fixed causal program.",
            "The verifier is observable and heuristic; it does not itself guarantee native safety.",
            "The event is first goal before native cost, environment stop, or horizon.",
            "No observation attack or change to dynamics, verifier, or controller is certified.",
        ],
    }
    destination = args.output or Path(protocol["output_json"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "trajectory_count_per_arm": result["trajectory_count_per_arm"],
        "summary": result["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
