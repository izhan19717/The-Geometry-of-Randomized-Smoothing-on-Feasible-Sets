"""Development-only trajectory smoothing with a learned Push1 controller.

The feasible-action test reads only the current observation and candidate
action. Unlike the earlier exact restored-state query, it does not consult
future simulator outcomes. A later held-out protocol must freeze the event,
filter, noise scale, attack rule, and cohort before any confirmation run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.iclr2027_safety_gym_push1_all_controller_integration import (
    CONTROLLER_SEEDS,
    ENVIRONMENT,
    _artifact_paths,
    load_push1_actor_bundle,
    verify_controller_artifacts,
)
from experiments.iclr2027_safety_gym_push1_trajectory_development import stable_seed


DEFAULT_CONTROLLER_ROOT = Path("safe_controller_push1_runs_v1_20260901")
DEFAULT_GOAL2_CONTROLLER_ROOT = Path(
    "Fast-Autoencoder-Based-Projections/output/"
    "iclr2027_safety_point_goal_controller_seed1637115473"
)
DEFAULT_FREEZE = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "SAFETY_GYM_PUSH1_ALL_CONTROLLER_INTEGRATION_PROTOCOL_V1_20260903.json"
)
DEFAULT_OUTPUT = Path("outputs/iclr2027_push1_observable_trajectory_pilot.json")
HORIZON = 500
NOISE_SCALES = (0.1, 0.2)
METHODS = ("matched_single", "correlated_four")
RHO_FOUR = 59.0 / 63.0
KAPPA = 1.05
NOISE_SEED = 60128749


def observation_action_caps(observation: np.ndarray) -> tuple[float, float]:
    """Fixed, observable state-dependent action constraints.

    Safety-Gymnasium 1.0.0 flattens the Point observations followed by four
    16-value lidar blocks in alphabetical key order. Hazards occupy [28:44].
    A larger hazard return reduces admissible forward/backward force. This is
    an action-limiting heuristic, not a verified predictor of native cost.
    """

    observation = np.asarray(observation, dtype=float)
    if observation.shape not in ((60,), (76,)) or not np.all(np.isfinite(observation)):
        raise ValueError("expected finite Point observation of dimension 60 or 76")
    hazard_signal = float(np.clip(np.max(observation[28:44]), 0.0, 1.0))
    return 0.98 - 0.10 * hazard_signal, 1.0


def is_admissible(observation: np.ndarray, action: np.ndarray) -> bool:
    forward_cap, steering_cap = observation_action_caps(observation)
    candidate = np.asarray(action, dtype=float)
    return bool(
        candidate.shape == (2,)
        and np.all(np.isfinite(candidate))
        and abs(candidate[0]) <= forward_cap
        and abs(candidate[1]) <= steering_cap
    )


def proposal_block(
    center: np.ndarray,
    shift: np.ndarray,
    sigma: float,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if method == "matched_single":
        marginal_sigma = sigma / math.sqrt(KAPPA)
        innovations = rng.standard_normal((1, 2))
        raw = center[None, :] + shift[None, :] + marginal_sigma * innovations
        multiplier = 1.0 / marginal_sigma**2
    elif method == "correlated_four":
        shared = rng.standard_normal(2)
        individual = rng.standard_normal((4, 2))
        noise = math.sqrt(RHO_FOUR) * shared[None, :]
        noise = noise + math.sqrt(1.0 - RHO_FOUR) * individual
        raw = center[None, :] + shift[None, :] + sigma * noise
        multiplier = KAPPA / sigma**2
    else:
        raise ValueError("unknown method")
    if not math.isclose(multiplier, KAPPA / sigma**2, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("method comparison energies differ")
    return np.clip(raw, -1.0, 1.0), multiplier


def predictable_shift(center: np.ndarray, budget: float, horizon: int) -> np.ndarray:
    if budget < 0 or horizon < 1:
        raise ValueError("invalid attack budget or horizon")
    norm = float(np.linalg.norm(center))
    direction = -center / norm if norm > 1e-12 else np.array([-1.0, 0.0])
    return (budget / math.sqrt(horizon)) * direction


def run_episode(
    env: Any,
    actor: Any,
    torch: Any,
    controller_seed: int,
    environment_seed: int,
    sigma: float,
    method: str,
    budget: float,
    horizon: int,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=environment_seed)
    observation = np.asarray(observation, dtype=float)
    noise_rng = np.random.default_rng(
        np.random.SeedSequence([NOISE_SEED, controller_seed, environment_seed])
    )
    selected_proposals = 0
    rejected_proposals = 0
    fallback_steps = 0
    executed_steps = 0
    total_energy = 0.0
    total_reward = 0.0
    event = False
    goal = False
    cost_hit = False
    stop_reason = "horizon"
    for step in range(horizon):
        raw_center, _ = actor.infer(observation, torch)
        center = np.clip(np.asarray(raw_center, dtype=float), -1.0, 1.0)
        shift = predictable_shift(center, budget, horizon)
        block, energy_multiplier = proposal_block(
            center, shift, sigma, method, noise_rng
        )
        total_energy += energy_multiplier * float(shift @ shift)
        action = None
        for candidate in block:
            if is_admissible(observation, candidate):
                action = candidate
                selected_proposals += 1
                break
            rejected_proposals += 1
        if action is None:
            forward_cap, steering_cap = observation_action_caps(observation)
            action = np.clip(center, [-forward_cap, -steering_cap], [forward_cap, steering_cap])
            if not is_admissible(observation, action):
                raise RuntimeError("declared fallback failed the common verifier")
            fallback_steps += 1
        next_observation, reward, cost, terminated, truncated, info = env.step(action)
        if not np.all(np.isfinite(next_observation)) or not math.isfinite(float(cost)):
            stop_reason = "nonfinite_transition"
            break
        executed_steps += 1
        total_reward += float(reward)
        observation = np.asarray(next_observation, dtype=float)
        if float(cost) > 0:
            cost_hit = True
            stop_reason = "native_cost"
            break
        if bool(terminated) or bool(truncated):
            stop_reason = "environment_stop"
            break
        if bool(info.get("goal_met", False)):
            goal = True
            event = True
            stop_reason = "first_goal_without_cost"
            break
    bound = KAPPA * budget**2 / sigma**2
    if total_energy > bound + 1e-10:
        raise RuntimeError("pathwise energy budget violated")
    return {
        "controller_seed": controller_seed,
        "environment_seed": environment_seed,
        "sigma": sigma,
        "method": method,
        "attack_budget": budget,
        "horizon": horizon,
        "event": int(event),
        "goal": int(goal),
        "native_cost_hit": int(cost_hit),
        "stop_reason": stop_reason,
        "executed_steps": executed_steps,
        "selected_proposals": selected_proposals,
        "rejected_proposals": rejected_proposals,
        "fallback_steps": fallback_steps,
        "total_energy": total_energy,
        "pathwise_energy_bound": bound,
        "total_reward": total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", type=Path, default=DEFAULT_CONTROLLER_ROOT)
    parser.add_argument("--goal2-controller-root", type=Path, default=DEFAULT_GOAL2_CONTROLLER_ROOT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-seeds", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--controller-count", type=int, default=3)
    parser.add_argument("--attack-budget", type=float, default=0.0)
    parser.add_argument("--task", choices=("push1", "goal2"), default="push1")
    args = parser.parse_args()
    if not 1 <= args.pilot_seeds <= 32 or not 1 <= args.horizon <= 1000:
        raise ValueError("pilot size or horizon outside development bounds")
    if not 1 <= args.controller_count <= len(CONTROLLER_SEEDS):
        raise ValueError("invalid controller count")
    if args.attack_budget < 0:
        raise ValueError("negative attack budget")
    freeze_bytes = args.freeze.read_bytes() if args.task == "push1" else b""
    freeze = json.loads(freeze_bytes) if freeze_bytes else None
    labels = [
        f"{args.task}-observable-trajectory-open-development-{i}"
        for i in range(args.pilot_seeds)
    ]
    seeds = [stable_seed(label)[1] for label in labels]
    import safety_gymnasium

    records: list[dict[str, Any]] = []
    started = time.time()
    environment_name = ENVIRONMENT if args.task == "push1" else "SafetyPointGoal2-v0"
    env = safety_gymnasium.make(environment_name)
    try:
        keys = tuple(env.unwrapped.obs_space_dict.spaces)
        expected_prefix = (
            "accelerometer", "velocimeter", "gyro", "magnetometer",
            "goal_lidar", "hazards_lidar",
        )
        expected_keys = expected_prefix + (
            ("pillars_lidar", "push_box_lidar")
            if args.task == "push1" else ("vases_lidar",)
        )
        if keys != expected_keys:
            raise RuntimeError("observation flattening order changed")
        seeds_to_run = CONTROLLER_SEEDS[: args.controller_count] if args.task == "push1" else (1637115473,)
        for controller_seed in seeds_to_run:
            if args.task == "push1":
                paths = _artifact_paths(args.controller_root, controller_seed)
                expected = freeze["controller_artifact_sha256"][str(controller_seed)]
                verify_controller_artifacts(paths, expected)
                actor, torch = load_push1_actor_bundle(
                    paths["model499.pt"], paths["state499.pkl"],
                    expected_checkpoint_sha256=expected["model499.pt"],
                    expected_normalizer_sha256=expected["state499.pkl"],
                )
            else:
                from experiments.iclr2027_safety_gym_engineering_smoke import load_actor_bundle
                actor, torch = load_actor_bundle(
                    args.goal2_controller_root / "model149.pt",
                    args.goal2_controller_root / "state149.pkl",
                )
            for sigma in NOISE_SCALES:
                for environment_seed in seeds:
                    for method in METHODS:
                        records.append(
                            run_episode(
                                env, actor, torch, controller_seed, environment_seed,
                                sigma, method, args.attack_budget, args.horizon,
                            )
                        )
    finally:
        env.close()
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["sigma"], row["method"])].append(row)
    summary = {
        f"sigma={sigma}/{method}": {
            "trajectories": len(group),
            "event_count": sum(row["event"] for row in group),
            "cost_count": sum(row["native_cost_hit"] for row in group),
            "mean_rejections": float(np.mean([row["rejected_proposals"] for row in group])),
            "mean_fallback_steps": float(np.mean([row["fallback_steps"] for row in group])),
            "mean_executed_steps": float(np.mean([row["executed_steps"] for row in group])),
        }
        for (sigma, method), group in sorted(groups.items())
    }
    result = {
        "status": "open_development_only",
        "paper_eligibility": "none",
        "task": environment_name,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "controller_freeze_sha256": sha256(freeze_bytes).hexdigest(),
        "pilot_labels": labels,
        "pilot_environment_seeds": seeds,
        "horizon": args.horizon,
        "attack_budget": args.attack_budget,
        "elapsed_seconds": time.time() - started,
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "This opened-data run may select a later protocol but is not confirmation evidence.",
            "The observation-based action constraint is a fixed heuristic, not a native-cost guarantee.",
            "Nominal event sampling alone does not test the adaptive attack comparison.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "summary")}, indent=2))


if __name__ == "__main__":
    main()
