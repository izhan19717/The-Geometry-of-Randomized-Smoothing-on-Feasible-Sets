"""Engineering-only restored-state mechanics for SafetyPointPush1-v0.

This program uses only the previously opened Safety-Gym engineering seed.  It
does not load a controller and does not open a development, validation,
confirmation, or attack seed.  It extends the frozen PointGoal2 serializer
with the mutable state specific to Push1 and the stateful wrapper flags that
change on the first call to ``step``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from experiments.iclr2027_safety_gym_engineering_smoke import (
    SimulatorSnapshot,
    TransitionRecord,
    canonical_digest,
    file_sha256,
    restore_simulator,
    snapshot_simulator,
)


ENGINEERING_SEED = 1570860784
ENVIRONMENT = "SafetyPointPush1-v0"
SERIALIZER_SOURCE = Path("experiments/iclr2027_safety_gym_engineering_smoke.py")
DEFAULT_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "SAFETY_GYM_PUSH1_ENGINEERING_PROTOCOL_V1_20260902.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "output/iclr2027_safety_gym_push1_engineering_smoke_v1/result.json"
)
DEFAULT_OUTPUT_MD = Path(
    "output/iclr2027_safety_gym_push1_engineering_smoke_v1/result.md"
)

FIXED_ACTIONS = np.asarray(
    [
        [0.0, 0.0],
        [0.25, 0.0],
        [-0.25, 0.0],
        [0.0, 0.25],
        [0.0, -0.25],
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
    ],
    dtype=np.float64,
)

WRAPPER_STATE_ATTRIBUTES = {
    "safety_gymnasium.wrappers.time_limit.SafeTimeLimit": (
        "_elapsed_steps",
    ),
    "gymnasium.wrappers.order_enforcing.OrderEnforcing": (
        "_has_reset",
        "_disable_render_order_enforcing",
    ),
    "safety_gymnasium.wrappers.env_checker.SafePassiveEnvChecker": (
        "checked_reset",
        "checked_step",
        "checked_render",
    ),
}

TASK_SOURCE_PATHS = (
    "tasks/safe_navigation/push/push_level0.py",
    "tasks/safe_navigation/push/push_level1.py",
    "builder.py",
    "bases/base_task.py",
)


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _wrapper_chain(env: Any) -> list[Any]:
    chain = []
    current = env
    while hasattr(current, "env"):
        chain.append(current)
        current = current.env
    return chain


def snapshot_wrapper_state(env: Any) -> list[dict[str, Any]]:
    """Capture every stateful flag in the declared wrapper chain."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, wrapper in enumerate(_wrapper_chain(env)):
        qualified = _qualified_name(wrapper)
        if qualified not in WRAPPER_STATE_ATTRIBUTES:
            continue
        attributes = WRAPPER_STATE_ATTRIBUTES[qualified]
        missing = [name for name in attributes if not hasattr(wrapper, name)]
        if missing:
            raise RuntimeError(f"wrapper {qualified} lacks declared state {missing}")
        records.append(
            {
                "chain_index": index,
                "qualified_name": qualified,
                "attributes": {
                    name: copy.deepcopy(getattr(wrapper, name))
                    for name in attributes
                },
            }
        )
        seen.add(qualified)
    if seen != set(WRAPPER_STATE_ATTRIBUTES):
        missing = sorted(set(WRAPPER_STATE_ATTRIBUTES) - seen)
        raise RuntimeError(f"declared stateful wrappers were not found: {missing}")
    return records


def restore_wrapper_state(env: Any, records: list[dict[str, Any]]) -> None:
    """Restore wrapper state after checking the frozen wrapper topology."""

    chain = _wrapper_chain(env)
    for record in records:
        index = int(record["chain_index"])
        if index >= len(chain):
            raise RuntimeError("wrapper chain is shorter than the snapshot")
        wrapper = chain[index]
        if _qualified_name(wrapper) != record["qualified_name"]:
            raise RuntimeError("wrapper topology differs from the snapshot")
        for name, value in record["attributes"].items():
            setattr(wrapper, name, copy.deepcopy(value))


def push_box_state(env: Any) -> dict[str, Any]:
    """Read the free-joint and derived world state of the movable box."""

    task = env.unwrapped.task
    model = task.agent.engine.model
    data = task.agent.engine.data
    joint = model.joint("push_box")
    qpos_address = int(joint.qposadr[0])
    dof_address = int(joint.dofadr[0])
    return {
        "qpos_address": qpos_address,
        "dof_address": dof_address,
        "qpos": np.array(data.qpos[qpos_address : qpos_address + 7], copy=True),
        "qvel": np.array(data.qvel[dof_address : dof_address + 6], copy=True),
        "world_position": np.array(data.body("push_box").xpos, copy=True),
    }


def goal_state(env: Any) -> dict[str, Any]:
    """Read the goal model position and the matching layout entry."""

    task = env.unwrapped.task
    model = task.agent.engine.model
    data = task.agent.engine.data
    return {
        "model_body_position": np.array(model.body("goal").pos, copy=True),
        "world_position": np.array(data.body("goal").xpos, copy=True),
        "layout_position": np.array(task.world_info.layout["goal"], copy=True),
    }


@dataclass
class Push1SimulatorSnapshot:
    """Complete mutable state needed for exact Push1 one-step replay."""

    core: SimulatorSnapshot
    wrapper_state: list[dict[str, Any]]
    builder_seed: int
    builder_first_reset: bool
    task_last_dist_box: float
    task_last_box_goal: float
    box: dict[str, Any]
    goal: dict[str, Any]

    def serializable_state(self) -> dict[str, Any]:
        return {
            "core": self.core.serializable_state(),
            "wrapper_state": self.wrapper_state,
            "builder_seed": self.builder_seed,
            "builder_first_reset": self.builder_first_reset,
            "task_last_dist_box": self.task_last_dist_box,
            "task_last_box_goal": self.task_last_box_goal,
            "box": self.box,
            "goal": self.goal,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.serializable_state())


def snapshot_push1(env: Any) -> Push1SimulatorSnapshot:
    """Capture the PointGoal2 state plus all Push1-specific mutable state."""

    builder = env.unwrapped
    task = builder.task
    if not hasattr(task, "last_dist_box") or not hasattr(task, "last_box_goal"):
        raise RuntimeError("the declared Push1 task state is unavailable")
    return Push1SimulatorSnapshot(
        core=snapshot_simulator(env),
        wrapper_state=snapshot_wrapper_state(env),
        builder_seed=int(builder._seed),
        builder_first_reset=bool(builder.first_reset),
        task_last_dist_box=float(task.last_dist_box),
        task_last_box_goal=float(task.last_box_goal),
        box=push_box_state(env),
        goal=goal_state(env),
    )


def restore_push1(
    env: Any, snapshot: Push1SimulatorSnapshot, mujoco: Any
) -> None:
    """Restore Push1 state and require exact box and goal reconstruction."""

    restore_simulator(env, snapshot.core, mujoco)
    builder = env.unwrapped
    task = builder.task
    builder._seed = snapshot.builder_seed
    builder.first_reset = snapshot.builder_first_reset
    task.last_dist_box = snapshot.task_last_dist_box
    task.last_box_goal = snapshot.task_last_box_goal
    restore_wrapper_state(env, snapshot.wrapper_state)

    if canonical_digest(push_box_state(env)) != canonical_digest(snapshot.box):
        raise RuntimeError("restored Push1 box state differs from the snapshot")
    if canonical_digest(goal_state(env)) != canonical_digest(snapshot.goal):
        raise RuntimeError("restored Push1 goal state differs from the snapshot")


def _transition_once_push1(env: Any, action: np.ndarray) -> TransitionRecord:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,):
        raise ValueError("the declared SafetyPointPush1 action must have shape (2,)")
    if not np.all(np.isfinite(action)) or np.any(action < -1.0) or np.any(action > 1.0):
        return TransitionRecord(
            action=action,
            observation=None,
            reward=None,
            cost=None,
            terminated=None,
            truncated=None,
            info={},
            exception="action_outside_declared_box",
            finite=False,
            simulator_exception=False,
            terminal_state_digest=None,
        )
    try:
        observation, reward, cost, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float64)
        reward_float = float(reward)
        cost_float = float(cost)
        finite = bool(
            np.all(np.isfinite(observation))
            and np.isfinite(reward_float)
            and np.isfinite(cost_float)
        )
        simulator_exception = bool(float(info.get("cost_exception", 0.0)) > 0.0)
        return TransitionRecord(
            action=action,
            observation=observation,
            reward=reward_float,
            cost=cost_float,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=copy.deepcopy(info),
            exception=None,
            finite=finite,
            simulator_exception=simulator_exception,
            terminal_state_digest=snapshot_push1(env).digest,
        )
    except Exception as error:
        return TransitionRecord(
            action=action,
            observation=None,
            reward=None,
            cost=None,
            terminated=None,
            truncated=None,
            info={},
            exception=f"{type(error).__name__}: {error}",
            finite=False,
            simulator_exception=True,
            terminal_state_digest=None,
        )


def oracle_query_push1(
    env: Any,
    snapshot: Push1SimulatorSnapshot,
    action: np.ndarray,
    mujoco: Any,
) -> TransitionRecord:
    """Evaluate one Push1 action and restore every captured state field."""

    restore_push1(env, snapshot, mujoco)
    if snapshot_push1(env).digest != snapshot.digest:
        raise RuntimeError("pre-query Push1 restoration differs from the snapshot")
    try:
        record = _transition_once_push1(env, action)
    finally:
        restore_push1(env, snapshot, mujoco)
    if snapshot_push1(env).digest != snapshot.digest:
        raise RuntimeError("post-query Push1 restoration differs from the snapshot")
    return record


def execute_verified_push1(
    env: Any,
    snapshot: Push1SimulatorSnapshot,
    queried: TransitionRecord,
) -> tuple[TransitionRecord, bool]:
    """Execute an already queried action from the identical Push1 state."""

    if snapshot_push1(env).digest != snapshot.digest:
        raise RuntimeError("Push1 state advanced between query and execution")
    actual = _transition_once_push1(env, queried.action)
    return actual, actual.digest == queried.digest


def set_push_box_at_goal(env: Any, mujoco: Any) -> None:
    """Place the free box at the current goal on the engineering fixture."""

    task = env.unwrapped.task
    model = task.agent.engine.model
    data = task.agent.engine.data
    joint = model.joint("push_box")
    qpos_address = int(joint.qposadr[0])
    dof_address = int(joint.dofadr[0])
    data.qpos[qpos_address : qpos_address + 2] = np.asarray(task.goal.pos[:2])
    data.qvel[dof_address : dof_address + 6] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    task.last_dist_box = float(task.dist_box())
    task.last_box_goal = float(task.dist_box_goal())
    task.last_dist_goal = float(task.dist_goal())
    if not task.goal_achieved:
        raise RuntimeError("forced Push1 box position did not reach the goal")


def _json_transition(record: TransitionRecord) -> dict[str, Any]:
    return {
        "action": record.action.tolist(),
        "reward": record.reward,
        "cost": record.cost,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "goal_met": bool(record.info.get("goal_met", False)),
        "exception": record.exception,
        "finite": record.finite,
        "simulator_exception": record.simulator_exception,
        "feasible": record.feasible,
        "digest": record.digest,
    }


def verify_protocol(args: argparse.Namespace, safety_gymnasium: Any) -> dict[str, Any]:
    """Bind the mechanics run to its frozen source and single seed."""

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_push1_engineering_smoke_v1":
        raise RuntimeError("Push1 engineering protocol is not frozen")
    if args.environment != protocol["environment"] or args.environment != ENVIRONMENT:
        raise RuntimeError("environment differs from the frozen Push1 protocol")
    if args.engineering_seed != protocol["engineering_seed"]:
        raise RuntimeError("engineering seed differs from the frozen protocol")
    if args.engineering_seed != ENGINEERING_SEED:
        raise RuntimeError("only the previously opened engineering seed is permitted")
    if protocol["fixed_actions"] != FIXED_ACTIONS.tolist():
        raise RuntimeError("fixed action sequence differs from the frozen protocol")
    if file_sha256(Path(__file__)) != protocol["source_sha256"]:
        raise RuntimeError("Push1 mechanics source differs from the frozen protocol")
    if file_sha256(SERIALIZER_SOURCE) != protocol["serializer_source_sha256"]:
        raise RuntimeError("imported serializer differs from the frozen protocol")

    package_root = Path(safety_gymnasium.__file__).resolve().parent
    for relative_path, expected_hash in protocol["task_source_sha256"].items():
        if relative_path not in TASK_SOURCE_PATHS:
            raise RuntimeError("protocol contains an undeclared task source")
        if file_sha256(package_root / relative_path) != expected_hash:
            raise RuntimeError(f"task source differs at {relative_path}")
    if set(protocol["task_source_sha256"]) != set(TASK_SOURCE_PATHS):
        raise RuntimeError("protocol omits a declared task source")
    return protocol


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run exact mechanics checks without loading a policy or opening a seed."""

    import mujoco
    import safety_gymnasium

    protocol = verify_protocol(args, safety_gymnasium)
    env = safety_gymnasium.make(args.environment)
    try:
        first_observation, _ = env.reset(seed=args.engineering_seed)
        first_snapshot = snapshot_push1(env)
        second_observation, _ = env.reset(seed=args.engineering_seed)
        second_snapshot = snapshot_push1(env)
        reset_exact = bool(
            np.array_equal(first_observation, second_observation)
            and first_snapshot.digest == second_snapshot.digest
        )

        transition_checks: list[dict[str, Any]] = []
        for index, action in enumerate(FIXED_ACTIONS):
            snapshot = snapshot_push1(env)
            queried = oracle_query_push1(env, snapshot, action, mujoco)
            actual, equivalent = execute_verified_push1(env, snapshot, queried)
            transition_checks.append(
                {
                    "index": index,
                    "snapshot_digest": snapshot.digest,
                    "query": _json_transition(queried),
                    "actual": _json_transition(actual),
                    "oracle_execute_exact": equivalent,
                }
            )
            if not equivalent or actual.observation is None:
                break
            if actual.terminated or actual.truncated:
                break

        env.reset(seed=args.engineering_seed)
        task = env.unwrapped.task
        old_goal = np.asarray(task.goal.pos, dtype=np.float64).copy()
        set_push_box_at_goal(env, mujoco)
        goal_snapshot = snapshot_push1(env)
        goal_query = oracle_query_push1(env, goal_snapshot, np.zeros(2), mujoco)
        goal_actual, goal_exact = execute_verified_push1(
            env, goal_snapshot, goal_query
        )
        new_goal = np.asarray(task.goal.pos, dtype=np.float64).copy()

        required_wrapper_fields = {
            name: list(attributes)
            for name, attributes in WRAPPER_STATE_ATTRIBUTES.items()
        }
        gates = {
            "same_seed_reset_exact": reset_exact,
            "all_fixed_transitions_completed": len(transition_checks)
            == len(FIXED_ACTIONS),
            "all_fixed_oracle_execute_pairs_exact": bool(transition_checks)
            and all(row["oracle_execute_exact"] for row in transition_checks),
            "all_fixed_transitions_finite": bool(transition_checks)
            and all(row["actual"]["finite"] for row in transition_checks),
            "goal_hit_occurred": bool(goal_actual.info.get("goal_met", False)),
            "goal_relocated": bool(np.any(new_goal != old_goal)),
            "goal_hit_oracle_execute_pair_exact": goal_exact,
        }
        payload = {
            "experiment": "iclr2027_safety_gym_push1_engineering_smoke_v1",
            "paper_eligibility": "none",
            "environment": args.environment,
            "engineering_seed": args.engineering_seed,
            "protocol_sha256": file_sha256(args.protocol),
            "source_sha256": file_sha256(Path(__file__)),
            "serializer_source_sha256": file_sha256(SERIALIZER_SOURCE),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "safety_gymnasium": safety_gymnasium.__version__,
                "mujoco": mujoco.__version__,
            },
            "state_contract": {
                "core_data_arrays": sorted(first_snapshot.core.data_arrays),
                "core_model_arrays": sorted(first_snapshot.core.model_arrays),
                "wrapper_fields": required_wrapper_fields,
                "builder_fields_added": ["_seed", "first_reset"],
                "push_task_fields_added": ["last_dist_box", "last_box_goal"],
                "box_joint_qpos_length": len(first_snapshot.box["qpos"]),
                "box_joint_qvel_length": len(first_snapshot.box["qvel"]),
            },
            "reset": {
                "first_snapshot_digest": first_snapshot.digest,
                "second_snapshot_digest": second_snapshot.digest,
                "exact": reset_exact,
            },
            "fixed_transition_checks": transition_checks,
            "goal_relocation_check": {
                "old_goal": old_goal.tolist(),
                "new_goal": new_goal.tolist(),
                "pre_transition_snapshot_digest": goal_snapshot.digest,
                "query": _json_transition(goal_query),
                "actual": _json_transition(goal_actual),
                "oracle_execute_exact": goal_exact,
            },
            "gates": gates,
            "pass": all(gates.values()),
            "scope": [
                "This result checks serializer and one-step oracle mechanics only.",
                "It uses privileged simulator lookahead.",
                "It gives no utility, safety, transfer, or certificate result.",
                "No controller checkpoint or held-out benchmark seed was opened.",
            ],
            "protocol_status": protocol["status"],
        }
        return payload
    finally:
        env.close()


def render_markdown(payload: dict[str, Any]) -> str:
    gates = payload["gates"]
    lines = [
        "# SafetyPointPush1 engineering mechanics result",
        "",
        f"Overall pass is `{payload['pass']}`.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {name} is `{value}`" for name, value in gates.items())
    lines.extend(
        [
            "",
            "## Scope",
            "",
            *[f"- {sentence}" for sentence in payload["scope"]],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment", default=ENVIRONMENT)
    parser.add_argument("--engineering-seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"pass": payload["pass"], "gates": payload["gates"]}, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
