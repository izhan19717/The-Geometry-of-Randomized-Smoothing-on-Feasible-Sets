"""Engineering-only audit for the learned Safety-Gym trajectory study.

This program is deliberately not a development, validation, or confirmation
experiment.  It verifies the mechanics that a later frozen benchmark would
depend on:

* load the preselected final actor and its frozen observation normalizer;
* snapshot and restore all integration, layout, goal, counter, and RNG state
  used by ``SafetyPointGoal2-v0``;
* query the one-step fixed-state feasibility fibre without advancing the live
  rollout;
* execute a queried action and require exact agreement with the oracle replay;
* exercise the state-only 121-by-121 deterministic fallback; and
* test restoration across a goal hit, where Safety-Gymnasium relocates the
  goal and consumes environment randomness.

Safety-Gymnasium, MuJoCo, Torch, Joblib, and SafePO are imported lazily.  The
pure helpers therefore remain testable on machines without the simulator.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import struct
import time
from typing import Any, Iterable, Iterator

import numpy as np


ANONYMOUS_SOURCE_PATH = Path(
    "experiments/iclr2027_safety_gym_engineering_smoke.py"
)
DEFAULT_PROTOCOL_PATH = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "SAFETY_GYM_ENGINEERING_PROTOCOL_V2_20260831.json"
)
DEFAULT_CHECKPOINT = Path(
    "output/iclr2027_safety_point_goal_controller_seed1637115473/model149.pt"
)
DEFAULT_NORMALIZER = Path(
    "output/iclr2027_safety_point_goal_controller_seed1637115473/state149.pkl"
)
CHECKPOINT_SHA256 = "ca2d08be73da17b14e18dc69747a96c3de93d5a7aedc3091533dc10c01c14ee0"
NORMALIZER_SHA256 = "c3335d52c610b17e4397017ed34cc9fc1c89bbab44c527cb951a7c0a9f2f21a6"

# These are MuJoCo integration inputs, as opposed to derived quantities that
# ``mj_forward`` deterministically reconstructs.
DATA_ARRAY_NAMES = (
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "plugin_state",
)

# Goal resampling mutates body_pos in this task.  The broader list makes the
# serializer explicit and robust to other movable model objects in the same
# environment version.
MODEL_ARRAY_NAMES = (
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "site_pos",
    "site_quat",
    "light_pos",
    "light_dir",
    "cam_pos",
    "cam_quat",
    "eq_data",
)


def file_sha256(path: Path) -> str:
    """Hash one file without depending on shell utilities."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    """Update a hash with a deterministic encoding of nested numeric state."""

    if value is None:
        digest.update(b"N")
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b"B1" if bool(value) else b"B0")
    elif isinstance(value, (int, np.integer)):
        encoded = str(int(value)).encode("ascii")
        digest.update(b"I" + struct.pack(">Q", len(encoded)) + encoded)
    elif isinstance(value, (float, np.floating)):
        digest.update(b"F" + struct.pack(">d", float(value)))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + struct.pack(">Q", len(encoded)) + encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y" + struct.pack(">Q", len(value)) + value)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        _update_digest(digest, "A")
        _update_digest(digest, array.dtype.str)
        _update_digest(digest, tuple(array.shape))
        _update_digest(digest, array.tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"D")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        digest.update(b"d")
    elif isinstance(value, tuple):
        digest.update(b"T")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"t")
    elif isinstance(value, list):
        digest.update(b"L")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"l")
    else:
        raise TypeError(f"unsupported digest type: {type(value)!r}")


def canonical_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for supported nested state."""

    digest = sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenNormalizer:
    """Immutable copy of the training observation statistics."""

    mean: np.ndarray
    variance: np.ndarray
    count: float
    epsilon: float = 1e-8

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != self.mean.shape:
            raise ValueError(
                f"observation shape {observation.shape} does not match "
                f"normalizer shape {self.mean.shape}"
            )
        return (observation - self.mean) / np.sqrt(self.variance + self.epsilon)

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "mean": self.mean,
                "variance": self.variance,
                "count": self.count,
                "epsilon": self.epsilon,
            }
        )


@dataclass
class ActorBundle:
    actor: Any
    normalizer: FrozenNormalizer
    action_standard_deviation: np.ndarray

    def infer(self, raw_observation: np.ndarray, torch: Any) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.normalizer.normalize(raw_observation)
        tensor = torch.as_tensor(normalized[None, :], dtype=torch.float32)
        with torch.no_grad():
            distribution = self.actor(tensor)
        center = distribution.loc.detach().cpu().numpy()[0].astype(np.float64)
        scale = distribution.scale.detach().cpu().numpy()
        if scale.ndim == 2:
            scale = scale[0]
        return center, np.asarray(scale, dtype=np.float64)


def load_actor_bundle(
    checkpoint_path: Path,
    normalizer_path: Path,
    *,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
    expected_normalizer_sha256: str = NORMALIZER_SHA256,
) -> tuple[ActorBundle, Any]:
    """Load the final actor state dict and copy, rather than update, its RMS."""

    if file_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError("actor checkpoint SHA-256 does not match the frozen manifest")
    if file_sha256(normalizer_path) != expected_normalizer_sha256:
        raise RuntimeError("normalizer SHA-256 does not match the frozen manifest")

    import joblib
    import torch
    from safepo.common.model import Actor

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    actor = Actor(obs_dim=60, act_dim=2, hidden_sizes=[64, 64])
    actor.load_state_dict(state_dict, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    saved_state = joblib.load(normalizer_path)
    if set(saved_state) != {"Normalizer"}:
        raise RuntimeError("unexpected keys in the frozen normalizer checkpoint")
    running = saved_state["Normalizer"]
    normalizer = FrozenNormalizer(
        mean=np.array(running.mean, dtype=np.float64, copy=True),
        variance=np.array(running.var, dtype=np.float64, copy=True),
        count=float(running.count),
    )
    if normalizer.mean.shape != (60,) or normalizer.variance.shape != (60,):
        raise RuntimeError("the frozen normalizer is not the expected 60-vector")
    standard_deviation = (
        torch.exp(actor.log_std).detach().cpu().numpy().astype(np.float64)
    )
    return ActorBundle(actor, normalizer, standard_deviation), torch


@dataclass
class SimulatorSnapshot:
    data_arrays: dict[str, np.ndarray]
    model_arrays: dict[str, np.ndarray]
    time: float
    elapsed_steps: int
    builder_steps: int
    builder_terminated: bool
    builder_truncated: bool
    builder_cost: dict[str, Any]
    builder_rng_state: dict[str, Any]
    task_rng_state: tuple[Any, ...]
    task_last_dist_goal: float
    layout: dict[str, Any]
    reset_layout: dict[str, Any]
    world_config_dict: dict[str, Any]

    def serializable_state(self) -> dict[str, Any]:
        return {
            "data_arrays": self.data_arrays,
            "model_arrays": self.model_arrays,
            "time": self.time,
            "elapsed_steps": self.elapsed_steps,
            "builder_steps": self.builder_steps,
            "builder_terminated": self.builder_terminated,
            "builder_truncated": self.builder_truncated,
            "builder_cost": self.builder_cost,
            "builder_rng_state": self.builder_rng_state,
            "task_rng_state": self.task_rng_state,
            "task_last_dist_goal": self.task_last_dist_goal,
            "layout": self.layout,
            "reset_layout": self.reset_layout,
            "world_config_dict": self.world_config_dict,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.serializable_state())


def snapshot_simulator(env: Any) -> SimulatorSnapshot:
    """Capture the complete mutable state used by the declared PointGoal task."""

    builder = env.unwrapped
    task = builder.task
    data = task.agent.engine.data
    model = task.agent.engine.model
    data_arrays = {
        name: np.array(getattr(data, name), copy=True)
        for name in DATA_ARRAY_NAMES
        if hasattr(data, name)
    }
    model_arrays = {
        name: np.array(getattr(model, name), copy=True)
        for name in MODEL_ARRAY_NAMES
        if hasattr(model, name)
    }
    return SimulatorSnapshot(
        data_arrays=data_arrays,
        model_arrays=model_arrays,
        time=float(data.time),
        elapsed_steps=int(env._elapsed_steps),
        builder_steps=int(builder.steps),
        builder_terminated=bool(builder.terminated),
        builder_truncated=bool(builder.truncated),
        builder_cost=copy.deepcopy(builder.cost),
        builder_rng_state=copy.deepcopy(builder.np_random.bit_generator.state),
        task_rng_state=copy.deepcopy(
            task.random_generator.random_generator.get_state()
        ),
        task_last_dist_goal=float(task.last_dist_goal),
        layout=copy.deepcopy(task.world_info.layout),
        reset_layout=copy.deepcopy(task.world_info.reset_layout),
        world_config_dict=copy.deepcopy(task.world_info.world_config_dict),
    )


def restore_simulator(env: Any, snapshot: SimulatorSnapshot, mujoco: Any) -> None:
    """Restore one snapshot while preserving the task's layout alias."""

    builder = env.unwrapped
    task = builder.task
    data = task.agent.engine.data
    model = task.agent.engine.model

    for name, value in snapshot.model_arrays.items():
        getattr(model, name)[:] = value
    for name, value in snapshot.data_arrays.items():
        getattr(data, name)[:] = value
    data.time = snapshot.time

    env._elapsed_steps = snapshot.elapsed_steps
    builder.steps = snapshot.builder_steps
    builder.terminated = snapshot.builder_terminated
    builder.truncated = snapshot.builder_truncated
    builder.cost = copy.deepcopy(snapshot.builder_cost)
    builder.np_random.bit_generator.state = copy.deepcopy(snapshot.builder_rng_state)
    task.random_generator.random_generator.set_state(
        copy.deepcopy(snapshot.task_rng_state)
    )
    task.last_dist_goal = snapshot.task_last_dist_goal

    # RandomGenerator.layout and WorldInfo.layout are aliases in version 1.0.0.
    # Recreate that alias explicitly after restoring the two mutable mappings.
    layout = copy.deepcopy(snapshot.layout)
    task.world_info.layout = layout
    task.random_generator.layout = layout
    task.world_info.reset_layout = copy.deepcopy(snapshot.reset_layout)
    task.world_info.world_config_dict = copy.deepcopy(snapshot.world_config_dict)
    mujoco.mj_forward(model, data)
    restore_warmstart_after_forward(data, snapshot)


def restore_warmstart_after_forward(
    data: Any, snapshot: SimulatorSnapshot
) -> None:
    """Restore solver warm-start bytes after reconstructing derived state.

    MuJoCo 2.3.3's ``mj_forward`` perturbs ``qacc_warmstart`` at roundoff
    scale after a non-initial transition.  The array is an input to the next
    solver step, so exact replay requires restoring it *after* ``mj_forward``.
    """

    # MuJoCo 2.3.3's mj_forward perturbs qacc_warmstart at roundoff scale
    if "qacc_warmstart" in snapshot.data_arrays:
        data.qacc_warmstart[:] = snapshot.data_arrays["qacc_warmstart"]


@dataclass
class TransitionRecord:
    action: np.ndarray
    observation: np.ndarray | None
    reward: float | None
    cost: float | None
    terminated: bool | None
    truncated: bool | None
    info: dict[str, Any]
    exception: str | None
    finite: bool
    simulator_exception: bool
    terminal_state_digest: str | None

    @property
    def feasible(self) -> bool:
        return (
            self.exception is None
            and self.finite
            and not self.simulator_exception
            and self.cost == 0.0
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "action": self.action,
                "observation": self.observation,
                "reward": self.reward,
                "cost": self.cost,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "info": self.info,
                "exception": self.exception,
                "finite": self.finite,
                "simulator_exception": self.simulator_exception,
                "terminal_state_digest": self.terminal_state_digest,
            }
        )


def _transition_once(env: Any, action: np.ndarray) -> TransitionRecord:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,):
        raise ValueError("the declared SafetyPointGoal2 action must have shape (2,)")
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
        terminal_digest = snapshot_simulator(env).digest
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
            terminal_state_digest=terminal_digest,
        )
    except Exception as error:  # The engineering result records, never hides, this.
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


def oracle_query(
    env: Any, snapshot: SimulatorSnapshot, action: np.ndarray, mujoco: Any
) -> TransitionRecord:
    """Evaluate one action on a fixed fibre and restore it in every case."""

    restore_simulator(env, snapshot, mujoco)
    if snapshot_simulator(env).digest != snapshot.digest:
        raise RuntimeError("pre-query restoration did not reproduce the snapshot")
    try:
        record = _transition_once(env, action)
    finally:
        restore_simulator(env, snapshot, mujoco)
    if snapshot_simulator(env).digest != snapshot.digest:
        raise RuntimeError("post-query restoration did not reproduce the snapshot")
    return record


def execute_verified(
    env: Any,
    snapshot: SimulatorSnapshot,
    queried: TransitionRecord,
) -> tuple[TransitionRecord, bool]:
    """Execute one already queried action from the identical live state."""

    if snapshot_simulator(env).digest != snapshot.digest:
        raise RuntimeError("the live state advanced between query and execution")
    actual = _transition_once(env, queried.action)
    return actual, actual.digest == queried.digest


def ordered_fallback_grid(
    policy_center: np.ndarray, *, resolution: int = 121
) -> Iterator[np.ndarray]:
    """Order the fixed grid around the unperturbed state-only policy mean.

    The center must be generated inside the common causal program from the
    current state.  It must not contain the residual perturbation and must not
    be replaced by any failed noisy proposal.
    """

    if resolution < 2:
        raise ValueError("fallback resolution must be at least two")
    target = np.asarray(policy_center, dtype=np.float64)
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        raise ValueError("policy center must be a finite two-vector")
    values = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    candidates = [
        (
            float((first - target[0]) ** 2 + (second - target[1]) ** 2),
            float(first),
            float(second),
        )
        for first in values
        for second in values
    ]
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _, first, second in candidates:
        yield np.array([first, second], dtype=np.float64)


def verified_grid_fallback(
    env: Any,
    snapshot: SimulatorSnapshot,
    policy_center: np.ndarray,
    *,
    resolution: int,
    mujoco: Any,
) -> dict[str, Any]:
    """Query and execute the first verified action in the fixed grid."""

    query_count = 0
    for candidate in ordered_fallback_grid(policy_center, resolution=resolution):
        query_count += 1
        queried = oracle_query(env, snapshot, candidate, mujoco)
        if queried.feasible:
            actual, equivalent = execute_verified(env, snapshot, queried)
            return {
                "found": True,
                "query_count": query_count,
                "selected_action": candidate.tolist(),
                "selected_cost": actual.cost,
                "actual_observation": (
                    actual.observation.tolist()
                    if actual.observation is not None
                    else None
                ),
                "actual_terminated": actual.terminated,
                "actual_truncated": actual.truncated,
                "oracle_execute_equivalent": equivalent,
                "queried_digest": queried.digest,
                "actual_digest": actual.digest,
            }
    return {
        "found": False,
        "query_count": query_count,
        "selected_action": None,
        "selected_cost": None,
        "actual_observation": None,
        "actual_terminated": None,
        "actual_truncated": None,
        "oracle_execute_equivalent": False,
        "queried_digest": None,
        "actual_digest": None,
    }


def _set_agent_position(env: Any, target_xy: np.ndarray, mujoco: Any) -> None:
    """Place the Point body at an absolute xy in the already sampled world."""

    builder = env.unwrapped
    task = builder.task
    data = task.agent.engine.data
    base_xy = np.asarray(task.world.config["agent_xy"], dtype=np.float64)
    rotation = float(task.world.config["agent_rot"])
    cosine, sine = math.cos(rotation), math.sin(rotation)
    inverse_rotation = np.array([[cosine, sine], [-sine, cosine]])
    data.qpos[:2] = inverse_rotation @ (np.asarray(target_xy) - base_xy)
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.ctrl[:] = 0.0
    data.time = 0.0
    mujoco.mj_forward(task.agent.engine.model, data)
    task.last_dist_goal = float(
        np.linalg.norm(np.asarray(task.goal.pos[:2]) - np.asarray(task.agent.pos[:2]))
    )
    builder.steps = 0
    builder.terminated = False
    builder.truncated = False
    env._elapsed_steps = 0


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


def _check_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if int(protocol.get("manifest_version", -1)) != 2:
        raise RuntimeError("engineering smoke requires corrected manifest version 2")
    if protocol["status"] != "frozen_before_recorded_engineering_smoke_v2":
        raise RuntimeError("corrected engineering protocol is not frozen")
    if int(protocol["engineering_seed"]) != args.engineering_seed:
        raise RuntimeError("command seed differs from frozen engineering seed")
    if int(protocol["fallback_resolution"]) != args.fallback_resolution:
        raise RuntimeError("command fallback resolution differs from protocol")
    if int(protocol["rollout_steps"]) != args.rollout_steps:
        raise RuntimeError("command rollout length differs from protocol")
    if int(protocol["activation_steps"]) != args.activation_steps:
        raise RuntimeError("command activation length differs from protocol")
    if int(protocol["activation_noise_seed"]) != args.activation_noise_seed:
        raise RuntimeError("command activation seed differs from protocol")
    if protocol["source_sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("source differs from the protocol-frozen source")
    return protocol


def run_engineering_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run only the frozen engineering checks on the one engineering seed."""

    import mujoco
    import safety_gymnasium

    protocol = _check_protocol(args)
    bundle, torch = load_actor_bundle(args.checkpoint, args.normalizer)
    normalizer_digest_before = bundle.normalizer.digest

    env = safety_gymnasium.make(args.environment)
    try:
        raw_observation, _ = env.reset(seed=args.engineering_seed)
        initial_snapshot = snapshot_simulator(env)
        initial_observation_digest = canonical_digest(np.asarray(raw_observation))

        center_first, scale_first = bundle.infer(raw_observation, torch)
        center_second, scale_second = bundle.infer(raw_observation, torch)
        actor_repeat_exact = bool(
            np.array_equal(center_first, center_second)
            and np.array_equal(scale_first, scale_second)
        )
        actor_scale_matches_checkpoint = bool(
            np.array_equal(scale_first, bundle.action_standard_deviation)
        )

        # Reset with the same engineering seed and require an identical layout,
        # state, raw observation, and RNG position.
        repeated_observation, _ = env.reset(seed=args.engineering_seed)
        repeated_snapshot = snapshot_simulator(env)
        reset_deterministic = bool(
            canonical_digest(np.asarray(repeated_observation))
            == initial_observation_digest
            and repeated_snapshot.digest == initial_snapshot.digest
        )

        # Query-before-execute on a short deterministic learned-policy rollout.
        rollout_checks: list[dict[str, Any]] = []
        current_observation = np.asarray(repeated_observation, dtype=np.float64)
        for step in range(args.rollout_steps):
            center, _ = bundle.infer(current_observation, torch)
            action = np.clip(center, -1.0, 1.0)
            snapshot = snapshot_simulator(env)
            queried = oracle_query(env, snapshot, action, mujoco)
            actual, equivalent = execute_verified(env, snapshot, queried)
            rollout_checks.append(
                {
                    "step": step,
                    "center": center.tolist(),
                    "action": action.tolist(),
                    "query": _json_transition(queried),
                    "actual": _json_transition(actual),
                    "oracle_execute_equivalent": equivalent,
                }
            )
            if not equivalent or actual.observation is None:
                break
            current_observation = actual.observation
            if actual.terminated or actual.truncated:
                break

        # Force a goal hit to cover update_layout, goal model mutation, task RNG
        # consumption, and goal-position resampling in the restore contract.
        env.reset(seed=args.engineering_seed)
        task = env.unwrapped.task
        old_goal = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
        _set_agent_position(env, old_goal, mujoco)
        goal_snapshot = snapshot_simulator(env)
        goal_query = oracle_query(env, goal_snapshot, np.zeros(2), mujoco)
        goal_actual, goal_equivalent = execute_verified(env, goal_snapshot, goal_query)
        new_goal = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
        goal_relocated = bool(np.any(new_goal != old_goal))

        # Construct a hazard-boundary fibre.  The failed proposal is recorded
        # only as a mechanism check: fallback ordering remains a function of the
        # unperturbed policy mean, never of this proposal.
        env.reset(seed=args.engineering_seed)
        task = env.unwrapped.task
        hazard_names = sorted(
            name for name in task.world.geoms if name.startswith("hazard")
        )
        hazard = task.world.geoms[hazard_names[0]]
        hazard_center = np.asarray(hazard["pos"][:2], dtype=np.float64)
        hazard_radius = float(np.asarray(hazard["size"], dtype=np.float64)[0])
        boundary_target = hazard_center + np.array(
            [hazard_radius + args.boundary_offset, 0.0]
        )
        _set_agent_position(env, boundary_target, mujoco)
        fibre_snapshot = snapshot_simulator(env)
        coarse_actions = [
            np.array([first, second], dtype=np.float64)
            for first in (-1.0, 0.0, 1.0)
            for second in (-1.0, 0.0, 1.0)
        ]
        coarse_records = [
            oracle_query(env, fibre_snapshot, action, mujoco)
            for action in coarse_actions
        ]
        unsafe_index = next(
            (index for index, record in enumerate(coarse_records) if not record.feasible),
            None,
        )
        first_proposal = (
            coarse_actions[unsafe_index]
            if unsafe_index is not None
            else np.array([1.0, 1.0], dtype=np.float64)
        )
        boundary_observation = task.obs()
        boundary_policy_center, _ = bundle.infer(boundary_observation, torch)
        fallback = verified_grid_fallback(
            env,
            fibre_snapshot,
            boundary_policy_center,
            resolution=args.fallback_resolution,
            mujoco=mujoco,
        )

        # Engineering-only activation audit.  Follow one K=1 filtered-policy
        # trajectory using the learned actor's frozen Gaussian scale.  A failed
        # proposal invokes the state-only policy-mean fallback above.  This is a
        # diagnostic of whether the method activates, not an algorithm result.
        activation_rng = np.random.default_rng(args.activation_noise_seed)
        activation_observation, _ = env.reset(seed=args.engineering_seed)
        activation_fallbacks = 0
        activation_oracle_failures = 0
        activation_equivalence_failures = 0
        activation_completed_steps = 0
        activation_query_counts: list[int] = []
        for _ in range(args.activation_steps):
            policy_center, policy_scale = bundle.infer(
                activation_observation, torch
            )
            proposal = policy_center + policy_scale * activation_rng.standard_normal(2)
            activation_snapshot = snapshot_simulator(env)
            proposal_query = oracle_query(
                env, activation_snapshot, proposal, mujoco
            )
            if proposal_query.feasible:
                actual, equivalent = execute_verified(
                    env, activation_snapshot, proposal_query
                )
                activation_query_counts.append(1)
            else:
                activation_fallbacks += 1
                fallback_step = verified_grid_fallback(
                    env,
                    activation_snapshot,
                    policy_center,
                    resolution=args.fallback_resolution,
                    mujoco=mujoco,
                )
                activation_query_counts.append(1 + int(fallback_step["query_count"]))
                if not fallback_step["found"]:
                    activation_oracle_failures += 1
                    break
                equivalent = bool(fallback_step["oracle_execute_equivalent"])
                actual_observation = fallback_step["actual_observation"]
                actual = None
            if not equivalent:
                activation_equivalence_failures += 1
                break
            activation_completed_steps += 1
            if actual is not None:
                if actual.observation is None:
                    break
                activation_observation = actual.observation
                terminal = bool(actual.terminated or actual.truncated)
            else:
                activation_observation = np.asarray(
                    actual_observation, dtype=np.float64
                )
                terminal = bool(
                    fallback_step["actual_terminated"]
                    or fallback_step["actual_truncated"]
                )
            if terminal:
                break
        activation_rate = (
            activation_fallbacks / activation_completed_steps
            if activation_completed_steps
            else 0.0
        )

        normalizer_digest_after = bundle.normalizer.digest
        gates = {
            "protocol_and_artifact_hashes_match": True,
            "actor_inference_repeats_exactly": actor_repeat_exact,
            "actor_scale_matches_loaded_checkpoint": actor_scale_matches_checkpoint,
            "frozen_normalizer_unchanged": (
                normalizer_digest_before == normalizer_digest_after
            ),
            "same_seed_reset_is_exact": reset_deterministic,
            "all_rollout_oracle_execute_pairs_exact": bool(rollout_checks)
            and all(item["oracle_execute_equivalent"] for item in rollout_checks),
            "goal_hit_occurred": bool(goal_actual.info.get("goal_met", False)),
            "goal_was_relocated": goal_relocated,
            "goal_hit_oracle_execute_pair_exact": goal_equivalent,
            "fallback_found_verified_action": bool(fallback["found"]),
            "fallback_executed_zero_cost": fallback["selected_cost"] == 0.0,
            "fallback_oracle_execute_pair_exact": bool(
                fallback["oracle_execute_equivalent"]
            ),
            "activation_audit_completed_declared_steps": (
                activation_completed_steps == args.activation_steps
            ),
            "activation_audit_has_no_oracle_failure": (
                activation_oracle_failures == 0
            ),
            "activation_audit_has_exact_query_execute_pairs": (
                activation_equivalence_failures == 0
            ),
        }
        result = {
            "status": (
                "engineering_smoke_passed"
                if all(gates.values())
                else "engineering_smoke_failed"
            ),
            "paper_eligibility": "none; engineering mechanics only",
            "protocol_path": str(args.protocol),
            "protocol_sha256": file_sha256(args.protocol),
            "protocol_manifest_version": protocol["manifest_version"],
            "source_path": str(ANONYMOUS_SOURCE_PATH),
            "source_sha256": file_sha256(Path(__file__)),
            "environment": args.environment,
            "engineering_seed": args.engineering_seed,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "safety_gymnasium": safety_gymnasium.__version__,
                "mujoco": mujoco.__version__,
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": file_sha256(args.checkpoint),
            },
            "normalizer": {
                "path": str(args.normalizer),
                "sha256": file_sha256(args.normalizer),
                "count": bundle.normalizer.count,
                "state_digest": normalizer_digest_before,
            },
            "actor": {
                "initial_center": center_first.tolist(),
                "standard_deviation": scale_first.tolist(),
            },
            "same_seed_reset": {
                "initial_observation_digest": initial_observation_digest,
                "initial_state_digest": initial_snapshot.digest,
                "repeated_observation_digest": canonical_digest(
                    np.asarray(repeated_observation)
                ),
                "repeated_state_digest": repeated_snapshot.digest,
            },
            "rollout_checks": rollout_checks,
            "goal_relocation_check": {
                "old_goal": old_goal.tolist(),
                "new_goal": new_goal.tolist(),
                "query": _json_transition(goal_query),
                "actual": _json_transition(goal_actual),
                "oracle_execute_equivalent": goal_equivalent,
            },
            "fixed_fibre_check": {
                "hazard_name": hazard_names[0],
                "hazard_center": hazard_center.tolist(),
                "hazard_radius": hazard_radius,
                "boundary_target": boundary_target.tolist(),
                "coarse_feasible": [record.feasible for record in coarse_records],
                "first_proposal": first_proposal.tolist(),
                "fallback_policy_center": boundary_policy_center.tolist(),
                "fallback_is_independent_of_first_proposal": True,
                "fallback_resolution": args.fallback_resolution,
                "fallback": fallback,
            },
            "k1_activation_audit": {
                "trajectory": "K=1 filtered learned-policy trajectory",
                "proposal_scale": "frozen learned actor standard deviation",
                "noise_seed": args.activation_noise_seed,
                "declared_steps": args.activation_steps,
                "completed_steps": activation_completed_steps,
                "fallback_count": activation_fallbacks,
                "fallback_rate": activation_rate,
                "activation_warning_below_10_percent": activation_rate < 0.10,
                "oracle_failures": activation_oracle_failures,
                "oracle_execute_equivalence_failures": (
                    activation_equivalence_failures
                ),
                "mean_total_oracle_queries_per_step": (
                    float(np.mean(activation_query_counts))
                    if activation_query_counts
                    else None
                ),
                "max_total_oracle_queries_in_step": (
                    max(activation_query_counts) if activation_query_counts else None
                ),
            },
            "gates": gates,
            "passes": all(gates.values()),
            "claim_boundary": [
                "This opens one engineering seed only.",
                "It is not a controller evaluation or an algorithm comparison.",
                "The simulator oracle is privileged one-step lookahead.",
                "The fallback depends on state through the unperturbed policy mean; "
                "it is common only for the declared residual-plan threat where that "
                "mean is generated inside the common causal program.",
                "One-step feasibility does not prove forward invariance.",
                "No development, validation, or confirmation seed is opened.",
            ],
        }
        return result
    finally:
        env.close()


def _write_report(result: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Safety-Gym engineering smoke result",
        "",
        f"Status: **{result['status']}**.",
        "",
        "This artifact is engineering-only and has no paper-result eligibility.",
        "",
        "## Gates",
        "",
    ]
    for name, passed in result["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Fixed-fibre fallback",
            "",
            f"- grid: {result['fixed_fibre_check']['fallback_resolution']} x "
            f"{result['fixed_fibre_check']['fallback_resolution']}",
            f"- queries: {result['fixed_fibre_check']['fallback']['query_count']}",
            f"- selected action: "
            f"`{result['fixed_fibre_check']['fallback']['selected_action']}`",
            "- ordering center: the unperturbed frozen-policy mean at that state",
            "- failed/noisy proposals do not affect fallback ordering",
            "",
            "## K=1 activation diagnostic",
            "",
            f"- completed steps: {result['k1_activation_audit']['completed_steps']}",
            f"- fallback rate: {result['k1_activation_audit']['fallback_rate']:.3f}",
            f"- below-10% warning: "
            f"{result['k1_activation_audit']['activation_warning_below_10_percent']}",
            "",
            "## Scope",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="SafetyPointGoal2-v0")
    parser.add_argument("--engineering-seed", type=int, default=1570860784)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--fallback-resolution", type=int, default=121)
    parser.add_argument("--activation-steps", type=int, default=128)
    parser.add_argument("--activation-noise-seed", type=int, default=1368584074)
    parser.add_argument("--boundary-offset", type=float, default=2.5e-4)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/iclr2027_safety_gym_engineering_smoke_v2/result.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("output/iclr2027_safety_gym_engineering_smoke_v2/result.md"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_engineering_smoke(args)
    result["wall_time_seconds"] = time.perf_counter() - started
    _write_report(result, args.output_json, args.output_md)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
