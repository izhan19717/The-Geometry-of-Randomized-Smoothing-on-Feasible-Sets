"""Strict post-run audit of the frozen Goal2 safety-event study."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from scipy.special import ndtr, ndtri


DEFAULT_PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "GOAL2_SAFE_TRAJECTORY_CONFIRMATION_V1_20260919.json"
)
DEFAULT_INTEGRITY = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "GOAL2_SAFE_TRAJECTORY_POSTRUN_INTEGRITY_V1_20260919.json"
)
DEFAULT_RESULT = Path("outputs/iclr2027_goal2_safe_trajectory_confirmation_v1.json")
DEFAULT_AUDIT = Path("outputs/iclr2027_goal2_safe_trajectory_audit_v1.json")

RESULT_KEYS = {
    "status",
    "protocol_sha256",
    "source_sha256",
    "elapsed_seconds",
    "versions",
    "ordered_seed_sha256",
    "summary",
    "records",
}
RECORD_KEYS = {
    "controller_seed",
    "environment_seed",
    "sigma",
    "method",
    "attack_budget",
    "horizon",
    "event",
    "goal",
    "native_cost_hit",
    "stop_reason",
    "executed_steps",
    "selected_proposals",
    "rejected_proposals",
    "fallback_steps",
    "total_energy",
    "pathwise_energy_bound",
    "total_reward",
}
SUMMARY_KEYS = {
    "nominal_safety_count",
    "nominal_safety_fraction",
    "simultaneous_hoeffding_lower",
    "certified_adaptive_center_radius",
    "lower_safety_probability_at_tested_budget",
    "attacked_safety_count",
    "nominal_goal_before_cost_count",
    "attacked_goal_before_cost_count",
    "nominal_cost_count",
    "nominal_mean_executed_steps",
    "nominal_mean_fallback_steps",
    "max_attacked_energy",
    "pathwise_energy_limit",
}
ALLOWED_STOP_REASONS = {
    "first_goal_without_cost",
    "native_cost",
    "horizon",
    "environment_stop",
    "nonfinite_transition",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is not finite")
    return number


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{name} is not a nonnegative integer")
    return value


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-13
        )
    return left == right


def _stable_seeds(prefix: str, count: int) -> list[int]:
    from experiments.iclr2027_safety_gym_push1_trajectory_development import (
        stable_seed,
    )

    return [stable_seed(f"{prefix}{index}")[1] for index in range(count)]


def _validate_integrity_record(
    integrity: dict[str, Any],
    protocol_hash: str,
    result_hash: str,
    root: Path,
) -> tuple[dict[str, Any], set[int]]:
    if integrity.get("status") != "postrun_integrity_record_v1":
        raise RuntimeError("post-run integrity record has wrong status")
    if integrity.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("integrity record does not bind the protocol")
    if integrity.get("result_sha256") != result_hash:
        raise RuntimeError("raw result digest differs from the integrity record")
    for relative, expected in integrity["files_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise RuntimeError(f"post-run source closure differs at {relative}")

    integration_path = root / integrity["integration_protocol_path"]
    if file_sha256(integration_path) != integrity["integration_protocol_sha256"]:
        raise RuntimeError("integration protocol digest differs")
    integration = json.loads(integration_path.read_text())
    if integration.get("status") != "frozen_before_all_controller_engineering_gate":
        raise RuntimeError("integration protocol has wrong status")
    if integration.get("source_sha256") != integrity["files_sha256"][
        "experiments/iclr2027_safety_gym_all_controller_integration.py"
    ]:
        raise RuntimeError("integration source digest differs")
    for relative, expected in integration["runtime_source_sha256"].items():
        if integrity["files_sha256"].get(relative) != expected:
            raise RuntimeError(f"integration runtime closure differs at {relative}")

    previous_path = root / integrity["previous_confirmation_protocol_path"]
    if file_sha256(previous_path) != integrity["previous_confirmation_protocol_sha256"]:
        raise RuntimeError("previous confirmation protocol digest differs")
    previous = json.loads(previous_path.read_text())
    previous_seeds = _stable_seeds(
        previous["seed_prefix"], int(previous["trajectory_count"])
    )
    previous_digest = sha256(
        json.dumps(previous_seeds, separators=(",", ":")).encode()
    ).hexdigest()
    if previous_digest != previous["ordered_seed_sha256"]:
        raise RuntimeError("previous confirmation seeds do not match their freeze")

    development_path = root / integrity["opened_development_output_path"]
    if file_sha256(development_path) != integrity["opened_development_output_sha256"]:
        raise RuntimeError("opened development output digest differs")
    development = json.loads(development_path.read_text())
    if development.get("status") != "open_development_only":
        raise RuntimeError("opened development artifact has wrong status")
    declared_development = set(development["development_seeds"])
    reconstructed_development: set[int] = set()
    for item in integrity["opened_development_prefixes"]:
        reconstructed_development.update(
            _stable_seeds(item["prefix"], int(item["count"]))
        )
    if not declared_development.issubset(reconstructed_development):
        raise RuntimeError("opened development seeds do not match declared prefixes")
    excluded = set(previous_seeds) | reconstructed_development
    return integration, excluded


def _validate_schedule(protocol: dict[str, Any], integrity: dict[str, Any]) -> None:
    base_sigma = float(protocol["sigma"])
    base_kappa = float(protocol["kappa"])
    schedules = integrity["method_schedules"]
    expected_four = {
        "proposal_count": 4,
        "sigma": base_sigma,
        "rho": float(protocol["rho_four"]),
        "kappa": base_kappa,
    }
    expected_single = {
        "proposal_count": 1,
        "sigma": base_sigma / math.sqrt(base_kappa),
        "rho": 0.0,
        "kappa": 1.0,
    }
    if set(schedules) != {"correlated_four", "matched_single"}:
        raise RuntimeError("method schedule set differs")
    for method, expected in (
        ("correlated_four", expected_four),
        ("matched_single", expected_single),
    ):
        if set(schedules[method]) != set(expected):
            raise RuntimeError(f"{method} schedule schema differs")
        for key, value in expected.items():
            if not _same_value(schedules[method][key], value):
                raise RuntimeError(f"{method} schedule differs at {key}")
        charge = schedules[method]["kappa"] / schedules[method]["sigma"] ** 2
        if not math.isclose(
            charge,
            float(integrity["common_energy_charge"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{method} energy charge differs")


def _safety_event(row: dict[str, Any]) -> bool:
    return not row["native_cost_hit"] and row["stop_reason"] in {
        "first_goal_without_cost",
        "horizon",
    }


def _validate_record(
    row: dict[str, Any], protocol: dict[str, Any], expected_charge: float
) -> None:
    if set(row) != RECORD_KEYS:
        raise RuntimeError("trajectory record schema differs")
    for name in ("event", "goal", "native_cost_hit"):
        if type(row[name]) is not int or row[name] not in (0, 1):
            raise RuntimeError(f"{name} is not a binary integer")
    if row["event"] != row["goal"]:
        raise RuntimeError("raw goal event and goal flag differ")
    if row["stop_reason"] not in ALLOWED_STOP_REASONS:
        raise RuntimeError("unknown trajectory stop reason")
    for name in (
        "executed_steps",
        "selected_proposals",
        "rejected_proposals",
        "fallback_steps",
    ):
        _nonnegative_integer(row[name], name)
    horizon = int(protocol["horizon"])
    if row["horizon"] != horizon or row["executed_steps"] > horizon:
        raise RuntimeError("trajectory length or horizon differs")
    if row["selected_proposals"] + row["fallback_steps"] != row["executed_steps"]:
        raise RuntimeError("executed-step accounting differs")
    sigma = _finite_number(row["sigma"], "sigma")
    if sigma != float(protocol["sigma"]):
        raise RuntimeError("stored base noise scale differs")
    budget = _finite_number(row["attack_budget"], "attack_budget")
    total_energy = _finite_number(row["total_energy"], "total_energy")
    energy_bound = _finite_number(
        row["pathwise_energy_bound"], "pathwise_energy_bound"
    )
    _finite_number(row["total_reward"], "total_reward")
    if budget < 0 or total_energy < -1e-15 or energy_bound < 0:
        raise RuntimeError("negative budget or energy")
    declared = expected_charge * budget**2
    if not math.isclose(energy_bound, declared, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("pathwise energy declaration differs")
    if total_energy > declared + 1e-10:
        raise RuntimeError("pathwise energy bound is violated")
    if budget == 0.0 and abs(total_energy) > 1e-15:
        raise RuntimeError("nominal trajectory has nonzero attack energy")

    reason = row["stop_reason"]
    if reason == "first_goal_without_cost":
        if row["event"] != 1 or row["native_cost_hit"] != 0:
            raise RuntimeError("goal stop is inconsistent")
    elif reason == "native_cost":
        if row["event"] != 0 or row["native_cost_hit"] != 1:
            raise RuntimeError("native-cost stop is inconsistent")
    else:
        if row["event"] != 0 or row["native_cost_hit"] != 0:
            raise RuntimeError("nongoal stop is inconsistent")
    if reason == "horizon" and row["executed_steps"] != horizon:
        raise RuntimeError("horizon success ended before the horizon")


def audit(
    result: dict[str, Any],
    protocol: dict[str, Any],
    protocol_hash: str,
    result_hash: str,
    integrity: dict[str, Any],
    integrity_hash: str,
    root: Path = Path("."),
) -> dict[str, Any]:
    if set(result) != RESULT_KEYS:
        raise RuntimeError("raw result schema differs")
    if result["status"] != "held_out_confirmation_complete":
        raise RuntimeError("run did not complete")
    _finite_number(result["elapsed_seconds"], "elapsed_seconds")
    if result["elapsed_seconds"] < 0:
        raise RuntimeError("elapsed time is negative")
    if result["protocol_sha256"] != protocol_hash:
        raise RuntimeError("protocol hash differs from frozen artifact")
    if result["source_sha256"] != protocol["confirmation_source_sha256"]:
        raise RuntimeError("confirmation source hash differs from protocol")
    if result["ordered_seed_sha256"] != protocol["ordered_seed_sha256"]:
        raise RuntimeError("seed digest differs from protocol")
    if result["versions"] != integrity["result_versions"]:
        raise RuntimeError("recorded runtime versions differ")
    if file_sha256(root / integrity["protocol_path"]) != protocol_hash:
        raise RuntimeError("protocol path does not match supplied protocol")

    integration, excluded_seeds = _validate_integrity_record(
        integrity, protocol_hash, result_hash, root
    )
    if integration["controller_artifact_sha256"] != protocol["controller_artifact_sha256"]:
        raise RuntimeError("controller artifacts differ between frozen manifests")
    if integration["controller_seeds"] != protocol["controller_seeds"]:
        raise RuntimeError("controller order differs between frozen manifests")
    _validate_schedule(protocol, integrity)

    count = int(protocol["trajectory_count"])
    seeds = _stable_seeds(protocol["seed_prefix"], count)
    digest = sha256(json.dumps(seeds, separators=(",", ":")).encode()).hexdigest()
    if digest != protocol["ordered_seed_sha256"] or len(set(seeds)) != count:
        raise RuntimeError("seed reconstruction failed")
    if set(seeds) & excluded_seeds:
        raise RuntimeError("confirmation cohort overlaps known prior or development seeds")

    controllers = protocol["controller_seeds"]
    methods = protocol["methods"]
    budget = float(protocol["attack_budget"])
    base_sigma = float(protocol["sigma"])
    base_kappa = float(protocol["kappa"])
    expected_charge = base_kappa / base_sigma**2
    expected_keys = {
        (seed, controller, method, attack)
        for seed in seeds
        for controller in controllers
        for method in methods
        for attack in (0.0, budget)
    }
    records = result["records"]
    if not isinstance(records, list):
        raise RuntimeError("records are not a list")
    actual_keys = []
    for row in records:
        if not isinstance(row, dict):
            raise RuntimeError("trajectory record is not an object")
        _validate_record(row, protocol, expected_charge)
        actual_keys.append(
            (
                row["environment_seed"],
                row["controller_seed"],
                row["method"],
                row["attack_budget"],
            )
        )
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise RuntimeError("trajectory arms are missing or extra")
    if len(set(actual_keys)) != len(actual_keys):
        raise RuntimeError("duplicate trajectory arm")

    expected_summary_names = {
        f"seed={controller}/{method}"
        for controller in controllers
        for method in methods
    }
    if set(result["summary"]) != expected_summary_names:
        raise RuntimeError("summary arms are missing or extra")
    error_per_arm = protocol["familywise_error"] / len(expected_summary_names)
    penalty = math.sqrt(math.log(1 / error_per_arm) / (2 * count))
    q = float(protocol["target_safety_probability"])
    rows: dict[str, dict[str, Any]] = {}
    for controller in controllers:
        for method in methods:
            nominal = [
                row
                for row in records
                if row["controller_seed"] == controller
                and row["method"] == method
                and row["attack_budget"] == 0.0
            ]
            attacked = [
                row
                for row in records
                if row["controller_seed"] == controller
                and row["method"] == method
                and row["attack_budget"] == budget
            ]
            if len(nominal) != count or len(attacked) != count:
                raise RuntimeError("trajectory arm has wrong sample count")
            successes = sum(_safety_event(row) for row in nominal)
            attacked_successes = sum(_safety_event(row) for row in attacked)
            lower = max(0.0, successes / count - penalty)
            radius = (
                base_sigma * (ndtri(lower) - ndtri(q)) / math.sqrt(base_kappa)
                if lower > q
                else None
            )
            tested_lower = (
                float(ndtr(ndtri(lower) - math.sqrt(base_kappa) * budget / base_sigma))
                if lower > 0
                else 0.0
            )
            expected_summary = {
                "nominal_safety_count": successes,
                "nominal_safety_fraction": successes / count,
                "simultaneous_hoeffding_lower": lower,
                "certified_adaptive_center_radius": radius,
                "lower_safety_probability_at_tested_budget": tested_lower,
                "attacked_safety_count": attacked_successes,
                "nominal_goal_before_cost_count": sum(row["event"] for row in nominal),
                "attacked_goal_before_cost_count": sum(row["event"] for row in attacked),
                "nominal_cost_count": sum(row["native_cost_hit"] for row in nominal),
                "nominal_mean_executed_steps": sum(
                    row["executed_steps"] for row in nominal
                ) / count,
                "nominal_mean_fallback_steps": sum(
                    row["fallback_steps"] for row in nominal
                ) / count,
                "max_attacked_energy": max(row["total_energy"] for row in attacked),
                "pathwise_energy_limit": expected_charge * budget**2,
            }
            key = f"seed={controller}/{method}"
            recorded = result["summary"][key]
            if set(recorded) != SUMMARY_KEYS:
                raise RuntimeError(f"summary schema differs for {key}")
            for field, expected in expected_summary.items():
                if not _same_value(recorded[field], expected):
                    raise RuntimeError(f"summary differs for {key} at {field}")
            goal_count = expected_summary["nominal_goal_before_cost_count"]
            horizon_count = sum(row["stop_reason"] == "horizon" for row in nominal)
            rows[key] = {
                **expected_summary,
                "nominal_horizon_without_cost_count": horizon_count,
                "nominal_early_stop_failure_count": sum(
                    row["stop_reason"] in {"environment_stop", "nonfinite_transition"}
                    for row in nominal
                ),
                "nominal_goal_plus_horizon_count": goal_count + horizon_count,
                "tested_attack_budget": budget,
                "tested_budget_within_certificate": bool(
                    radius is not None and budget <= radius
                ),
            }

    return {
        "status": "strict_postrun_audit_passed",
        "target_initial_law": "uniform mixture over the 256 frozen reset seeds",
        "certified_event": integrity["field_interpretation"]["certified_event"],
        "raw_event_field": integrity["field_interpretation"]["records.event"],
        "primary_confidence_method": (
            "one-sided Hoeffding for six fixed-cohort nominal arms"
        ),
        "public_trace_scope": (
            "arm-blind controller, selection, fallback, dynamics, and stop outputs; "
            "attack budget and energy diagnostics are experimental metadata"
        ),
        "protocol_sha256": protocol_hash,
        "integrity_record_sha256": integrity_hash,
        "result_sha256": result_hash,
        "confirmation_source_sha256": result["source_sha256"],
        "record_count": len(records),
        "known_exclusion_seed_count": len(excluded_seeds),
        "method_schedules": integrity["method_schedules"],
        "postrun_remote_environment": integrity["postrun_remote_environment"],
        "summary": rows,
        "limitations": integrity["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    integrity_bytes = args.integrity.read_bytes()
    result_bytes = args.result.read_bytes()
    report = audit(
        json.loads(result_bytes),
        json.loads(protocol_bytes),
        sha256(protocol_bytes).hexdigest(),
        sha256(result_bytes).hexdigest(),
        json.loads(integrity_bytes),
        sha256(integrity_bytes).hexdigest(),
    )
    report["audit_source_sha256"] = file_sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
