"""Integrity checks for the frozen Goal2 trajectory audit."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from experiments.iclr2027_goal2_safe_trajectory_audit import (
    DEFAULT_INTEGRITY,
    DEFAULT_PROTOCOL,
    DEFAULT_RESULT,
    audit,
)


def _artifacts() -> tuple[dict, dict, bytes, dict, bytes]:
    protocol_bytes = DEFAULT_PROTOCOL.read_bytes()
    result_bytes = DEFAULT_RESULT.read_bytes()
    integrity_bytes = DEFAULT_INTEGRITY.read_bytes()
    return (
        json.loads(result_bytes),
        json.loads(protocol_bytes),
        protocol_bytes,
        json.loads(integrity_bytes),
        integrity_bytes,
    )


def _audit_with_bound_digest(result: dict) -> dict:
    original, protocol, protocol_bytes, integrity, integrity_bytes = _artifacts()
    return audit(
        result,
        protocol,
        sha256(protocol_bytes).hexdigest(),
        sha256(DEFAULT_RESULT.read_bytes()).hexdigest(),
        integrity,
        sha256(integrity_bytes).hexdigest(),
    )


def test_strict_audit_reproduces_all_six_certificates() -> None:
    result, _, _, _, _ = _artifacts()
    report = _audit_with_bound_digest(result)
    assert report["status"] == "strict_postrun_audit_passed"
    assert report["record_count"] == 3 * 2 * 2 * 256
    assert len(report["summary"]) == 6
    assert all(row["tested_budget_within_certificate"] for row in report["summary"].values())
    assert sum(row["nominal_horizon_without_cost_count"] for row in report["summary"].values()) == 714


@pytest.mark.parametrize("tamper", ("version", "extra_summary", "nan_energy"))
def test_strict_audit_rejects_previously_accepted_tampering(tamper: str) -> None:
    result, _, _, _, _ = _artifacts()
    changed = deepcopy(result)
    if tamper == "version":
        changed["versions"]["safety_gymnasium"] = "999.0"
    elif tamper == "extra_summary":
        changed["summary"]["seed=bogus/radius=999"] = {}
    else:
        changed["records"][0]["total_energy"] = float("nan")
    with pytest.raises(RuntimeError):
        _audit_with_bound_digest(changed)
