import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.iclr2027_safety_gym_all_controller_integration import (
    CONTROLLER_SEEDS,
    DEFAULT_PROTOCOL,
    _coarse_actions,
    _negative_selection,
    _terminal_row,
    verify_controller_artifacts,
)
from experiments.iclr2027_safety_gym_engineering_smoke import file_sha256


def test_frozen_protocol_binds_sources_and_controller_artifacts() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_all_controller_engineering_gate"
    assert tuple(protocol["controller_seeds"]) == CONTROLLER_SEEDS
    assert file_sha256(Path(protocol["source_path"])) == protocol["source_sha256"]
    assert file_sha256(Path(protocol["test_path"])) == protocol["test_sha256"]
    for path, expected in protocol["runtime_source_sha256"].items():
        assert file_sha256(Path(path)) == expected
    root = Path("output/iclr2027_safety_gym_safe_controllers_v2")
    for seed in CONTROLLER_SEEDS:
        expected = protocol["controller_artifact_sha256"][str(seed)]
        for name, expected_sha256 in expected.items():
            assert file_sha256(root / f"seed{seed}" / name) == expected_sha256


def test_negative_fixture_totalizes_after_complete_inspection() -> None:
    actions = _coarse_actions()
    inspected: list[tuple[float, float]] = []

    def reject(action: np.ndarray) -> bool:
        inspected.append(tuple(action))
        return False

    selection = _negative_selection(actions, reject)
    assert selection.abstained
    assert selection.action is None
    assert selection.proposal_inspections == 1
    assert selection.fallback_inspections == 8
    assert inspected == [tuple(action) for action in actions]


def test_terminal_row_requires_complete_horizon(tmp_path: Path) -> None:
    path = tmp_path / "progress.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Train/Epoch", "Train/TotalSteps"])
        writer.writeheader()
        for epoch in range(1, 501):
            writer.writerow(
                {"Train/Epoch": epoch, "Train/TotalSteps": epoch * 20_000}
            )
    final = _terminal_row(path)
    assert final["Train/Epoch"] == "500"
    assert final["Train/TotalSteps"] == "10000000"


def test_controller_artifacts_bind_exact_hashes(tmp_path: Path) -> None:
    names = ("model499.pt", "state499.pkl", "config.json", "progress.csv")
    paths = {}
    expected = {}
    for index, name in enumerate(names):
        path = tmp_path / name
        path.write_bytes(f"artifact-{index}".encode())
        paths[name] = path
        expected[name] = file_sha256(path)
    assert verify_controller_artifacts(paths, expected) == expected
    expected["model499.pt"] = "0" * 64
    with pytest.raises(RuntimeError, match="model499.pt"):
        verify_controller_artifacts(paths, expected)
