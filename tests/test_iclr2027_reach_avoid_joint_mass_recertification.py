from pathlib import Path

import pytest

from experiments.iclr2027_reach_avoid_joint_mass_recertification import recertify


PROTOCOL = Path(
    "_ICLR_2027__Feasibility_Breaks_Smoothing/research/"
    "REACH_AVOID_CONTROLLER_PROTOCOL_V2_20260830.json"
)


def test_frozen_reference_state_recertification_reproduces_audit():
    result = recertify(PROTOCOL)
    summary = result["summary"]
    assert result["state_count"] == 448
    assert result["layout_count"] == 64
    assert result["positive_joint_mass_radius_count"] == 448
    assert result["imported_radius_crossing_count"] == 448
    assert result["joint_mass_radius_crossing_count"] == 0
    assert summary["joint_mass_radius"]["minimum"] == pytest.approx(
        0.0017083, abs=5e-8
    )
    assert summary["joint_mass_radius"]["median"] == pytest.approx(
        0.2266669, abs=5e-8
    )
    assert summary["joint_to_imported_ratio"]["median"] == pytest.approx(
        0.68716, abs=5e-6
    )
    assert summary["joint_to_boundary_ratio"]["maximum"] == pytest.approx(
        0.86279, abs=5e-6
    )
