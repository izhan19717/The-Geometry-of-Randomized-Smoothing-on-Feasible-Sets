from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/iclr2027_auditvotes_threshold_selection.py"
)
SPEC = spec_from_file_location("auditvotes_threshold_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selection_uses_declared_mean_and_lower_threshold_tie_break() -> None:
    summary = {
        "0.0": {"certified_accuracy": {"0.25": 0.4, "0.5": 0.3}},
        "0.5": {"certified_accuracy": {"0.25": 0.4, "0.5": 0.3}},
        "0.9": {"certified_accuracy": {"0.25": 0.5, "0.5": 0.1}},
    }
    selected, scores = MODULE._select_threshold(
        summary, (0.0, 0.5, 0.9), (0.25, 0.5)
    )
    assert selected == 0.0
    assert scores == {"0.0": 0.35, "0.5": 0.35, "0.9": 0.3}


def test_threshold_summary_requires_a_strictly_positive_zero_radius() -> None:
    positive = MODULE.ThresholdResult(
        threshold=0.0,
        selected_label=1,
        selection_retained=100,
        estimation_retained=1000,
        acceptance_rate=1.0,
        selected_joint_count=600,
        selected_joint_lower=0.55,
        runner_joint_upper=0.35,
        joint_radius=0.1,
        correct=True,
    )
    abstained = MODULE.ThresholdResult(
        threshold=0.0,
        selected_label=1,
        selection_retained=100,
        estimation_retained=1000,
        acceptance_rate=1.0,
        selected_joint_count=400,
        selected_joint_lower=0.35,
        runner_joint_upper=0.45,
        joint_radius=0.0,
        correct=True,
    )
    rows = [
        MODULE.ImageResult(0, 1, 0.1, (positive,)),
        MODULE.ImageResult(1, 1, 0.1, (abstained,)),
    ]
    result = MODULE._threshold_summary(rows, (0.0,), (0.0,))["0.0"]
    assert result["selected_label_accuracy"] == 1.0
    assert result["positive_certificate_fraction"] == 0.5
    assert result["certified_accuracy"]["0.0"] == 0.5
