import numpy as np

from experiments.iclr2027_auditvotes_boundary_search import (
    confidence_log_margin_numpy,
    derived_seed,
    pairwise_winner_pvalues,
    project_l2_box_numpy,
)
from experiments.plot_iclr2027_filtered_smoothing import retained_pair_margin


def test_confidence_margin_matches_softmax_threshold() -> None:
    logits = np.asarray([[4.0, 0.0, -1.0], [1.0, 0.9, 0.8]])
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    margin = confidence_log_margin_numpy(logits, 0, 0.9)
    assert np.array_equal(margin > 0, probabilities[:, 0] > 0.9)


def test_projection_enforces_box_and_radius() -> None:
    original = np.asarray([0.2, 0.8, 0.5])
    proposed = np.asarray([-2.0, 3.0, 0.9])
    result = project_l2_box_numpy(original, proposed, 0.25)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    assert np.linalg.norm(result - original) <= 0.25 + 1e-14


def test_seed_derivation_is_stable_and_separates_streams() -> None:
    assert derived_seed(17, 4, "a") == derived_seed(17, 4, "a")
    assert derived_seed(17, 4, "a") != derived_seed(17, 4, "b")


def test_pairwise_exact_tests_accept_clear_winner() -> None:
    pvalues = pairwise_winner_pvalues([900, 50, 50], 0)
    assert len(pvalues) == 2
    assert max(pvalues) < 1e-20


def test_pairwise_exact_tests_reject_tie() -> None:
    pvalues = pairwise_winner_pvalues([50, 50, 0], 0)
    assert max(pvalues) >= 0.5


def test_retained_pair_margin_has_declared_orientation() -> None:
    assert retained_pair_margin([20, 5, 10], 35, 0, 2) == 10 / 35
    assert retained_pair_margin([5, 20, 10], 35, 0, 2) == -5 / 35
