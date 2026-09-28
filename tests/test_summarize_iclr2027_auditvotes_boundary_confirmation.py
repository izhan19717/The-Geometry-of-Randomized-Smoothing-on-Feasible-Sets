import pytest

from experiments.summarize_iclr2027_auditvotes_boundary_confirmation import (
    binary_substituted_radius_lower,
    exact_winner_pvalues,
    retained_margin,
)


def test_exact_winner_pvalues_respect_competitor_order() -> None:
    pvalues = exact_winner_pvalues([900, 50, 40], 0)
    assert len(pvalues) == 2
    assert max(pvalues) < 1e-20


def test_binary_radius_lower_is_positive_for_clear_majority() -> None:
    lower, radius = binary_substituted_radius_lower(900, 1000, 0.25, 1e-4)
    assert 0.5 < lower < 0.9
    assert radius > 0.0


def test_retained_margin_orientation() -> None:
    assert retained_margin([15, 2, 5], 22, 0, 2) == pytest.approx(10 / 22)
    assert retained_margin([2, 15, 5], 22, 0, 2) == pytest.approx(-3 / 22)
