import numpy as np
from types import SimpleNamespace

from experiments.iclr2027_safety_gym_engineering_smoke import (
    FrozenNormalizer,
    canonical_digest,
    ordered_fallback_grid,
    restore_warmstart_after_forward,
)


def test_canonical_digest_is_sensitive_to_array_metadata_and_values() -> None:
    first = np.array([[1.0, 2.0]], dtype=np.float64)
    second = np.array([[1.0, 3.0]], dtype=np.float64)
    assert canonical_digest(first) == canonical_digest(first.copy())
    assert canonical_digest(first) != canonical_digest(second)
    assert canonical_digest(first) != canonical_digest(first.astype(np.float32))
    assert canonical_digest(first) != canonical_digest(first.reshape(2, 1))


def test_frozen_normalizer_matches_declared_formula_without_mutation() -> None:
    normalizer = FrozenNormalizer(
        mean=np.array([1.0, -1.0]),
        variance=np.array([4.0, 9.0]),
        count=12.5,
    )
    before = normalizer.digest
    normalized = normalizer.normalize(np.array([5.0, 5.0]))
    expected = np.array([4.0, 6.0]) / np.sqrt(np.array([4.0, 9.0]) + 1e-8)
    np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=0.0)
    assert normalizer.digest == before


def test_frozen_normalizer_rejects_wrong_shape() -> None:
    normalizer = FrozenNormalizer(
        mean=np.zeros(2), variance=np.ones(2), count=1.0
    )
    with np.testing.assert_raises(ValueError):
        normalizer.normalize(np.zeros(3))


def test_fallback_grid_uses_distance_then_lexicographic_tie_break() -> None:
    ordered = list(ordered_fallback_grid(np.array([0.0, 0.0]), resolution=3))
    expected_first_five = [
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
    ]
    for actual, expected in zip(ordered[:5], expected_first_five):
        np.testing.assert_array_equal(actual, expected)
    assert len(ordered) == 9


def test_fallback_grid_uses_unclipped_policy_center() -> None:
    first = next(ordered_fallback_grid(np.array([4.0, -3.0]), resolution=5))
    np.testing.assert_array_equal(first, np.array([1.0, -1.0]))


def test_fallback_grid_rejects_invalid_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        list(ordered_fallback_grid(np.zeros(2), resolution=1))
    with np.testing.assert_raises(ValueError):
        list(ordered_fallback_grid(np.array([np.nan, 0.0]), resolution=3))


def test_warmstart_bytes_are_reapplied_after_forward() -> None:
    expected = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    data = SimpleNamespace(qacc_warmstart=expected + np.finfo(float).eps)
    snapshot = SimpleNamespace(data_arrays={"qacc_warmstart": expected.copy()})
    restore_warmstart_after_forward(data, snapshot)
    np.testing.assert_array_equal(data.qacc_warmstart, expected)
