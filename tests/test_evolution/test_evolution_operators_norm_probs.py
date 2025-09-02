"""Cover _normalize_probs branches in operators."""

from ggnes.evolution.operators import _normalize_probs


def test_normalize_probs_zero_total_nonempty():
    # Sum is zero, but non-empty list -> uniform distribution
    assert _normalize_probs([0.0, 0.0, 0.0]) == [1/3, 1/3, 1/3]


def test_normalize_probs_empty():
    assert _normalize_probs([]) == []


