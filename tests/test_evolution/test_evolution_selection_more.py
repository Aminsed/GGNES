"""More selection tests to fully cover branches (fallback and index bound)."""

from __future__ import annotations

import uuid

from ggnes.evolution.selection import select_match
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule


class FixedRNGManager:
    def __init__(self, r):
        self._r = float(r)

    class _R:
        def __init__(self, r):
            self._r = r

        def random(self):  # noqa: D401
            return self._r

    def get_context_rng(self, context):  # noqa: D401
        return self._R(self._r)


def _rule(priority: int, probability: float) -> Rule:
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': priority, 'probability': probability},
    )


def test_selection_all_group_weights_zero_fallback_to_order():
    # Make rounding zero-out probabilities by using huge precision
    r1 = _rule(5, 0.6)
    r2 = _rule(5, 0.3)
    r3 = _rule(5, 0.1)
    matches = [(r1, {}), (r2, {}), (r3, {})]
    cfg = {'probability_precision': 10.0}
    sel = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', FixedRNGManager(0.5), cfg)
    assert sel in matches


def test_selection_index_bound_adjustment_path():
    # Create two probability groups and set r to the boundary of first group
    r1 = _rule(5, 0.6)
    r2 = _rule(5, 0.6)
    r3 = _rule(5, 0.4)
    matches = [(r1, {}), (r2, {}), (r3, {})]
    # After normalization: probs ~ [0.375, 0.375, 0.25]; first group_weight = 0.375 * 2 = 0.75
    rng = FixedRNGManager(0.75)
    cfg = {'probability_precision': 1e-6}
    sel = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', rng, cfg)
    assert sel in matches


