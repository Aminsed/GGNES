"""Additional selection tests to cover tie/fallback and zero-probability cases."""

from __future__ import annotations

import uuid

from ggnes.evolution.selection import select_match
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _rule(prio: int, prob: float):
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': prio, 'probability': prob},
    )


def test_selection_equal_zero_probabilities_fallback_to_order():
    # All probabilities zero -> equal weights -> group_weight zero -> fallback path
    r1 = _rule(5, 0.0)
    r2 = _rule(5, 0.0)
    r3 = _rule(5, 0.0)
    matches = [(r1, {}), (r2, {}), (r3, {})]
    rng = RNGManager(seed=42)
    cfg = {'probability_precision': 1e-6}
    sel = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', rng, cfg)
    assert sel in matches


