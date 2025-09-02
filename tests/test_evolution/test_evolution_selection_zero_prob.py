"""Cover selection group_weight == 0 continue and final fallback."""

from __future__ import annotations

import uuid

from ggnes.evolution.selection import select_match
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _mk(p):
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': 5, 'probability': p},
    )


def test_selection_zero_probability_group_and_fallback():
    # One zero-probability group should trigger group_weight == 0 and continue
    r_hi = _mk(1.0)
    r_zero = _mk(0.0)
    matches = [(r_hi, {}), (r_zero, {})]
    cfg = {'probability_precision': 1e-6}
    rng = RNGManager(seed=7)
    sel = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', rng, cfg)
    assert sel in matches


