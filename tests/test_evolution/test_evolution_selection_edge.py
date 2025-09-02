"""Edge tests for selection grouping to cover end-group path."""

from __future__ import annotations

import uuid

from ggnes.evolution.selection import select_match
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule


def _r(p, pr):
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': pr, 'probability': p},
    )


class Rng:
    def __init__(self, r):
        self._r = r

    class X:
        def __init__(self, r):
            self._r = r

        def random(self):
            return self._r

    def get_context_rng(self, _):
        return self.X(self._r)


def test_selection_last_group_fallback():
    # Two groups with clear ordering. r is large to force last-group selection path
    m = [(_r(0.7, 5), {}), (_r(0.2, 5), {}), (_r(0.1, 5), {})]
    rng = Rng(0.999)
    cfg = {'probability_precision': 1e-6}
    sel = select_match(m, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', rng, cfg)
    assert sel in m
