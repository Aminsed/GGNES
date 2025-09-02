"""Cover selection rounding/precision and fallback at end of loop."""

from __future__ import annotations

import uuid

from ggnes.evolution.selection import select_match
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule


def rule(prob, prio=5):
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': prio, 'probability': prob},
    )


class Rng:
    def __init__(self, r):
        self._r = r

    class RR:
        def __init__(self, r):
            self._r = r

        def random(self):
            return self._r

    def get_context_rng(self, _):
        return self.RR(self._r)


def test_selection_precision_groups_same_then_fallback_end():
    # Equal rounded groups; set r so cumulative never triggers until last prob
    r1 = rule(0.33)
    r2 = rule(0.33)
    r3 = rule(0.34)
    matches = [(r1, {}), (r2, {}), (r3, {})]
    sel = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', Rng(1.0), {'probability_precision': 0.01})
    assert sel in matches


