"""Force selection loop to not return and hit final fallback line 63."""

from __future__ import annotations

import uuid

from ggnes.evolution.selection import select_match
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule


def r(prob):
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': 5, 'probability': prob},
    )


class ZeroRNG:
    class R:
        def random(self):
            return 2.0  # beyond any cumulative + group_weight

    def get_context_rng(self, _):
        return self.R()


def test_selection_final_fallback():
    # Large rounding precision zeros out probs; rng.random() too large to trigger selection
    matches = [(r(0.3), {}), (r(0.3), {}), (r(0.4), {})]
    sel = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', ZeroRNG(), {'probability_precision': 10.0})
    assert sel in matches


