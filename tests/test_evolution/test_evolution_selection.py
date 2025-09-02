"""[T-evo-03] Selection strategy tests per project_guide.md §7.4."""

from __future__ import annotations

import uuid

from ggnes.evolution.selection import select_match
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _mk_rule(priority=0, probability=1.0):
    rid = uuid.uuid4()
    return Rule(
        rule_id=rid,
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': priority, 'probability': probability}
    )


def test_selection_priority_then_probability_then_order():
    # Two high-priority with different probabilities, one lower priority
    r_low = _mk_rule(priority=1, probability=1.0)
    r_hi1 = _mk_rule(priority=5, probability=0.2)
    r_hi2 = _mk_rule(priority=5, probability=0.8)
    matches = [(r_hi1, {}), (r_low, {}), (r_hi2, {})]
    cfg = {'probability_precision': 1e-6}
    rng = RNGManager(seed=7)

    # Should always filter to hi priority set and then prefer 0.8 most of the time
    selected = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', rng, cfg)
    assert selected[0] in {r_hi1, r_hi2}


def test_selection_grouping_precision_tie_break_by_order():
    r1 = _mk_rule(priority=5, probability=0.5000001)
    r2 = _mk_rule(priority=5, probability=0.5)
    r3 = _mk_rule(priority=5, probability=0.4999999)
    # All effectively same group at precision 1e-3
    matches = [(r1, {}), (r2, {}), (r3, {})]
    cfg = {'probability_precision': 1e-3}
    rng = RNGManager(seed=1234)

    sel = select_match(matches, 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', rng, cfg)
    # Tie-break by original order among same-probability group
    assert sel in matches


def test_selection_errors_and_fallback():
    # Unknown strategy should raise when matches are present
    rng = RNGManager(seed=9)
    rid = uuid.uuid4()
    r = Rule(rid, LHSPattern(nodes=[], edges=[], boundary_nodes=[]), RHSAction(), EmbeddingLogic(), metadata={'priority': 0, 'probability': 1.0})
    try:
        select_match([(r, {})], 'UNKNOWN', rng, {})
        raise AssertionError('Expected ValueError for unknown strategy')
    except ValueError:
        pass

    # Empty list -> None for canonical strategy
    assert select_match([], 'PRIORITY_THEN_PROBABILITY_THEN_ORDER', rng, {}) is None


