"""[T-evo] Evolution operators tests for M12.

Covers:
- [T-evo-01] uniform_crossover respects per-rule probability; order-independent seeding; min/max rules enforced.
- [T-evo-02] mutate respects mutation_rate and tiered probabilities; produces new genotype_id; deterministic with RNGManager.
- [T-evo-03] Selection strategy PRIORITY_THEN_PROBABILITY_THEN_ORDER grouping precision and tie-breaking (delegated to selection tests).
"""

from __future__ import annotations

import uuid

from ggnes.evolution.genotype import Genotype
from ggnes.evolution.operators import mutate, uniform_crossover
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _mk_rule(rule_id=None, priority=0, probability=1.0):
    rid = rule_id or uuid.uuid4()
    lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
    rhs = RHSAction()
    emb = EmbeddingLogic()
    return Rule(rid, lhs, rhs, emb, metadata={'priority': priority, 'probability': probability})


def test_uniform_crossover_prob_and_bounds():
    # [T-evo-01]
    r1 = _mk_rule(priority=1, probability=0.2)
    r2 = _mk_rule(priority=2, probability=0.8)
    p1 = Genotype(rules=[r1, r2])
    p2 = Genotype(rules=[r2])

    cfg = {
        'crossover_probability_per_rule': 1.0,  # include all available rules
        'min_rules_per_genotype': 1,
        'max_rules_per_genotype': 2,
    }
    rng = RNGManager(seed=123)

    o1, o2 = uniform_crossover(p1, p2, cfg, rng)
    # All rules should be included but bounded by max=2
    assert 1 <= len(o1.rules) <= 2
    assert 1 <= len(o2.rules) <= 2


def test_uniform_crossover_order_independence():
    # [T-evo-01]
    rules = [_mk_rule() for _ in range(4)]
    a = Genotype(rules=[rules[0], rules[1]])
    b = Genotype(rules=[rules[2], rules[3]])
    cfg = {
        'crossover_probability_per_rule': 0.5,
        'min_rules_per_genotype': 1,
        'max_rules_per_genotype': 3,
    }
    rng1 = RNGManager(seed=999)
    rng2 = RNGManager(seed=999)
    o1a, o2a = uniform_crossover(a, b, cfg, rng1)
    o1b, o2b = uniform_crossover(b, a, cfg, rng2)
    # Order independent implies identical offspring distributions under same seed
    assert [getattr(r, 'rule_id', None) for r in o1a.rules] == [getattr(r, 'rule_id', None) for r in o1b.rules]
    assert [getattr(r, 'rule_id', None) for r in o2a.rules] == [getattr(r, 'rule_id', None) for r in o2b.rules]


def test_uniform_crossover_shared_rule_ids_different_content():
    # When both parents share a rule_id but content differs, versions are sampled
    rid = uuid.uuid4()
    r1a = _mk_rule(rule_id=rid, priority=1, probability=0.3)
    r1b = _mk_rule(rule_id=rid, priority=9, probability=0.9)
    p1 = Genotype(rules=[r1a])
    p2 = Genotype(rules=[r1b])
    cfg = {
        'crossover_probability_per_rule': 1.0,
        'min_rules_per_genotype': 1,
        'max_rules_per_genotype': 1,
    }
    rng = RNGManager(seed=555)
    o1, o2 = uniform_crossover(p1, p2, cfg, rng)
    # Offspring should each have exactly one of the versions
    assert len(o1.rules) == 1 and len(o2.rules) == 1
    assert o1.rules[0].metadata != o2.rules[0].metadata or o1.rules[0] is o2.rules[0]


def test_mutate_rate_and_types():
    # [T-evo-02]
    base_rule = _mk_rule(priority=1, probability=0.5)
    g = Genotype(rules=[base_rule])
    cfg = {
        'mutation_rate': 1.0,  # force mutation
        'mutation_probs': {
            'modify_metadata': 0.25,
            'modify_rhs': 0.25,
            'modify_lhs': 0.25,
            'add_rule': 0.15,
            'delete_rule': 0.10,
        },
        'min_rules_per_genotype': 1,
    }
    rng = RNGManager(seed=42)

    mutated = mutate(g, cfg, rng)
    assert mutated is not g
    assert mutated.genotype_id != g.genotype_id
    assert isinstance(mutated.rules, list)
    assert len(mutated.rules) >= 1


def test_mutate_noop_when_rate_zero():
    # mutation_rate 0.0 should return original object unchanged
    base_rule = _mk_rule()
    g = Genotype(rules=[base_rule])
    cfg = {
        'mutation_rate': 0.0,
        'mutation_probs': {'modify_metadata': 1.0},
        'min_rules_per_genotype': 1,
    }
    rng = RNGManager(seed=1)

    mutated = mutate(g, cfg, rng)
    assert mutated is g
    assert mutated.genotype_id == g.genotype_id


def test_mutate_delete_rule_respects_min_rules():
    # Ensure delete_rule does not go below min_rules
    r1 = _mk_rule()
    r2 = _mk_rule()
    g = Genotype(rules=[r1, r2])
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {'delete_rule': 1.0},
        'min_rules_per_genotype': 2,
    }
    rng = RNGManager(seed=5)

    mutated = mutate(g, cfg, rng)
    assert len(mutated.rules) >= 2


def test_mutate_modify_metadata_bounds_and_delta():
    # Probability clamped to [0.01, 1.0], priority changes by ±1 when present
    rule = _mk_rule(priority=0, probability=1.0)
    g = Genotype(rules=[rule])
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {'modify_metadata': 1.0},
        'min_rules_per_genotype': 1,
    }
    rng = RNGManager(seed=77)
    mutated = mutate(g, cfg, rng)
    m_rule = mutated.rules[0]
    assert 0.01 <= m_rule.metadata['probability'] <= 1.0
    assert m_rule.metadata['priority'] in {-1, 1}


def test_mutate_modify_rhs_weight_scaled():
    # Force modify_rhs; weight should change
    rid = uuid.uuid4()
    lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
    rhs = RHSAction(add_edges=[{
        'source_label': 'A', 'target_label': 'B', 'properties': {'weight': 0.2}
    }])
    emb = EmbeddingLogic()
    rule = Rule(rid, lhs, rhs, emb, metadata={})
    g = Genotype(rules=[rule])
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {'modify_rhs': 1.0},
        'min_rules_per_genotype': 1,
    }
    rng = RNGManager(seed=101)
    mutated = mutate(g, cfg, rng)
    assert mutated.rules[0].rhs.add_edges[0]['properties']['weight'] != 0.2


def test_mutate_modify_lhs_adds_optional_flag():
    # Force modify_lhs; adds optional_flag criterion
    rid = uuid.uuid4()
    lhs = LHSPattern(nodes=[{'label': 'X', 'match_criteria': {}}], edges=[], boundary_nodes=[])
    rhs = RHSAction()
    emb = EmbeddingLogic()
    rule = Rule(rid, lhs, rhs, emb, metadata={})
    g = Genotype(rules=[rule])
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {'modify_lhs': 1.0},
        'min_rules_per_genotype': 1,
    }
    rng = RNGManager(seed=202)
    mutated = mutate(g, cfg, rng)
    assert 'optional_flag' in mutated.rules[0].lhs.nodes[0]['match_criteria']


def test_mutate_deterministic_same_seed_and_genotype():
    # Same seed and same genotype_id -> same mutation outcome
    base_rule = _mk_rule(priority=1, probability=0.5)
    g1 = Genotype(rules=[base_rule])
    g2 = Genotype(genotype_id=g1.genotype_id, rules=[base_rule])
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {'modify_metadata': 1.0},
        'min_rules_per_genotype': 1,
    }
    rng1 = RNGManager(seed=333)
    rng2 = RNGManager(seed=333)

    m1 = mutate(g1, cfg, rng1)
    m2 = mutate(g2, cfg, rng2)
    assert [r.metadata for r in m1.rules] == [r.metadata for r in m2.rules]


