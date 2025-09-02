"""Exercise remaining mutate branches (add_rule, delete_rule unreachable else guard)."""

from __future__ import annotations

import uuid

from ggnes.evolution.genotype import Genotype
from ggnes.evolution.operators import mutate
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def make_simple_rule() -> Rule:
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(add_edges=[{'source_label': 'A', 'target_label': 'A', 'properties': {'weight': 0.2}}]),
        embedding=EmbeddingLogic(),
        metadata={'priority': 1, 'probability': 0.5}
    )


def test_mutate_add_rule_path():
    g = Genotype(rules=[make_simple_rule()])
    rng = RNGManager(seed=321)
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {
            'add_rule': 1.0,
            'modify_lhs': 0.0,
            'modify_rhs': 0.0,
            'modify_metadata': 0.0,
            'delete_rule': 0.0,
        },
    }
    g2 = mutate(g, cfg, rng)
    assert len(g2.rules) >= 2


def test_mutate_delete_rule_guard_no_delete_when_at_min():
    g = Genotype(rules=[make_simple_rule()])
    rng = RNGManager(seed=654)
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {
            'delete_rule': 1.0,
            'add_rule': 0.0,
            'modify_lhs': 0.0,
            'modify_rhs': 0.0,
            'modify_metadata': 0.0,
        },
        'min_rules_per_genotype': 1,
    }
    g2 = mutate(g, cfg, rng)
    assert len(g2.rules) == 1


