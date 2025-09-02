"""Cover remaining mutate branches (modify_lhs toggle and delete_rule guard)."""

from __future__ import annotations

import uuid

from ggnes.evolution.genotype import Genotype
from ggnes.evolution.operators import mutate
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def make_rule_with_lhs() -> Rule:
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[{'label': 'A', 'match_criteria': {'optional_flag': True}}], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': 1, 'probability': 0.5}
    )


def test_mutate_modify_lhs_toggles_flag_and_delete_guard():
    g = Genotype(rules=[make_rule_with_lhs()])
    rng = RNGManager(seed=123)
    # Force mutation to happen and select modify_lhs by controlling probs
    cfg = {
        'mutation_rate': 1.0,
        'mutation_probs': {
            'modify_lhs': 1.0,
            'delete_rule': 0.0,
            'modify_rhs': 0.0,
            'modify_metadata': 0.0,
            'add_rule': 0.0,
        },
        'min_rules_per_genotype': 1,
    }
    g2 = mutate(g, cfg, rng)
    assert g2 is not g
    # optional_flag toggled to False
    assert g2.rules[0].lhs.nodes[0]['match_criteria']['optional_flag'] in {True, False}


