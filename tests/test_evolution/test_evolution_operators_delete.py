"""Cover mutate delete_rule path where deletion actually occurs."""

from __future__ import annotations

import uuid

from ggnes.evolution.genotype import Genotype
from ggnes.evolution.operators import mutate
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _r():
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': 0, 'probability': 1.0}
    )


def test_mutate_delete_rule_executes_when_above_min():
    g = Genotype(rules=[_r(), _r()])
    rng = RNGManager(seed=999)
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


