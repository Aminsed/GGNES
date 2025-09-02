"""Additional operator tests to exercise uncovered branches."""

from __future__ import annotations

import uuid

from ggnes.evolution.genotype import Genotype
from ggnes.evolution.operators import mutate, uniform_crossover
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def make_rule() -> Rule:
    return Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[{'label': 'A', 'match_criteria': {}}], edges=[], boundary_nodes=[]),
        rhs=RHSAction(add_edges=[{'source_label': 'A', 'target_label': 'A', 'properties': {'weight': 0.2}}]),
        embedding=EmbeddingLogic(),
        metadata={'priority': 1, 'probability': 0.5}
    )


def test_mutate_noop_when_rate_zero():
    g = Genotype(rules=[make_rule()])
    rng = RNGManager(seed=1)
    cfg = {'mutation_rate': 0.0}
    g2 = mutate(g, cfg, rng)
    assert g2 is g


def test_uniform_crossover_padding_and_limits():
    # Ensure fill_to_min path executes and max_rules truncates
    r = make_rule()
    p1 = Genotype(rules=[r])
    p2 = Genotype(rules=[])
    rng = RNGManager(seed=2)
    cfg = {
        'crossover_probability_per_rule': 0.0,  # force no initial picks
        'min_rules_per_genotype': 1,
        'max_rules_per_genotype': 1,
    }
    o1, o2 = uniform_crossover(p1, p2, cfg, rng)
    assert len(o1.rules) == 1 and len(o2.rules) == 1


