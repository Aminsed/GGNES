import uuid

from ggnes.evolution.genotype import Genotype
from ggnes.evolution.operators import mutate, uniform_crossover
from ggnes.utils.rng_manager import RNGManager


class DummyRule:
    def __init__(self, rid=None):
        self.rule_id = rid or uuid.uuid4()
        self.metadata = {'priority': 0, 'probability': 0.5}
        self.rhs = type('RHS', (), {'add_edges': [{'properties': {'weight': 0.1}}]})()
        self.lhs = type('LHS', (), {'nodes': [{'label': 'A', 'match_criteria': {}}]})()


def test_mutate_noop_when_rate_zero():
    r = DummyRule()
    g = Genotype(rules=[r])
    cfg = {'mutation_rate': 0.0}
    rng = RNGManager(seed=123)
    out = mutate(g, cfg, rng)
    assert out is g


def test_uniform_crossover_fill_to_min_and_sort_pool():
    # Use different rule objects without rule_id attr to exercise fallback repr sort branch
    class RuleNoId:
        def __init__(self):
            self.metadata = {}

    p1 = Genotype(rules=[RuleNoId(), RuleNoId()])
    p2 = Genotype(rules=[RuleNoId()])
    cfg = {'crossover_probability_per_rule': 0.0, 'min_rules_per_genotype': 2, 'max_rules_per_genotype': 3}
    rng = RNGManager(seed=42)
    o1, o2 = uniform_crossover(p1, p2, cfg, rng)
    assert len(o1.rules) == 2
    assert len(o2.rules) == 2

