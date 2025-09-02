"""Cover remaining branches in generate_network (no selected; warning paths)."""

from __future__ import annotations

import uuid

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.generation import network_gen
from ggnes.generation.network_gen import generate_network
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def axiom():
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    assert g.add_edge(i, o, {'weight': 0.1}) is not None
    return g


def test_selection_returns_none_breaks_loop():
    # Monkeypatch select_match to return None to hit the break path
    orig = network_gen.select_match
    network_gen.select_match = lambda *args, **kwargs: None
    try:
        r = Rule(uuid.uuid4(), LHSPattern([], [], []), RHSAction(), EmbeddingLogic(), {'priority': 1, 'probability': 1.0})

        class Genotype:
            def __init__(self, rules):
                self.rules = rules

        g, m = generate_network(Genotype([r]), axiom(), {
            'max_iterations': 3,
            'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER'
        }, RNGManager(seed=99))
        assert m['iterations'] == 0
    finally:
        network_gen.select_match = orig


def test_skip_and_reselect_breaks_on_consecutive_limit():
    # Configure oscillation always and immediate break via consecutive limit
    r = Rule(uuid.uuid4(), LHSPattern([], [], []), RHSAction(), EmbeddingLogic(), {'rule_type': 'A', 'priority': 1, 'probability': 1.0})

    class Genotype:
        def __init__(self, rules):
            self.rules = rules

    g, m = generate_network(Genotype([r]), axiom(), {
        'max_iterations': 50,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_strategy': 'SIMPLE_REVERSAL',
        'oscillation_action': 'SKIP_AND_RESELECT',
        'rule_reverse_pairs': {'A': 'A'},
        'cooldown_iterations': 0,
        'max_consecutive_oscillation_skips': 0,
        'max_total_oscillation_skips': 999,
    }, RNGManager(seed=7))
    assert m['oscillation_skips'] >= 1


def test_skip_and_reselect_breaks_on_total_limit():
    r = Rule(uuid.uuid4(), LHSPattern([], [], []), RHSAction(), EmbeddingLogic(), {'rule_type': 'A', 'priority': 1, 'probability': 1.0})

    class Genotype:
        def __init__(self, rules):
            self.rules = rules

    g, m = generate_network(Genotype([r]), axiom(), {
        'max_iterations': 50,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_strategy': 'SIMPLE_REVERSAL',
        'oscillation_action': 'SKIP_AND_RESELECT',
        'rule_reverse_pairs': {'A': 'A'},
        'cooldown_iterations': 0,
        'max_consecutive_oscillation_skips': 999,
        'max_total_oscillation_skips': 1,
    }, RNGManager(seed=8))
    assert m['oscillation_skips'] >= 2


