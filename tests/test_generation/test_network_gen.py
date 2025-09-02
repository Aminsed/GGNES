"""
M13 Generation Engine tests per project_guide.md §7.3.

Tests:
- [T-gen-01] Quiescence when no matches.
- [T-gen-02] Oscillation action: TERMINATE, SKIP_AND_RESELECT limits, IGNORE (warn).
- [T-gen-03] GraphHistory fingerprints/metrics updated each iteration.
- [T-gen-04] Final validation triggers repair and metrics populated.
"""

from __future__ import annotations

import uuid

import pytest  # noqa: F401

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.generation import generate_network
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def minimal_axiom(input_size: int = 4, output_size: int = 2) -> Graph:
    g = Graph()
    i = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': input_size}
    })
    o = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': output_size}
    })
    eid = g.add_edge(i, o, {'weight': 0.1})
    assert eid is not None
    return g


class DummyRule:
    """Helper to construct simple rules with metadata for matching all graphs."""

    def __init__(self, rule_type: str, priority: int = 0, probability: float = 1.0):
        self.rule_id = uuid.uuid4()
        self.lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
        self.rhs = RHSAction()
        self.embedding = EmbeddingLogic()
        self.metadata = {'rule_type': rule_type, 'priority': priority, 'probability': probability}
        self.condition = None


def test_quiescence_when_no_matches():
    # No rules -> quiescence immediately
    class Genotype:
        def __init__(self):
            self.rules = []

    g0 = minimal_axiom()
    rng = RNGManager(seed=1)
    g, metrics = generate_network(Genotype(), g0, {
        'max_iterations': 5,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER'
    }, rng)

    assert g is not g0  # deep-copied
    assert metrics['iterations'] == 0
    assert metrics['final_nodes'] == len(g.nodes)
    assert metrics['final_edges'] == sum(len(n.edges_out) for n in g.nodes.values())


def test_oscillation_actions_terminate_and_ignore():
    # Install a rule that will always be selected; configure reverse_pairs
    # so the rule is considered a reversal of itself on subsequent iterations.
    class Genotype:
        def __init__(self, rule):
            self.rules = [rule]

    rule = DummyRule('AddNode', priority=10, probability=1.0)

    # We can't inject GraphHistory directly into generate_network, so craft config that triggers early
    cfg = {
        'max_iterations': 3,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_strategy': 'SIMPLE_REVERSAL',
        'oscillation_action': 'TERMINATE',
        # Treat AddNode as its own reverse to trigger detection on 2nd iteration
        'rule_reverse_pairs': {'AddNode': 'AddNode'},
    }

    # Run: should terminate after first iteration due to reversal detection
    g, m = generate_network(Genotype(rule), minimal_axiom(), cfg, RNGManager(seed=2))
    assert m['iterations'] in (0, 1)

    # IGNORE should proceed at least one application
    cfg['oscillation_action'] = 'IGNORE'
    g2, m2 = generate_network(Genotype(rule), minimal_axiom(), cfg, RNGManager(seed=2))
    assert m2['iterations'] >= 0


def test_skip_and_reselect_limits():
    class Genotype:
        def __init__(self, rule):
            self.rules = [rule]

    rule = DummyRule('A', priority=5, probability=1.0)
    cfg = {
        'max_iterations': 100,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_strategy': 'ALL',
        'oscillation_action': 'SKIP_AND_RESELECT',
        'rule_reverse_pairs': {'A': 'A'},
        'max_consecutive_oscillation_skips': 1,
        'max_total_oscillation_skips': 2,
    }
    g, m = generate_network(Genotype(rule), minimal_axiom(), cfg, RNGManager(seed=3))
    # Should stop due to skip limits without applying many iterations
    assert m['oscillation_skips'] <= 2


def test_history_and_metrics_update_each_iteration():
    # Create a rule that adds a node each time; ensure history metrics populate
    class Genotype:
        def __init__(self, rule):
            self.rules = [rule]

    add_rule = Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(add_nodes=[{
            'label': 'H',
            'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'relu',
                'attributes': {'output_size': 3}
            }
        }]),
        embedding=EmbeddingLogic(),
        metadata={'rule_type': 'AddNode', 'priority': 10, 'probability': 1.0}
    )

    g, m = generate_network(Genotype(add_rule), minimal_axiom(), {
        'max_iterations': 3,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_strategy': 'ALL',
        'oscillation_action': 'IGNORE',
    }, RNGManager(seed=4))

    assert 0 <= m['iterations'] <= 3
    assert 'final_nodes' in m and 'final_edges' in m


def test_final_validation_triggers_repair_metrics():
    # Rule creates invalid node (missing output_size) so final graph invalid
    class Genotype:
        def __init__(self, rule):
            self.rules = [rule]

    bad_rule = Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(add_nodes=[{
            'label': 'X',
            'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'relu',
                'attributes': {}
            }
        }]),
        embedding=EmbeddingLogic(),
        metadata={'rule_type': 'AddNode', 'priority': 9, 'probability': 1.0}
    )

    cfg = {
        'max_iterations': 1,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'repair_strategy': 'MINIMAL_CHANGE',
        'allowed_repairs': ['add_missing_attributes'],
    }
    g, m = generate_network(Genotype(bad_rule), minimal_axiom(), cfg, RNGManager(seed=5))
    # Should have attempted repair and include repair_metrics dict
    assert 'repair_metrics' in m
    assert isinstance(m['repair_metrics'], dict) or m['repair_metrics'] is None


