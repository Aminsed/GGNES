"""Additional tests for network_gen components to raise coverage to 100%."""

from __future__ import annotations

import uuid

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.generation.network_gen import GraphHistory, RuleCooldown, generate_network
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _axiom():
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    assert g.add_edge(i, o, {'weight': 0.1}) is not None
    return g


def test_rulecooldown_expiry_deletes_entries():
    rc = RuleCooldown(cooldown_iterations=1)
    rid = 'r'
    rc.add_cooldown(rid)
    # First update should drop to zero and delete
    rc.update()
    assert rc.is_cooled_down(rid)
    # Also cover clear when id not present and when present
    rc.clear_cooldown('missing')  # no-op
    rc.add_cooldown(rid)
    rc.clear_cooldown(rid)
    assert rc.is_cooled_down(rid)


def test_graphhistory_methods():
    gh = GraphHistory()
    gh.add_fingerprint('fp')
    gh.add_action('Add', 'rid', set([1]))
    gh.add_metrics({'num_nodes': 1})
    assert gh.fingerprints == ['fp'] and gh.action_history and gh.metrics


def test_repair_disable_path_and_graph_context_keys():
    # Create a rule that invalidates the graph
    bad = Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(add_nodes=[{'label': 'X', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {}}}]),
        embedding=EmbeddingLogic(),
        metadata={'rule_type': 'AddNode', 'priority': 1, 'probability': 1.0}
    )

    class Genotype:
        def __init__(self, r):
            self.rules = [r]

    cfg = {
        'max_iterations': 1,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'repair_strategy': 'DISABLE',  # force unsuccessful repair path
        'graph_context_keys': ['custom_metric'],
    }

    # Use a Graph subclass that always validates False to force repair path
    class BadGraph(Graph):
        def validate(self, collect_errors=None, collect_warnings=None):  # noqa: D401
            if collect_errors is not None:
                collect_errors.append({'x': 1})
            return False

    # Build axiom using BadGraph
    g_bad = BadGraph()
    i = g_bad.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    o = g_bad.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    assert g_bad.add_edge(i, o, {'weight': 0.1}) is not None

    g, m = generate_network(Genotype(bad), g_bad, cfg, RNGManager(seed=10))
    assert m['repair_metrics'] is None or isinstance(m['repair_metrics'], dict)


def test_condition_filters_matches_to_quiescence():
    # Rule with condition that always returns False should lead to quiescence
    class Genotype:
        def __init__(self, rule):
            self.rules = [rule]

    def always_false(graph, bindings, ctx):  # noqa: D401
        return False

    r = Rule(
        rule_id=uuid.uuid4(),
        lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
        rhs=RHSAction(),
        embedding=EmbeddingLogic(),
        metadata={'priority': 5, 'probability': 1.0},
        condition=always_false,
    )
    g, m = generate_network(Genotype(r), _axiom(), {
        'max_iterations': 3,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER'
    }, RNGManager(seed=11))
    assert m['iterations'] == 0


