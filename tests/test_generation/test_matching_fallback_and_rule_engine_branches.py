import re

import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import Direction, Distribution, EmbeddingLogic, RHSAction


def _add_node(g: Graph, node_type: NodeType, act: str = 'relu', size: int = 4):
    return g.add_node({'node_type': node_type, 'activation_function': act, 'attributes': {'output_size': size}})


def test_rule_engine_unknown_in_error():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    z = _add_node(g, NodeType.HIDDEN)
    # external incoming to X
    g.add_edge(z, x, {'weight': 0.1})

    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={}, unknown_direction_handling='ERROR')
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'rin'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    with pytest.raises(ValueError):
        engine.apply_rule(rule, {'X': x})


def test_rule_engine_embedding_out_numeric_maps_to_first_external_target():
    g = Graph()
    # Boundary node X with outgoing edge to Y
    x = _add_node(g, NodeType.HIDDEN)
    y = _add_node(g, NodeType.HIDDEN)
    g.add_edge(x, y, {'weight': 0.2})

    # LHS: boundary 'X'
    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    # RHS: add node 'T'
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={("X", Direction.OUT): [("T", 0.5)]})
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r1'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is True
    # New node 'T' added; ensure it connects to Y
    # Find newly added node id (highest id)
    t_id = max(g.nodes.keys())
    assert g.find_edge_by_endpoints(t_id, y) is not None


def test_rule_engine_embedding_out_excess_error_on_connect_single_with_two_targets():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    y1 = _add_node(g, NodeType.HIDDEN)
    y2 = _add_node(g, NodeType.HIDDEN)
    g.add_edge(x, y1, {'weight': 0.2})
    g.add_edge(x, y2, {'weight': 0.3})

    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={("X", Direction.OUT): [("T", Distribution.CONNECT_SINGLE)]}, excess_connection_handling='ERROR')
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r2'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    with pytest.raises(ValueError, match=re.escape('Excess outgoing connections for boundary X')):
        engine.apply_rule(rule, {'X': x})


def test_rule_engine_embedding_out_invalid_distribution_is_ignored():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    y = _add_node(g, NodeType.HIDDEN)
    g.add_edge(x, y, {'weight': 0.2})

    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    # invalid distribution object triggers exception and continue
    emb = EmbeddingLogic(connection_map={("X", Direction.OUT): [("T", object())]})
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r3'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is True
    # Ensure we did not create an edge from T to Y via invalid dist
    t_id = max(g.nodes.keys())
    assert g.find_edge_by_endpoints(t_id, y) is None


def test_rule_engine_boundary_ignore_short_circuit():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={}, boundary_handling='IGNORE')
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r4'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is True


def test_rule_engine_unknown_in_warning_succeeds():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    z1 = _add_node(g, NodeType.HIDDEN)
    # external incoming to X
    g.add_edge(z1, x, {'weight': 0.1})

    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={}, unknown_direction_handling='WARNING')
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'rin_warn'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is True


def test_rule_engine_commit_exception_rolls_back(monkeypatch):
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={})
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r5'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)

    def _boom():
        raise RuntimeError('boom')

    monkeypatch.setattr(engine.tm, 'commit', _boom)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is False


def test_rule_engine_gates_condition_and_cooldown():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)

    # Condition gate blocks
    lhs = type('LHS', (), {'boundary_nodes': [], 'edges': []})()
    rhs = RHSAction()
    emb = EmbeddingLogic(connection_map={})
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.rule_id = 'r6'

    def cond(graph, bindings, ctx):
        return False

    rule.condition = cond
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    assert engine.apply_rule(rule, {'X': x}) is False


def test_rule_engine_unknown_out_error():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    y = _add_node(g, NodeType.HIDDEN)
    # external outgoing from X
    g.add_edge(x, y, {'weight': 0.1})

    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={}, unknown_direction_handling='ERROR')
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r_out_err'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    with pytest.raises(ValueError):
        engine.apply_rule(rule, {'X': x})


def test_rule_engine_embedding_in_numeric_no_external_connects_boundary_to_target():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={("X", Direction.IN): [("T", 0.7)]})
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r_in_num'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is True
    t_id = max(g.nodes.keys())
    # Numeric IN path with no external_in should connect X -> T
    assert g.find_edge_by_endpoints(x, t_id) is not None


def test_rule_engine_delete_edge_fallback_via_edge_label():
    g = Graph()
    a = _add_node(g, NodeType.HIDDEN)
    b = _add_node(g, NodeType.HIDDEN)
    g.add_edge(a, b, {'weight': 0.5})

    # LHS includes edge label 'E' between A->B
    lhs = type('LHS', (), {'boundary_nodes': [], 'edges': [
        {'source_label': 'A', 'target_label': 'B', 'edge_label': 'E'}
    ]})()
    rhs = RHSAction(delete_edges=[{'edge_label': 'E'}])
    emb = EmbeddingLogic(connection_map={})
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r_del_edge'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'A': a, 'B': b})
    assert ok is True
    # Edge A->B should be deleted via fallback resolution
    assert g.find_edge_by_endpoints(a, b) is None


def test_rule_engine_embedding_out_numeric_no_external_connects_to_boundary():
    g = Graph()
    x = _add_node(g, NodeType.HIDDEN)
    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    # No external_out from X; numeric OUT should connect T -> X
    emb = EmbeddingLogic(connection_map={("X", Direction.OUT): [("T", 0.4)]})
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r_out_num_noext'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is True
    t_id = max(g.nodes.keys())
    assert g.find_edge_by_endpoints(t_id, x) is not None


def test_rule_engine_excess_in_warning_allows_success():
    g = Graph()
    # Setup two incoming edges to X to trigger excess under CONNECT_SINGLE
    x = _add_node(g, NodeType.HIDDEN)
    z1 = _add_node(g, NodeType.HIDDEN)
    z2 = _add_node(g, NodeType.HIDDEN)
    g.add_edge(z1, x, {'weight': 0.1})
    g.add_edge(z2, x, {'weight': 0.2})

    lhs = type('LHS', (), {'boundary_nodes': ['X'], 'edges': []})()
    rhs = RHSAction(add_nodes=[{'label': 'T', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    emb = EmbeddingLogic(connection_map={("X", Direction.IN): [("T", Distribution.CONNECT_SINGLE)]}, excess_connection_handling='WARNING')
    rule = type('RuleObj', (), {})()
    rule.lhs = lhs
    rule.rhs = rhs
    rule.embedding = emb
    rule.condition = None
    rule.rule_id = 'r_excess_in_warn'

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok is True

    # Cooldown gate blocks
    engine._cooldowns[rule.rule_id] = 1
    rule.condition = None
    assert engine.apply_rule(rule, {'X': x}) is False


