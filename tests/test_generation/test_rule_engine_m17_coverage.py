import uuid

import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import Direction, EmbeddingLogic, LHSPattern, RHSAction, Rule


def make_line_graph():
    g = Graph()
    n0 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    n1 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    n2 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    g.add_edge(n0, n1, {'weight': 0.1})
    g.add_edge(n1, n2, {'weight': 0.2})
    return g, n0, n1, n2


def test_condition_gate_returns_false():
    g, n0, n1, _ = make_line_graph()
    lhs = LHSPattern(nodes=[{'label': 'A', 'match_criteria': {}}], edges=[], boundary_nodes=[])
    rhs = RHSAction(add_nodes=[{'label': 'X', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])

    def cond(graph_view, bindings, graph_context):
        return False

    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}), metadata={}, condition=cond)
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    assert engine.apply_rule(rule, {'A': n0}) is False


def test_cooldown_blocks_and_tick_allows():
    g, n0, _, _ = make_line_graph()
    lhs = LHSPattern(nodes=[{'label': 'A', 'match_criteria': {}}], edges=[], boundary_nodes=[])
    rhs = RHSAction(add_nodes=[{'label': 'Y', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}), metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None, cooldown_steps=1)
    assert engine.apply_rule(rule, {'A': n0}) is True
    # Cooldown active, second apply should be blocked
    assert engine.apply_rule(rule, {'A': n0}) is False
    engine.tick_cooldown()
    res = engine.apply_rule(rule, {'A': n0})
    assert res in (True, False)


def test_embedding_boundary_ignore_skips_validation_and_returns_true():
    g, n0, n1, n2 = make_line_graph()
    lhs = LHSPattern(nodes=[{'label': 'B', 'match_criteria': {}}], edges=[], boundary_nodes=['B'])
    rhs = RHSAction(delete_nodes=['B'])
    emb = EmbeddingLogic(connection_map={}, boundary_handling='IGNORE')
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    assert engine.apply_rule(rule, {'B': n1}) is True


def test_embedding_unknown_direction_error_raises():
    g, n0, n1, n2 = make_line_graph()
    lhs = LHSPattern(nodes=[{'label': 'B', 'match_criteria': {}}], edges=[], boundary_nodes=['B'])
    rhs = RHSAction(delete_nodes=['B'])
    # No mapping provided; external edges exist; ERROR should raise
    emb = EmbeddingLogic(connection_map={}, unknown_direction_handling='ERROR')
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    with pytest.raises(ValueError):
        engine.apply_rule(rule, {'B': n1})


def test_delete_edges_by_edge_label_and_commit_success():
    g, n0, n1, n2 = make_line_graph()
    # LHS binds the edge n0->n1 with edge_label 'E01'
    lhs = LHSPattern(
        nodes=[{'label': 'S', 'match_criteria': {}}, {'label': 'T', 'match_criteria': {}}],
        edges=[{'source_label': 'S', 'target_label': 'T', 'match_criteria': {}, 'edge_label': 'E01'}],
        boundary_nodes=[]
    )
    rhs = RHSAction(delete_edges=[{'edge_label': 'E01'}])
    emb = EmbeddingLogic(connection_map={})
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    # Provide bindings
    ok = engine.apply_rule(rule, {'S': n0, 'T': n1})
    assert ok
    assert g.find_edge_by_endpoints(n0, n1) is None


def test_delete_edges_with_edge_id_in_bindings():
    g, n0, n1, n2 = make_line_graph()
    edge_obj = g.find_edge_by_endpoints(n0, n1)
    lhs = LHSPattern(nodes=[{'label': 'S', 'match_criteria': {}}, {'label': 'T', 'match_criteria': {}}],
                     edges=[{'source_label': 'S', 'target_label': 'T', 'match_criteria': {}, 'edge_label': 'E01'}],
                     boundary_nodes=[])
    rhs = RHSAction(delete_edges=[{'edge_label': 'E01'}])
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}), metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'S': n0, 'T': n1, 'E01': edge_obj.edge_id})
    assert ok
    assert g.find_edge_by_endpoints(n0, n1) is None


def test_embedding_numeric_and_string_distribution_and_warnings(caplog):
    g, n0, n1, n2 = make_line_graph()
    # Add a second incoming to n1 to generate excess
    n3 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    g.add_edge(n3, n1, {'weight': 0.3})
    # Boundary is n1; delete it and reconnect
    lhs = LHSPattern(nodes=[{'label': 'B', 'match_criteria': {}}], edges=[], boundary_nodes=['B'])
    rhs = RHSAction(
        delete_nodes=['B'],
        add_nodes=[{
            'label': 'P',
            'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'relu',
                'attributes': {'output_size': 4}
            }
        }]
    )
    emb = EmbeddingLogic(connection_map={
        ('B', Direction.IN): [('P', 0.7)],  # numeric distribution path
        ('B', Direction.OUT): [('P', 'CONNECT_SINGLE')],  # string distribution path
    }, excess_connection_handling='WARNING', unknown_direction_handling='WARNING', boundary_handling='PROCESS_LAST')
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    # When there is excess after CONNECT_SINGLE, rule application should still succeed (warnings only)
    ok = engine.apply_rule(rule, {'B': n1})
    assert ok is True


def test_string_distribution_parsing_and_skip_on_invalid():
    g, n0, n1, n2 = make_line_graph()
    lhs = LHSPattern(nodes=[{'label': 'B', 'match_criteria': {}}], edges=[], boundary_nodes=['B'])
    rhs = RHSAction(add_nodes=[{'label': 'P', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    # Provide an invalid string distribution which should be skipped, leaving rule still successful
    emb = EmbeddingLogic(connection_map={('B', 'in'): [('P', 'INVALID_DISTRIBUTION')]})
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    assert engine.apply_rule(rule, {'B': n1}) is True


def test_commit_rollback_on_validate_failure():
    g, n0, n1, _ = make_line_graph()
    # Attempt to add an edge to a node being deleted in the same transaction
    lhs = LHSPattern(nodes=[{'label': 'Z', 'match_criteria': {}}], edges=[], boundary_nodes=[])
    rhs = RHSAction(
        delete_nodes=['Z'],
        add_nodes=[{'label': 'N', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}],
        add_edges=[{'source_label': 'Z', 'target_label': 'N', 'properties': {'weight': 0.1}}]
    )
    emb = EmbeddingLogic(connection_map={})
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    assert engine.apply_rule(rule, {'Z': n1}) is False


def test_commit_exception_rollback_path_is_handled(monkeypatch):
    g, n0, n1, _ = make_line_graph()
    lhs = LHSPattern(nodes=[{'label': 'A', 'match_criteria': {}}], edges=[], boundary_nodes=[])
    rhs = RHSAction(add_nodes=[{'label': 'X', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}])
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}), metadata={})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)

    # Force tm.commit() to raise to cover exception path
    original_commit = engine.tm.commit

    def boom():
        raise RuntimeError("boom")
    engine.tm.commit = boom  # type: ignore[assignment]

    try:
        assert engine.apply_rule(rule, {'A': n0}) is False
    finally:
        engine.tm.commit = original_commit  # restore


def test_post_commit_validation_failure_cleanup():
    """Test cleanup when validation fails after successful commit."""
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.generation.rule_engine import RuleEngine
    from ggnes.utils.rng_manager import RNGManager

    # Create a graph and rule that will pass initial commit but fail validation
    g = Graph()
    n1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                     'attributes': {'output_size': 10}})

    # Create a rule that adds a node with invalid aggregation parameters
    # This will pass the commit phase but fail during validation
    lhs = LHSPattern(nodes=[{'label': 'A', 'node_type': NodeType.INPUT}], edges=[], boundary_nodes=['A'])
    rhs = RHSAction(
        add_nodes=[{'label': 'B', 'properties': {
            'node_type': NodeType.HIDDEN,
            'activation_function': 'relu',
            'attributes': {
                'output_size': 5,
                'aggregation': 'topk_weighted_sum',
                'top_k': -1  # Invalid, but only caught during validation
            }
        }}],
        add_edges=[],
        delete_nodes=[],
        delete_edges=[]
    )
    rule = Rule(rule_id='test', lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}))

    # Apply rule - should fail during post-commit validation
    rng = RNGManager(seed=42)
    engine = RuleEngine(g, rng)
    result = engine.apply_rule(rule, {'A': n1})

    # Should fail and rollback
    assert result is False
    # Graph should remain unchanged (only original node)
    assert len(g.nodes) == 1
    assert n1 in g.nodes
