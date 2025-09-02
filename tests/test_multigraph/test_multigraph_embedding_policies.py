import uuid

import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _graph_with_boundary_in_and_out():
    g = Graph(config={"multigraph": True})
    s = g.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 2}})
    x = g.add_node({"node_type": NodeType.HIDDEN, "activation_function": "linear", "attributes": {"output_size": 2}})
    t = g.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 2}})
    # inbound
    g.add_edge(s, x, {"weight": 0.1})
    g.add_edge(s, x, {"weight": 0.2})
    # outbound
    g.add_edge(x, t, {"weight": 0.3})
    return g, s, x, t


def test_embedding_unknown_direction_error_for_in():
    g, s, x, t = _graph_with_boundary_in_and_out()
    rng = RNGManager(seed=7)
    engine = RuleEngine(g, rng)

    lhs = LHSPattern(nodes=[{"label": "X"}], edges=[], boundary_nodes=["X"])
    rhs = RHSAction(
        delete_nodes=["X"],
        add_nodes=[{
            "label": "P",
            "properties": {
                "node_type": NodeType.HIDDEN,
                "activation_function": "linear",
                "attributes": {"output_size": 2}
            }
        }]
    )
    emb = EmbeddingLogic(connection_map={("X", 'OUT'): [("P", 'COPY_ALL')]}, unknown_direction_handling='ERROR')
    rule = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={})

    with pytest.raises(ValueError):
        engine.apply_rule(rule, {"X": x})


def test_embedding_excess_connections_error_on_connect_single():
    g, s, x, t = _graph_with_boundary_in_and_out()
    rng = RNGManager(seed=8)
    engine = RuleEngine(g, rng)

    lhs = LHSPattern(nodes=[{"label": "X"}], edges=[], boundary_nodes=["X"])
    rhs = RHSAction(
        delete_nodes=["X"],
        add_nodes=[{
            "label": "P",
            "properties": {
                "node_type": NodeType.HIDDEN,
                "activation_function": "linear",
                "attributes": {"output_size": 2}
            }
        }]
    )
    emb = EmbeddingLogic(connection_map={("X", 'IN'): [("P", 'CONNECT_SINGLE')]}, excess_connection_handling='ERROR')
    rule = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={})

    with pytest.raises(ValueError):
        engine.apply_rule(rule, {"X": x})


def test_embedding_boundary_handling_ignore_skips_reconnections_and_deletes():
    g, s, x, t = _graph_with_boundary_in_and_out()
    rng = RNGManager(seed=9)
    engine = RuleEngine(g, rng)

    lhs = LHSPattern(nodes=[{"label": "X"}], edges=[], boundary_nodes=["X"])
    rhs = RHSAction(
        delete_nodes=["X"],
        add_nodes=[{
            "label": "P",
            "properties": {
                "node_type": NodeType.HIDDEN,
                "activation_function": "linear",
                "attributes": {"output_size": 2}
            }
        }]
    )
    emb = EmbeddingLogic(connection_map={}, boundary_handling='IGNORE')
    rule = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={})

    assert engine.apply_rule(rule, {"X": x}) is True
    # X deleted, P added, but no reconnections
    assert x not in g.nodes
    p_candidates = [nid for nid in g.nodes if nid not in (s, t)]
    assert len(p_candidates) == 1
    p = p_candidates[0]
    assert g.find_edges_by_endpoints(s, p) == []
    assert g.find_edges_by_endpoints(p, t) == []


