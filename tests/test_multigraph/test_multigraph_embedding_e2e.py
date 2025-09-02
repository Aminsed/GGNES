import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def test_embedding_e2e_copy_all_and_connect_single_with_parallel_edges():
    # Build a graph where node X has parallel inbound and outbound edges
    g = Graph(config={"multigraph": True})
    s = g.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 2}})
    x = g.add_node({"node_type": NodeType.HIDDEN, "activation_function": "linear", "attributes": {"output_size": 2}})
    t = g.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 2}})

    # Two inbound s->x and two outbound x->t
    g.add_edge(s, x, {"weight": 0.1})
    g.add_edge(s, x, {"weight": 0.2})
    g.add_edge(x, t, {"weight": 0.3})
    g.add_edge(x, t, {"weight": 0.4})

    # Rule: delete X, add P, reconnect:
    # - IN: COPY_ALL (both s->x edges become s->P)
    # - OUT: CONNECT_SINGLE (only first x->t edge becomes P->t)
    lhs = LHSPattern(nodes=[{"label": "X", "match_criteria": {"node_type": NodeType.HIDDEN}}], edges=[], boundary_nodes=["X"])
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
    emb = EmbeddingLogic(connection_map={("X", 'IN'): [("P", 'COPY_ALL')], ("X", 'OUT'): [("P", 'CONNECT_SINGLE')]})
    rule = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={})

    rng = RNGManager(seed=1)
    engine = RuleEngine(g, rng)
    # Bindings: label X -> node id x
    success = engine.apply_rule(rule, {"X": x})
    assert success

    # After application, there should be a new node P and:
    # - two s->P edges (copied inbound)
    # - exactly one P->t edge (connect single)
    p_candidates = [nid for nid, n in g.nodes.items() if nid != s and nid != t and n.node_type == NodeType.HIDDEN]
    assert len(p_candidates) == 1
    p = p_candidates[0]

    in_edges = g.find_edges_by_endpoints(s, p)
    out_edges = g.find_edges_by_endpoints(p, t)

    assert len(in_edges) == 2
    assert len(out_edges) == 1


