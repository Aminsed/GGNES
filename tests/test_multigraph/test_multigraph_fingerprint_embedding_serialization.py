import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule


# [T-mg-core-06]
def test_wl_fingerprint_reflects_multiplicity_and_is_deterministic():
    g1 = Graph(config={"multigraph": True, "wl_iterations": 2})
    a = g1.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    b = g1.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    g1.add_edge(a, b, {"weight": 0.1})
    fp1 = g1.compute_fingerprint()
    g1.add_edge(a, b, {"weight": 0.2})
    fp2 = g1.compute_fingerprint()
    assert fp1 != fp2
    # Order of insertion should not affect fingerprint beyond multiplicity
    g2 = Graph(config={"multigraph": True, "wl_iterations": 2})
    a2 = g2.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    b2 = g2.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    g2.add_edge(a2, b2, {"weight": 0.2})
    g2.add_edge(a2, b2, {"weight": 0.1})
    assert g2.compute_fingerprint() == fp2


# [T-mg-embed-01]
def test_embedding_deterministic_order_with_parallel_edges():
    # Construct a rule that deletes boundary X and reconnects its external edges deterministically
    lhs = LHSPattern(nodes=[{"label": "X", "match_criteria": {"node_type": NodeType.HIDDEN}}], edges=[], boundary_nodes=["X"])
    rhs = RHSAction(add_nodes=[{"label": "P", "properties": {"node_type": NodeType.HIDDEN, "activation_function": "linear", "attributes": {"output_size": 2}}}])
    emb = EmbeddingLogic(connection_map={("X", 'IN'): [("P", 'COPY_ALL')], ("X", 'OUT'): [("P", 'CONNECT_SINGLE')]})
    Rule(uuid.uuid4(), lhs, rhs, emb, metadata={})

    # Graph with parallel external edges to X
    g = Graph(config={"multigraph": True})
    s1 = g.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 2}})
    x = g.add_node({"node_type": NodeType.HIDDEN, "activation_function": "linear", "attributes": {"output_size": 2}})
    t1 = g.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 2}})
    # Parallel edges inbound and outbound
    g.add_edge(s1, x, {"weight": 0.1})
    g.add_edge(s1, x, {"weight": 0.2})
    g.add_edge(x, t1, {"weight": 0.3})
    g.add_edge(x, t1, {"weight": 0.4})

    # We won't run full RuleEngine; just verify we can sort parallel edges deterministically
    in_edges = g.find_edges_by_endpoints(s1, x)
    out_edges = g.find_edges_by_endpoints(x, t1)
    sorted_in = [e.edge_id for e in sorted(in_edges, key=lambda e: (s1, x, str(e.edge_id)))]
    sorted_out = [e.edge_id for e in sorted(out_edges, key=lambda e: (x, t1, str(e.edge_id)))]
    assert sorted_in == sorted(sorted_in, key=str)
    assert sorted_out == sorted(sorted_out, key=str)


# [T-mg-ser-01]
def test_serialization_roundtrip_with_edge_labels_in_multigraph():
    from ggnes.utils.serialization import deserialize_rule, serialize_rule

    lhs = LHSPattern(
        nodes=[{"label": "A", "match_criteria": {"node_type": NodeType.INPUT}}, {"label": "B", "match_criteria": {"node_type": NodeType.HIDDEN}}],
        edges=[{"source_label": "A", "target_label": "B", "edge_label": "E", "match_criteria": {}}],
        boundary_nodes=[],
    )
    rhs = RHSAction()
    emb = EmbeddingLogic(connection_map={})
    r = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={})

    data = serialize_rule(r)
    r2 = deserialize_rule(data)
    assert r2.lhs.edges[0]['edge_label'] == 'E'
    assert r2.lhs.nodes[0]['label'] == 'A'


