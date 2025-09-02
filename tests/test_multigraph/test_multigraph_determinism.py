from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.matching import find_subgraph_matches


def test_list_edges_deterministic_ordering():
    g = Graph(config={"multigraph": True})
    a = g.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    b = g.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    # Add several edges and ensure list_edges yields in edge_id order globally
    ids = [g.add_edge(a, b, {"weight": 0.1}) for _ in range(5)]
    # Insertion order differs from edge_id string order potentially; check determinism
    listed = [e.edge_id for e in g.list_edges()]
    assert listed == sorted(ids, key=str)


def test_matching_edge_instance_binding_deterministic():
    g = Graph(config={"multigraph": True})
    a = g.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    b = g.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    # Parallel edges
    [g.add_edge(a, b, {"weight": 0.1}), g.add_edge(a, b, {"weight": 0.2}), g.add_edge(a, b, {"weight": 0.15})]

    lhs = {
        "nodes": [
            {"label": "A", "match_criteria": {"node_type": NodeType.INPUT}},
            {"label": "B", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {"source_label": "A", "target_label": "B", "edge_label": "E", "match_criteria": {}},
        ],
    }

    m1 = list(find_subgraph_matches(g, lhs, timeout_ms=100))
    m2 = list(find_subgraph_matches(g, lhs, timeout_ms=100))
    # Every run must bind the same first-by-edge_id edge deterministically
    assert m1 and m2
    bound1 = [m["E"].edge_id for m in m1]
    bound2 = [m["E"].edge_id for m in m2]
    assert bound1 == bound2


