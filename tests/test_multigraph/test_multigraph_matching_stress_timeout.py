import time

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.matching import find_subgraph_matches


def test_matching_timeout_with_many_parallel_edges():
    g = Graph(config={"multigraph": True})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1},
    })
    b = g.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1},
    })
    # Add many parallel edges to stress matching
    for _ in range(200):
        g.add_edge(a, b, {"weight": 0.1})

    lhs = {
        "nodes": [
            {"label": "A", "match_criteria": {"node_type": NodeType.INPUT}},
            {"label": "B", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {"source_label": "A", "target_label": "B", "edge_label": "E", "match_criteria": {}},
        ],
    }

    start = time.time()
    results = list(find_subgraph_matches(g, lhs, timeout_ms=5))
    elapsed_ms = (time.time() - start) * 1000.0
    # Should respect timeout and return quickly
    assert elapsed_ms <= 50.0
    # Some results may be present, but not required; ensure it didn't blow up
    assert isinstance(results, list)


