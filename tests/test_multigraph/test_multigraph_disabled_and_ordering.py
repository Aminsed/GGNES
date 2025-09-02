from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def test_disabled_parallel_edges_excluded_from_algorithms():
    g = Graph(config={"multigraph": True, "wl_iterations": 2})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })
    b = g.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })

    g.add_edge(a, b, {"weight": 0.1, "enabled": True})
    e2 = g.add_edge(a, b, {"weight": 0.2, "enabled": True})

    fp_enabled = g.compute_fingerprint()
    order_enabled = g.topological_sort()

    # Disable one edge, algorithms should reflect reduced multiplicity
    edge2 = g.find_edge_by_id(e2)
    edge2.enabled = False

    fp_disabled = g.compute_fingerprint()
    order_disabled = g.topological_sort()

    assert fp_disabled != fp_enabled
    # Topological sort should remain valid ordering of nodes
    assert set(order_disabled) == set(order_enabled) == set(g.nodes.keys())

    # list_edges still yields disabled edges, but algorithms exclude them
    all_edges = list(g.list_edges())
    assert any(e.edge_id == e2 and e.enabled is False for e in all_edges)


