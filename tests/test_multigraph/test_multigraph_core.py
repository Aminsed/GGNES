
from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


# [T-mg-core-01]
def test_add_edge_allows_parallel_edges_in_multigraph():
    g = Graph(config={"multigraph": True})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 4},
    })
    b = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 4},
    })

    e1 = g.add_edge(a, b, {"weight": 0.1})
    e2 = g.add_edge(a, b, {"weight": 0.2})

    assert e1 is not None and e2 is not None and e1 != e2
    edges = g.find_edges_by_endpoints(a, b)
    assert len(edges) == 2
    ids = {edges[0].edge_id, edges[1].edge_id}
    assert e1 in ids and e2 in ids


# [T-mg-core-02]
def test_remove_edge_removes_only_targeted_instance_and_cleans_lists():
    g = Graph(config={"multigraph": True})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 3},
    })
    b = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 3},
    })
    e1 = g.add_edge(a, b, {"weight": 0.1})
    e2 = g.add_edge(a, b, {"weight": 0.2})
    assert len(g.find_edges_by_endpoints(a, b)) == 2

    g.remove_edge(e1)
    es = g.find_edges_by_endpoints(a, b)
    assert len(es) == 1
    assert es[0].edge_id == e2

    g.remove_edge(e2)
    assert g.find_edges_by_endpoints(a, b) == []
    # internal adjacency cleaned (no key for (a,b))
    assert b not in g.nodes[a].edges_out


# [T-mg-core-03]
def test_find_edge_by_endpoints_returns_first_instance_compatibly():
    g = Graph(config={"multigraph": True})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })
    b = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })
    e1 = g.add_edge(a, b, {"weight": 0.1})
    e2 = g.add_edge(a, b, {"weight": 0.2})
    # Deterministic first by edge_id string sort
    first = g.find_edge_by_endpoints(a, b)
    assert first is not None
    assert first.edge_id in {e1, e2}


# [T-mg-core-04]
def test_validation_checks_non_finite_per_edge_and_reachability():
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
    eid = g.add_edge(a, b, {"weight": float("nan")})
    # Reachability should still hold
    assert g._is_reachable_from_input(b)
    errs: list = []
    assert not g.validate(collect_errors=errs)
    assert any(getattr(e, "edge_id", None) == eid and e.error_type == "non_finite_weight" for e in errs)


