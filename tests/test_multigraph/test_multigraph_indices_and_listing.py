from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


# [T-mg-core-05]
def test_global_edge_index_and_list_edges_api_exist_and_work():
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
    c = g.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })

    e1 = g.add_edge(a, b, {"weight": 0.1})
    e2 = g.add_edge(a, b, {"weight": 0.2})
    e3 = g.add_edge(b, c, {"weight": 0.3})

    # Global index should resolve them quickly and deterministically
    e1_obj = g.find_edge_by_id(e1)
    e2_obj = g.find_edge_by_id(e2)
    e3_obj = g.find_edge_by_id(e3)
    assert e1_obj and e2_obj and e3_obj

    # list_edges should filter by endpoints or return all
    all_edges = list(g.list_edges())
    assert len(all_edges) == 3
    ab_edges = list(g.list_edges(src=a, tgt=b))
    assert len(ab_edges) == 2
    bc_edges = list(g.list_edges(src=b, tgt=c))
    assert len(bc_edges) == 1 and bc_edges[0].edge_id == e3


