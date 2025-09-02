from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def test_global_index_cleaned_on_remove_node_multigraph():
    g = Graph(config={"multigraph": True})
    a = g.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    b = g.add_node({"node_type": NodeType.HIDDEN, "activation_function": "linear", "attributes": {"output_size": 1}})
    c = g.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 1}})

    e1 = g.add_edge(a, b, {"weight": 0.1})
    e2 = g.add_edge(a, b, {"weight": 0.2})
    e3 = g.add_edge(b, c, {"weight": 0.3})

    assert g.find_edge_by_id(e1) is not None
    assert g.find_edge_by_id(e2) is not None
    assert g.find_edge_by_id(e3) is not None

    g.remove_node(b)

    assert g.find_edge_by_id(e1) is None
    assert g.find_edge_by_id(e2) is None
    assert g.find_edge_by_id(e3) is None
    # Adjacency cleaned
    assert c not in g.nodes.get(b, {}).get('edges_out', {})
    assert b not in g.nodes[a].edges_out


def test_global_index_cleaned_on_remove_node_simple():
    g = Graph(config={"multigraph": False})
    a = g.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    b = g.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    e = g.add_edge(a, b, {"weight": 0.5})
    assert g.find_edge_by_id(e) is not None
    g.remove_node(b)
    assert g.find_edge_by_id(e) is None
    assert b not in g.nodes[a].edges_out


