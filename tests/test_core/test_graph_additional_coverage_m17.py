from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def build_simple():
    g = Graph()
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 3}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 3}})
    g.add_edge(a, b, {'weight': 0.1})
    return g, a, b


def build_multigraph():
    g = Graph(config={'multigraph': True})
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 3}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 3}})
    g.add_edge(a, b, {'weight': 0.1})
    g.add_edge(a, b, {'weight': 0.2})
    return g, a, b


def test_find_edges_by_endpoints_missing_source_and_present():
    g, a, b = build_simple()
    assert g.find_edges_by_endpoints(999, b) == []
    lst = g.find_edges_by_endpoints(a, b)
    assert len(lst) == 1 and lst[0] is not None


def test_list_edges_tgt_simple_and_multigraph():
    g, a, b = build_simple()
    # Simple mode: list edges by tgt
    edges = list(g.list_edges(tgt=b))
    assert len(edges) == 1

    mg, ma, mb = build_multigraph()
    edges_mg = list(mg.list_edges(tgt=mb))
    # Two parallel edges
    assert len(edges_mg) == 2


def test_list_edges_src_and_both_args_and_global_listing():
    g, a, b = build_simple()
    # src-only
    edges_from_a = list(g.list_edges(src=a))
    assert len(edges_from_a) == 1
    # both src and tgt
    both = list(g.list_edges(src=a, tgt=b))
    assert len(both) == 1
    # global listing
    all_edges = list(g.list_edges())
    assert len(all_edges) == 1


def test_list_edges_src_missing_and_src_multigraph():
    # Missing src should yield empty
    g, a, b = build_simple()
    assert list(g.list_edges(src=9999)) == []

    # Multigraph src-only should list all parallel edges
    mg, ma, mb = build_multigraph()
    edges_from_ma = list(mg.list_edges(src=ma))
    assert len(edges_from_ma) == 2


def test_validate_adv_agg_multigraph_fanin_path():
    g = Graph(config={'multigraph': True})
    i1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    i2 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4, 'aggregation': 'attention', 'head_dim': 4}})
    g.add_edge(i1, h, {'weight': 0.1})
    g.add_edge(i2, h, {'weight': 0.2})
    # Duplicate parallel edge to ensure multigraph fan_in path is exercised
    g.add_edge(i2, h, {'weight': 0.3})
    errs = []
    assert g.validate(collect_errors=errs)
    # Cover warnings branch: output node with no incoming
    g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    errs2 = []
    warns = []
    assert not g.validate(collect_errors=errs2, collect_warnings=warns)


def test_list_edges_tgt_missing():
    """Test list_edges with tgt=missing to cover line 748."""
    g = Graph()
    edges = list(g.list_edges(tgt="missing_node"))
    assert edges == []  # Should return empty when node doesn't exist
