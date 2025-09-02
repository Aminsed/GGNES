import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model


def make_base_graph():
    g = Graph()
    a = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 8}
    })
    b = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'linear',
        'attributes': {'output_size': 8}
    })
    c = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 8}
    })
    g.add_edge(a, b, {'weight': 0.5})
    g.add_edge(b, c, {'weight': 0.5})
    return g, a, b, c


def _make_graph_with_agg(aggregation: str, attrs: dict):
    g = Graph()
    i1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 8}})
    i2 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 8}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8, 'aggregation': aggregation, **attrs}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 8}})
    g.add_edge(i1, h, {'weight': 0.1})
    g.add_edge(i2, h, {'weight': 0.2})
    g.add_edge(h, o, {'weight': 0.3})
    return g


def test_registry_has_advanced_aggregations():
    from ggnes.core.primitives import PrimitivesLibrary
    for name in ['attention', 'multi_head_attention', 'gated_sum', 'topk_weighted_sum', 'moe', 'attn_pool']:
        assert PrimitivesLibrary.is_valid_aggregation(name)


def test_attention_shapes_and_forward():
    g = _make_graph_with_agg('attention', {'head_dim': 8, 'temperature': 1.0})
    model = to_pytorch_model(g)
    import torch
    x = torch.randn(2, 16)
    y = model(x, reset_states=True)
    assert list(y.shape) == [2, 8]


def test_multi_head_attention_concat_heads():
    g = _make_graph_with_agg('multi_head_attention', {'num_heads': 2, 'head_dim': 4})
    model = to_pytorch_model(g)
    import torch
    x = torch.randn(2, 16)
    y = model(x, reset_states=True)
    assert list(y.shape) == [2, 8]


def test_topk_weighted_sum_respects_topk():
    g = _make_graph_with_agg('topk_weighted_sum', {'top_k': 1})
    model = to_pytorch_model(g)
    import torch
    x = torch.randn(2, 16)
    y = model(x, reset_states=True)
    assert list(y.shape) == [2, 8]


def test_gated_sum_works():
    g = _make_graph_with_agg('gated_sum', {})
    model = to_pytorch_model(g)
    import torch
    x = torch.randn(2, 16)
    y = model(x, reset_states=True)
    assert list(y.shape) == [2, 8]


def test_moe_softmax_and_topk():
    g_soft = _make_graph_with_agg('moe', {'router_type': 'softmax'})
    g_topk = _make_graph_with_agg('moe', {'router_type': 'topk', 'top_k': 1})
    import torch
    m1 = to_pytorch_model(g_soft)
    y1 = m1(torch.randn(2, 16), reset_states=True)
    m2 = to_pytorch_model(g_topk)
    y2 = m2(torch.randn(2, 16), reset_states=True)
    assert list(y1.shape) == [2, 8]
    assert list(y2.shape) == [2, 8]


def test_attn_pool_shapes():
    g = _make_graph_with_agg('attn_pool', {'pool_heads': 2})
    model = to_pytorch_model(g)
    import torch
    x = torch.randn(2, 16)
    y = model(x, reset_states=True)
    assert list(y.shape) == [2, 8]


@pytest.mark.parametrize("pname,value", [
    ("dropout_p", 1.5),
    ("num_heads", 0),
    ("head_dim", 0),
    ("temperature", -1.0),
    ("attn_eps", 0.0),
    ("attn_type", "invalid"),
])
def test_invalid_params_are_reported_by_validate(pname, value):
    g = _make_graph_with_agg('attention', {'head_dim': 8})
    # mutate attribute to invalid
    hid = [nid for nid, n in g.nodes.items() if n.node_type == NodeType.HIDDEN][0]
    g.nodes[hid].attributes[pname] = value
    errs = []
    ok = g.validate(collect_errors=errs)
    assert not ok
    assert any(e.error_type == 'invalid_agg_param' for e in errs)


def test_repair_sets_defaults_for_invalid_params():
    g = _make_graph_with_agg('attention', {'head_dim': 8})
    hid = [nid for nid, n in g.nodes.items() if n.node_type == NodeType.HIDDEN][0]
    g.nodes[hid].attributes['temperature'] = -1
    errs = []
    ok = g.validate(collect_errors=errs)
    assert not ok
    from ggnes.repair.repair import repair
    success, metrics = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert success
    assert any('set_default_temperature' == a for a in metrics['repairs_attempted'])


