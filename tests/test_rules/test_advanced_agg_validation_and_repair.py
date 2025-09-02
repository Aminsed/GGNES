import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def build_graph_with_hidden(aggregation: str, attrs: dict, fan_in: int = 2):
    g = Graph()
    inputs = []
    for _ in range(fan_in):
        inputs.append(
            g.add_node({
                'node_type': NodeType.INPUT,
                'activation_function': 'linear',
                'attributes': {'output_size': 4},
            })
        )
    h = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'attributes': {'output_size': 8, 'aggregation': aggregation, **attrs},
    })
    o = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 8},
    })
    for i in inputs:
        g.add_edge(i, h, {'weight': 0.1})
    g.add_edge(h, o, {'weight': 0.2})
    return g, h


@pytest.mark.parametrize(
    'pname,value',
    [
        ('dropout_p', 2.0),
        ('post_projection', 'x'),
        ('normalize', 'y'),
    ],
)
def test_common_param_validation_and_repair(pname, value):
    g, h = build_graph_with_hidden('attention', {'head_dim': 4}, fan_in=2)
    g.nodes[h].attributes[pname] = value
    errs = []
    ok = g.validate(collect_errors=errs)
    assert not ok
    assert any(e.error_type == 'invalid_agg_param' for e in errs)
    from ggnes.repair.repair import repair
    success, metrics = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert success
    # Defaults per spec
    defaults = {
        'dropout_p': 0.0,
        'post_projection': True,
        'normalize': False,
    }
    assert g.nodes[h].attributes[pname] == defaults[pname]


def test_attention_param_bounds_and_repair():
    g, h = build_graph_with_hidden('attention', {'head_dim': 0, 'temperature': -1.0, 'attn_eps': 0.0, 'attn_type': 'bad', 'top_k': 5}, fan_in=2)
    errs = []
    ok = g.validate(collect_errors=errs)
    # Expect multiple invalids
    assert not ok
    kinds = [e.details.get('param_name') for e in errs if hasattr(e, 'details')]
    for expected in ['head_dim', 'temperature', 'attn_eps', 'attn_type', 'top_k']:
        assert expected in kinds
    from ggnes.repair.repair import repair
    success, _ = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert success
    # Revalidate ok
    errs2 = []
    assert g.validate(collect_errors=errs2)


def test_multi_head_attention_num_heads_repair():
    g, h = build_graph_with_hidden('multi_head_attention', {'num_heads': 0, 'head_dim': 4}, fan_in=2)
    errs = []
    assert not g.validate(collect_errors=errs)
    assert any(e.details.get('param_name') == 'num_heads' for e in errs)
    from ggnes.repair.repair import repair
    success, _ = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert success
    assert g.nodes[h].attributes['num_heads'] == 1


def test_topk_weighted_sum_topk_out_of_range():
    g, h = build_graph_with_hidden('topk_weighted_sum', {'top_k': 3}, fan_in=2)
    errs = []
    assert not g.validate(collect_errors=errs)
    assert any(e.details.get('param_name') == 'top_k' for e in errs)
    from ggnes.repair.repair import repair
    success, _ = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert success
    assert g.nodes[h].attributes['top_k'] is None


def test_moe_validation_and_repair():
    g, h = build_graph_with_hidden('moe', {
        'router_type': 'invalid',
        'experts': 0,
        'capacity_factor': -1.0,
        'router_temperature': -2.0,
        'top_k': 10,
    }, fan_in=2)
    errs = []
    assert not g.validate(collect_errors=errs)
    names = [e.details.get('param_name') for e in errs if hasattr(e, 'details')]
    for expected in ['router_type', 'experts', 'capacity_factor', 'router_temperature', 'top_k']:
        assert expected in names
    from ggnes.repair.repair import repair
    success, _ = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert success
    assert g.nodes[h].attributes['router_type'] == 'softmax'
    assert g.nodes[h].attributes['experts'] == 1
    assert g.nodes[h].attributes['capacity_factor'] == 1.0
    assert g.nodes[h].attributes['router_temperature'] == 1.0
    assert g.nodes[h].attributes['top_k'] is None


def test_attn_pool_pool_heads_repair():
    g, h = build_graph_with_hidden('attn_pool', {'pool_heads': 0}, fan_in=2)
    errs = []
    assert not g.validate(collect_errors=errs)
    assert any(e.details.get('param_name') == 'pool_heads' for e in errs)
    from ggnes.repair.repair import repair
    success, _ = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert success
    assert g.nodes[h].attributes['pool_heads'] == 1


