# ruff: noqa: I001
import pytest


_torch_missing = (
    "torch" not in __import__('sys').modules
    and __import__('importlib').util.find_spec('torch') is None
)


def build_simple_graph():
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType

    g = Graph()
    in_id = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 4},
    })
    h_id = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'bias': 0.1,
        'attributes': {'output_size': 4, 'aggregation': 'sum'},
    })
    out_id = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 4, 'aggregation': 'sum'},
    })
    g.add_edge(in_id, h_id, {'weight': 0.5})
    g.add_edge(h_id, out_id, {'weight': 1.0})
    g.input_node_ids = [in_id]
    g.output_node_ids = [out_id]
    return g, in_id, h_id, out_id


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_basic_forward_sum_aggregation_matches_shape():
    import torch
    from ggnes.translation import to_pytorch_model

    g, in_id, h_id, out_id = build_simple_graph()
    model = to_pytorch_model(g)

    x = torch.ones(2, 4)
    y = model(x)
    assert y.shape == (2, 4)


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_input_width_mismatch_raises():
    import torch
    from ggnes.translation import to_pytorch_model

    g, *_ = build_simple_graph()
    model = to_pytorch_model(g)
    with pytest.raises(ValueError):
        _ = model(torch.ones(1, 3))


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_per_edge_projection_applied():
    import torch
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    g = Graph()
    a = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 3}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear',
                    'attributes': {'output_size': 5, 'aggregation': 'sum'}})
    g.add_edge(a, b, {'weight': 1.0})
    g.input_node_ids = [a]
    g.output_node_ids = [b]

    model = to_pytorch_model(g)
    y = model(torch.ones(2, 3))
    assert y.shape == (2, 5)


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_concat_and_post_projection():
    import torch
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    g = Graph()
    i1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                     'attributes': {'output_size': 2}})
    i2 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                     'attributes': {'output_size': 3}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear',
                    'attributes': {'output_size': 4, 'aggregation': 'concat'}})
    g.add_edge(i1, h, {'weight': 1.0})
    g.add_edge(i2, h, {'weight': 1.0})
    g.input_node_ids = [i1, i2]
    g.output_node_ids = [h]

    model = to_pytorch_model(g)
    x = torch.ones(2, 5)
    y = model(x)
    assert y.shape == (2, 4)


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_recurrent_uses_prev_output_with_state_manager():
    import torch
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'tanh',
                    'bias': 0.0, 'attributes': {'output_size': 2, 'aggregation': 'sum'}})
    g.add_edge(i, h, {'weight': 0.5})
    e = g.add_edge(h, h, {'weight': 0.9})
    edge_obj = g.find_edge_by_id(e)
    edge_obj.attributes['is_recurrent'] = True
    g.input_node_ids = [i]
    g.output_node_ids = [h]

    model = to_pytorch_model(g)

    x = torch.ones(1, 2)
    y1 = model(x, reset_states=True)
    y2 = model(x)

    assert not torch.allclose(y1, y2)


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_device_and_dtype_configuration_applied():
    import torch
    from ggnes.translation import to_pytorch_model
    g, *_ = build_simple_graph()
    model = to_pytorch_model(g, {'device': 'cpu', 'dtype': torch.float32})
    for p in model.parameters():
        assert p.dtype == torch.float32


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_hierarchical_submodule_cache_reuse_does_not_change_outputs():
    import torch
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model
    from ggnes.translation.pytorch_impl import clear_translation_cache

    g = Graph()
    # Node mimicking hierarchical derivation tag
    h = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'bias': 0.1,
        'attributes': {'output_size': 4, 'derivation_uuid': '123e4567-e89b-12d3-a456-426614174000'}
    })
    o = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 2}
    })
    g.add_edge(h, o, {'weight': 0.3})
    g.input_node_ids = []
    g.output_node_ids = [o]

    clear_translation_cache()
    m1 = to_pytorch_model(g)
    x = torch.randn(2, 0)
    y1 = m1(x, reset_states=True)
    m2 = to_pytorch_model(g)
    y2 = m2(x, reset_states=True)
    assert y1.shape == y2.shape
