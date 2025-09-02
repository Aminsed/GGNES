# ruff: noqa: I001
import pytest


_torch_missing = (
    "torch" not in __import__('sys').modules
    and __import__('importlib').util.find_spec('torch') is None
)


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_mean_and_max_aggregation():
    import torch
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    g = Graph()
    i1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                     'attributes': {'output_size': 3}})
    i2 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                     'attributes': {'output_size': 3}})
    h_mean = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear',
                         'attributes': {'output_size': 3, 'aggregation': 'mean'}})
    h_max = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                        'attributes': {'output_size': 3, 'aggregation': 'max'}})

    g.add_edge(i1, h_mean, {'weight': 1.0})
    g.add_edge(i2, h_mean, {'weight': 1.0})
    g.add_edge(h_mean, h_max, {'weight': 1.0})

    g.input_node_ids = [i1, i2]
    g.output_node_ids = [h_max]

    model = to_pytorch_model(g)
    x = torch.ones(2, 6)
    y = model(x)
    assert y.shape == (2, 3)


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_matrix_product_flattens_multiple_inputs():
    import torch
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    g = Graph()
    i1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                     'attributes': {'output_size': 2}})
    i2 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                     'attributes': {'output_size': 3}})
    h = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 5, 'aggregation': 'matrix_product'}})
    g.add_edge(i1, h, {'weight': 1.0})
    g.add_edge(i2, h, {'weight': 1.0})

    g.input_node_ids = [i1, i2]
    g.output_node_ids = [h]

    model = to_pytorch_model(g)
    x = torch.ones(2, 5)
    y = model(x)
    assert y.shape == (2, 5)


@pytest.mark.skipif(_torch_missing, reason="requires torch")
def test_node_with_no_inputs_uses_zeros():
    import torch
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    h = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    # No edge from i to h: h should receive zeros and output zeros (linear)
    g.input_node_ids = [i]
    g.output_node_ids = [h]

    model = to_pytorch_model(g)
    y = model(torch.ones(1, 2))
    assert y.shape == (1, 2)
    assert torch.allclose(y, torch.zeros_like(y))
