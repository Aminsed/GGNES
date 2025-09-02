import pytest


def test_input_node_incoming_edges_aggregate_with_original_slice_no_bias_activation():
    # Aligns with guide §5.3 INPUT Node behavior steps
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    torch = pytest.importorskip('torch')

    g = Graph()
    inp = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'bias': 1.23,
                      'attributes': {'output_size': 6}})
    src = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'bias': 0.0,
                      'attributes': {'output_size': 6}})

    # Connect HIDDEN(4) -> INPUT(6), requires per-edge projection 4->6, weight applied
    e = g.add_edge(src, inp, {'weight': 2.0, 'enabled': True})
    assert e is not None

    # Add OUTPUT so the model yields a tensor
    out = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'bias': 0.0,
                      'attributes': {'output_size': 6}})
    e2 = g.add_edge(inp, out, {'weight': 1.0, 'enabled': True})
    assert e2 is not None

    model = to_pytorch_model(g)

    x = torch.randn(3, 6)
    y = model(x, reset_states=True)

    # Output shape should follow OUTPUT(6)
    assert y.shape == (3, 6)
    # With source producing zeros and weight 2.0, INPUT output equals original slice
    # and no extra bias/activation is applied at INPUT
    assert torch.allclose(y, x)


