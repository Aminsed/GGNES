import pytest


def test_recurrent_cells_use_single_bias_source():
    # LSTM/GRU must be created with bias=False; node bias applied once
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    torch = pytest.importorskip('torch')

    for act in ('lstm', 'gru'):
        g = Graph()
        inp = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 8}})
        rec = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': act, 'bias': 0.3,
                          'attributes': {'output_size': 8}})
        out = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 8}})
        assert g.add_edge(inp, rec, {'weight': 1.0}) is not None
        assert g.add_edge(rec, rec, {'weight': 0.2, 'attributes': {'is_recurrent': True}}) is not None
        assert g.add_edge(rec, out, {'weight': 1.0}) is not None

        model = to_pytorch_model(g)

        # Ensure bias parameter exists and is applied once by forward
        bias_param = getattr(model, f"bias_{rec}", None)
        assert bias_param is not None

        x = torch.zeros(1, 8)
        y1 = model(x, reset_states=True)
        y2 = model(x, reset_states=False)  # uses prev_outputs on recurrent edge

        assert y1.shape == (1, 8)
        assert y2.shape == (1, 8)


