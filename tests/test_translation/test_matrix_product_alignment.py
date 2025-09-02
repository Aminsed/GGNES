import pytest


def test_matrix_product_concat_flatten_then_project():
    # Validates §8.3 and §13.6 semantics
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    torch = pytest.importorskip('torch')

    g = Graph()
    inp1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 10}})
    inp2 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 20}})
    tgt = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 15}})
    out = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 15}})

    # Edges into target with matrix_product aggregation
    g.nodes[tgt].attributes['aggregation'] = 'matrix_product'
    assert g.add_edge(inp1, tgt, {'weight': 0.5}) is not None
    assert g.add_edge(inp2, tgt, {'weight': 0.5}) is not None
    assert g.add_edge(tgt, out, {'weight': 1.0}) is not None

    model = to_pytorch_model(g)

    # Inputs concatenate at source: since there are two INPUT nodes, compose combined x appropriately
    x = torch.randn(2, 30)  # 10 + 20 distributed according to input_node_ids
    y = model(x, reset_states=True)
    assert y.shape == (2, 15)


