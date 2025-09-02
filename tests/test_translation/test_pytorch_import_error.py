# ruff: noqa: I001
import sys
import pytest


def test_to_pytorch_model_import_error_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    if 'ggnes.translation.pytorch_impl' in sys.modules:
        del sys.modules['ggnes.translation.pytorch_impl']
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation.pytorch import to_pytorch_model

    g = Graph()
    i = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 1},
    })
    o = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 1},
    })
    g.add_edge(i, o, {'weight': 1.0})
    g.input_node_ids = [i]
    g.output_node_ids = [o]

    with pytest.raises(ImportError):
        _ = to_pytorch_model(g)
