"""
Backend Parity Harness Tutorial

Goals:
- Show how to invoke PyTorch translation and note parity harness exists in tests
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
import torch


def main():
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 3}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 3}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    g.add_edge(i, h, {'weight': 0.2})
    g.add_edge(h, o, {'weight': 0.3})

    model = to_pytorch_model(g, {'device': 'cpu', 'dtype': torch.float32})
    y = model(torch.randn(4, 3), reset_states=True)
    print('output_shape:', tuple(y.shape))
    print('note: see tests/test_translation for parity harness examples')


if __name__ == '__main__':
    main()


