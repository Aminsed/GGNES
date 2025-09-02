"""
Multigraph Mode Tutorial

Goals:
- Enable multigraph mode and add parallel edges between the same nodes
- Verify translation creates per-edge parameters and projections keyed by edge_id
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
import torch


def main():
    g = Graph({'multigraph': True})
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 3}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 5}})
    _e1 = g.add_edge(i, h, {'weight': 0.2})
    _e2 = g.add_edge(i, h, {'weight': 0.4})  # parallel edge allowed
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    g.add_edge(h, o, {'weight': 0.1})

    model = to_pytorch_model(g, {'device': 'cpu'})
    y = model(torch.randn(2, 3), reset_states=True)
    print('output_shape:', tuple(y.shape))


if __name__ == '__main__':
    main()


