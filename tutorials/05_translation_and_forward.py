"""
Translation and Forward Tutorial

Goals:
- Demonstrate per-edge projection and aggregation behavior
- Show a simple recurrent edge usage across two timesteps

We build input→hidden→output, plus a recurrent hidden→hidden edge.
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
import torch


def build_graph() -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'tanh', 'bias': 0.0, 'attributes': {'output_size': 6}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'bias': 0.0, 'attributes': {'output_size': 3}})
    _ = g.add_edge(i, h, {'weight': 0.4})
    _ = g.add_edge(h, o, {'weight': 0.2})
    # Add a recurrent self-edge on hidden
    _ = g.add_edge(h, h, {'weight': 0.1, 'attributes': {'is_recurrent': True}})
    return g


def main():
    g = build_graph()
    model = to_pytorch_model(g, {'device': 'cpu'})

    x = torch.randn(2, 4)
    # Timestep 1: recurrent path uses zeros as previous hidden output
    y1 = model(x, reset_states=True)
    # Timestep 2: recurrent path uses previous hidden output from timestep 1
    y2 = model(x, reset_states=False)
    print('y1_shape:', tuple(y1.shape), 'y2_shape:', tuple(y2.shape))


if __name__ == '__main__':
    main()


