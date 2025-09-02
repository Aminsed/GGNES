from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
import torch


def main():
    # Quickstart goal:
    # 1) Build the simplest possible graph (INPUT → HIDDEN → OUTPUT)
    # 2) Translate it to a PyTorch model
    # 3) Run a forward pass to confirm dimensions and a working pipeline

    # Create an empty graph.
    g = Graph()

    # Add an INPUT node.
    # The INPUT node's output_size must match the width of the tensor you will feed later.
    inp = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',  # INPUT nodes are linear passthrough in our translator
        'attributes': {'output_size': 4},  # will feed x with shape [batch, 4]
    })

    # Add a simple HIDDEN node.
    # HIDDEN nodes can have a bias and an activation function (ReLU here).
    hid = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'bias': 0.0,
        'attributes': {'output_size': 4},  # keep the same size for simplicity
    })

    # Add an OUTPUT node.
    # OUTPUT nodes contribute to the final concatenated output of the network.
    out = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 2},  # final dimensionality will be 2
    })

    # Connect the nodes with weighted edges.
    # Edges carry a scalar weight; dimensional projections are inserted automatically if needed.
    _ = g.add_edge(inp, hid, {'weight': 0.5})
    _ = g.add_edge(hid, out, {'weight': 0.3})

    # Translate to a PyTorch model. Device/dtype can be customized via config.
    model = to_pytorch_model(g, {'device': 'cpu'})

    # Prepare a random input with batch size 2 and width equal to the sum of INPUT node sizes (4 here).
    x = torch.randn(2, 4)

    # Run one forward pass. reset_states=True is safe for non-recurrent graphs as well.
    y = model(x, reset_states=True)

    # Print a simple confirmation of the output shape.
    print('output_shape:', tuple(y.shape))


if __name__ == '__main__':
    main()


