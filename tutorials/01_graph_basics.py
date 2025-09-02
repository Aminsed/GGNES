"""
Graph Basics Tutorial

Goals:
- Create a small graph with multiple INPUT nodes and one OUTPUT node
- Add edges and validate the graph
- Compute and display the Weisfeiler–Lehman fingerprint
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def main():
    # Create an empty graph
    g = Graph()

    # Two INPUT nodes with different output sizes; a single OUTPUT node
    i1 = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 2},
    })
    i2 = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 3},
    })
    o1 = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 1},
    })

    # Connect both inputs to the output
    _ = g.add_edge(i1, o1, {'weight': 0.1})
    _ = g.add_edge(i2, o1, {'weight': 0.2})

    # Validate the graph and print concise results
    errors, warnings = [], []
    is_valid = g.validate(errors, warnings)
    print('valid:', is_valid)
    print('errors:', [str(e) for e in errors])
    print('warnings:', [str(w) for w in warnings])

    # Compute WL fingerprint (deterministic structural hash)
    fp = g.compute_fingerprint()
    print('fingerprint:', fp)


if __name__ == '__main__':
    main()


