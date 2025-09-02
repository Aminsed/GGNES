"""
Validation and Repair Tutorial

Goals:
- Trigger simple validation errors and observe structured messages
- Invoke the repair function to fix issues and revalidate

We create a graph with one problematic element (e.g., non-finite weight),
then call repair with MINIMAL_CHANGE.
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.repair import repair as repair_fn


def build_invalid_graph() -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    # Introduce an invalid (non-finite) weight to trigger validation error
    _ = g.add_edge(i, o, {'weight': float('nan')})
    return g


def main():
    g = build_invalid_graph()
    errors = []
    valid = g.validate(errors)
    print('valid_before:', valid)
    print('errors_before:', [str(e) for e in errors])

    ok, metrics = repair_fn(g, {'strategy': 'MINIMAL_CHANGE'})
    print('repaired:', ok, metrics)

    errors_after = []
    valid_after = g.validate(errors_after)
    print('valid_after:', valid_after)
    print('errors_after:', [str(e) for e in errors_after])


if __name__ == '__main__':
    main()


