"""
Transactions and Safety Tutorial

Goals:
- Stage node and edge additions using TransactionManager
- Commit safely, or rollback on errors

Design tips:
- Keep staged changes small and self-contained
- Validate assumptions early; handle exceptions explicitly
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.transaction import TransactionManager


def build_seed_graph() -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    _ = g.add_edge(i, o, {'weight': 0.1})
    return g


def main():
    g = build_seed_graph()
    print('initial_nodes:', sorted(g.nodes.keys()))

    tx = TransactionManager(g, rng_manager=None)
    tx.begin()
    try:
        # Stage: add a hidden node and connect input→hidden and hidden→output
        h = tx.buffer.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'tanh', 'attributes': {'output_size': 2}})
        tx.buffer.add_edge(source=g.input_node_ids[0], target=h, properties={'weight': 0.05})
        tx.buffer.add_edge(source=h, target=g.output_node_ids[0], properties={'weight': 0.07})

        # Validate and commit atomically
        mapping = tx.commit()
        print('commit_mapping:', mapping)
        print('post_nodes:', sorted(g.nodes.keys()))
    except Exception as exc:
        # On errors, rollback keeps graph unchanged
        print('error:', exc)
        tx.rollback()


if __name__ == '__main__':
    main()


