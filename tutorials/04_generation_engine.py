"""
Generation Engine Tutorial

Goals:
- Show how to invoke the generation engine `generate_network`
- Demonstrate quiescence (no rules → zero iterations)

We use an empty Genotype (no rules). This keeps the example minimal while
illustrating the API and returned metrics.
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.network_gen import generate_network
from ggnes.evolution.genotype import Genotype
from ggnes.utils.rng_manager import RNGManager


def build_axiom() -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    _ = g.add_edge(i, o, {'weight': 0.1})
    return g


def main():
    axiom = build_axiom()
    genotype = Genotype()  # empty rule list
    rng = RNGManager(seed=42)
    config = {
        'max_iterations': 5,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_action': 'TERMINATE',
    }
    phenotype, metrics = generate_network(genotype, axiom, config, rng)

    print('iterations:', metrics['iterations'])  # expect 0
    print('oscillation_skips:', metrics['oscillation_skips'])
    print('final_nodes:', metrics['final_nodes'])
    print('final_edges:', metrics['final_edges'])


if __name__ == '__main__':
    main()


