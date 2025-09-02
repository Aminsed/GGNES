"""
Rules and Matching Tutorial (Concept Intro)

Goals:
- Define a minimal LHSPattern
- Use find_subgraph_matches to locate simple subgraphs in a small graph

We keep it simple: INPUT → OUTPUT edges only, with node-type criteria.
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.rules.rule import LHSPattern
from ggnes.generation.matching import find_subgraph_matches


def build_graph():
    g = Graph()
    i1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    o1 = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    _ = g.add_edge(i1, o1, {'weight': 0.2})
    return g


def main():
    g = build_graph()

    # Define a minimal LHS that finds INPUT→OUTPUT pairs
    lhs = LHSPattern(
        nodes=[
            {'label': 'A', 'match_criteria': {'node_type': NodeType.INPUT}},
            {'label': 'B', 'match_criteria': {'node_type': NodeType.OUTPUT}},
        ],
        edges=[
            {'source_label': 'A', 'target_label': 'B', 'match_criteria': {}},
        ],
        boundary_nodes=[],
    )

    matches = find_subgraph_matches(g, lhs)
    print('matches:', matches)


if __name__ == '__main__':
    main()


