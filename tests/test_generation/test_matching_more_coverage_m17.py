import re

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.matching import find_subgraph_matches
from ggnes.rules.predicates import PredicateRegistry


def test_node_name_regex_and_edge_predicate_factory_path():
    g = Graph()
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 2, 'name': 'foo'}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 2, 'name': 'bar'}})
    g.add_edge(a, b, {'weight': 0.25, 'enabled': True})

    # Ensure factory exists in registry for this test
    @PredicateRegistry.register_factory('greater_than')
    def greater_than_factory(threshold):
        def predicate(value):
            return value > threshold
        return predicate

    lhs = {
        'nodes': [
            {'label': 'A', 'match_criteria': {'name_regex': re.compile('^f.*')}} ,
            {'label': 'B', 'match_criteria': {'name_regex': re.compile('^b.*')}} ,
        ],
        'edges': [
            {'source_label': 'A', 'target_label': 'B', 'match_criteria': {'weight_predicate': ('greater_than', {'threshold': 0.2})}, 'edge_label': 'EAB'}
        ]
    }

    results = list(find_subgraph_matches(g, lhs, timeout_ms=10))
    assert results and results[0]['A'] == a and results[0]['B'] == b and 'EAB' in results[0]

