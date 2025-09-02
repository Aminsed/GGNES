from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.matching import find_subgraph_matches
from ggnes.rules.predicates import PredicateRegistry


# [T-mg-match-02]
def test_matching_selects_edge_instance_by_weight_predicate():
    # Ensure the needed factory is registered locally for this test
    @PredicateRegistry.register_factory('greater_than')
    def greater_than_factory(threshold):
        def predicate(value):
            return value > threshold
        predicate._factory_name = 'greater_than'
        predicate._factory_params = {'threshold': threshold}
        return predicate

    g = Graph(config={"multigraph": True})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1},
    })
    b = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 1},
    })
    g.add_edge(a, b, {"weight": 0.1})
    g.add_edge(a, b, {"weight": 0.9})

    lhs = {
        "nodes": [
            {"label": "A", "match_criteria": {"node_type": NodeType.INPUT}},
            {"label": "B", "match_criteria": {"node_type": NodeType.HIDDEN}},
        ],
        "edges": [
            {
                "source_label": "A",
                "target_label": "B",
                "edge_label": "E",
                "match_criteria": {"weight_predicate": ("greater_than", {"threshold": 0.5})},
            }
        ],
    }

    matches = list(find_subgraph_matches(g, lhs, timeout_ms=100))
    assert matches
    # The bound edge should be the heavier one
    for m in matches:
        assert m["E"].weight > 0.5


