import pytest


def test_graph_id_strategy_default_and_normalization_and_invalid():
    from ggnes.core.graph import Graph, IDStrategy

    # Default should be HYBRID when not provided
    g_default = Graph()
    assert g_default.config['id_strategy'] == IDStrategy.HYBRID

    # String should normalize to enum
    g_global = Graph(config={'id_strategy': 'GLOBAL_ONLY'})
    assert g_global.config['id_strategy'] == IDStrategy.GLOBAL_ONLY

    # Invalid string should raise ValueError
    with pytest.raises(ValueError):
        Graph(config={'id_strategy': 'NOT_A_STRATEGY'})


def test_graph_fingerprint_ignores_edge_weight_and_considers_aggregation():
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType

    g = Graph()
    a = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    g.add_edge(a, b, {'weight': 0.1})

    fp1 = g.compute_fingerprint()

    # Changing weight should not affect fingerprint
    edge = g.find_edge_by_endpoints(a, b)
    edge.weight = 123.456
    fp2 = g.compute_fingerprint()
    assert fp1 == fp2

    # Changing aggregation should affect fingerprint (part of node_dim)
    g.nodes[b].attributes['aggregation'] = 'mean'
    fp3 = g.compute_fingerprint()
    assert fp3 != fp2


