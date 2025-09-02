from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def test_global_only_strategy_with_deterministic_uuids():
    g = Graph(config={
        'id_strategy': 'GLOBAL_ONLY',
        'deterministic_uuids': True,
        'graph_provenance_uuid': 'cccccccc-cccc-cccc-cccc-cccccccccccc',
    })
    n1 = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    n2 = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    e = g.add_edge(n1, n2, {'weight': 0.1})
    assert isinstance(e, uuid.UUID)
    assert isinstance(g.nodes[n1].global_id, uuid.UUID)


def test_multigraph_parallel_edges_have_distinct_uuids_deterministic():
    g = Graph(config={
        'id_strategy': 'HYBRID',
        'deterministic_uuids': True,
        'multigraph': True,
        'graph_provenance_uuid': 'dddddddd-dddd-dddd-dddd-dddddddddddd',
    })
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    e1 = g.add_edge(a, b, {'weight': 0.1})
    e2 = g.add_edge(a, b, {'weight': 0.2})
    assert e1 != e2
    # Deterministic: rebuilding produces same pair of UUIDs (order-insensitive set compare)
    g2 = Graph(config={
        'id_strategy': 'HYBRID',
        'deterministic_uuids': True,
        'multigraph': True,
        'graph_provenance_uuid': 'dddddddd-dddd-dddd-dddd-dddddddddddd',
    })
    a2 = g2.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    b2 = g2.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    f1 = g2.add_edge(a2, b2, {'weight': 0.1})
    f2 = g2.add_edge(a2, b2, {'weight': 0.2})
    assert {e1, e2} == {f1, f2}

