import logging
import uuid

import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def test_invalid_id_strategy_raises_value_error():
    with pytest.raises(ValueError):
        Graph(config={'id_strategy': 'INVALID'})


def test_remove_node_nonexistent_noop():
    g = Graph()
    # Add a single valid node to ensure graph is initialized
    g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'attributes': {'output_size': 4},
    })
    # Removing a non-existent node should be a no-op (no exception)
    g.remove_node(999)
    assert len(g.nodes) == 1


def test_add_edge_with_missing_endpoints_raises():
    g = Graph()
    with pytest.raises(ValueError):
        g.add_edge(0, 1)


def test_remove_edge_unknown_is_noop():
    g = Graph()
    # Use a UUID since default id_strategy is HYBRID
    g.remove_edge(uuid.uuid4())
    # No exception and still empty
    assert len(g.nodes) == 0


def test_find_edge_by_endpoints_unknown_source_returns_none():
    g = Graph()
    # Create one node so graph isn't entirely empty
    n0 = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'attributes': {'output_size': 2},
    })
    # Source id 999 does not exist
    assert g.find_edge_by_endpoints(999, n0) is None


def test_topological_sort_warns_on_cycle(caplog):
    g = Graph()
    a = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'attributes': {'output_size': 3},
    })
    b = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'attributes': {'output_size': 3},
    })
    g.add_edge(a, b, {'weight': 0.1})
    g.add_edge(b, a, {'weight': 0.1})

    caplog.set_level(logging.WARNING)
    order = g.topological_sort()  # detect_cycles not called → cycle remains
    # Incomplete ordering triggers a warning
    assert any('Incomplete topological sort' in r.message for r in caplog.records)
    assert len(order) < len(g.nodes)


