"""
Targeted coverage tests for ggnes.core.graph to reach 100%.
"""

from __future__ import annotations

import logging
import uuid

import pytest

from ggnes.core.graph import Graph, NodeType


def test_invalid_id_strategy_raises():
    with pytest.raises(ValueError):
        Graph(config={'id_strategy': 'BOGUS'})


def test_remove_nonexistent_node_is_noop():
    g = Graph()
    # Should not raise
    g.remove_node(999)


def test_add_edge_missing_nodes_raises():
    g = Graph()
    with pytest.raises(ValueError):
        g.add_edge(0, 1)


def test_remove_edge_nonexistent_is_noop():
    g = Graph()
    # Use a random UUID to ensure not found path
    g.remove_edge(uuid.uuid4())


def test_find_edge_by_endpoints_when_source_missing_returns_none():
    g = Graph()
    assert g.find_edge_by_endpoints(1, 2) is None


def test_validate_dangling_target_edge():
    g = Graph()
    n1 = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 4}
    })
    # Manually create edge to non-existent target
    edge_id = g.add_edge(n1, n1, {'weight': 0.1})
    # Remove the target endpoint to create dangling target entry
    g.nodes[n1].edges_out.pop(n1)
    # Insert a fake mapping to a missing target id with required attributes
    g.nodes[n1].edges_out[999] = type('E', (), {
        'edge_id': edge_id,
        'enabled': True,
        'weight': 0.1,
    })()

    errors = []
    valid = g.validate(collect_errors=errors)
    assert not valid
    # Ensure a dangling_edge error referencing target exists
    assert any(e.error_type == 'dangling_edge' and 'Target node' in e.message for e in errors)


def test_detect_cycles_skips_disabled_edges():
    g = Graph()
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 2}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 2}})
    e1 = g.add_edge(a, b, {'weight': 0.1, 'attributes': {}})
    e2 = g.add_edge(b, a, {'weight': 0.1, 'attributes': {}})
    # Disable edges so DFS branch 'continue' executes
    g.find_edge_by_id(e1).enabled = False
    g.find_edge_by_id(e2).enabled = False

    g.detect_cycles()
    # Disabled edges should not be marked recurrent
    assert g.find_edge_by_id(e1).attributes.get('is_recurrent', False) is False
    assert g.find_edge_by_id(e2).attributes.get('is_recurrent', False) is False


def test_topological_sort_warns_on_cycle(caplog):
    g = Graph()
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 2}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 2}})
    g.add_edge(a, b)
    g.add_edge(b, a)
    with caplog.at_level(logging.WARNING):
        order = g.topological_sort(ignore_recurrent=True)
    assert len(order) < len(g.nodes)
    assert any('Incomplete topological sort' in rec.message for rec in caplog.records)


def test_has_path_source_equals_target_true():
    g = Graph()
    a = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    assert g._has_path(a, a) is True


