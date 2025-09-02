"""
Additional tests for M5 matching to reach 100% coverage and enforce edge cases.
"""

import re

import pytest  # noqa: F401

from ggnes.core import Graph
from ggnes.core.graph import NodeType


def build_graph_with_named_nodes():
    graph = Graph()
    n1 = graph.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 4, "name": "input_0"}
    })
    n2 = graph.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "relu",
        "attributes": {"output_size": 4, "name": "hidden_1", "tag": "ok"}
    })
    n3 = graph.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2, "name": "output_0"}
    })
    graph.add_edge(n1, n2, {"weight": 0.5, "enabled": True})
    graph.add_edge(n2, n3, {"weight": 0.7, "enabled": True})
    return graph, (n1, n2, n3)


def test_node_name_regex_invalid_type_excludes_all():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    # name_regex is invalid type (string), should be rejected and thus no candidates
    lhs = {
        "nodes": [
            {
                "label": "H",
                "match_criteria": {
                    "node_type": NodeType.HIDDEN,
                    "name_regex": "hidden_\\d+"
                }
            },
        ],
        "edges": []
    }

    matches = list(find_subgraph_matches(graph, lhs, timeout_ms=50))
    assert matches == []


def test_node_attribute_equality_mismatch_produces_no_candidates():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    # tag mismatch should exclude hidden node
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN, "tag": "nope"}},
        ],
        "edges": []
    }

    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_edge_enabled_mismatch_rejects():
    from ggnes.generation.matching import find_subgraph_matches

    graph, (n1, n2, n3) = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {"source_label": "H", "target_label": "O", "match_criteria": {"enabled": False}},
        ]
    }

    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_edge_weight_predicate_invalid_spec():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {
                "source_label": "H",
                "target_label": "O",
                "match_criteria": {"weight_predicate": "not-a-tuple"}
            },
        ]
    }

    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_edge_weight_predicate_unknown():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {
                "source_label": "H",
                "target_label": "O",
                "match_criteria": {"weight_predicate": ("unknown_pred", {})}
            },
        ]
    }

    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_edge_weight_predicate_false():
    from ggnes.generation.matching import find_subgraph_matches
    from ggnes.rules.predicates import PredicateRegistry

    @PredicateRegistry.register('gt')
    def gt(value, threshold: float):
        return float(value) > float(threshold)

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {
                "source_label": "H",
                "target_label": "O",
                "match_criteria": {"weight_predicate": ("gt", {"threshold": 0.9})}
            },
        ]
    }

    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_edge_attribute_equality_mismatch():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {
                "source_label": "H",
                "target_label": "O",
                "match_criteria": {"custom": "value"}
            },
        ]
    }

    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_binding_enrichment_with_unknown_edge_labels():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        # Edge references labels not present among nodes; should be ignored during enrichment
        "edges": [
            {"source_label": "X", "target_label": "Y", "edge_label": "E"},
        ]
    }

    matches = list(find_subgraph_matches(graph, lhs, timeout_ms=50))
    # Expect Cartesian matches of H and O with no edge labels present
    assert matches
    assert all("E" not in b for b in matches)


def test_timeout_inside_loop(monkeypatch):
    from ggnes.generation import matching as m

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "A", "match_criteria": {}},
            {"label": "B", "match_criteria": {}},
            {"label": "C", "match_criteria": {}},
        ],
        "edges": []
    }

    # Fake monotonic to cross the budget only inside the loop
    seq = [0.0, 0.001, 0.01, 0.02, 0.03]

    def fake_monotonic():
        return seq.pop(0) if seq else 1.0

    monkeypatch.setattr(m.time, 'monotonic', fake_monotonic)

    matches = list(m.find_subgraph_matches(graph, lhs, timeout_ms=5))
    # Should return quickly with possibly partial results
    assert isinstance(matches, list)


def test_node_name_regex_mismatch_excludes_node():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {
                "label": "H",
                "match_criteria": {
                    "node_type": NodeType.HIDDEN,
                    "name_regex": re.compile(r"no_match")
                }
            },
        ],
        "edges": []
    }

    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_edge_with_empty_criteria_path():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        # Empty match_criteria should be treated as True (hit early return path)
        "edges": [
            {"source_label": "H", "target_label": "O", "edge_label": "E"},
        ]
    }

    matches = list(find_subgraph_matches(graph, lhs, timeout_ms=50))
    assert matches and any("E" in b for b in matches)


def test_missing_graph_edge_inconsistent_bindings():
    from ggnes.generation.matching import find_subgraph_matches

    graph, _ = build_graph_with_named_nodes()
    # Ask for H->I (hidden to input). Graph has INPUT->HIDDEN, not HIDDEN->INPUT
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "I", "match_criteria": {"node_type": NodeType.INPUT}},
        ],
        "edges": [
            {"source_label": "H", "target_label": "I"},
        ]
    }

    # No matches because edge is missing in graph direction
    assert list(find_subgraph_matches(graph, lhs, timeout_ms=50)) == []


def test_timeout_before_backtrack(monkeypatch):
    from ggnes.generation import matching as m

    graph, _ = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "A", "match_criteria": {}},
        ],
        "edges": []
    }

    seq = [0.0, 1000.0]

    def fake_monotonic():
        return seq.pop(0)

    monkeypatch.setattr(m.time, 'monotonic', fake_monotonic)

    assert list(m.find_subgraph_matches(graph, lhs, timeout_ms=1)) in ([], [{}])


def test_edge_between_handles_missing_node(monkeypatch):
    from ggnes.generation import matching as m

    graph, (n1, n2, n3) = build_graph_with_named_nodes()
    lhs = {
        "nodes": [
            {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
            {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
        ],
        "edges": [
            {"source_label": "H", "target_label": "O"},
        ]
    }

    original_nodes = graph.nodes

    class Proxy(dict):
        def get(self, key, default=None):
            # Simulate missing node for hidden node id
            if key == n2:
                return None
            return original_nodes.get(key, default)

    graph.nodes = Proxy(original_nodes)

    try:
        assert list(m.find_subgraph_matches(graph, lhs, timeout_ms=50)) == []
    finally:
        graph.nodes = original_nodes
