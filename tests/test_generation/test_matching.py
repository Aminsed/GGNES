"""
M5 Matching tests to strictly adhere to project_guide.md.

Covers:
- [T-match-01] Empty LHS matches with [{}]
- [T-match-02] Most-constrained starting node heuristic determinism
- [T-match-03] Edge criteria including regex and predicate matching
- [T-match-04] Timeout respected; partial results acceptable
- [T-match-05] Edge labels added to bindings when specified
"""

import re
import time

import pytest  # noqa: F401

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.rules.predicates import PredicateRegistry


def build_linear_graph():
    graph = Graph()
    n1 = graph.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 4}
    })
    n2 = graph.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "relu",
        "attributes": {"output_size": 4}
    })
    n3 = graph.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2}
    })
    graph.add_edge(n1, n2, {"weight": 0.5})
    graph.add_edge(n2, n3, {"weight": 0.7, "enabled": True, "label": "out_edge"})
    return graph, (n1, n2, n3)


class TestMatching:
    def test_empty_lhs_matches_empty_binding(self):
        """[T-match-01] Empty LHS must yield a single empty binding {}."""
        from ggnes.generation.matching import find_subgraph_matches

        graph, _ = build_linear_graph()
        lhs = {"nodes": [], "edges": []}

        matches = list(find_subgraph_matches(graph, lhs, timeout_ms=100))
        assert matches == [{}]

    def test_heuristic_determinism_most_constrained_start(self):
        """[T-match-02] Deterministic ordering via most-constrained starting node."""
        from ggnes.generation.matching import find_subgraph_matches

        graph, (n1, n2, n3) = build_linear_graph()

        # Pattern: one HIDDEN node connected to OUTPUT; HIDDEN is more constrained
        lhs = {
            "nodes": [
                {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
                {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
            ],
            "edges": [
                {"source_label": "H", "target_label": "O", "match_criteria": {"enabled": True}},
            ],
        }

        m1 = list(find_subgraph_matches(graph, lhs, timeout_ms=100))
        m2 = list(find_subgraph_matches(graph, lhs, timeout_ms=100))
        assert m1 == m2
        # And ensure binding maps labels correctly
        assert all("H" in b and "O" in b for b in m1)

    def test_edge_criteria_regex_and_predicate(self):
        """[T-match-03] Edge criteria support regex and predicate factories/funcs."""
        from ggnes.generation.matching import find_subgraph_matches

        # Register a simple predicate: weight between 0.6 and 0.8 (inclusive)
        @PredicateRegistry.register('between_inclusive')
        def between_inclusive(value, low: float = 0.6, high: float = 0.8):
            return isinstance(value, int | float) and low <= float(value) <= high

        graph, (n1, n2, n3) = build_linear_graph()
        # Add label to node to test regex
        graph.nodes[n2].attributes['name'] = 'hidden_12'

        lhs = {
            "nodes": [
                {"label": "I", "match_criteria": {"node_type": NodeType.INPUT}},
                {
                    "label": "H",
                    "match_criteria": {
                        "node_type": NodeType.HIDDEN,
                        "name_regex": re.compile(r"hidden_\d+")
                    }
                },
                {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
            ],
            "edges": [
                {
                    "source_label": "H",
                    "target_label": "O",
                    "edge_label": "E",
                    "match_criteria": {
                        "enabled": True,
                        "weight_predicate": (
                            "between_inclusive",
                            {"low": 0.6, "high": 0.8}
                        )
                    }
                }
            ],
        }

        matches = list(find_subgraph_matches(graph, lhs, timeout_ms=100))
        assert matches, "Expected at least one match"
        # Verify predicate and regex constraints held
        for b in matches:
            edge = b.get("E")
            assert edge is not None
            assert 0.6 <= edge.weight <= 0.8
            name_val = graph.nodes[b["H"]].attributes.get('name', '')
            assert re.match(r"hidden_\d+", name_val)

    def test_timeout_respected_partial_results_ok(self):
        """[T-match-04] Timeout should be respected and may return partial results."""
        from ggnes.generation.matching import find_subgraph_matches

        graph, _ = build_linear_graph()

        # Create a pattern that would normally have very many combinations
        lhs = {
            "nodes": [
                {"label": "N0", "match_criteria": {}},
                {"label": "N1", "match_criteria": {}},
                {"label": "N2", "match_criteria": {}},
            ],
            "edges": []
        }

        start = time.time()
        matches = list(find_subgraph_matches(graph, lhs, timeout_ms=1))
        elapsed_ms = (time.time() - start) * 1000.0
        # generous upper bound accounting for env variability
        assert elapsed_ms <= 50.0

        assert isinstance(matches, list)

    def test_edge_labels_added_to_bindings(self):
        """[T-match-05] Edge labels must be included in the bindings when provided."""
        from ggnes.generation.matching import find_subgraph_matches

        graph, (n1, n2, n3) = build_linear_graph()
        lhs = {
            "nodes": [
                {"label": "H", "match_criteria": {"node_type": NodeType.HIDDEN}},
                {"label": "O", "match_criteria": {"node_type": NodeType.OUTPUT}},
            ],
            "edges": [
                {
                    "source_label": "H",
                    "target_label": "O",
                    "edge_label": "E",
                    "match_criteria": {"enabled": True}
                },
            ],
        }

        matches = list(find_subgraph_matches(graph, lhs, timeout_ms=100))
        assert matches
        assert any("E" in b for b in matches)
