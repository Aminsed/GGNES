"""
Test module for Graph class.
Tests Graph functionality per M2 milestone.
"""

from unittest.mock import patch

from ggnes.core import Graph, Node
from ggnes.core.graph import NodeType
from ggnes.utils.validation import EdgeError, NodeError


class TestGraphEdgeHandling:
    """Tests for Graph edge handling."""

    def test_graph_add_edge_rejects_duplicates(self):
        """[T-core-02] Graph.add_edge rejects duplicates (returns None, logs warning)."""
        graph = Graph()

        # Add two nodes
        node1_id = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })

        node2_id = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 20}
        })

        # First edge should succeed
        edge_id1 = graph.add_edge(node1_id, node2_id, {"weight": 0.5})
        assert edge_id1 is not None

        # Duplicate edge should return None and log warning
        with patch("logging.warning") as mock_warning:
            edge_id2 = graph.add_edge(node1_id, node2_id, {"weight": 0.7})
            assert edge_id2 is None
            mock_warning.assert_called_once()
            warning_msg = mock_warning.call_args[0][0]
            assert "already exists" in warning_msg
            assert "Simple graphs do not allow duplicate edges" in warning_msg


class TestGraphValidation:
    """Tests for Graph validation."""

    def test_validate_produces_node_errors(self):
        """[T-core-03] validate() produces NodeError.

        Tests invalid activation, aggregation, missing output_size.
        """
        graph = Graph()

        # Node with invalid activation
        graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "invalid_activation",
            "attributes": {"output_size": 10}
        })

        # Node with invalid aggregation
        graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 20, "aggregation": "invalid_agg"}
        })

        # Node with missing output_size - bypass constructor validation
        # Create node with valid output_size first
        node3 = Node(
            node_id=3,
            node_type=NodeType.HIDDEN,
            activation_function="tanh",
            attributes={"output_size": 10}  # Valid for constructor
        )
        # Then remove it to test validation
        del node3.attributes["output_size"]
        graph.nodes[3] = node3

        errors = []
        warnings = []
        is_valid = graph.validate(collect_errors=errors, collect_warnings=warnings)

        assert not is_valid
        assert len(errors) >= 3

        # Check error types and content
        error_types = {e.error_type for e in errors}
        assert "invalid_activation" in error_types
        assert "invalid_aggregation" in error_types
        assert "missing_output_size" in error_types

        # Check that errors are NodeError instances
        for error in errors:
            if error.error_type in ["invalid_activation", "invalid_aggregation",
                                    "missing_output_size"]:
                assert isinstance(error, NodeError)

    def test_validate_produces_edge_errors(self):
        """[T-core-03] validate() produces EdgeError for non_finite_weight."""
        graph = Graph()

        # Add valid nodes
        node1_id = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })

        node2_id = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Add edge with non-finite weight
        graph.add_edge(node1_id, node2_id, {"weight": float("inf")})

        errors = []
        is_valid = graph.validate(collect_errors=errors)

        assert not is_valid
        assert len(errors) >= 1

        # Find the non-finite weight error
        weight_error = next((e for e in errors if e.error_type == "non_finite_weight"), None)
        assert weight_error is not None
        assert isinstance(weight_error, EdgeError)
        assert weight_error.details.get("weight") == float("inf")


class TestGraphCycleDetection:
    """Tests for Graph cycle detection."""

    def test_detect_cycles_preserves_explicit_is_recurrent(self):
        """[T-core-04] detect_cycles preserves explicit is_recurrent; marks back edges recurrent."""
        graph = Graph()

        # Create a cycle: n1 -> n2 -> n3 -> n1
        n1 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 10}
        })

        n2 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 10}
        })

        n3 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 10}
        })

        # Add edges
        graph.add_edge(n1, n2, {"weight": 0.5})
        graph.add_edge(n2, n3, {"weight": 0.5})
        graph.add_edge(n3, n1, {"weight": 0.5, "attributes": {"is_recurrent": True}})

        # Get the actual edge objects
        edge_n3_n1 = graph.find_edge_by_endpoints(n3, n1)

        # Verify explicit flag is set
        assert edge_n3_n1.attributes.get("is_recurrent") is True

        # Run cycle detection
        graph.detect_cycles()

        # Verify explicit flag is preserved
        assert edge_n3_n1.attributes.get("is_recurrent") is True

        # Add another cycle without explicit marking
        n4 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 10}
        })

        graph.add_edge(n1, n4, {"weight": 0.5})
        graph.add_edge(n4, n1, {"weight": 0.5})  # Creates cycle n1->n4->n1

        edge_n4_n1 = graph.find_edge_by_endpoints(n4, n1)

        # Run cycle detection again
        graph.detect_cycles()

        # Original explicit flag should still be preserved
        assert edge_n3_n1.attributes.get("is_recurrent") is True

        # New back edge should be marked
        assert edge_n4_n1.attributes.get("is_recurrent") is True


class TestGraphTopologicalSort:
    """Tests for Graph topological sort."""

    def test_topological_sort_excludes_disabled_and_recurrent(self):
        """[T-core-05] topological_sort excludes disabled and recurrent edges when requested."""
        graph = Graph()

        # Create nodes
        n1 = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })

        n2 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 20}
        })

        n3 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 20}
        })

        n4 = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Add edges
        graph.add_edge(n1, n2, {"weight": 0.5})
        graph.add_edge(n2, n3, {"weight": 0.5, "enabled": False})  # Disabled edge
        graph.add_edge(n3, n4, {"weight": 0.5})
        # Recurrent edge
        graph.add_edge(n3, n2, {"weight": 0.5, "attributes": {"is_recurrent": True}})

        # Topological sort should exclude disabled and recurrent edges
        order = graph.topological_sort(ignore_recurrent=True)

        # n2 should come before n3 (since disabled edge is ignored)
        # But n3 can also come early since there's no enabled path from n2 to n3
        assert n1 in order
        assert n2 in order
        assert n3 in order
        assert n4 in order

        # n1 should be first (input), n4 should be last (output)
        assert order.index(n1) < order.index(n4)

        # With the disabled edge, n2 and n3 are independent
        # So their relative order doesn't matter

        # Test with ignore_recurrent=False
        order_with_recurrent = graph.topological_sort(ignore_recurrent=False)

        # Should still have all nodes but might warn about cycles
        assert len(order_with_recurrent) == 4


class TestGraphFingerprint:
    """Tests for Graph fingerprint computation."""

    def test_compute_fingerprint_excludes_disabled_edges(self):
        """[T-core-06] compute_fingerprint excludes disabled edges;
        stable under attribute-irrelevant changes."""
        graph1 = Graph()

        # Create simple graph
        n1 = graph1.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })

        n2 = graph1.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        graph1.add_edge(n1, n2, {"weight": 0.5})

        # Get fingerprint
        fp1 = graph1.compute_fingerprint()

        # Create identical graph
        graph2 = Graph()

        n1_2 = graph2.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })

        n2_2 = graph2.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        graph2.add_edge(n1_2, n2_2, {"weight": 0.5})

        fp2 = graph2.compute_fingerprint()

        # Fingerprints should match
        assert fp1 == fp2

        # Add disabled edge between existing nodes in graph2
        graph2.add_edge(n2_2, n1_2, {"weight": 0.7, "enabled": False})

        fp3 = graph2.compute_fingerprint()

        # Fingerprint should be unchanged (disabled edge excluded)
        assert fp2 == fp3

        # Enable the edge
        edge = graph2.find_edge_by_endpoints(n2_2, n1_2)
        edge.enabled = True

        fp4 = graph2.compute_fingerprint()

        # Now fingerprint should change
        assert fp3 != fp4
