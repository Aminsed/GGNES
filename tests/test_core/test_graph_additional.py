"""
Additional tests for Graph class to ensure complete adherence to project_guide.md.
"""

import uuid

import pytest

from ggnes.core import Edge, Graph, Node
from ggnes.core.graph import NodeType
from ggnes.utils.validation import EdgeError, NodeError


class TestGraphIDStrategies:
    """Tests for different ID strategies as per project_guide.md."""

    def test_local_only_strategy(self):
        """Test LOCAL_ONLY ID strategy uses only integers."""
        graph = Graph(config={'id_strategy': 'LOCAL_ONLY'})

        # Add nodes
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

        # Check node IDs are integers
        assert isinstance(n1, int)
        assert isinstance(n2, int)
        assert n1 == 0
        assert n2 == 1

        # Check nodes don't have global_id
        assert graph.nodes[n1].global_id is None
        assert graph.nodes[n2].global_id is None

        # Add edge
        edge_id = graph.add_edge(n1, n2, {"weight": 0.5})

        # Check edge ID is integer
        assert isinstance(edge_id, int)
        assert edge_id == 0

        # Check edge doesn't have local_edge_id (only used in HYBRID)
        edge = graph.find_edge_by_id(edge_id)
        assert edge.local_edge_id is None

    def test_hybrid_strategy(self):
        """Test HYBRID ID strategy uses both local and global IDs."""
        graph = Graph(config={'id_strategy': 'HYBRID'})

        # Add nodes
        n1 = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })
        n2 = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Check node IDs are integers
        assert isinstance(n1, int)
        assert isinstance(n2, int)

        # Check nodes have global_id
        assert isinstance(graph.nodes[n1].global_id, uuid.UUID)
        assert isinstance(graph.nodes[n2].global_id, uuid.UUID)

        # Add edge
        edge_id = graph.add_edge(n1, n2, {"weight": 0.5})

        # Check edge ID is UUID
        assert isinstance(edge_id, uuid.UUID)

        # Check edge has local_edge_id
        edge = graph.find_edge_by_id(edge_id)
        assert isinstance(edge.local_edge_id, int)
        assert edge.local_edge_id == 0

    def test_global_only_strategy(self):
        """Test GLOBAL_ONLY ID strategy uses UUIDs."""
        graph = Graph(config={'id_strategy': 'GLOBAL_ONLY'})

        # Add nodes
        n1 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "tanh",
            "attributes": {"output_size": 15}
        })

        # Node ID is still int internally
        assert isinstance(n1, int)

        # But node has global_id
        assert isinstance(graph.nodes[n1].global_id, uuid.UUID)

        # Add edge
        n2 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "elu",
            "attributes": {"output_size": 15}
        })
        edge_id = graph.add_edge(n1, n2, {"weight": 0.3})

        # Edge ID is UUID
        assert isinstance(edge_id, uuid.UUID)

        # No local_edge_id in GLOBAL_ONLY
        edge = graph.find_edge_by_id(edge_id)
        assert edge.local_edge_id is None


class TestGraphNodeManagement:
    """Tests for node management methods."""

    def test_remove_node_updates_tracking_lists(self):
        """Test that removing nodes updates input/output tracking lists."""
        graph = Graph()

        # Add input, hidden, output nodes
        n_in = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })
        graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 20}
        })
        n_out = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "softmax",
            "attributes": {"output_size": 5}
        })

        # Verify tracking lists
        assert n_in in graph.input_node_ids
        assert n_out in graph.output_node_ids
        assert len(graph.input_node_ids) == 1
        assert len(graph.output_node_ids) == 1

        # Remove nodes
        graph.remove_node(n_in)
        graph.remove_node(n_out)

        # Verify removed from tracking lists
        assert n_in not in graph.input_node_ids
        assert n_out not in graph.output_node_ids
        assert len(graph.input_node_ids) == 0
        assert len(graph.output_node_ids) == 0

    def test_remove_node_removes_all_edges(self):
        """Test that removing a node removes all its edges."""
        graph = Graph()

        # Create a small network
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
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Add edges
        graph.add_edge(n1, n2, {"weight": 0.5})
        graph.add_edge(n2, n3, {"weight": 0.7})
        graph.add_edge(n1, n3, {"weight": 0.3})  # Skip connection

        # Verify edges exist
        assert n2 in graph.nodes[n1].edges_out
        assert n3 in graph.nodes[n2].edges_out
        assert n1 in graph.nodes[n2].edges_in
        assert n2 in graph.nodes[n3].edges_in

        # Remove middle node
        graph.remove_node(n2)

        # Verify node is gone
        assert n2 not in graph.nodes

        # Verify edges are cleaned up
        assert n2 not in graph.nodes[n1].edges_out
        assert n2 not in graph.nodes[n3].edges_in

        # Skip connection should remain
        assert n3 in graph.nodes[n1].edges_out
        assert n1 in graph.nodes[n3].edges_in


class TestGraphValidationAdditional:
    """Additional validation tests per project_guide.md."""

    def test_validate_non_finite_bias_error(self):
        """Test validation detects non-finite bias values."""
        graph = Graph()

        # Create node with valid values first
        n1 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "bias": 0.5,
            "attributes": {"output_size": 10}
        })

        # Manually set non-finite bias
        graph.nodes[n1].bias = float('inf')

        errors = []
        is_valid = graph.validate(collect_errors=errors)

        assert not is_valid

        # Find the non-finite bias error
        bias_error = next((e for e in errors if e.error_type == 'non_finite_bias'), None)
        assert bias_error is not None
        assert isinstance(bias_error, NodeError)
        assert bias_error.node_id == n1
        assert bias_error.details['bias'] == float('inf')

    def test_validate_dangling_edges(self):
        """Test validation detects dangling edges."""
        graph = Graph()

        # Add nodes
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

        # Add edge
        edge = Edge(
            edge_id=0,
            source_node_id=n1,
            target_node_id=n2,
            weight=0.5
        )
        graph.nodes[n1].edges_out[n2] = edge
        graph.nodes[n2].edges_in[n1] = edge

        # Manually create dangling edge by adding fake edge
        fake_edge = Edge(
            edge_id=1,
            source_node_id=999,  # Non-existent node
            target_node_id=n2,
            weight=0.5
        )
        graph.nodes[n2].edges_in[999] = fake_edge

        errors = []
        is_valid = graph.validate(collect_errors=errors)

        assert not is_valid

        # Find dangling edge error
        dangling_error = next((e for e in errors if e.error_type == 'dangling_edge'), None)
        assert dangling_error is not None
        assert isinstance(dangling_error, EdgeError)
        assert "999" in dangling_error.message

    def test_validate_warnings_for_output_nodes(self):
        """Test validation produces warnings (not errors) for output nodes
        without incoming edges."""
        graph = Graph()

        # Add input node
        n_in = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })

        # Add output node
        n_out = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Connect them so output IS reachable from input (no unreachable_output error)
        graph.add_edge(n_in, n_out, {"weight": 0.5})

        # Add another output node without incoming edges
        n_out2 = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "softmax",
            "attributes": {"output_size": 10}
        })

        # Connect it from input so it's reachable
        graph.add_edge(n_in, n_out2, {"weight": 0.3})

        # Now remove the edge to n_out2 to create the warning condition
        edge = graph.find_edge_by_endpoints(n_in, n_out2)
        graph.remove_edge(edge.edge_id)

        errors = []
        warnings = []
        is_valid = graph.validate(collect_errors=errors, collect_warnings=warnings)

        # Should NOT be valid because n_out2 is now unreachable
        assert not is_valid

        # Should have unreachable_output error
        unreachable_error = next((e for e in errors if e.error_type == 'unreachable_output'), None)
        assert unreachable_error is not None
        assert unreachable_error.node_id == n_out2

        # Should also have warning about no incoming edges
        warning = next((w for w in warnings if w.error_type == 'no_incoming'), None)
        assert warning is not None
        assert warning.node_id == n_out2


class TestGraphMiscellaneous:
    """Tests for miscellaneous Graph methods."""

    def test_reset_id_counter_with_empty_graph(self):
        """Test reset_id_counter with empty graph."""
        graph = Graph()

        # Add and remove some nodes to advance counter
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

        assert graph._node_id_counter == 2

        # Remove all nodes
        graph.remove_node(n1)
        graph.remove_node(n2)

        # Reset counter
        graph.reset_id_counter()

        # Should reset to 0 for empty graph
        assert graph._node_id_counter == 0

    def test_reset_id_counter_with_nodes(self):
        """Test reset_id_counter with existing nodes."""
        graph = Graph()

        # Manually set node IDs to test reset
        graph._node_id_counter = 10

        # Add nodes with manual IDs
        for i in [5, 7, 15, 3]:
            node = Node(
                node_id=i,
                node_type=NodeType.HIDDEN,
                activation_function="relu",
                attributes={"output_size": 10}
            )
            graph.nodes[i] = node

        # Reset counter
        graph.reset_id_counter()

        # Should be max ID + 1
        assert graph._node_id_counter == 16

    def test_reset_id_counter_with_start_at(self):
        """Test reset_id_counter with explicit start_at value."""
        graph = Graph()

        # Reset to specific value
        graph.reset_id_counter(start_at=100)
        assert graph._node_id_counter == 100

        # Add node should use this counter
        n1 = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })

        assert n1 == 100
        assert graph._node_id_counter == 101

    def test_find_edge_by_endpoints(self):
        """Test finding edges by their endpoints."""
        graph = Graph()

        n1 = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })
        n2 = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Add edge
        edge_id = graph.add_edge(n1, n2, {"weight": 0.5, "attributes": {"custom": "data"}})

        # Find by endpoints
        edge = graph.find_edge_by_endpoints(n1, n2)
        assert edge is not None
        assert edge.edge_id == edge_id
        assert edge.weight == 0.5
        assert edge.attributes["custom"] == "data"

        # Non-existent edge
        edge = graph.find_edge_by_endpoints(n2, n1)  # Reverse direction
        assert edge is None

    def test_graph_has_unique_id(self):
        """Test that each graph instance has a unique graph_id."""
        graph1 = Graph()
        graph2 = Graph()

        assert isinstance(graph1.graph_id, uuid.UUID)
        assert isinstance(graph2.graph_id, uuid.UUID)
        assert graph1.graph_id != graph2.graph_id


class TestNodeAdditional:
    """Additional tests for Node class."""

    def test_node_get_degree(self):
        """Test Node.get_degree() method."""
        graph = Graph()

        # Create small network
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
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Add edges
        graph.add_edge(n1, n2)
        graph.add_edge(n2, n3)
        graph.add_edge(n1, n3)

        # Test degrees
        assert graph.nodes[n1].get_degree() == 2  # 0 in, 2 out
        assert graph.nodes[n2].get_degree() == 2  # 1 in, 1 out
        assert graph.nodes[n3].get_degree() == 2  # 2 in, 0 out

    def test_node_set_attribute(self):
        """Test Node.set_attribute() method."""
        node = Node(
            node_id=0,
            node_type=NodeType.HIDDEN,
            activation_function="relu",
            attributes={"output_size": 10}
        )

        # Set regular attribute
        node.set_attribute("custom_key", "custom_value")
        assert node.attributes["custom_key"] == "custom_value"

        # Set output_size through set_attribute (should validate)
        node.set_attribute("output_size", 32)
        assert node.output_size == 32

        # Invalid output_size through set_attribute
        with pytest.raises(ValueError):
            node.set_attribute("output_size", -5)


class TestEdgeAdditional:
    """Additional tests for Edge class."""

    def test_edge_attributes(self):
        """Test Edge attributes handling."""
        edge = Edge(
            edge_id=0,
            source_node_id=1,
            target_node_id=2,
            weight=0.5,
            enabled=True,
            attributes={"is_recurrent": True, "custom": "data"}
        )

        assert edge.attributes["is_recurrent"] is True
        assert edge.attributes["custom"] == "data"
        assert edge.enabled is True
        assert edge.weight == 0.5
