"""
Test module for Node class.
Tests Node validation and behavior per M2 milestone.
"""

import pytest

from ggnes.core.node import Node, NodeType


class TestNode:
    """Tests for Node class."""

    def test_node_rejects_invalid_output_size_zero(self):
        """[T-core-01] Node rejects output_size <= 0."""
        with pytest.raises(ValueError, match="output_size must be positive int"):
            Node(
                node_id=0,
                node_type=NodeType.HIDDEN,
                activation_function="relu",
                attributes={"output_size": 0}
            )

    def test_node_rejects_invalid_output_size_negative(self):
        """[T-core-01] Node rejects negative output_size."""
        with pytest.raises(ValueError, match="output_size must be positive int"):
            Node(
                node_id=0,
                node_type=NodeType.HIDDEN,
                activation_function="relu",
                attributes={"output_size": -5}
            )

    def test_node_rejects_invalid_output_size_non_int(self):
        """[T-core-01] Node rejects non-int output_size."""
        with pytest.raises(ValueError, match="output_size must be positive int"):
            Node(
                node_id=0,
                node_type=NodeType.HIDDEN,
                activation_function="relu",
                attributes={"output_size": 3.14}
            )

    def test_node_rejects_missing_output_size(self):
        """[T-core-01] Node rejects missing output_size."""
        with pytest.raises(ValueError, match="output_size must be positive int"):
            Node(
                node_id=0,
                node_type=NodeType.HIDDEN,
                activation_function="relu",
                attributes={}
            )

    def test_node_accepts_valid_output_size(self):
        """[T-core-01] Node accepts valid positive int output_size."""
        node = Node(
            node_id=0,
            node_type=NodeType.HIDDEN,
            activation_function="relu",
            attributes={"output_size": 32}
        )
        assert node.output_size == 32
        assert node.attributes["output_size"] == 32

    def test_node_output_size_property_setter(self):
        """[T-core-01] Node output_size property setter validates."""
        node = Node(
            node_id=0,
            node_type=NodeType.HIDDEN,
            activation_function="relu",
            attributes={"output_size": 16}
        )

        # Valid change
        node.output_size = 64
        assert node.output_size == 64

        # Invalid change
        with pytest.raises(ValueError, match="output_size must be positive int"):
            node.output_size = 0

        with pytest.raises(ValueError, match="output_size must be positive int"):
            node.output_size = "invalid"
