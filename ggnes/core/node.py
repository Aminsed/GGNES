"""Node class and related functionality."""

from enum import Enum
from uuid import UUID


class NodeType(Enum):
    """Types of nodes in the neural network graph."""
    INPUT = "INPUT"
    HIDDEN = "HIDDEN"
    OUTPUT = "OUTPUT"


class Node:
    """Neural network node representation.

    All nodes MUST have 'output_size' in attributes.
    """

    def __init__(self, node_id: int, node_type: NodeType, activation_function: str,
                 bias: float = 0.0, attributes: dict = None, global_id: UUID = None):
        self.node_id = node_id  # MUST be int (local ID)
        self.node_type = node_type  # MUST be NodeType enum
        self.activation_function = activation_function
        self.bias = bias
        self.attributes = attributes or {}
        self.global_id = global_id
        self.edges_in = {}  # {source_node_id: Edge}
        self.edges_out = {}  # {target_node_id: Edge}

        # Validate required attributes
        output_size = self.attributes.get('output_size')
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError(f"output_size must be positive int, got {output_size}")

    def get_degree(self) -> int:
        """Return total degree (in + out)."""
        return len(self.edges_in) + len(self.edges_out)

    @property
    def output_size(self) -> int:
        """Get output size with validation."""
        return self.attributes.get('output_size', 0)

    @output_size.setter
    def output_size(self, value: int):
        """Set output size with validation."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"output_size must be positive int, got {value}")
        self.attributes['output_size'] = value

    def set_attribute(self, key: str, value):
        """Safely set attribute with validation."""
        if key == 'output_size':
            self.output_size = value
        else:
            self.attributes[key] = value
