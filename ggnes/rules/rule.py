"""Rule, LHSPattern, RHSAction, and EmbeddingLogic classes."""

from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from typing import Any
from uuid import UUID


class EmbeddingStrategy(Enum):
    """Embedding strategy for boundary node reconnection."""
    MAP_BOUNDARY_CONNECTIONS = "MAP_BOUNDARY_CONNECTIONS"


class Direction(Enum):
    """Direction relative to a boundary node for embedding reconnection."""
    IN = "IN"
    OUT = "OUT"


class Distribution(Enum):
    """Distribution policy for mapping boundary connections."""
    COPY_ALL = "COPY_ALL"
    CONNECT_SINGLE = "CONNECT_SINGLE"


class LHSPattern:
    """Left-hand side pattern for matching.

    Defines the subgraph pattern to match in the host graph.
    """

    def __init__(self, nodes: list[dict], edges: list[dict], boundary_nodes: list[str]):
        """Initialize LHS pattern.

        Args:
            nodes: List of node specs with 'label' and 'match_criteria'
            edges: List of edge specs with 'source_label', 'target_label',
                   'match_criteria', and optional 'edge_label'
            boundary_nodes: List of node labels that are boundary nodes
        """
        self.nodes = nodes  # List of {'label': str, 'match_criteria': dict}
        self.edges = edges  # List of edge specs
        self.boundary_nodes = boundary_nodes  # List of node labels


class RHSAction:
    """Right-hand side actions to apply.

    Defines the modifications to make to the matched subgraph.
    """

    def __init__(
        self,
        add_nodes: list[dict] | None = None,
        add_edges: list[dict] | None = None,
        delete_nodes: list[str] | None = None,
        delete_edges: list[dict] | None = None,
        modify_nodes: list[dict] | None = None,
        modify_edges: list[dict] | None = None
    ):
        """Initialize RHS action.

        Args:
            add_nodes: List of nodes to add with 'label' and 'properties'
            add_edges: List of edges to add with 'source_label', 'target_label', and 'properties'
            delete_nodes: List of node labels to delete
            delete_edges: List of edge specs to delete
            modify_nodes: List of node modifications with 'label' and 'properties'
            modify_edges: List of edge modifications
        """
        self.add_nodes = add_nodes or []  # List of {'label': str, 'properties': dict}
        self.add_edges = add_edges or []  # List of edges to add
        self.delete_nodes = delete_nodes or []  # List of node labels
        self.delete_edges = delete_edges or []  # List of edge specs
        self.modify_nodes = modify_nodes or []  # List of {'label': str, 'properties': dict}
        self.modify_edges = modify_edges or []  # List of edge modification specs


class EmbeddingLogic:
    """Controls reconnection when boundary nodes are deleted.

    Specifies how to reconnect edges when boundary nodes are removed.
    """

    def __init__(
        self,
        strategy: EmbeddingStrategy = EmbeddingStrategy.MAP_BOUNDARY_CONNECTIONS,
        connection_map: dict | None = None,
        excess_connection_handling: str = 'WARNING',
        unknown_direction_handling: str = 'WARNING',
        boundary_handling: str = 'PROCESS_LAST'
    ):
        """Initialize embedding logic.

        Args:
            strategy: Must be MAP_BOUNDARY_CONNECTIONS
            connection_map: OrderedDict mapping (boundary_label, direction) to
                          list of (rhs_label, distribution) tuples
            excess_connection_handling: How to handle excess connections (DROP/WARNING/ERROR)
            unknown_direction_handling: How to handle unknown directions (DROP/WARNING/ERROR)
            boundary_handling: When to process boundary nodes (PROCESS_FIRST/PROCESS_LAST/IGNORE)
        """
        self.strategy = strategy  # MUST be MAP_BOUNDARY_CONNECTIONS
        self.connection_map = OrderedDict(connection_map or {})

        self.excess_connection_handling = excess_connection_handling  # DROP/WARNING/ERROR
        self.unknown_direction_handling = unknown_direction_handling  # DROP/WARNING/ERROR
        self.boundary_handling = boundary_handling  # PROCESS_FIRST/PROCESS_LAST/IGNORE


class Rule:
    """A graph rewriting rule.

    Encapsulates a complete graph transformation rule with pattern,
    actions, embedding logic, and optional conditions.
    """

    def __init__(
        self,
        rule_id: UUID,
        lhs: LHSPattern,
        rhs: RHSAction,
        embedding: EmbeddingLogic,
        metadata: dict[str, Any] | None = None,
        condition: Callable | None = None
    ):
        """Initialize rule.

        Args:
            rule_id: Unique identifier for the rule
            lhs: Left-hand side pattern to match
            rhs: Right-hand side actions to apply
            embedding: Logic for reconnecting edges when deleting boundary nodes
            metadata: Optional metadata (e.g., priority, probability)
            condition: Optional callable(graph_view, bindings, graph_context) -> bool
        """
        self.rule_id = rule_id
        self.lhs = lhs
        self.rhs = rhs
        self.embedding = embedding
        self.metadata = metadata or {}
        self.condition = condition  # Optional callable(graph_view, bindings, graph_context) -> bool


def validate_connection_map(
    connection_map: dict,
    lhs: LHSPattern,
    rhs: RHSAction
) -> None:
    """Validate that connection_map references valid labels.

    MUST fail on non-existent labels.
    Duplicates: first wins, others ignored with warning.

    This validation SHOULD be called:
    1. At Rule construction time (recommended)
    2. During rule deserialization
    3. Before rule application (as a safety check)

    Args:
        connection_map: The connection map to validate
        lhs: LHS pattern containing boundary nodes
        rhs: RHS action containing added nodes

    Raises:
        ValueError: If invalid labels are referenced
    """
    # Check all boundary labels exist in LHS
    for (boundary_label, direction), targets in connection_map.items():
        if boundary_label not in lhs.boundary_nodes:
            raise ValueError(
                f"Connection map references non-existent boundary node: {boundary_label}"
            )

        # Check all RHS labels exist
        rhs_labels = {node['label'] for node in rhs.add_nodes}
        for rhs_label, dist in targets:
            if rhs_label not in rhs_labels:
                raise ValueError(
                    f"Connection map references non-existent RHS node: {rhs_label}"
                )
