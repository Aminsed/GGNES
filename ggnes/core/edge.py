"""Edge class and related functionality."""

from uuid import UUID


class Edge:
    """Directed edge between nodes."""

    def __init__(self, edge_id: int | UUID, source_node_id: int, target_node_id: int,
                 weight: float = 0.1, enabled: bool = True, attributes: dict = None):
        self.edge_id = edge_id  # int if LOCAL_ONLY, UUID otherwise
        self.source_node_id = source_node_id  # MUST be int
        self.target_node_id = target_node_id  # MUST be int
        self.weight = weight  # MUST be finite float
        self.enabled = enabled
        self.attributes = attributes or {}
        self.local_edge_id = None  # MAY be set in HYBRID mode
