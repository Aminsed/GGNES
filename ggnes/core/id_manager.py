"""IDManager for managing ID mappings across different contexts."""

import uuid

from .graph import IDStrategy


class IDManager:
    """Manages ID mappings across different contexts.

    Supports three strategies:
    - LOCAL_ONLY: No global IDs or mappings
    - HYBRID: Both local (int) and global (UUID) IDs with bidirectional mapping
    - GLOBAL_ONLY: Only global IDs (UUID) with identity mapping
    """

    def __init__(self, id_strategy: IDStrategy = IDStrategy.HYBRID):
        """Initialize IDManager with specified strategy.

        Args:
            id_strategy: ID management strategy (defaults to HYBRID)
        """
        self.id_strategy = id_strategy
        self.global_to_local = {}  # {context_id: {global_id: local_id}}
        self.local_to_global = {}  # {context_id: {element_type: {local_id: global_id}}}

    def register_node(self, node, context_id: str) -> None:
        """Register a node's IDs in the appropriate mappings.

        Args:
            node: Node to register
            context_id: Context identifier
        """
        if self.id_strategy == IDStrategy.LOCAL_ONLY:
            return  # No global IDs to track

        if node.global_id is None:
            node.global_id = uuid.uuid4()

        # Initialize context if needed
        if context_id not in self.global_to_local:
            self.global_to_local[context_id] = {}
            self.local_to_global[context_id] = {'node': {}, 'edge': {}}

        # Register mappings
        self.global_to_local[context_id][node.global_id] = node.node_id
        self.local_to_global[context_id]['node'][node.node_id] = node.global_id

    def register_edge(self, edge, context_id: str) -> None:
        """Register an edge's IDs in the appropriate mappings.

        Args:
            edge: Edge to register
            context_id: Context identifier
        """
        if self.id_strategy == IDStrategy.LOCAL_ONLY:
            return  # No global IDs to track

        # Initialize context if needed
        if context_id not in self.global_to_local:
            self.global_to_local[context_id] = {}
            self.local_to_global[context_id] = {'node': {}, 'edge': {}}

        if self.id_strategy == IDStrategy.HYBRID:
            # HYBRID mode: edge has UUID edge_id and int local_edge_id
            if hasattr(edge, 'local_edge_id') and edge.local_edge_id is not None:
                self.global_to_local[context_id][edge.edge_id] = edge.local_edge_id
                self.local_to_global[context_id]['edge'][edge.local_edge_id] = edge.edge_id
        elif self.id_strategy == IDStrategy.GLOBAL_ONLY:
            # GLOBAL_ONLY mode: edge_id is UUID, identity mapping
            self.global_to_local[context_id][edge.edge_id] = edge.edge_id
            # No local->global mapping needed for GLOBAL_ONLY edges

    def get_local_node_id(self, global_id: uuid.UUID, context_id: str) -> int | None:
        """Get local node ID from global ID.

        Args:
            global_id: Global UUID
            context_id: Context identifier

        Returns:
            Local node ID or None if not found
        """
        if context_id not in self.global_to_local:
            return None
        return self.global_to_local[context_id].get(global_id)

    def get_global_node_id(self, local_id: int, context_id: str) -> uuid.UUID | None:
        """Get global node ID from local ID.

        Args:
            local_id: Local node ID
            context_id: Context identifier

        Returns:
            Global UUID or None if not found
        """
        if context_id not in self.local_to_global:
            return None
        return self.local_to_global[context_id].get('node', {}).get(local_id)

    def get_local_edge_id(
        self, global_id: uuid.UUID, context_id: str
    ) -> int | uuid.UUID | None:
        """Get local edge ID from global ID.

        Args:
            global_id: Global UUID
            context_id: Context identifier

        Returns:
            Local edge ID (int in HYBRID, UUID in GLOBAL_ONLY) or None
        """
        if context_id not in self.global_to_local:
            return None
        return self.global_to_local[context_id].get(global_id)

    def get_global_edge_id(
        self, local_id: int | uuid.UUID, context_id: str
    ) -> uuid.UUID | None:
        """Get global edge ID from local ID.

        Args:
            local_id: Local edge ID
            context_id: Context identifier

        Returns:
            Global UUID or None if not found
        """
        if context_id not in self.local_to_global:
            return None
        return self.local_to_global[context_id].get('edge', {}).get(local_id)
