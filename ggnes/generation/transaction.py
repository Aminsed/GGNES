"""Transaction management for staged graph modifications.

Implements a simple TransactionManager with a ChangeBuffer that:
- Stages node/edge additions and node deletions
- On begin(), snapshots RNGManager state
- On commit(), validates staged operations, applies them to the Graph, and
  registers created entities with IDManager (if provided), returning a mapping
  of temp handles to final IDs
- On rollback(), clears staged operations and restores RNGManager state

Per project_guide.md M6.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any


class ChangeBuffer:
    """In-memory buffer for staging graph mutations prior to commit."""

    def __init__(self) -> None:
        # temp handles → node properties
        self._temp_nodes: dict[str, dict[str, Any]] = {}
        # edges: list of (src_handle_or_id, dst_handle_or_id, properties)
        self._temp_edges: list[tuple[Any, Any, dict[str, Any]]] = []
        # nodes (final id or temp) staged for deletion
        self._delete_nodes: set[Any] = set()
        # edges (by id) staged for deletion
        self._delete_edges: set[Any] = set()

    def reset(self) -> None:
        self._temp_nodes.clear()
        self._temp_edges.clear()
        self._delete_nodes.clear()
        self._delete_edges.clear()

    def add_node(self, properties: dict[str, Any]) -> str:
        handle = f"tmp:{uuid.uuid4()}"
        self._temp_nodes[handle] = properties
        return handle

    def add_edge(self, source: Any, target: Any, properties: dict[str, Any]) -> None:
        self._temp_edges.append((source, target, properties))

    def delete_node(self, node: Any) -> None:
        self._delete_nodes.add(node)

    def delete_edge(self, edge_id: Any) -> None:
        self._delete_edges.add(edge_id)


@dataclass
class TransactionManager:
    graph: Any
    rng_manager: Any
    id_manager: Any | None = None
    context_id: str = "default"
    buffer: ChangeBuffer = field(default_factory=ChangeBuffer)

    _rng_state_snapshot: bytes | None = None

    def begin(self) -> None:
        """Begin a transaction by snapshotting RNG state and clearing buffer."""
        if self.rng_manager is not None:
            self._rng_state_snapshot = self.rng_manager.get_state()
        self.buffer.reset()
        # When deterministic UUIDs are enabled, set UUID context so new IDs
        # include transaction information if provided by caller via graph.set_uuid_context()

    # Validation helpers
    def _resolve_node_ref(self, ref: Any, temp_to_real: dict[str, int]) -> int | None:
        if isinstance(ref, str) and ref.startswith("tmp:"):
            return temp_to_real.get(ref)
        if isinstance(ref, int):
            return ref
        return None

    def _validate(self) -> None:
        # If any edge touches a node staged for deletion, fail
        delete_set = set(self.buffer._delete_nodes)
        for src, dst, _ in self.buffer._temp_edges:
            if src in delete_set or dst in delete_set:
                raise ValueError("Cannot add edge to node staged for deletion")

        # Detect duplicate staged edges (temp-temp)
        seen = set()
        for src, dst, _ in self.buffer._temp_edges:
            key = (src, dst)
            if key in seen:
                # Allow detection; logging happens in commit
                continue
            seen.add(key)

    def rollback(self) -> None:
        """Discard staged operations and restore RNG state."""
        self.buffer.reset()
        if self._rng_state_snapshot is not None and self.rng_manager is not None:
            self.rng_manager.set_state(self._rng_state_snapshot)
            self._rng_state_snapshot = None

    def commit(self) -> dict[str, int]:
        """Validate and apply staged operations. Returns temp→real node id mapping."""
        self._validate()

        temp_to_real: dict[str, int] = {}

        # Apply deletions first (for existing ids only)
        for e_id in list(self.buffer._delete_edges):
            self.graph.remove_edge(e_id)
        for ref in list(self.buffer._delete_nodes):
            if isinstance(ref, int):
                # Remove node if exists
                self.graph.remove_node(ref)

        # Create nodes
        for handle, props in self.buffer._temp_nodes.items():
            node_id = self.graph.add_node(props)
            temp_to_real[handle] = node_id
            if self.id_manager is not None:
                try:
                    node_obj = self.graph.nodes[node_id]
                    self.id_manager.register_node(node_obj, self.context_id)
                except Exception:  # pragma: no cover - defensive
                    pass

        # Create edges
        # Resolve source/target referencing temp handles or real ids
        for src_ref, dst_ref, props in self.buffer._temp_edges:
            # Duplicate edge detection against existing graph
            src_real = self._resolve_node_ref(src_ref, temp_to_real)
            dst_real = self._resolve_node_ref(dst_ref, temp_to_real)
            if src_real is None or dst_real is None:
                raise ValueError("Invalid edge endpoint reference")

            existing = self.graph.find_edge_by_endpoints(src_real, dst_real)
            if existing is not None and not self.graph.config.get('multigraph'):
                logging.warning("Duplicate edge attempt detected during commit")
                continue

            edge_id = self.graph.add_edge(src_real, dst_real, props)
            if edge_id is None:
                # Graph rejected duplicate (simple-graph constraint)
                logging.warning("Duplicate edge attempt detected during commit")
                continue

            if self.id_manager is not None:
                try:
                    edge_obj = self.graph.find_edge_by_endpoints(src_real, dst_real)
                    if edge_obj is not None:
                        self.id_manager.register_edge(edge_obj, self.context_id)
                except Exception:  # pragma: no cover - defensive
                    pass

        # Done - clear buffer and RNG snapshot remains as current state
        self.buffer.reset()
        return temp_to_real
