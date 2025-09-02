"""Hierarchical embedding planner (project_guide.md §20.5).

This module plans reconnections for boundary nodes across module levels using
MAP_BOUNDARY_CONNECTIONS along with COPY_ALL, CONNECT_SINGLE, and numeric
distribution policies. It produces a deterministic plan that upstream code can
apply via transactions. The plan itself is a pure data structure for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ggnes.core.graph import Graph
from ggnes.rules.rule import Direction  # reuse Direction enum from rules


@dataclass(frozen=True)
class ExternalEdge:
    source_id: int
    target_id: int
    weight: float
    enabled: bool
    attributes: Dict[str, Any]


@dataclass(frozen=True)
class BoundaryInfo:
    node_id: int
    external_in: Tuple[ExternalEdge, ...]
    external_out: Tuple[ExternalEdge, ...]


@dataclass(frozen=True)
class PlanEdge:
    source_id: int
    target_id: int
    weight: float
    enabled: bool
    attributes: Dict[str, Any]


def _sorted_in(edges: Iterable[ExternalEdge]) -> List[ExternalEdge]:
    return sorted(edges, key=lambda e: (e.source_id, e.target_id, str(e.attributes.get("edge_id", ""))))


def _sorted_out(edges: Iterable[ExternalEdge]) -> List[ExternalEdge]:
    return sorted(edges, key=lambda e: (e.source_id, e.target_id, str(e.attributes.get("edge_id", ""))))


def plan_embedding(
    graph: Graph,
    connection_map: Mapping[Tuple[str, Direction], List[Tuple[str, Any]]],
    boundary_label_to_info: Mapping[str, BoundaryInfo],
    rhs_bindings: Mapping[str, int],
    *,
    excess_policy: str = "WARNING",
    unknown_policy: str = "WARNING",
    boundary_handling: str = "PROCESS_LAST",
) -> Tuple[List[PlanEdge], List[str]]:
    """Create a deterministic reconnection plan.

    Args:
        graph: Graph context (unused; reserved for multigraph variants)
        connection_map: mapping from (boundary_label, Direction) to list of
            tuples (rhs_label, distribution), where distribution is one of
            {"COPY_ALL", "CONNECT_SINGLE", int} where int selects exactly that
            many edges deterministically.
        boundary_label_to_info: collected external connections for each boundary
        rhs_bindings: mapping from RHS labels to concrete node IDs
        excess_policy: 'DROP' | 'WARNING' | 'ERROR'
        unknown_policy: 'DROP' | 'WARNING' | 'ERROR'

    Returns:
        (plan_edges, warnings) where plan_edges are edges to add.
    """
    warnings: List[str] = []
    plan: List[PlanEdge] = []

    def handle_unknown(msg: str) -> None:
        if unknown_policy == "ERROR":
            raise ValueError(msg)
        if unknown_policy == "WARNING":
            warnings.append(msg)

    def handle_excess(msg: str) -> None:
        if excess_policy == "ERROR":
            raise ValueError(msg)
        if excess_policy == "WARNING":
            warnings.append(msg)

    # Determine boundary processing order
    boundary_items = list(connection_map.items())
    if boundary_handling == "PROCESS_LAST":
        # reverse order for deterministic alternate strategy
        boundary_items = list(reversed(boundary_items))
    elif boundary_handling == "IGNORE":
        return [], []
    elif boundary_handling == "PROCESS_FIRST":
        # keep original order
        pass
    else:
        handle_unknown(f"Unknown boundary handling: {boundary_handling}")

    for (boundary_label, direction), targets in boundary_items:
        info = boundary_label_to_info.get(boundary_label)
        if info is None:
            handle_unknown(f"Unknown boundary label: {boundary_label}")
            continue

        if direction == Direction.IN:
            ext = _sorted_in(info.external_in)
            handled: set[Tuple[int, int]] = set()
            for rhs_label, distribution in targets:
                tgt = rhs_bindings.get(rhs_label)
                if tgt is None:
                    handle_unknown(f"Unknown RHS label: {rhs_label}")
                    continue
                if distribution == "COPY_ALL":
                    for e in ext:
                        plan.append(PlanEdge(e.source_id, tgt, e.weight, e.enabled, dict(e.attributes)))
                        handled.add((e.source_id, info.node_id))
                elif distribution == "CONNECT_SINGLE":
                    for e in ext:
                        key = (e.source_id, info.node_id)
                        if key not in handled:
                            plan.append(PlanEdge(e.source_id, tgt, e.weight, e.enabled, dict(e.attributes)))
                            handled.add(key)
                            break
                elif isinstance(distribution, int):
                    if distribution <= 0:
                        continue
                    count = 0
                    for e in ext:
                        key = (e.source_id, info.node_id)
                        if key not in handled:
                            plan.append(PlanEdge(e.source_id, tgt, e.weight, e.enabled, dict(e.attributes)))
                            handled.add(key)
                            count += 1
                            if count >= distribution:
                                break
                else:
                    handle_unknown(f"Unknown distribution: {distribution}")
            all_in = {(e.source_id, info.node_id) for e in ext}
            excess = all_in - handled
            if excess:
                handle_excess(f"Excess IN connections for {boundary_label}: {len(excess)}")

        elif direction == Direction.OUT:
            ext = _sorted_out(info.external_out)
            handled: set[Tuple[int, int]] = set()
            for rhs_label, distribution in targets:
                src = rhs_bindings.get(rhs_label)
                if src is None:
                    handle_unknown(f"Unknown RHS label: {rhs_label}")
                    continue
                if distribution == "COPY_ALL":
                    for e in ext:
                        plan.append(PlanEdge(src, e.target_id, e.weight, e.enabled, dict(e.attributes)))
                        handled.add((info.node_id, e.target_id))
                elif distribution == "CONNECT_SINGLE":
                    for e in ext:
                        key = (info.node_id, e.target_id)
                        if key not in handled:
                            plan.append(PlanEdge(src, e.target_id, e.weight, e.enabled, dict(e.attributes)))
                            handled.add(key)
                            break
                elif isinstance(distribution, int):
                    if distribution <= 0:
                        continue
                    count = 0
                    for e in ext:
                        key = (info.node_id, e.target_id)
                        if key not in handled:
                            plan.append(PlanEdge(src, e.target_id, e.weight, e.enabled, dict(e.attributes)))
                            handled.add(key)
                            count += 1
                            if count >= distribution:
                                break
                else:
                    handle_unknown(f"Unknown distribution: {distribution}")
            all_out = {(info.node_id, e.target_id) for e in ext}
            excess = all_out - handled
            if excess:
                handle_excess(f"Excess OUT connections for {boundary_label}: {len(excess)}")
        else:
            handle_unknown(f"Unknown direction: {direction}")

    return plan, warnings


