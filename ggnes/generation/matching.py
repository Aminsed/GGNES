"""Subgraph matching algorithm.

Implements a simple, deterministic subgraph matcher that respects:
- Node match criteria (node_type, name_regex)
- Edge match criteria (enabled, weight_predicate)
- Optional edge labels added to bindings
- Timeout (returns partial results if exceeded)

The function returns an iterator of bindings (dicts) mapping LHS labels to
graph node_ids and optional edge labels to Edge instances.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from ggnes.rules.predicates import PredicateRegistry


def _node_matches_criteria(node: Any, criteria: dict[str, Any]) -> bool:
    """Check if a graph node matches provided criteria.

    Supported keys:
    - node_type: NodeType enum
    - name_regex: compiled regex matched against node.attributes['name'] (if present)
    - any other key is treated as attribute equality on node.attributes
    """
    if not criteria:
        return True

    # node_type constraint
    if 'node_type' in criteria:
        if node.node_type != criteria['node_type']:
            return False

    # name_regex constraint
    name_regex = criteria.get('name_regex')
    if name_regex is not None:
        name_value = node.attributes.get('name', '')
        if not getattr(name_regex, 'match', None):
            return False
        if not name_regex.match(str(name_value)):
            return False

    # Equality checks for remaining keys (exclude handled keys)
    for key, expected in criteria.items():
        if key in {'node_type', 'name_regex'}:
            continue
        if node.attributes.get(key) != expected:
            return False

    return True


def _edge_matches_criteria(edge: Any, criteria: dict[str, Any]) -> bool:
    """Check if an edge matches edge criteria.

    Supported keys:
    - enabled: bool
    - weight_predicate: (name, params_dict) referencing either a registered predicate
      (called as pred(value, **params)) or a factory (created then called as pred(value))
    - any other key is treated as attribute equality on edge.attributes
    """
    if not criteria:
        return True

    if 'enabled' in criteria and bool(edge.enabled) != bool(criteria['enabled']):
        return False

    if 'weight_predicate' in criteria:
        pred_spec = criteria['weight_predicate']
        if not isinstance(pred_spec, tuple) or len(pred_spec) != 2:
            return False
        pred_name, pred_params = pred_spec
        predicate = PredicateRegistry.get(pred_name)
        if predicate is not None:
            if not predicate(edge.weight, **(pred_params or {})):
                return False
        else:
            # Try factory
            factory_pred = PredicateRegistry.create(pred_name, **(pred_params or {}))
            if factory_pred is None or not factory_pred(edge.weight):
                return False

    for key, expected in criteria.items():
        if key in {'enabled', 'weight_predicate'}:
            continue
        if edge.attributes.get(key) != expected:
            return False

    return True


def _enumerate_candidates(graph: Any, lhs_nodes: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Enumerate candidate graph node IDs for each LHS node label deterministically."""
    label_to_candidates: dict[str, list[int]] = {}
    for node_spec in lhs_nodes:
        label = node_spec['label']
        criteria = node_spec.get('match_criteria', {})

        # Gather all node_ids that satisfy criteria
        candidates: list[int] = []
        for node_id, node in graph.nodes.items():
            if _node_matches_criteria(node, criteria):
                candidates.append(node_id)

        candidates.sort()
        label_to_candidates[label] = candidates

    return label_to_candidates


def _order_labels_by_constraint(lhs_nodes: list[dict[str, Any]],
                                label_to_candidates: dict[str, list[int]]) -> list[str]:
    """Order LHS labels by most-constrained heuristic with deterministic tie-breaks."""
    def constraint_score(node_spec: dict[str, Any]) -> int:
        criteria = node_spec.get('match_criteria', {})
        # More keys implies more constrained
        return len(criteria or {})

    # Sort by (candidate_count asc, -constraint_score desc, label asc)
    return [
        spec['label']
        for spec in sorted(
            lhs_nodes,
            key=lambda s: (
                len(label_to_candidates.get(s['label'], [])),
                -constraint_score(s),
                s['label']
            )
        )
    ]


def find_subgraph_matches(graph: Any,
                          lhs: dict[str, Any],
                          timeout_ms: int = 100) -> Iterator[dict[str, Any]]:
    """Find subgraph matches for the given LHS pattern.

    Args:
        graph: The Graph instance.
        lhs: Dict with keys 'nodes' and 'edges'.
        timeout_ms: Milliseconds budget; partial results allowed after timeout.

    Yields:
        Dict[str, Any]: Bindings mapping node labels to node_ids, and edge labels
        (if provided) to Edge instances.
    """
    start = time.monotonic()
    budget = max(0.0, float(timeout_ms) / 1000.0)

    lhs_nodes: list[dict[str, Any]] = lhs.get('nodes', []) or []
    lhs_edges: list[dict[str, Any]] = lhs.get('edges', []) or []

    # Trivial case: empty pattern matches empty binding
    if not lhs_nodes and not lhs_edges:
        yield {}
        return

    label_to_candidates = _enumerate_candidates(graph, lhs_nodes)
    order = _order_labels_by_constraint(lhs_nodes, label_to_candidates)

    # Backtracking search with injective mapping
    bindings: dict[str, int] = {}
    used_nodes: set[int] = set()

    # Pre-index edges by endpoints for faster checks
    def edges_between(u: int, v: int) -> list[Any]:
        node_u = graph.nodes.get(u)
        if not node_u:
            return []
        edges_obj = node_u.edges_out.get(v)
        if edges_obj is None:
            return []
        if isinstance(edges_obj, list):
            return list(edges_obj)
        return [edges_obj]

    def consistent_with_edges() -> bool:
        # Check all edges whose endpoints are both bound
        for e_spec in lhs_edges:
            s_lbl = e_spec['source_label']
            t_lbl = e_spec['target_label']
            if s_lbl in bindings and t_lbl in bindings:
                candidates = edges_between(bindings[s_lbl], bindings[t_lbl])
                if not candidates:
                    return False
                crit = e_spec.get('match_criteria', {})
                if crit:
                    if not any(_edge_matches_criteria(e, crit) for e in candidates):
                        return False
        return True

    results: list[dict[str, Any]] = []

    def backtrack(idx: int) -> None:
        # Timeout check
        if (time.monotonic() - start) > budget:
            return

        if idx == len(order):
            # All nodes assigned; enrich with edge label bindings
            binding_out: dict[str, Any] = dict(bindings)
            for e_spec in lhs_edges:
                e_label = e_spec.get('edge_label')
                if not e_label:
                    continue
                s_id = bindings.get(e_spec['source_label'])
                t_id = bindings.get(e_spec['target_label'])
                if s_id is None or t_id is None:
                    continue
                es = edges_between(s_id, t_id)
                if es:
                    # Prefer an instance that satisfies criteria if provided
                    crit = e_spec.get('match_criteria', {})
                    filtered = [e for e in es if _edge_matches_criteria(e, crit)] if crit else es
                    es_sorted = sorted(filtered, key=lambda e: str(getattr(e, 'edge_id', '')))
                    binding_out[e_label] = es_sorted[0]
            results.append(binding_out)
            return

        label = order[idx]
        candidates = label_to_candidates.get(label, [])
        for node_id in candidates:
            # Timeout check within loop
            if (time.monotonic() - start) > budget:
                return

            if node_id in used_nodes:
                continue

            bindings[label] = node_id
            used_nodes.add(node_id)

            if consistent_with_edges():
                backtrack(idx + 1)

            # Undo
            used_nodes.remove(node_id)
            del bindings[label]

    backtrack(0)

    # Yield collected (possibly partial under timeout) deterministically
    yield from results
