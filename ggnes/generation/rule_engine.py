"""RuleEngine for applying rules using TransactionManager and embedding logic."""

from __future__ import annotations

from typing import Any

from ggnes.generation.transaction import TransactionManager
from ggnes.rules.rule import Direction, Distribution


class RuleEngine:
    """Applies a Rule to the graph with cooldown, condition, and embedding."""

    def __init__(self, graph: Any, rng_manager: Any, id_manager: Any | None = None,
                 context_id: str = "default", cooldown_steps: int = 0) -> None:
        self.graph = graph
        self.tm = TransactionManager(graph=graph, rng_manager=rng_manager,
                                     id_manager=id_manager, context_id=context_id)
        self.cooldown_steps = max(0, int(cooldown_steps))
        self._cooldowns: dict[Any, int] = {}
        self._last_bindings: dict[str, Any] | None = None

    def tick_cooldown(self) -> None:
        to_del = []
        for rid, val in self._cooldowns.items():
            nv = max(0, val - 1)
            if nv == 0:
                to_del.append(rid)
            else:
                self._cooldowns[rid] = nv
        for rid in to_del:
            del self._cooldowns[rid]

    def _on_success(self, rule_id: Any) -> None:
        if self.cooldown_steps > 0 and rule_id is not None:
            self._cooldowns[rule_id] = self.cooldown_steps

    def _embedding_apply(self, rule, rhs_label_to_handle: dict[str, Any],
                         bindings: dict[str, Any]) -> None:
        # Only MAP_BOUNDARY_CONNECTIONS is supported; connection_map is ordered
        conn_map = getattr(rule.embedding, 'connection_map', {}) or {}
        excess_handling = getattr(rule.embedding, 'excess_connection_handling', 'WARNING')
        unknown_handling = getattr(rule.embedding, 'unknown_direction_handling', 'WARNING')
        boundary_handling = getattr(rule.embedding, 'boundary_handling', 'PROCESS_LAST')

        # Build per-boundary mapping with normalized directions
        boundary_to_targets: dict[str, dict[Direction, list[tuple[str, Any]]]] = {}
        for (boundary_label, direction), targets in conn_map.items():
            if not isinstance(direction, Direction):
                try:
                    if isinstance(direction, str) and direction.lower() in ("in", "out"):
                        direction = Direction[direction.upper()]
                    else:
                        direction = Direction[direction]  # type: ignore[index]
                except Exception:
                    continue
            boundary_to_targets.setdefault(boundary_label, {}).setdefault(direction, []).extend(targets)

        # Ensure we process all LHS boundary nodes even if no mapping provided
        all_boundary_labels = set(getattr(rule.lhs, 'boundary_nodes', []) or []) | set(boundary_to_targets.keys())
        items = [(lbl, boundary_to_targets.get(lbl, {})) for lbl in all_boundary_labels]

        # Boundary processing order
        if isinstance(boundary_handling, str) and boundary_handling.upper() == 'PROCESS_LAST':
            items = list(reversed(items))
        ignore_reconnections = isinstance(boundary_handling, str) and boundary_handling.upper() == 'IGNORE'

        for boundary_label, dir_map in items:
            boundary_node_id = bindings.get(boundary_label)
            if boundary_node_id is None:
                continue
            node = self.graph.nodes[boundary_node_id]

            # Collect external edges safely even if no edges present
            if node.edges_in:
                probe_in = next(iter(node.edges_in.values()))
                if isinstance(probe_in, list):
                    external_in = [e for lst in node.edges_in.values() for e in lst]
                else:
                    external_in = list(node.edges_in.values())
            else:
                external_in = []
            if node.edges_out:
                probe_out = next(iter(node.edges_out.values()))
                if isinstance(probe_out, list):
                    external_out = [e for lst in node.edges_out.values() for e in lst]
                else:
                    external_out = list(node.edges_out.values())
            else:
                external_out = []

            # Unknown direction handling
            if not ignore_reconnections:
                if not dir_map.get(Direction.IN) and external_in:
                    if isinstance(unknown_handling, str) and unknown_handling.upper() == 'ERROR':
                        raise ValueError(f"No connection mapping for boundary node '{boundary_label}' direction IN")
                if not dir_map.get(Direction.OUT) and external_out:
                    if isinstance(unknown_handling, str) and unknown_handling.upper() == 'ERROR':
                        raise ValueError(f"No connection mapping for boundary node '{boundary_label}' direction OUT")

            if ignore_reconnections:
                # Skip reconnections for this boundary; deletions are handled later in apply_rule
                continue

            # IN processing
            handled_in = 0
            for rhs_label, distribution in dir_map.get(Direction.IN, []):
                target_handle = rhs_label_to_handle.get(rhs_label)
                if not target_handle:
                    continue
                if isinstance(distribution, (int, float)):
                    external_in_sorted = sorted(external_in, key=lambda e: (e.source_node_id, str(e.edge_id)))
                    if isinstance(distribution, int):
                        # Integer: select first k incoming sources deterministically (copy edge attributes)
                        k = max(0, int(distribution))
                        take = min(k, len(external_in_sorted))
                        for idx in range(take):
                            e = external_in_sorted[idx]
                            self.tm.buffer.add_edge(
                                e.source_node_id,
                                target_handle,
                                {"weight": float(e.weight), "enabled": bool(e.enabled), "attributes": dict(e.attributes)}
                            )
                        handled_in = max(handled_in, take)
                    else:
                        # Float: single edge with provided weight
                        if external_in_sorted:
                            e = external_in_sorted[0]
                            self.tm.buffer.add_edge(e.source_node_id, target_handle, {"weight": float(distribution)})
                            handled_in = min(len(external_in_sorted), 1)
                        else:
                            # No external_in: connect boundary -> target with provided weight
                            self.tm.buffer.add_edge(boundary_node_id, target_handle, {"weight": float(distribution)})
                            handled_in = 1
                    continue
                if not isinstance(distribution, Distribution):
                    try:
                        distribution = Distribution[distribution] if isinstance(distribution, str) else Distribution(distribution)  # type: ignore[arg-type]
                    except Exception:
                        continue
                external_in_sorted = sorted(external_in, key=lambda e: (e.source_node_id, str(e.edge_id)))
                if distribution == Distribution.COPY_ALL:
                    for e in external_in_sorted:
                        self.tm.buffer.add_edge(
                            e.source_node_id,
                            target_handle,
                            {"weight": float(e.weight), "enabled": bool(e.enabled), "attributes": dict(e.attributes)}
                        )
                    handled_in = max(handled_in, len(external_in_sorted))
                else:  # CONNECT_SINGLE
                    if handled_in < len(external_in_sorted):
                        e = external_in_sorted[handled_in]
                        self.tm.buffer.add_edge(
                            e.source_node_id,
                            target_handle,
                            {"weight": float(e.weight), "enabled": bool(e.enabled), "attributes": dict(e.attributes)}
                        )
                        handled_in += 1
            if handled_in < len(external_in):
                if isinstance(excess_handling, str) and excess_handling.upper() == 'ERROR':
                    raise ValueError(f"Excess incoming connections for boundary {boundary_label}: {len(external_in) - handled_in} unhandled")
                elif isinstance(excess_handling, str) and excess_handling.upper() == 'WARNING':
                    import logging
                    logging.warning(f"Excess incoming connections for boundary {boundary_label}: {len(external_in) - handled_in} unhandled")

            # OUT processing
            handled_out = 0
            for rhs_label, distribution in dir_map.get(Direction.OUT, []):
                target_handle = rhs_label_to_handle.get(rhs_label)
                if not target_handle:
                    continue
                if isinstance(distribution, (int, float)):
                    external_out_sorted = sorted(external_out, key=lambda e: (e.target_node_id, str(e.edge_id)))
                    if isinstance(distribution, int):
                        # Integer: select first k outgoing targets deterministically (copy edge attributes)
                        k = max(0, int(distribution))
                        take = min(k, len(external_out_sorted))
                        for idx in range(take):
                            e = external_out_sorted[idx]
                            self.tm.buffer.add_edge(
                                target_handle,
                                e.target_node_id,
                                {"weight": float(e.weight), "enabled": bool(e.enabled), "attributes": dict(e.attributes)}
                            )
                        handled_out = max(handled_out, take)
                    else:
                        # Float: single edge with provided weight
                        if external_out_sorted:
                            e = external_out_sorted[0]
                            self.tm.buffer.add_edge(target_handle, e.target_node_id, {"weight": float(distribution)})
                            handled_out = min(len(external_out_sorted), 1)
                        else:
                            # No external_out: connect target -> boundary with provided weight
                            self.tm.buffer.add_edge(target_handle, boundary_node_id, {"weight": float(distribution)})
                            handled_out = 1
                    continue
                if not isinstance(distribution, Distribution):
                    try:
                        distribution = Distribution[distribution] if isinstance(distribution, str) else Distribution(distribution)  # type: ignore[arg-type]
                    except Exception:
                        continue
                external_out_sorted = sorted(external_out, key=lambda e: (e.target_node_id, str(e.edge_id)))
                if distribution == Distribution.COPY_ALL:
                    for e in external_out_sorted:
                        self.tm.buffer.add_edge(
                            target_handle,
                            e.target_node_id,
                            {"weight": float(e.weight), "enabled": bool(e.enabled), "attributes": dict(e.attributes)}
                        )
                    handled_out = max(handled_out, len(external_out_sorted))
                else:  # CONNECT_SINGLE
                    if handled_out < len(external_out_sorted):
                        e = external_out_sorted[handled_out]
                        self.tm.buffer.add_edge(
                            target_handle,
                            e.target_node_id,
                            {"weight": float(e.weight), "enabled": bool(e.enabled), "attributes": dict(e.attributes)}
                        )
                        handled_out += 1
            if handled_out < len(external_out):
                if isinstance(excess_handling, str) and excess_handling.upper() == 'ERROR':
                    raise ValueError(f"Excess outgoing connections for boundary {boundary_label}: {len(external_out) - handled_out} unhandled")
                elif isinstance(excess_handling, str) and excess_handling.upper() == 'WARNING':
                    import logging
                    logging.warning(f"Excess outgoing connections for boundary {boundary_label}: {len(external_out) - handled_out} unhandled")

    def apply_rule(self, rule, bindings: dict[str, Any]) -> bool:
        # Cooldown gate
        if rule.rule_id in self._cooldowns and self._cooldowns[rule.rule_id] > 0:
            return False

        # Condition gate
        if rule.condition is not None:
            graph_context = {"num_nodes": len(self.graph.nodes)}
            if not rule.condition(self.graph, dict(bindings), graph_context):
                return False

        # Provide UUID derivation context for deterministic UUIDs (§19)
        try:
            if hasattr(self.graph, 'set_uuid_context'):
                self.graph.set_uuid_context({
                    'rule_id': str(getattr(rule, 'rule_id', '')),
                    'binding_signature': str(sorted(bindings.items())) if bindings else '',
                })
        except Exception:  # pragma: no cover - defensive
            pass

        self.tm.begin()
        self._last_bindings = dict(bindings) if bindings else {}

        # Stage RHS nodes
        rhs_label_to_handle: dict[str, Any] = {}
        for n in getattr(rule.rhs, 'add_nodes', []) or []:
            label = n.get('label')
            props = n.get('properties', {})
            handle = self.tm.buffer.add_node(props)
            if label:
                rhs_label_to_handle[label] = handle

        # Stage RHS edges
        for e in getattr(rule.rhs, 'add_edges', []) or []:
            s_lbl = e.get('source_label')
            t_lbl = e.get('target_label')
            props = e.get('properties', {})

            s_ref = rhs_label_to_handle.get(s_lbl, bindings.get(s_lbl))
            t_ref = rhs_label_to_handle.get(t_lbl, bindings.get(t_lbl))
            if s_ref is None or t_ref is None:
                continue
            self.tm.buffer.add_edge(s_ref, t_ref, props)

        # Apply embedding reconnections (must occur before deletions to access external edges)
        self._embedding_apply(rule, rhs_label_to_handle, bindings)

        # Stage deletions (edges first, then nodes)
        for edge_spec in getattr(rule.rhs, 'delete_edges', []) or []:
            # Resolve by either explicit edge_label binding or source/target binding
            e_id = None
            e_label = edge_spec.get('edge_label') if isinstance(edge_spec, dict) else None
            if e_label and e_label in bindings:
                e_id = bindings[e_label]
            else:
                s_lbl = edge_spec.get('source_label') if isinstance(edge_spec, dict) else None
                t_lbl = edge_spec.get('target_label') if isinstance(edge_spec, dict) else None
                s_id = bindings.get(s_lbl)
                t_id = bindings.get(t_lbl)
                if s_id is not None and t_id is not None:
                    edge_obj = self.graph.find_edge_by_endpoints(s_id, t_id)
                    e_id = getattr(edge_obj, 'edge_id', None)
                # Fallback: resolve by consulting LHS edges for this edge_label
                if e_id is None and e_label and isinstance(rule.lhs.edges, list):
                    for lhs_e in rule.lhs.edges:
                        if lhs_e.get('edge_label') == e_label:
                            s = bindings.get(lhs_e.get('source_label'))
                            t = bindings.get(lhs_e.get('target_label'))
                            if s is not None and t is not None:
                                edge_obj = self.graph.find_edge_by_endpoints(s, t)
                                e_id = getattr(edge_obj, 'edge_id', None)
                            break
            if e_id is not None:
                self.tm.buffer.delete_edge(e_id)

        for label in getattr(rule.rhs, 'delete_nodes', []) or []:
            nid = bindings.get(label)
            if nid is not None:
                self.tm.buffer.delete_node(nid)

        # Commit staged changes (guard against constructor errors)
        try:
            mapping = self.tm.commit()
        except Exception:
            # Rollback RNG/buffer and report failure
            self.tm.rollback()
            self._last_bindings = None
            return False

        # If boundary_handling is IGNORE, skip validation and accept changes
        emb = getattr(rule, 'embedding', None)
        if emb is not None and isinstance(getattr(emb, 'boundary_handling', ''), str) and emb.boundary_handling.upper() == 'IGNORE':
            self._on_success(rule.rule_id)
            self._last_bindings = None
            return True

        # Validate graph; if invalid, undo applied nodes and return False
        errors: list = []
        valid = self.graph.validate(collect_errors=errors)
        if not valid:
            # Remove newly added nodes
            for _, node_id in mapping.items():
                self.graph.remove_node(node_id)
            # Restore RNG state via rollback (buffer already reset)
            self.tm.rollback()
            self._last_bindings = None
            return False

        # Success
        self._on_success(rule.rule_id)
        self._last_bindings = None
        return True
