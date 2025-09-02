"""Graph class and related functionality."""

import hashlib
import logging
import math
import uuid
from enum import Enum

from ..utils.uuid_provider import (
    DeterministicUUIDProvider,
    provider_from_graph_config,
)
from ..utils.validation import EdgeError, NodeError
from .edge import Edge
from .node import Node, NodeType
from .primitives import PrimitivesLibrary


class IDStrategy(Enum):
    """ID management strategies."""
    LOCAL_ONLY = "LOCAL_ONLY"
    HYBRID = "HYBRID"
    GLOBAL_ONLY = "GLOBAL_ONLY"


class Graph:
    """Simple directed graph (no multiple edges between same node pair)."""

    def __init__(self, config: dict = None):
        self.nodes = {}  # {node_id: Node}
        self.input_node_ids = []  # List[int]
        self.output_node_ids = []  # List[int]
        self._node_id_counter = 0
        self._edge_id_counter = 0  # For LOCAL_ONLY mode
        self.config = config or {}
        self.graph_id = uuid.uuid4()  # Unique identifier for this graph instance
        self._edges_by_id = {}  # Global edge index: edge_id -> Edge
        self._uuid_provider: DeterministicUUIDProvider | None = None
        self._uuid_context: dict | None = None

        # Set default ID strategy to HYBRID if not specified
        if 'id_strategy' not in self.config:
            self.config['id_strategy'] = 'HYBRID'

        # Multigraph feature flag (default: False → simple graph)
        if 'multigraph' not in self.config:
            self.config['multigraph'] = False

        # Normalize ID strategy from string to enum
        if isinstance(self.config['id_strategy'], str):
            try:
                self.config['id_strategy'] = IDStrategy[self.config['id_strategy']]
            except KeyError:
                raise ValueError(
                    f"Invalid id_strategy: {self.config['id_strategy']}. "
                    "Must be one of: 'LOCAL_ONLY', 'HYBRID', 'GLOBAL_ONLY'"
                )

        # Deterministic UUIDs configuration (project_guide.md §19)
        # Defaults
        if 'deterministic_uuids' not in self.config:
            self.config['deterministic_uuids'] = False
        # Optional fixed provenance UUID for determinism across runs
        prov = self.config.get('graph_provenance_uuid')
        if prov:
            try:
                self.graph_id = uuid.UUID(str(prov))
            except Exception:
                # Keep auto-generated graph_id on invalid input
                pass
        if self.config.get('deterministic_uuids'):
            self._uuid_provider = provider_from_graph_config(self.config)

    def add_node(self, properties: dict) -> int:
        """Add a node to the graph.

        Args:
            properties: Node properties including node_type, activation_function, etc.

        Returns:
            int: The assigned node_id
        """
        node_id = self._node_id_counter
        self._node_id_counter += 1

        node = Node(
            node_id=node_id,
            node_type=properties['node_type'],
            activation_function=properties['activation_function'],
            bias=properties.get('bias', 0.0),
            attributes=properties.get('attributes', {})
        )

        # Assign global_id based on id_strategy / deterministic UUIDs
        if self.config.get('id_strategy') != IDStrategy.LOCAL_ONLY:
            if self._uuid_provider is not None:
                inputs = {
                    'graph_provenance_uuid': str(self.graph_id),
                    'local_node_id': node_id,
                    'node_type': node.node_type.value,
                    'activation_function': node.activation_function,
                    'bias': float(node.bias),
                    'attributes': node.attributes,
                }
                if self._uuid_context:
                    inputs['context'] = dict(self._uuid_context)
                node.global_id = self._uuid_provider.derive_uuid('node', inputs)
            else:
                node.global_id = uuid.uuid4()

        self.nodes[node_id] = node

        # Track input/output nodes
        if node.node_type == NodeType.INPUT:
            self.input_node_ids.append(node_id)
        elif node.node_type == NodeType.OUTPUT:
            self.output_node_ids.append(node_id)

        return node_id

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all its edges.

        Args:
            node_id: ID of node to remove
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove all connected edges via remove_edge to keep global index consistent
        # Collect edge_ids first to avoid mutation during iteration
        edge_ids_to_remove = []
        # Incoming edges
        if not self.config.get('multigraph'):
            edge_ids_to_remove.extend(
                [edge.edge_id for edge in node.edges_in.values()]
            )
            edge_ids_to_remove.extend(
                [edge.edge_id for edge in node.edges_out.values()]
            )
        else:
            for lst in node.edges_in.values():
                for e in lst:
                    edge_ids_to_remove.append(e.edge_id)
            for lst in node.edges_out.values():
                for e in lst:
                    edge_ids_to_remove.append(e.edge_id)
        for eid in edge_ids_to_remove:
            self.remove_edge(eid)

        # Remove from tracking lists
        if node_id in self.input_node_ids:
            self.input_node_ids.remove(node_id)
        if node_id in self.output_node_ids:
            self.output_node_ids.remove(node_id)

        del self.nodes[node_id]

    def _iter_out_edges(self, node):
        """Yield (target_id, edge) for all outgoing edges, handling multigraph lists."""
        if not self.config.get('multigraph'):
            for target_id, edge in node.edges_out.items():
                yield target_id, edge
        else:
            for target_id, edges in node.edges_out.items():
                for edge in edges:
                    yield target_id, edge

    def _iter_in_edges(self, node):
        """Yield (source_id, edge) for all incoming edges, handling multigraph lists."""
        if not self.config.get('multigraph'):
            for source_id, edge in node.edges_in.items():
                yield source_id, edge
        else:
            for source_id, edges in node.edges_in.items():
                for edge in edges:
                    yield source_id, edge

    def add_edge(self, source_id: int, target_id: int, properties: dict = None):
        """Add an edge between two nodes.

        Simple graph constraint: MUST reject second edge between same pair with warning.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            properties: Edge properties

        Returns:
            edge_id (type depends on id_strategy) or None if duplicate
        """
        properties = properties or {}

        # Check nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Cannot add edge: nodes {source_id} or {target_id} do not exist")

        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]

        # Simple graph constraint: reject duplicate edges when multigraph disabled
        if not self.config.get('multigraph'):
            if target_id in source_node.edges_out:
                logging.warning(
                    f"Edge from {source_id} to {target_id} already exists. "
                    "Simple graphs do not allow duplicate edges."
                )
                return None  # Caller MUST handle None return

        # Create edge with appropriate ID type
        if self.config.get('id_strategy') == IDStrategy.LOCAL_ONLY:
            edge_id = self._edge_id_counter
            self._edge_id_counter += 1
        else:
            if self._uuid_provider is not None:
                # Use source/target global IDs if available for canonical inputs
                src_node = self.nodes[source_id]
                tgt_node = self.nodes[target_id]
                inputs = {
                    'graph_provenance_uuid': str(self.graph_id),
                    'source_global_uuid': str(getattr(src_node, 'global_id', '')),
                    'target_global_uuid': str(getattr(tgt_node, 'global_id', '')),
                    'local_edge_index': self._edge_id_counter,
                    'properties': properties,
                }
                if self._uuid_context:
                    inputs['context'] = dict(self._uuid_context)
                edge_id = self._uuid_provider.derive_uuid('edge', inputs)
            else:
                edge_id = uuid.uuid4()

        edge = Edge(
            edge_id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            weight=properties.get('weight', 0.1),
            enabled=properties.get('enabled', True),
            attributes=properties.get('attributes', {})
        )

        # Set local_edge_id in HYBRID mode
        if self.config.get('id_strategy') == IDStrategy.HYBRID:
            edge.local_edge_id = self._edge_id_counter
            self._edge_id_counter += 1

        # Add to adjacency (list in multigraph mode; single in simple)
        if not self.config.get('multigraph'):
            source_node.edges_out[target_id] = edge
            target_node.edges_in[source_id] = edge
        else:
            out_list = source_node.edges_out.get(target_id)
            if out_list is None:
                out_list = []
                source_node.edges_out[target_id] = out_list
            out_list.append(edge)

            in_list = target_node.edges_in.get(source_id)
            if in_list is None:
                in_list = []
                target_node.edges_in[source_id] = in_list
            in_list.append(edge)

        # Index globally
        self._edges_by_id[edge_id] = edge

        return edge_id

    # UUID context controls
    def set_uuid_context(self, context: dict | None) -> None:
        """Set ephemeral UUID derivation context (e.g., rule_id, binding_signature, iteration_index).

        Context will be embedded into deterministic UUID inputs for subsequent
        add_node/add_edge calls until changed. Passing None clears the context.
        """
        self._uuid_context = dict(context) if context else None

    def remove_edge(self, edge_id) -> None:
        """Remove an edge by ID.

        Args:
            edge_id: ID of edge to remove
        """
        edge = self.find_edge_by_id(edge_id)
        if not edge:
            return

        source_node = self.nodes.get(edge.source_node_id)
        target_node = self.nodes.get(edge.target_node_id)

        if not self.config.get('multigraph'):
            if source_node and edge.target_node_id in source_node.edges_out:
                del source_node.edges_out[edge.target_node_id]
            if target_node and edge.source_node_id in target_node.edges_in:
                del target_node.edges_in[edge.source_node_id]
        else:
            # Remove only this edge instance from lists, and cleanup empty lists
            if source_node:
                lst = source_node.edges_out.get(edge.target_node_id, [])
                source_node.edges_out[edge.target_node_id] = [e for e in lst if e.edge_id != edge.edge_id]
                if not source_node.edges_out[edge.target_node_id]:
                    del source_node.edges_out[edge.target_node_id]
            if target_node:
                lst = target_node.edges_in.get(edge.source_node_id, [])
                target_node.edges_in[edge.source_node_id] = [e for e in lst if e.edge_id != edge.edge_id]
                if not target_node.edges_in[edge.source_node_id]:
                    del target_node.edges_in[edge.source_node_id]

        # Remove from global index
        if edge.edge_id in self._edges_by_id:
            del self._edges_by_id[edge.edge_id]

    def find_edge_by_id(self, edge_id) -> Edge | None:
        """Find an edge by its ID.

        Args:
            edge_id: Edge ID to search for

        Returns:
            Edge or None if not found
        """
        return self._edges_by_id.get(edge_id)

    def find_edge_by_endpoints(self, source_id: int, target_id: int) -> Edge | None:
        """Find an edge by its endpoints.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Edge or None if not found
        """
        source_node = self.nodes.get(source_id)
        if not source_node:
            return None
        if not self.config.get('multigraph'):
            return source_node.edges_out.get(target_id)
        lst = source_node.edges_out.get(target_id)
        if not lst:
            return None
        # Compatibility: return first edge instance deterministically by edge_id
        return sorted(lst, key=lambda e: str(e.edge_id))[0]

    def find_edges_by_endpoints(self, source_id: int, target_id: int) -> list[Edge]:
        """Find all edges by endpoints (multigraph-aware).

        Returns empty list if none found. In simple graph mode, returns a list
        of length 0 or 1.
        """
        source_node = self.nodes.get(source_id)
        if not source_node:
            return []
        if not self.config.get('multigraph'):
            edge = source_node.edges_out.get(target_id)
            return [edge] if edge else []
        lst = source_node.edges_out.get(target_id)
        return list(lst) if lst else []

    def validate(self, collect_errors=None, collect_warnings=None) -> bool:
        """Validate graph integrity.

        Args:
            collect_errors: Optional list to collect ValidationError objects
            collect_warnings: Optional list to collect ValidationError objects

        Returns:
            bool: True if valid (no errors), False otherwise
        """
        errors = []
        warnings = []

        # Check for dangling edges
        for node in self.nodes.values():
            for source_id, _ in self._iter_in_edges(node):
                if source_id not in self.nodes:
                    errors.append(EdgeError(
                        None, 'dangling_edge',
                        f"Source node {source_id} does not exist",
                        source_id=source_id
                    ))
            for target_id, _ in self._iter_out_edges(node):
                if target_id not in self.nodes:
                    errors.append(EdgeError(
                        None, 'dangling_edge',
                        f"Target node {target_id} does not exist",
                        target_id=target_id
                    ))

        # Check node attributes
        for node_id, node in self.nodes.items():
            # Check activation function
            if not PrimitivesLibrary.is_valid_activation(node.activation_function):
                errors.append(NodeError(
                    node_id, 'invalid_activation',
                    f"Invalid activation function: {node.activation_function}",
                    activation=node.activation_function
                ))

            # Check aggregation function if specified
            aggregation = node.attributes.get('aggregation', 'sum')
            if not PrimitivesLibrary.is_valid_aggregation(aggregation):
                errors.append(NodeError(
                    node_id, 'invalid_aggregation',
                    f"Invalid aggregation function: {aggregation}",
                    aggregation=aggregation
                ))

            # Check output_size for all nodes
            output_size = node.attributes.get('output_size')
            if not isinstance(output_size, int) or output_size <= 0:
                errors.append(NodeError(
                    node_id, 'missing_output_size',
                    "Missing or invalid output_size",
                    output_size=output_size
                ))

            # Check finite values
            if not math.isfinite(node.bias):
                errors.append(NodeError(
                    node_id, 'non_finite_bias',
                    f"Non-finite bias: {node.bias}",
                    bias=node.bias
                ))

            # Advanced aggregation parameter validation (project_guide.md §8.4)
            adv_aggs = {
                'attention', 'multi_head_attention', 'gated_sum',
                'topk_weighted_sum', 'moe', 'attn_pool'
            }
            if aggregation in adv_aggs:
                # Helper: report invalid param
                def _param_error(pname: str, value, reason: str):
                    errors.append(NodeError(
                        node_id, 'invalid_agg_param',
                        f"Invalid '{pname}' for aggregation '{aggregation}'",
                        param_name=pname, value=value, reason=reason, aggregation=aggregation
                    ))

                # Compute fan-in for constraints (enabled edges only)
                if not self.config.get('multigraph'):
                    fan_in = sum(1 for e in node.edges_in.values() if e.enabled)
                else:
                    fan_in = sum(1 for lst in node.edges_in.values() for e in lst if e.enabled)

                # Common params
                p = node.attributes
                if 'dropout_p' in p:
                    dp = p['dropout_p']
                    if not isinstance(dp, int | float) or not (0.0 <= dp < 1.0):
                        _param_error('dropout_p', dp, 'must_be_in_[0,1)')
                if 'post_projection' in p and not isinstance(p['post_projection'], bool):
                    _param_error('post_projection', p['post_projection'], 'must_be_bool')
                if 'normalize' in p and not isinstance(p['normalize'], bool):
                    _param_error('normalize', p['normalize'], 'must_be_bool')

                # Attention-like
                if aggregation in {'attention', 'multi_head_attention'}:
                    num_heads = p.get('num_heads', 1)
                    if not isinstance(num_heads, int) or num_heads <= 0:
                        _param_error('num_heads', num_heads, 'value_must_be_positive_int')
                    if 'head_dim' in p:
                        hd = p['head_dim']
                        if not isinstance(hd, int) or hd <= 0:
                            _param_error('head_dim', hd, 'value_must_be_positive_int')
                    if 'top_k' in p:
                        tk = p['top_k']
                        if tk is not None and (not isinstance(tk, int) or tk <= 0 or tk > max(1, fan_in)):
                            _param_error('top_k', tk, 'top_k_out_of_range')
                    if 'temperature' in p:
                        t = p['temperature']
                        if not isinstance(t, int | float) or t <= 0:
                            _param_error('temperature', t, 'value_must_be_positive')
                    if 'attn_eps' in p:
                        eps = p['attn_eps']
                        if not isinstance(eps, int | float) or eps <= 0:
                            _param_error('attn_eps', eps, 'value_must_be_positive')
                    if 'attn_type' in p and p['attn_type'] not in {'dot', 'additive', 'gat'}:
                        _param_error('attn_type', p['attn_type'], 'invalid_choice')

                # Top-k weighted sum
                if aggregation == 'topk_weighted_sum':
                    if 'top_k' in p:
                        tk = p['top_k']
                        if tk is not None and (not isinstance(tk, int) or tk <= 0 or tk > max(1, fan_in)):
                            _param_error('top_k', tk, 'top_k_out_of_range')

                # MoE
                if aggregation == 'moe':
                    if 'router_type' in p and p['router_type'] not in {'softmax', 'topk'}:
                        _param_error('router_type', p['router_type'], 'invalid_choice')
                    if 'experts' in p:
                        ex = p['experts']
                        if not isinstance(ex, int) or ex <= 0:
                            _param_error('experts', ex, 'value_must_be_positive_int')
                    if 'capacity_factor' in p:
                        cf = p['capacity_factor']
                        if not isinstance(cf, int | float) or cf <= 0:
                            _param_error('capacity_factor', cf, 'value_must_be_positive')
                    if 'router_temperature' in p:
                        rt = p['router_temperature']
                        if not isinstance(rt, int | float) or rt <= 0:
                            _param_error('router_temperature', rt, 'value_must_be_positive')
                    if 'top_k' in p:
                        tk = p['top_k']
                        if tk is not None and (not isinstance(tk, int) or tk <= 0 or tk > max(1, fan_in)):
                            _param_error('top_k', tk, 'top_k_out_of_range')

                # attn_pool
                if aggregation == 'attn_pool':
                    if 'pool_heads' in p:
                        ph = p['pool_heads']
                        if not isinstance(ph, int) or ph <= 0:
                            _param_error('pool_heads', ph, 'value_must_be_positive_int')

        # Check edge weights
        for node in self.nodes.values():
            for _, edge in self._iter_out_edges(node):
                if not math.isfinite(edge.weight):
                    errors.append(EdgeError(
                        edge.edge_id, 'non_finite_weight',
                        "Non-finite weight",
                        weight=edge.weight
                    ))

        # Check output reachability
        for output_id in self.output_node_ids:
            if not self._is_reachable_from_input(output_id):
                errors.append(NodeError(
                    output_id, 'unreachable_output',
                    "Output node not reachable from inputs",
                    node_type=NodeType.OUTPUT
                ))

        # Warnings (not counted as errors)
        for output_id in self.output_node_ids:
            output_node = self.nodes[output_id]
            if not output_node.edges_in:
                warnings.append(NodeError(
                    output_id, 'no_incoming',
                    "Output node has no incoming connections",
                    node_type=NodeType.OUTPUT
                ))

        # Collect results
        if collect_errors is not None:
            collect_errors.extend(errors)
        if collect_warnings is not None:
            collect_warnings.extend(warnings)

        return len(errors) == 0

    def detect_cycles(self) -> None:
        """Detect cycles and mark recurrent edges.

        Preserves explicitly set is_recurrent flags and only updates edges
        that haven't been explicitly marked.
        """
        visited = set()
        path = set()

        # Only reset flags that weren't explicitly set
        for node in self.nodes.values():
            for _, edge in self._iter_out_edges(node):
                if 'is_recurrent' not in edge.attributes:
                    edge.attributes['is_recurrent'] = False

        # DFS from each node
        for node_id in list(self.nodes.keys()):
            if node_id not in visited:
                self._detect_cycles_dfs(node_id, visited, path)

    def _detect_cycles_dfs(self, node_id: int, visited: set, path: set) -> None:
        """DFS helper for cycle detection."""
        visited.add(node_id)
        path.add(node_id)

        node = self.nodes[node_id]
        for target_id, edge in self._iter_out_edges(node):
            if not edge.enabled:
                continue
            if target_id not in visited:
                self._detect_cycles_dfs(target_id, visited, path)
            elif target_id in path:
                # Back edge found - mark as recurrent
                edge.attributes['is_recurrent'] = True

        path.remove(node_id)

    def topological_sort(self, ignore_recurrent: bool = True) -> list[int]:
        """Return nodes in topological order.

        Prerequisites:
            If ignore_recurrent=True, detect_cycles() should be called first
            to mark recurrent edges.

        Args:
            ignore_recurrent: If True, ignore edges marked as recurrent

        Returns:
            List of node IDs in topological order
        """
        # Build adjacency for enabled, non-recurrent edges
        adj = {node_id: set() for node_id in self.nodes}
        for node_id, node in self.nodes.items():
            for target_id, edge in self._iter_out_edges(node):
                if not edge.enabled:  # Skip disabled edges
                    continue
                if ignore_recurrent and edge.attributes.get('is_recurrent', False):
                    continue
                adj[node_id].add(target_id)

        # Kahn's algorithm
        in_degree = {node_id: 0 for node_id in self.nodes}
        for node_id in self.nodes:
            for target_id in adj[node_id]:
                in_degree[target_id] += 1

        queue = [node_id for node_id in self.nodes if in_degree[node_id] == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for neighbor in adj[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Validate completeness
        if len(result) != len(self.nodes):
            logging.warning(
                f"Incomplete topological sort: {len(result)} of {len(self.nodes)} nodes ordered. "
                "Graph may have cycles with enabled edges."
            )

        return result

    def compute_fingerprint(self) -> str:
        """Compute Weisfeiler-Lehman hash of graph structure.

        Note: This method only considers enabled edges.

        Returns:
            str: Hexadecimal hash string
        """
        # Initial node labels
        labels = {}
        for node_id, node in self.nodes.items():
            # Count only enabled edges (respect multiplicity)
            enabled_in = sum(1 for _, edge in self._iter_in_edges(node) if edge.enabled)
            enabled_out = sum(1 for _, edge in self._iter_out_edges(node) if edge.enabled)
            label = (
                node.node_type.value,
                node.activation_function,
                enabled_in,
                enabled_out
            )
            labels[node_id] = str(label)

        # WL iterations
        k = 3  # Number of iterations
        k = int(self.config.get('wl_iterations', k))
        for iteration in range(k):
            new_labels = {}
            for node_id, node in self.nodes.items():
                # Collect neighbor labels (only enabled edges), multiplicity-aware
                neighbor_labels = []
                if not self.config.get('multigraph'):
                    for source_id, edge in sorted(node.edges_in.items()):
                        if edge.enabled:
                            neighbor_labels.append(('in', labels[source_id]))
                    for target_id, edge in sorted(node.edges_out.items()):
                        if edge.enabled:
                            neighbor_labels.append(('out', labels[target_id]))
                else:
                    # Compress neighbor multiset into (label, multiplicity) pairs deterministically
                    in_counts = {}
                    for source_id, edges in sorted(node.edges_in.items()):
                        for edge in sorted(edges, key=lambda e: str(e.edge_id)):
                            if edge.enabled:
                                key = ('in', labels[source_id])
                                in_counts[key] = in_counts.get(key, 0) + 1
                    out_counts = {}
                    for target_id, edges in sorted(node.edges_out.items()):
                        for edge in sorted(edges, key=lambda e: str(e.edge_id)):
                            if edge.enabled:
                                key = ('out', labels[target_id])
                                out_counts[key] = out_counts.get(key, 0) + 1
                    # Extend neighbor_labels with compressed tuples
                    for key, count in sorted(in_counts.items(), key=lambda kv: kv[0]):
                        neighbor_labels.append((key, count))
                    for key, count in sorted(out_counts.items(), key=lambda kv: kv[0]):
                        neighbor_labels.append((key, count))

                # Include dimensional attributes
                node_dim = (
                    node.attributes.get('output_size'),
                    node.attributes.get('aggregation', 'sum')
                )
                # Create new label (deterministic)
                combined = (labels[node_id], tuple(sorted(neighbor_labels)), node_dim)
                # Use SHA-256 for deterministic hashing
                combined_str = str(combined)
                new_labels[node_id] = hashlib.sha256(combined_str.encode()).hexdigest()[:16]

            labels = new_labels

        # Final hash using SHA-256
        sorted_labels = sorted(labels.values())
        fingerprint_data = (
            tuple(sorted_labels),
            len(self.input_node_ids),
            len(self.output_node_ids)
        )

        return hashlib.sha256(str(fingerprint_data).encode()).hexdigest()

    def reset_id_counter(self, start_at: int = None) -> None:
        """Reset the node ID counter.

        Args:
            start_at: Starting value for counter. If None, uses max existing ID + 1
        """
        if start_at is not None:
            self._node_id_counter = start_at
        elif self.nodes:
            self._node_id_counter = max(self.nodes.keys()) + 1
        else:
            self._node_id_counter = 0

    def _is_reachable_from_input(self, target_id: int) -> bool:
        """Check if a node is reachable from any input node."""
        for input_id in self.input_node_ids:
            if self._has_path(input_id, target_id):
                return True
        return False

    def _has_path(self, source_id: int, target_id: int) -> bool:
        """Check if path exists from source to target via enabled edges."""
        if source_id == target_id:
            return True

        visited = set()
        queue = [source_id]

        while queue:
            current = queue.pop(0)
            if current == target_id:
                return True

            visited.add(current)

            node = self.nodes.get(current)
            if node:
                if not self.config.get('multigraph'):
                    items = list(node.edges_out.items())
                else:
                    items = [
                        (tid, e)
                        for tid, lst in node.edges_out.items()
                        for e in lst
                    ]
                for next_id, edge in items:
                    # Only follow enabled edges for reachability
                    if edge.enabled and next_id not in visited:
                        queue.append(next_id)

        return False

    def list_edges(self, src: int | None = None, tgt: int | None = None):
        """Iterate over edges with optional endpoint filtering.

        In simple mode, yields single-edge instances.
        In multigraph, yields each parallel edge.
        """
        if src is None and tgt is None:
            # Use global index for efficiency
            for edge in sorted(self._edges_by_id.values(), key=lambda e: str(e.edge_id)):
                yield edge
            return

        if src is not None and tgt is not None:
            yield from self.find_edges_by_endpoints(src, tgt)
            return

        # One-sided filter
        if src is not None:
            node = self.nodes.get(src)
            if not node:
                return
            if not self.config.get('multigraph'):
                for _, edge in sorted(node.edges_out.items()):
                    yield edge
            else:
                for lst in node.edges_out.values():
                    for edge in sorted(lst, key=lambda e: str(e.edge_id)):
                        yield edge
            return

        if tgt is not None:
            node = self.nodes.get(tgt)
            if not node:
                return
            if not self.config.get('multigraph'):
                for _, edge in sorted(node.edges_in.items()):
                    yield edge
            else:
                for lst in node.edges_in.values():
                    for edge in sorted(lst, key=lambda e: str(e.edge_id)):
                        yield edge
