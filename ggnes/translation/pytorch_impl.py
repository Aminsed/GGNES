"""PyTorch translation implementation (requires torch)."""

from __future__ import annotations

from typing import Any

from ..core.node import NodeType
from .state_manager import StateManager

# Global cache for hierarchical submodules keyed by deterministic UUID
# When a node was produced by hierarchical derivation, it may contain
# attributes['derivation_uuid']. We reuse cached parameters/modules
# across translations to ensure parity per project_guide.md §23.
_SUBMODULE_CACHE: dict[str, dict[str, object]] = {}

# Simple metrics for cache observability (project_guide.md §20.13, §23)
_CACHE_METRICS: dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "entries": 0,
}

# Feature flag to enable/disable cache usage at runtime (semantic parity required)
_CACHE_ENABLED: bool = True

def clear_translation_cache():
    """Testing helper to clear the submodule cache."""
    _SUBMODULE_CACHE.clear()
    _CACHE_METRICS["entries"] = 0
    _CACHE_METRICS["hits"] = 0
    _CACHE_METRICS["misses"] = 0


def get_translation_cache_metrics() -> dict[str, int]:
    """Return a copy of translation cache metrics (hits/misses/entries)."""
    return dict(_CACHE_METRICS)


def set_translation_cache_enabled(enabled: bool) -> None:
    """Enable or disable translation cache usage globally."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = bool(enabled)


def to_pytorch_model(graph, config: dict | None = None):
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    class TranslatedModel(nn.Module):
        def __init__(self, graph, config: dict | None = None):
            super().__init__()
            self.graph = graph
            self.config = config or {}
            self.device = torch.device(self.config.get("device", "cpu"))
            self.dtype = self.config.get("dtype", torch.float32)
            self.state_manager = StateManager(
                max_sequence_length=self.config.get("max_sequence_length")
            )

            # Per-model cache flag (falls back to global default)
            self._cache_enabled = bool(self.config.get("enable_translation_cache", _CACHE_ENABLED))

            self.graph.detect_cycles()

            self.layers = nn.ModuleDict()
            self.projections = nn.ModuleDict()
            self.post_aggregation_projections = nn.ModuleDict()

            # Advanced aggregation helpers
            self._adv_aggs = {
                "attention", "multi_head_attention", "gated_sum",
                "topk_weighted_sum", "moe", "attn_pool"
            }
            self._agg_input_size: dict[int, int] = {}

            self.node_sizes: dict[int, int] = {}
            for node_id, node in graph.nodes.items():
                self.node_sizes[node_id] = node.attributes.get("output_size", 1)
                if node.node_type == NodeType.INPUT:
                    continue
                act_key = f"act_{node_id}"
                if node.activation_function == "relu":
                    self.layers[act_key] = nn.ReLU()
                elif node.activation_function == "sigmoid":
                    self.layers[act_key] = nn.Sigmoid()
                elif node.activation_function == "tanh":
                    self.layers[act_key] = nn.Tanh()
                elif node.activation_function == "softmax":
                    self.layers[act_key] = nn.Softmax(dim=1)
                elif node.activation_function == "leaky_relu":
                    self.layers[act_key] = nn.LeakyReLU()
                elif node.activation_function == "elu":
                    self.layers[act_key] = nn.ELU()
                elif node.activation_function == "gelu":
                    self.layers[act_key] = nn.GELU()
                elif node.activation_function == "lstm":
                    self.layers[f"lstm_{node_id}"] = nn.LSTMCell(
                        self.node_sizes[node_id], self.node_sizes[node_id], bias=False
                    )
                elif node.activation_function == "gru":
                    self.layers[f"gru_{node_id}"] = nn.GRUCell(
                        self.node_sizes[node_id], self.node_sizes[node_id], bias=False
                    )
                elif node.activation_function == "linear":
                    self.layers[act_key] = nn.Identity()
                else:
                    raise ValueError(f"Unknown activation: {node.activation_function}")
                # Bias parameter, reuse via cache if derivation_uuid present
                bias_name = f"bias_{node_id}"
                cached = None
                deriv_uuid = node.attributes.get("derivation_uuid")
                if self._cache_enabled and deriv_uuid is not None:
                    entry = _SUBMODULE_CACHE.setdefault(str(deriv_uuid), {})
                    cached = entry.get(bias_name)
                    if cached is not None:
                        _CACHE_METRICS["hits"] += 1
                    else:
                        _CACHE_METRICS["misses"] += 1
                else:
                    # Count as a miss for observability even if cache disabled or UUID absent
                    _CACHE_METRICS["misses"] += 1
                if cached is None:
                    param = nn.Parameter(torch.tensor(float(node.bias), device=self.device, dtype=self.dtype))
                    if self._cache_enabled and deriv_uuid is not None:
                        _SUBMODULE_CACHE[str(deriv_uuid)][bias_name] = param
                        _CACHE_METRICS["entries"] = sum(len(v) for v in _SUBMODULE_CACHE.values())
                else:
                    param = cached  # type: ignore[assignment]
                self.register_parameter(bias_name, param)  # type: ignore[arg-type]

                # Advanced aggregation per-node params
                aggregation = node.attributes.get("aggregation", "sum")
                if aggregation in self._adv_aggs:
                    if aggregation in {"attention", "multi_head_attention", "attn_pool"}:
                        num_heads = int(node.attributes.get("num_heads", 1))
                        head_dim = int(node.attributes.get("head_dim", self.node_sizes[node_id]))
                        self._agg_input_size[node_id] = head_dim
                        if aggregation == "multi_head_attention":
                            q_shape = (num_heads, head_dim)
                        elif aggregation == "attn_pool":
                            pool_heads = int(node.attributes.get("pool_heads", 1))
                            q_shape = (pool_heads, head_dim)
                        else:
                            q_shape = (head_dim,)
                        self.register_parameter(
                            f"q_{node_id}",
                            nn.Parameter(torch.randn(*q_shape, device=self.device, dtype=self.dtype) * 0.01)
                        )
                    else:
                        self._agg_input_size[node_id] = self.node_sizes[node_id]

            for source_id, source_node in graph.nodes.items():
                source_size = self.node_sizes[source_id]
                # Support multigraph adjacency
                out_map = source_node.edges_out
                if isinstance(out_map, dict) and out_map and isinstance(next(iter(out_map.values())), list):
                    items = [
                        (tid, e)
                        for tid, lst in out_map.items()
                        for e in lst
                    ]
                else:
                    items = list(out_map.items())
                for target_id, edge in items:
                    if not edge.enabled:
                        continue
                    target_node = graph.nodes[target_id]
                    if target_node.node_type != NodeType.INPUT:
                        target_input_size = self._get_edge_input_size(target_node)
                        if source_size != target_input_size:
                            if getattr(graph, 'config', {}).get('multigraph'):
                                proj_key = f"proj_{edge.edge_id}"
                            else:
                                proj_key = f"proj_{source_id}_{target_id}"
                            if proj_key not in self.projections:
                                self.projections[proj_key] = nn.Linear(
                                    source_size, target_input_size, bias=False
                                )
                    # Per-edge parameters disambiguated by endpoints (simple mode) or edge_id (multigraph)
                    if getattr(graph, 'config', {}).get('multigraph'):
                        weight_key = f"weight_{edge.edge_id}"
                    else:
                        weight_key = f"weight_{source_id}_{target_id}"
                    self.register_parameter(
                        weight_key,
                        nn.Parameter(
                            torch.tensor(float(edge.weight), device=self.device, dtype=self.dtype)
                        ),
                    )

                    # Additional per-edge params for advanced aggregations
                    aggregation = target_node.attributes.get("aggregation", "sum")
                    if aggregation == "gated_sum":
                        gate_key = f"gate_{edge.edge_id}" if getattr(graph, 'config', {}).get('multigraph') else f"gate_{source_id}_{target_id}"
                        if not hasattr(self, gate_key):
                            self.register_parameter(gate_key, nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.dtype)))
                    if aggregation == "moe":
                        router_key = f"router_{edge.edge_id}" if getattr(graph, 'config', {}).get('multigraph') else f"router_{source_id}_{target_id}"
                        if not hasattr(self, router_key):
                            self.register_parameter(router_key, nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.dtype)))

            for node_id, node in graph.nodes.items():
                aggregated_size = self._calculate_aggregated_size(node)
                output_size = self.node_sizes[node_id]
                if aggregated_size != output_size:
                    post_name = f"post_{node_id}"
                    deriv_uuid = graph.nodes[node_id].attributes.get("derivation_uuid")
                    cached = None
                    if self._cache_enabled and deriv_uuid is not None:
                        entry = _SUBMODULE_CACHE.setdefault(str(deriv_uuid), {})
                        cached = entry.get(post_name)
                        if cached is not None and isinstance(cached, nn.Linear) and cached.in_features == aggregated_size and cached.out_features == output_size:
                            _CACHE_METRICS["hits"] += 1
                        else:
                            _CACHE_METRICS["misses"] += 1
                    else:
                        _CACHE_METRICS["misses"] += 1
                    if cached is None or not isinstance(cached, nn.Linear) or cached.in_features != aggregated_size or cached.out_features != output_size:
                        layer = nn.Linear(aggregated_size, output_size, bias=False)
                        if self._cache_enabled and deriv_uuid is not None:
                            _SUBMODULE_CACHE[str(deriv_uuid)][post_name] = layer
                            _CACHE_METRICS["entries"] = sum(len(v) for v in _SUBMODULE_CACHE.values())
                    else:
                        layer = cached
                    self.post_aggregation_projections[post_name] = layer  # type: ignore[assignment]

            self.execution_order = graph.topological_sort(ignore_recurrent=True)
            self.to(self.device, self.dtype)

        def _get_edge_input_size(self, node) -> int:
            aggregation = node.attributes.get("aggregation", "sum")
            if aggregation in self._adv_aggs and node.node_id in self._agg_input_size:
                return self._agg_input_size[node.node_id]
            return self.node_sizes[node.node_id]

        def _calculate_aggregated_size(self, node) -> int:
            aggregation = node.attributes.get("aggregation", "sum")
            if aggregation in ["sum", "mean", "max"]:
                return self._get_edge_input_size(node)
            if aggregation in ["concat", "matrix_product"]:
                total_size = 0
                # Build list of incoming (source_id, edge) pairs to respect multiplicity
                in_map = node.edges_in
                if isinstance(in_map, dict) and in_map and isinstance(next(iter(in_map.values())), list):
                    in_items = [
                        (sid, e)
                        for sid, lst in in_map.items()
                        for e in lst
                    ]
                else:
                    in_items = [(sid, e) for sid, e in in_map.items()]

                for source_id, _edge in in_items:
                    source_size = self.node_sizes[source_id]
                    if getattr(self.graph, 'config', {}).get('multigraph'):
                        # In multigraph mode, projection may be keyed by edge_id; post-agg size is still target input size per edge
                        # For sizing, if a projection exists for any edge from source_id, count target size; else count source size
                        # We approximate by counting target size since per-edge projections are created when needed
                        total_size += self._get_edge_input_size(node) if source_size != self._get_edge_input_size(node) else source_size
                    else:
                        proj_key = f"proj_{source_id}_{node.node_id}"
                        if proj_key in self.projections:
                            total_size += self._get_edge_input_size(node)
                        else:
                            total_size += source_size
                return max(1, total_size)
            if aggregation == "attention":
                return self._agg_input_size.get(node.node_id, self.node_sizes[node.node_id])
            if aggregation == "multi_head_attention":
                num_heads = int(node.attributes.get("num_heads", 1))
                head_dim = int(self._agg_input_size.get(node.node_id, self.node_sizes[node.node_id]))
                return num_heads * head_dim
            if aggregation == "attn_pool":
                pool_heads = int(node.attributes.get("pool_heads", 1))
                return pool_heads * self._get_edge_input_size(node)
            if aggregation in {"gated_sum", "topk_weighted_sum", "moe"}:
                return self._get_edge_input_size(node)
            return self._get_edge_input_size(node)

        def forward(self, x, reset_states: bool = False):
            import torch

            batch_size = x.size(0)
            device = x.device
            if reset_states:
                self.state_manager.reset()
            self.state_manager.initialize(batch_size, device)

            node_outputs: dict[int, Any] = {}

            input_sizes = [self.node_sizes[i] for i in self.graph.input_node_ids]
            expected_input_width = sum(input_sizes)
            actual_input_width = x.size(1)
            if actual_input_width != expected_input_width:
                raise ValueError(
                    f"Input width mismatch: expected {expected_input_width} from INPUT nodes "
                    f"{self.graph.input_node_ids}, got {actual_input_width}"
                )
            if len(input_sizes) > 1:
                input_splits = torch.split(x, input_sizes, dim=1)
                for idx, input_id in enumerate(self.graph.input_node_ids):
                    node_outputs[input_id] = input_splits[idx]
            elif len(input_sizes) == 1:
                node_outputs[self.graph.input_node_ids[0]] = x

            for node_id in self.execution_order:
                if node_id in self.graph.input_node_ids:
                    node = self.graph.nodes[node_id]
                    has_enabled_in = any(
                        edge.enabled for edge in node.edges_in.values()
                    ) if not isinstance(
                        node.edges_in.get(next(iter(node.edges_in), None), None), list
                    ) else any(
                        e.enabled for lst in node.edges_in.values() for e in lst
                    )
                    if not has_enabled_in:
                        continue

                node = self.graph.nodes[node_id]
                inputs = []
                edge_meta = []
                routers = []
                if node.node_type == NodeType.INPUT and node_id in node_outputs:
                    inputs.append(node_outputs[node_id])

                # Multigraph-aware incoming edges iteration
                in_map = node.edges_in
                if isinstance(in_map, dict) and in_map and isinstance(next(iter(in_map.values())), list):
                    in_items = [
                        (sid, e)
                        for sid, lst in in_map.items()
                        for e in lst
                    ]
                else:
                    in_items = list(in_map.items())
                for source_id, edge in in_items:
                    if not edge.enabled:
                        continue
                    if edge.attributes.get("is_recurrent", False):
                        source_output = self.state_manager.get_prev_output(source_id)
                        if source_output is None:
                            source_output = torch.zeros(
                                batch_size, self.node_sizes[source_id], device=device
                            )
                    else:
                        source_output = node_outputs.get(source_id)
                        if source_output is None:
                            continue

                    # Apply per-edge projection if needed
                    if getattr(self.graph, 'config', {}).get('multigraph'):
                        proj_key_edge = f"proj_{edge.edge_id}"
                        if proj_key_edge in self.projections:
                            source_output = self.projections[proj_key_edge](source_output)
                    else:
                        proj_key = f"proj_{source_id}_{node_id}"
                        if proj_key in self.projections:
                            source_output = self.projections[proj_key](source_output)

                    if getattr(self.graph, 'config', {}).get('multigraph'):
                        weight_key = f"weight_{edge.edge_id}"
                    else:
                        weight_key = f"weight_{source_id}_{node_id}"
                    weight = getattr(self, weight_key)
                    weighted = source_output * weight

                    # Apply per-edge gate for gated_sum
                    aggregation = node.attributes.get("aggregation", "sum")
                    if aggregation == "gated_sum":
                        gate_key = f"gate_{edge.edge_id}" if getattr(self.graph, 'config', {}).get('multigraph') else f"gate_{source_id}_{node_id}"
                        gate = getattr(self, gate_key)
                        import torch as _torch
                        weighted = weighted * _torch.sigmoid(gate)

                    if aggregation == "moe":
                        router_key = f"router_{edge.edge_id}" if getattr(self.graph, 'config', {}).get('multigraph') else f"router_{source_id}_{node_id}"
                        routers.append(getattr(self, router_key))

                    inputs.append(weighted)
                    edge_meta.append((source_id, edge))

                if not inputs:
                    aggregated = torch.zeros(
                        batch_size, self._get_edge_input_size(node), device=device
                    )
                else:
                    aggregation = node.attributes.get("aggregation", "sum")
                    if aggregation == "sum":
                        aggregated = sum(inputs)
                    elif aggregation == "mean":
                        aggregated = sum(inputs) / len(inputs)
                    elif aggregation == "max":
                        aggregated = torch.stack(inputs).max(dim=0)[0]
                    elif aggregation == "concat":
                        aggregated = torch.cat(inputs, dim=1)
                    elif aggregation == "matrix_product":
                        if len(inputs) > 1:
                            stacked = torch.stack(inputs, dim=1)
                            flattened = stacked.view(stacked.size(0), -1)
                            aggregated = flattened
                        else:
                            aggregated = inputs[0]
                    elif aggregation == "attention":
                        import math as _math
                        q = getattr(self, f"q_{node_id}")
                        stacked = torch.stack(inputs, dim=1)
                        # q shape: [dim]
                        scores = (stacked @ q) / (_math.sqrt(max(1.0, float(q.numel()))) * float(node.attributes.get("temperature", 1.0)))
                        weights = torch.softmax(scores, dim=1)
                        aggregated = (weights.unsqueeze(-1) * stacked).sum(dim=1)
                    elif aggregation == "multi_head_attention":
                        import math as _math
                        q = getattr(self, f"q_{node_id}")  # [H, dim]
                        stacked = torch.stack(inputs, dim=1)
                        heads = []
                        for h in range(q.size(0)):
                            qh = q[h]
                            scores = (stacked @ qh) / (_math.sqrt(max(1.0, float(qh.numel()))) * float(node.attributes.get("temperature", 1.0)))
                            weights = torch.softmax(scores, dim=1)
                            head_out = (weights.unsqueeze(-1) * stacked).sum(dim=1)
                            heads.append(head_out)
                        aggregated = torch.cat(heads, dim=1)
                    elif aggregation == "topk_weighted_sum":
                        stacked = torch.stack(inputs, dim=1)
                        scores = stacked.mean(dim=2)
                        top_k = node.attributes.get("top_k")
                        if top_k is not None:
                            k = int(top_k)
                            k = max(1, min(k, scores.size(1)))
                            vals, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
                            sel = torch.gather(stacked, 1, idx.unsqueeze(-1).expand(-1, -1, stacked.size(2)))
                            weights = torch.softmax(vals, dim=1)
                            aggregated = (weights.unsqueeze(-1) * sel).sum(dim=1)
                        else:
                            weights = torch.softmax(scores, dim=1)
                            aggregated = (weights.unsqueeze(-1) * stacked).sum(dim=1)
                    elif aggregation == "moe":
                        import torch as _torch
                        if routers:
                            logits = _torch.stack(routers)
                            logits = logits.unsqueeze(0).expand(batch_size, -1)
                        else:
                            logits = _torch.zeros(len(inputs), device=device, dtype=self.dtype).unsqueeze(0).expand(batch_size, -1)
                        if node.attributes.get("router_type", "softmax") == "topk":
                            k = int(node.attributes.get("top_k", 1))
                            k = max(1, min(k, logits.size(1)))
                            vals, idx = _torch.topk(logits, k=k, dim=1, largest=True, sorted=True)
                            stacked = _torch.stack(inputs, dim=1)
                            sel = _torch.gather(stacked, 1, idx.unsqueeze(-1).expand(-1, -1, stacked.size(2)))
                            weights = _torch.softmax(vals, dim=1)
                            aggregated = (weights.unsqueeze(-1) * sel).sum(dim=1)
                        else:
                            weights = _torch.softmax(logits, dim=1)
                            stacked = _torch.stack(inputs, dim=1)
                            aggregated = (weights.unsqueeze(-1) * stacked).sum(dim=1)
                    elif aggregation == "attn_pool":
                        import math as _math
                        q = getattr(self, f"q_{node_id}")  # [H, dim]
                        stacked = torch.stack(inputs, dim=1)
                        heads = []
                        for h in range(q.size(0)):
                            qh = q[h]
                            scores = (stacked @ qh) / _math.sqrt(max(1.0, float(qh.numel())))
                            weights = torch.softmax(scores, dim=1)
                            head_out = (weights.unsqueeze(-1) * stacked).sum(dim=1)
                            heads.append(head_out)
                        aggregated = torch.cat(heads, dim=1)
                    else:
                        aggregated = sum(inputs)

                post_key = f"post_{node_id}"
                if post_key in self.post_aggregation_projections:
                    aggregated = self.post_aggregation_projections[post_key](aggregated)

                if node.node_type == NodeType.INPUT:
                    output = aggregated
                else:
                    bias = getattr(self, f"bias_{node_id}")
                    aggregated = aggregated + bias
                    if node.activation_function == "lstm":
                        state = self.state_manager.get_state(
                            node_id, self.node_sizes[node_id], "lstm"
                        )
                        h, c = self.layers[f"lstm_{node_id}"](aggregated, state)
                        self.state_manager.update_state(node_id, (h, c))
                        output = h
                    elif node.activation_function == "gru":
                        state = self.state_manager.get_state(
                            node_id, self.node_sizes[node_id], "default"
                        )
                        h = self.layers[f"gru_{node_id}"](aggregated, state)
                        self.state_manager.update_state(node_id, h)
                        output = h
                    else:
                        output = self.layers[f"act_{node_id}"](aggregated)

                node_outputs[node_id] = output

            for nid, out in node_outputs.items():
                self.state_manager.store_output(nid, out)

            outputs = [
                node_outputs[oid]
                for oid in self.graph.output_node_ids
                if oid in node_outputs
            ]
            if len(outputs) == 1:
                return outputs[0]
            if outputs:
                return torch.cat(outputs, dim=1)
            return torch.empty(batch_size, 0, device=device)

    return TranslatedModel(graph, config)
