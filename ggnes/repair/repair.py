"""Repair function and strategies (per project_guide.md §14)."""

from __future__ import annotations

from typing import Any

from ..core.node import NodeType


def repair(graph, config: dict | None = None, rng_manager=None) -> tuple[bool, dict[str, Any]]:
    """Attempt to repair an invalid graph.

    Repair happens exactly once after generation and before translation.
    All randomness MUST come from RNGManager for determinism.

    Returns:
        (success, repair_metrics)
    """
    if config is None:
        config = {'strategy': 'MINIMAL_CHANGE'}

    strategy = config.get('strategy', 'MINIMAL_CHANGE')
    repair_metrics: dict[str, Any] = {
        'repairs_attempted': [],
        'repairs_succeeded': [],
        'repair_impact_score': 0.0,
        'strategy': strategy,
    }

    # Collect validation errors as structured objects
    errors = []
    graph.validate(collect_errors=errors)

    # Nothing to do
    if not errors:
        return True, repair_metrics

    # DISABLE strategy: no repair, return immediately
    if strategy == 'DISABLE':
        return False, repair_metrics

    # RNG for repairs
    rng = rng_manager.get_context_rng('repair') if rng_manager else None

    allowed_repairs = config.get(
        'allowed_repairs', ['fix_weights', 'add_missing_attributes']
    )
    impact = 0.0

    for error in errors:
        repair_attempted = None
        repair_succeeded = False

        # Fix non-finite bias
        if error.error_type == 'non_finite_bias' and 'fix_weights' in allowed_repairs:
            repair_attempted = 'fix_non_finite_bias'
            node_id = error.node_id
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                if strategy == 'MINIMAL_CHANGE':
                    node.bias = 0.0
                else:
                    node.bias = rng.uniform(-0.1, 0.1) if rng else 0.0
                impact += 0.05
                repair_succeeded = True

        # Fix non-finite weight
        elif error.error_type == 'non_finite_weight' and 'fix_weights' in allowed_repairs:
            repair_attempted = 'fix_non_finite_weight'
            edge = graph.find_edge_by_id(error.edge_id)
            if edge:
                if strategy == 'MINIMAL_CHANGE':
                    edge.weight = 0.1
                else:
                    edge.weight = rng.uniform(0.01, 0.2) if rng else 0.1
                impact += 0.05
                repair_succeeded = True

        # Add missing output_size
        elif (
            error.error_type == 'missing_output_size'
            and 'add_missing_attributes' in allowed_repairs
        ):
            repair_attempted = 'add_output_size'
            node_id = error.node_id
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                if strategy == 'MINIMAL_CHANGE':
                    node.attributes['output_size'] = 16
                else:
                    node.attributes['output_size'] = (
                        rng.choice([8, 16, 32, 64]) if rng else 16
                    )
                impact += 0.1
                repair_succeeded = True

        # Connect unreachable outputs (AGGRESSIVE only)
        elif (
            error.error_type == 'unreachable_output' and strategy == 'AGGRESSIVE'
        ):
            repair_attempted = 'connect_unreachable'
            output_id = error.node_id
            hidden_nodes = [
                n for n in graph.nodes.values() if n.node_type == NodeType.HIDDEN
            ]
            if hidden_nodes and rng:
                source = rng.choice(hidden_nodes)
                edge_id = graph.add_edge(source.node_id, output_id, {'weight': 0.1})
                if edge_id is not None:
                    impact += 0.2
                    repair_succeeded = True

        # Clamp/repair invalid advanced aggregation parameters
        elif error.error_type == 'invalid_agg_param':
            node_id = getattr(error, 'node_id', None)
            pname = error.details.get('param_name') if hasattr(error, 'details') else None
            if node_id in graph.nodes and pname:
                node = graph.nodes[node_id]
                repair_attempted = f"set_default_{pname}"
                # Defaults per project_guide.md §8.4
                defaults = {
                    'temperature': 1.0,
                    'dropout_p': 0.0,
                    'attn_eps': 1e-6,
                    'post_projection': True,
                    'normalize': False,
                    'num_heads': 1,
                    'top_k': None,
                    'attn_type': 'dot',
                    'router_type': 'softmax',
                    'pool_heads': 1,
                }
                value = defaults.get(pname)
                # Generic clamps
                if pname in {'head_dim'}:
                    value = node.attributes.get('output_size', 1)
                if pname in {'experts', 'pool_heads'} and (value is None):
                    value = 1
                if pname in {'capacity_factor', 'router_temperature'} and (value is None):
                    value = 1.0
                node.attributes[pname] = value
                # Impact for moderate attribute fixes
                impact += 0.1
                repair_succeeded = True

        if repair_attempted:
            repair_metrics['repairs_attempted'].append(repair_attempted)
            if repair_succeeded:
                repair_metrics['repairs_succeeded'].append(repair_attempted)

    repair_metrics['repair_impact_score'] = min(impact, 1.0)

    # Revalidate after attempted repairs
    final_errors = []
    is_valid = graph.validate(collect_errors=final_errors)
    return is_valid, repair_metrics


def calculate_repair_penalty(repair_metrics: dict | None) -> float:
    """Calculate fitness penalty from repair impact (project_guide.md §14.3)."""
    if not repair_metrics:
        return 0.0

    impact = repair_metrics.get('repair_impact_score', 0.0)
    if impact <= 0.0:
        return 0.0
    if impact <= 0.1:
        return impact * 0.05
    if impact <= 0.5:
        return 0.005 + (impact - 0.1) * 0.1
    return 0.045 + (impact - 0.5) * 0.2
