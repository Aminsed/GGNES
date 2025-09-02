"""Compile-time projection planning (shape inference) for hierarchical modules.

Produces a deterministic, bias-free projection plan objects that can be
consumed by translators, without altering runtime semantics in this step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ggnes.hierarchical.module_spec import ModuleSpec


@dataclass(frozen=True)
class ProjectionPlan:
    target_node_id: int
    inputs: Tuple[Tuple[int, int], ...]  # (source_node_id, expected_size)
    post_aggregation_size: int


def plan_projections(
    spec: ModuleSpec,
    bound_params: Mapping[str, Any],
    incoming_edges: List[Tuple[int, int]],
    *,
    aggregation: str = 'sum',
) -> ProjectionPlan:
    """Create a deterministic projection plan for a module node.

    Args:
        spec: ModuleSpec (ports are consulted for sizes)
        bound_params: validated/bound params (from ModuleSpec)
        incoming_edges: list of (source_node_id, source_size)
        aggregation: aggregator; determines how sizes combine

    Returns:
        ProjectionPlan with per-input expected sizes and post-aggregation size.
    """
    # Determine target output size from ports (favor 'out')
    port_map = {p.name: p for p in spec.ports}
    out_size = port_map.get('out', next(iter(port_map.values()))).size if port_map else int(bound_params.get('model_dim', 1))

    expected_input_size = out_size
    inputs: List[Tuple[int, int]] = [(src, expected_input_size) for src, _sz in incoming_edges]

    if aggregation == 'concat':
        post_size = expected_input_size * max(1, len(incoming_edges))
    elif aggregation == 'matrix_product':
        post_size = expected_input_size * max(1, len(incoming_edges))
    else:
        post_size = expected_input_size

    return ProjectionPlan(target_node_id=-1, inputs=tuple(inputs), post_aggregation_size=post_size)


