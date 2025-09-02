"""Network generation per project_guide.md §7.3.

Implements:
- RuleCooldown: temporary cooldowns to avoid immediate reselection
- GraphHistory: fingerprints, action history, and metrics tracking
- generate_network: applies rules iteratively with oscillation detection,
  selection strategy, cooldowns, and optional repair on invalid final graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ggnes.core.graph import Graph
from ggnes.evolution.selection import select_match
from ggnes.generation.matching import find_subgraph_matches
from ggnes.generation.oscillation import detect_oscillation
from ggnes.generation.rule_engine import RuleEngine
from ggnes.repair.repair import repair as repair_fn
from ggnes.utils.rng_manager import RNGManager


class RuleCooldown:
    """Manages temporary rule cooldowns to prevent oscillation."""

    def __init__(self, cooldown_iterations: int = 5) -> None:
        self.cooldown_iterations = max(0, int(cooldown_iterations))
        self._cooldowns: dict[Any, int] = {}

    def add_cooldown(self, rule_id: Any) -> None:
        if self.cooldown_iterations > 0 and rule_id is not None:
            self._cooldowns[rule_id] = self.cooldown_iterations

    def update(self) -> None:
        to_delete: list[Any] = []
        for rid, val in self._cooldowns.items():
            nv = max(0, val - 1)
            if nv == 0:
                to_delete.append(rid)
            else:
                self._cooldowns[rid] = nv
        for rid in to_delete:
            del self._cooldowns[rid]

    def is_cooled_down(self, rule_id: Any) -> bool:
        return rule_id not in self._cooldowns

    def clear_cooldown(self, rule_id: Any) -> None:
        if rule_id in self._cooldowns:
            del self._cooldowns[rule_id]


@dataclass
class GraphHistory:
    """Tracks graph evolution during generation."""

    fingerprints: list[str] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)

    def add_fingerprint(self, fingerprint: str) -> None:
        self.fingerprints.append(fingerprint)

    def add_action(self, rule_type: str, rule_id: Any, affected_nodes: set[int] | None = None) -> None:
        self.action_history.append({
            'rule_info': (rule_type, rule_id),
            'affected_nodes': affected_nodes or set(),
        })

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.append(dict(metrics))


def generate_network(genotype: Any, axiom_graph: Graph, config: dict, rng_manager: RNGManager, id_manager: Any | None = None):
    """Generate a phenotype network from a genotype.

    Args:
        genotype: Object with a `rules` list. Each rule should have attributes:
                  rule_id, lhs (dict pattern), rhs, embedding, metadata, condition (optional).
        axiom_graph: Starting graph (will be deep-copied).
        config: Configuration dict (see project_guide.md §12.1 keys for Generation/Repair/Graph).
        rng_manager: RNGManager instance for determinism.
        id_manager: Optional ID manager for global ID tracking.

    Returns:
        (graph, generation_metrics)
    """
    import copy
    import logging

    graph = copy.deepcopy(axiom_graph)
    graph.reset_id_counter()

    iteration_count = 0
    graph_history = GraphHistory()
    oscillation_skips_this_iter = 0
    total_oscillation_skips = 0
    rule_cooldown = RuleCooldown(int(config.get('cooldown_iterations', 5)))

    # Graph context baseline
    graph_context: dict[str, Any] = {
        'num_nodes': len(graph.nodes),
        'num_edges': sum(len(n.edges_out) for n in graph.nodes.values()),
        'iteration': 0,
        'node_type_counts': {},  # optional; not used by our conditions directly
    }
    # Add configured custom metrics initialized to 0
    for key in config.get('graph_context_keys', []) or []:
        graph_context[key] = 0

    max_iterations = int(config.get('max_iterations', 50))
    selection_strategy = str(config.get('selection_strategy', 'PRIORITY_THEN_PROBABILITY_THEN_ORDER'))

    while iteration_count < max_iterations:
        # Update cooldowns
        rule_cooldown.update()

        # Gather matches across rules
        potential_matches: list[tuple[Any, dict]] = []
        for rule in getattr(genotype, 'rules', []) or []:
            if not rule_cooldown.is_cooled_down(getattr(rule, 'rule_id', None)):
                continue

            lhs = getattr(rule, 'lhs', None) or {'nodes': [], 'edges': []}
            # Support both dict LHS and LHSPattern objects from rules.rule
            if not isinstance(lhs, dict) and hasattr(lhs, 'nodes') and hasattr(lhs, 'edges'):
                lhs = {
                    'nodes': list(getattr(lhs, 'nodes', []) or []),
                    'edges': list(getattr(lhs, 'edges', []) or []),
                }
            matches_iter = find_subgraph_matches(graph, lhs, int(config.get('max_match_time_ms', 1000)))
            # Filter by condition if present
            for bindings in matches_iter:
                cond = getattr(rule, 'condition', None)
                if cond is not None and not cond(graph, dict(bindings), graph_context):
                    continue
                potential_matches.append((rule, bindings))

        # Quiescence
        if not potential_matches:
            logging.info('Quiescence: no rules match')
            break

        # Selection
        selected = select_match(potential_matches, selection_strategy, rng_manager, config)
        if not selected:
            break
        selected_rule, selected_bindings = selected

        # Oscillation detection
        detected, reason = detect_oscillation(graph_history, graph, selected_rule, selected_bindings, config)
        if detected:
            action = str(config.get('oscillation_action', 'TERMINATE'))
            if action == 'TERMINATE':
                logging.info(f"Oscillation detected: {reason}. Terminating.")
                break
            if action == 'SKIP_AND_RESELECT':
                rule_cooldown.add_cooldown(getattr(selected_rule, 'rule_id', None))
                oscillation_skips_this_iter += 1
                total_oscillation_skips += 1
                if oscillation_skips_this_iter > int(config.get('max_consecutive_oscillation_skips', 10)):
                    logging.info("Max consecutive oscillation skips reached. Terminating.")
                    break
                if total_oscillation_skips > int(config.get('max_total_oscillation_skips', 50)):
                    logging.info("Max total oscillation skips reached. Terminating.")
                    break
                # Reselect next iteration
                continue
            if action == 'IGNORE':
                logging.warning(f"Oscillation detected but ignored: {reason}")

        # Apply selected rule
        engine = RuleEngine(graph=graph, rng_manager=rng_manager, id_manager=id_manager)
        success = engine.apply_rule(selected_rule, selected_bindings)
        if not success:
            import logging as _logging
            _logging.error(f"Rule application failed for rule {getattr(selected_rule, 'rule_id', None)}")
            break

        # Track history and metrics after successful application
        graph_history.add_fingerprint(graph.compute_fingerprint())
        rule_type = getattr(selected_rule, 'metadata', {}).get('rule_type', 'unknown')
        rule_id = getattr(selected_rule, 'rule_id', None)
        graph_history.add_action(rule_type, rule_id, set())

        graph_context['num_nodes'] = len(graph.nodes)
        graph_context['num_edges'] = sum(len(n.edges_out) for n in graph.nodes.values())
        graph_context['iteration'] = iteration_count + 1
        graph_history.add_metrics({
            'num_nodes': graph_context['num_nodes'],
            'num_edges': graph_context['num_edges'],
            'iteration': graph_context['iteration'],
            'rule_applied': 1,
        })

        # Reset per-iteration skip counter on success
        oscillation_skips_this_iter = 0
        # Clear cooldown for successfully applied rule
        rule_cooldown.clear_cooldown(rule_id)

        iteration_count += 1

    # Final validation and optional repair
    errors: list[Any] = []
    is_valid = graph.validate(collect_errors=errors)
    repair_metrics: dict[str, Any] | None = None
    if not is_valid:
        # Construct repair config from main config
        repair_config = {
            'strategy': config.get('repair_strategy', 'MINIMAL_CHANGE'),
            'allowed_repairs': config.get('allowed_repairs', ['fix_weights', 'add_missing_attributes']),
            'max_repair_iterations': config.get('max_repair_iterations', 5),
        }
        repair_successful, repair_metrics = repair_fn(graph, repair_config, rng_manager)
        if not repair_successful:
            import logging as _logging
            _logging.warning('Repair failed, assigning minimal fitness (handled by caller)')

    generation_metrics = {
        'iterations': iteration_count,
        'oscillation_skips': total_oscillation_skips,
        'final_nodes': len(graph.nodes),
        'final_edges': sum(len(n.edges_out) for n in graph.nodes.values()),
        'repair_metrics': repair_metrics,
    }

    return graph, generation_metrics


__all__ = [
    'RuleCooldown',
    'GraphHistory',
    'generate_network',
]

