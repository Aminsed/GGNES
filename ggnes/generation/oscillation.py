"""Oscillation detection strategies per project_guide.md.

Implements:
- detect_oscillation (aggregator with precedence)
- detect_simple_reversal
- detect_state_fingerprinting
- detect_growth_monitoring
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def detect_oscillation(
    graph_history: Any,
    graph: Any,
    rule: Any,
    bindings: dict,
    config: dict,
) -> tuple[bool, str | None]:
    """Detect oscillation using configured strategies.

    Precedence: SIMPLE_REVERSAL → STATE_FINGERPRINTING → GROWTH_MONITORING.
    First positive detection wins.
    """
    strategies = config.get("oscillation_strategy", "ALL")

    if strategies == "ALL":
        strategies = [
            "SIMPLE_REVERSAL",
            "STATE_FINGERPRINTING",
            "GROWTH_MONITORING",
        ]
    elif not isinstance(strategies, list):
        strategies = [strategies]

    for strategy in strategies:
        if strategy == "SIMPLE_REVERSAL":
            detected, reason = detect_simple_reversal(
                rule, getattr(graph_history, "action_history", []), config
            )
            if detected:
                return True, f"[SIMPLE_REVERSAL] {reason}"
        elif strategy == "STATE_FINGERPRINTING":
            detected, reason = detect_state_fingerprinting(graph, graph_history, config)
            if detected:
                return True, f"[STATE_FINGERPRINTING] {reason}"
        elif strategy == "GROWTH_MONITORING":
            detected, reason = detect_growth_monitoring(
                getattr(graph_history, "metrics", []), config
            )
            if detected:
                return True, f"[GROWTH_MONITORING] {reason}"

    return False, None


def detect_simple_reversal(
    rule: Any, action_history: Iterable[dict], config: dict
) -> tuple[bool, str | None]:
    """Detect if rule undoes a previous action within a window.

    A rule is considered a reversal if its rule_type appears as a reverse pair
    to a recent action's rule_type (in either direction as per spec).
    """
    rule_type = getattr(rule, "metadata", {}).get("rule_type", "unknown")
    reverse_pairs = config.get("rule_reverse_pairs", {})

    reverse_types = []
    for key, val in reverse_pairs.items():
        if key == rule_type:
            reverse_types.append(val)
        elif val == rule_type:
            reverse_types.append(key)

    if not reverse_types:
        return False, None

    window = min(len(action_history), config.get("oscillation_history_window", 10))
    for i in range(1, window + 1):
        prev = action_history[-i]
        prev_type = prev.get("rule_info", ("unknown", None))[0]
        if prev_type in reverse_types:
            return True, f"Rule {rule_type} reverses previous {prev_type}"

    return False, None


def detect_state_fingerprinting(
    graph: Any, graph_history: Any, config: dict
) -> tuple[bool, str | None]:
    """Detect cycles via fingerprint matching.

    Compares current fingerprint to prior ones excluding a recent window,
    returning True with a cycle length message if a match is found.
    """
    current_fp = graph.compute_fingerprint()
    fingerprints = getattr(graph_history, "fingerprints", [])
    exclusion_window = config.get("fingerprint_exclusion_window", 1)

    if len(fingerprints) > exclusion_window:
        check_fingerprints = (
            fingerprints[:-exclusion_window] if exclusion_window > 0 else fingerprints
        )
        if current_fp in check_fingerprints:
            idx = fingerprints.index(current_fp)
            cycle_length = len(fingerprints) - idx
            return True, f"State cycle detected (length {cycle_length})"

    return False, None


def detect_growth_monitoring(metrics: Iterable[dict], config: dict) -> tuple[bool, str | None]:
    """Detect lack of productive growth when rules are being applied.

    Flags when the last N entries show no change in num_nodes and num_edges,
    and the sum of rule_applied counters is > 0.
    """
    metrics = list(metrics)
    check_depth = config.get("oscillation_check_depth", 5)

    if len(metrics) < check_depth:
        return False, None

    recent = metrics[-check_depth:]
    node_counts = [m.get("num_nodes") for m in recent]
    edge_counts = [m.get("num_edges") for m in recent]
    rules_applied = [m.get("rule_applied", 0) for m in recent]

    if all(n == node_counts[0] for n in node_counts) and all(
        e == edge_counts[0] for e in edge_counts
    ):
        total_applications = sum(rules_applied)
        if total_applications > 0:
            return (
                True,
                (
                    "No structural growth in "
                    f"{check_depth} iterations despite {total_applications} rule applications"
                ),
            )

    return False, None
