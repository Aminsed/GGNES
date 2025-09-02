"""Hierarchical matching utilities (project_guide.md §20.5).

Provides small, deterministic helpers to match ModuleSpec instances using
predicates over parameters, ports, and attributes. This is intentionally
minimal and independent from the core graph matcher so tests can strictly
validate hierarchical semantics without introducing heavy coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ggnes.hierarchical.module_spec import ModuleSpec, PortSpec


ParamPredicate = Callable[[Mapping[str, Any]], bool]
PortPredicate = Callable[[Mapping[str, PortSpec]], bool]
AttrPredicate = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class MatchCriteria:
    """Criteria for hierarchical matching.

    All predicates must return True for a successful match. Missing predicates
    are treated as vacuously true.
    """

    param_predicates: Tuple[ParamPredicate, ...] = ()
    port_predicates: Tuple[PortPredicate, ...] = ()
    attr_predicates: Tuple[AttrPredicate, ...] = ()


def evaluate_match(spec: ModuleSpec, overrides: Optional[Dict[str, Any]] = None, criteria: Optional[MatchCriteria] = None) -> bool:
    """Evaluate hierarchical match against a ModuleSpec.

    Args:
        spec: ModuleSpec to evaluate
        overrides: Parameter overrides used to bind values
        criteria: MatchCriteria with predicates over params/ports/attributes

    Returns:
        True if all provided predicates pass; False otherwise.
    """
    if criteria is None:
        return True

    # Bind parameters (validates invariants too via ModuleSpec)
    env = spec.validate_and_bind_params(overrides or {})

    # Build ports map by name
    ports: Dict[str, PortSpec] = {p.name: p for p in spec.ports}
    # Attributes as-is
    attrs: Dict[str, Any] = dict(spec.attributes)

    # Evaluate in a deterministic, side-effect free manner
    for pred in criteria.param_predicates:
        if not bool(pred(env)):
            return False
    for pred in criteria.port_predicates:
        if not bool(pred(ports)):
            return False
    for pred in criteria.attr_predicates:
        if not bool(pred(attrs)):
            return False

    return True


def make_param_predicate(fn: Callable[[Mapping[str, Any]], bool]) -> ParamPredicate:
    return fn


def make_port_predicate(fn: Callable[[Mapping[str, PortSpec]], bool]) -> PortPredicate:
    return fn


def make_attr_predicate(fn: Callable[[Mapping[str, Any]], bool]) -> AttrPredicate:
    return fn


def rank_module_candidates(
    specs: List[ModuleSpec],
    overrides: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[ModuleSpec, str]]:
    """Rank module candidates deterministically.

    Sort key: (name, version, binding_signature)

    Args:
        specs: list of ModuleSpec to rank
        overrides: optional list of overrides per spec (len matches specs or None)

    Returns:
        List of (spec, signature) sorted deterministically.
    """
    signatures: List[Tuple[ModuleSpec, str]] = []
    if overrides is None:
        overrides = [{} for _ in specs]
    for spec, ov in zip(specs, overrides):
        env = spec.validate_and_bind_params(ov or {})
        signatures.append((spec, spec.binding_signature(env)))
    signatures.sort(key=lambda t: (t[0].name, int(t[0].version), t[1]))
    return signatures


def rank_module_candidates_with_limit(
    specs: List[ModuleSpec],
    overrides: Optional[List[Dict[str, Any]]] = None,
    *,
    limit: Optional[int] = None,
) -> List[Tuple[ModuleSpec, str]]:
    """Rank module candidates deterministically and truncate to limit.

    Deterministic truncation preserves the sorted order. If limit is None or
    larger than the number of candidates, all are returned.
    """
    ranked = rank_module_candidates(specs, overrides)
    if limit is None:
        return ranked
    return ranked[: max(0, int(limit))]


