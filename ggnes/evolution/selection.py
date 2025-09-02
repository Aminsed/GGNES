"""Selection strategies (project_guide.md §7.4, §21.4 for M28).

This module implements:
- select_match: rule match selection (PRIORITY_THEN_PROBABILITY_THEN_ORDER)
- NSGA-II style multi-objective selection with deterministic tie-breakers

Determinism rules:
- All randomness comes from RNGManager contexts
- Canonical sorting keys used for breaking ties
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from ggnes.utils.rng_manager import RNGManager


def select_match(potential_matches: list[tuple[Any, dict]], strategy: str, rng_manager: RNGManager, config: dict):
    """Select a match based on strategy.

    Implements PRIORITY_THEN_PROBABILITY_THEN_ORDER per spec with grouping precision.
    Returns selected (rule, bindings) tuple or None.
    """
    if not potential_matches:
        return None

    if strategy != 'PRIORITY_THEN_PROBABILITY_THEN_ORDER':
        raise ValueError(f"Unknown selection strategy: {strategy}")

    # 1) Filter for highest priority
    max_priority = max(match[0].metadata.get('priority', 0) for match in potential_matches)
    filtered = [m for m in potential_matches if m[0].metadata.get('priority', 0) == max_priority]

    # 2) Select by probability weights
    probs = [float(m[0].metadata.get('probability', 1.0)) for m in filtered]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / len(filtered) for _ in filtered]

    # 3) Group by probability with precision, then tie-break by original order
    prob_groups: dict[float, list[tuple[int, tuple[Any, dict]]]] = {}
    precision = float(config.get('probability_precision', 1e-6))
    for m, p in zip(filtered, probs):
        rounded = round(p / precision) * precision
        idx = potential_matches.index(m)
        prob_groups.setdefault(rounded, []).append((idx, m))

    sorted_probs = sorted(prob_groups.keys(), reverse=True)

    # Deterministically pick group then index within the group using RNG
    rng = rng_manager.get_context_rng('selection')
    r = rng.random()
    cumulative = 0.0
    for prob in sorted_probs:
        group = prob_groups[prob]
        group.sort(key=lambda t: t[0])
        group_weight = prob * len(group)
        if group_weight <= 0:
            continue
        if cumulative + group_weight >= r or prob == sorted_probs[-1]:
            rel = (r - cumulative) / group_weight
            idx = int(rel * len(group))
            if idx >= len(group):
                idx = len(group) - 1
            return group[idx][1]
        cumulative += group_weight

    # Fallback: first by genotype order
    return min(filtered, key=lambda m: potential_matches.index(m))


__all__ = ["select_match"]


# ------------------------------
# M28: Multi-objective selection
# ------------------------------

def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return True if objective vector a Pareto-dominates b (minimization).

    a dominates b if a is no worse in all objectives and strictly better in at least one.
    """
    assert len(a) == len(b)
    not_worse_all = all(x <= y for x, y in zip(a, b))
    strictly_better_any = any(x < y for x, y in zip(a, b))
    return not_worse_all and strictly_better_any


def _fast_nondominated_sort(pop_objs: list[list[float]]) -> list[list[int]]:
    """Compute non-dominated fronts indices using Deb's fast non-dominated sort.

    Returns list of fronts, each a list of indices into pop_objs, in increasing rank order.
    Deterministic by processing indices in ascending order and sorting fronts.
    """
    n = len(pop_objs)
    S: list[list[int]] = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif _dominates(pop_objs[q], pop_objs[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    fronts[0].sort()
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        next_front = sorted(set(next_front))
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def _crowding_distance(front_indices: list[int], pop_objs: list[list[float]]) -> dict[int, float]:
    """Compute crowding distance for individuals in a front (minimization).

    Deterministic tie-breakers by sorting indices for each objective.
    Boundaries receive inf distance.
    """
    if not front_indices:
        return {}
    m = len(pop_objs[0])
    distance = {i: 0.0 for i in front_indices}
    for obj_idx in range(m):
        ordered = sorted(front_indices, key=lambda i: (pop_objs[i][obj_idx], i))
        f_min = pop_objs[ordered[0]][obj_idx]
        f_max = pop_objs[ordered[-1]][obj_idx]
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        if f_max == f_min:
            # All equal; distances remain as set (boundaries inf, middle 0)
            continue
        norm = f_max - f_min
        for k in range(1, len(ordered) - 1):
            prev_i = ordered[k - 1]
            next_i = ordered[k + 1]
            distance[ordered[k]] += (pop_objs[next_i][obj_idx] - pop_objs[prev_i][obj_idx]) / norm
    return distance


def nsga2_select(
    population: list[Any],
    objectives: Callable[[Any], Sequence[float]] | Iterable[Sequence[float]],
    k: int,
    rng_manager: RNGManager,
) -> list[Any]:
    """Select k individuals via deterministic NSGA-II (minimization objectives).

    Args:
        population: list of individuals
        objectives: either a callable mapping individual -> sequence of floats, or
                    a precomputed iterable aligned with population order
        k: number to select
        rng_manager: RNGManager for deterministic tie-breaking when needed

    Returns:
        Selected individuals in deterministic order: fronts by rank, then by
        descending crowding distance, breaking ties by stable index order.

    Determinism:
    - Stable index-based ordering within equal distances
    - RNG is not used by default; reserved for future stochastic tie-breaking hooks
    """
    if k <= 0 or not population:
        return []
    if callable(objectives):
        pop_objs = [list(objectives(ind)) for ind in population]
    else:
        pop_objs = [list(v) for v in objectives]  # type: ignore[arg-type]
        if len(pop_objs) != len(population):
            raise ValueError("objectives length must match population")
    fronts = _fast_nondominated_sort(pop_objs)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            # Need to choose subset from this front using crowding distance
            dist = _crowding_distance(front, pop_objs)
            # Sort by distance desc, then by index asc for determinism
            ordered = sorted(front, key=lambda i: (-dist.get(i, 0.0), i))
            remaining = k - len(selected)
            selected.extend(ordered[:remaining])
            break
    # Return individuals in the selected order
    return [population[i] for i in selected]


__all__.extend(["nsga2_select"]) 
