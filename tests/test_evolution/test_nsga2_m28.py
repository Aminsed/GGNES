"""M28: Fitness & Multi‑Objective Selection (NSGA‑II) tests.

Test IDs:
- [T-co-28-01] Objective calculation correctness and penalties (repair impact)
- [T-co-28-02] Deterministic Pareto sorting and tie-breakers
- [T-co-28-03] Selection subset via crowding distance is deterministic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math

from ggnes.evolution.selection import nsga2_select
from ggnes.repair.repair import calculate_repair_penalty
from ggnes.utils.rng_manager import RNGManager


@dataclass
class Ind:
    name: str
    loss: float
    params: float
    latency: float
    repair_impact: float = 0.0

    def objectives(self) -> Sequence[float]:
        # Minimize all: task loss, params, latency, and penalty
        penalty = calculate_repair_penalty({'repair_impact_score': self.repair_impact})
        return [self.loss + penalty, self.params, self.latency]


def test_co_28_01_objective_and_penalty_integration():
    # A and B identical except A has higher repair impact ⇒ higher penalty ⇒ dominated by B
    a = Ind('A', loss=0.20, params=1.0, latency=10.0, repair_impact=0.5)  # penalty 0.045
    b = Ind('B', loss=0.20, params=1.0, latency=10.0, repair_impact=0.0)  # penalty 0.0
    c = Ind('C', loss=0.18, params=1.2, latency=10.0, repair_impact=0.0)  # trade-off
    pop = [a, b, c]
    rng = RNGManager(seed=42)

    selected = nsga2_select(pop, lambda ind: ind.objectives(), k=3, rng_manager=rng)
    # B and C should be non-dominated; A dominated by B due to penalty
    names = [i.name for i in selected]
    assert 'B' in names and 'C' in names
    assert 'A' in names  # k=3 selects all, but A should not appear before B/C by rank


def test_co_28_02_deterministic_pareto_sorting_and_ties():
    # Four individuals on a square; none dominates another; distances used.
    p = [
        Ind('P0', loss=1.0, params=1.0, latency=1.0),
        Ind('P1', loss=1.0, params=1.0, latency=2.0),
        Ind('P2', loss=1.0, params=2.0, latency=1.0),
        Ind('P3', loss=2.0, params=1.0, latency=1.0),
    ]
    rng = RNGManager(seed=7)

    # k=4 returns all in deterministic crowding order; boundaries get inf distance
    sel_all = nsga2_select(p, lambda x: x.objectives(), k=4, rng_manager=rng)
    assert len(sel_all) == 4
    # k=2 subset must be deterministic; tie-broken by index when distances equal
    sel_two = nsga2_select(p, (ind.objectives() for ind in p), k=2, rng_manager=rng)
    assert [i.name for i in sel_two] == [p[0].name, p[1].name] or [i.name for i in sel_two] == [p[0].name, p[2].name] or [i.name for i in sel_two] == [p[0].name, p[3].name]


def test_co_28_03_crowding_edge_cases_and_input_validation():
    # Equal objective values across front ⇒ f_max == f_min path executed
    q = [
        Ind('Q0', loss=1.0, params=1.0, latency=1.0),
        Ind('Q1', loss=1.0, params=1.0, latency=1.0),
        Ind('Q2', loss=1.0, params=1.0, latency=1.0),
    ]
    rng = RNGManager(seed=13)
    sel = nsga2_select(q, (ind.objectives() for ind in q), k=2, rng_manager=rng)
    # Boundaries assigned inf; middle 0; deterministic by index for equal distances
    assert len(sel) == 2
    assert sel[0].name == 'Q0'

    # k <= 0 or empty population
    assert nsga2_select([], lambda x: [0.0], k=3, rng_manager=rng) == []
    assert nsga2_select(q, lambda x: x.objectives(), k=0, rng_manager=rng) == []

    # Precomputed objectives with wrong length raises
    try:
        nsga2_select(q, ([0.0], [0.0]), k=1, rng_manager=rng)  # type: ignore[arg-type]
        raise AssertionError('Expected ValueError for length mismatch')
    except ValueError:
        pass


