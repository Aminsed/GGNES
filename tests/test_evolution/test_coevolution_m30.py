"""M30: End-to-End Co‑Evolution – scalarization fallback and repro bundle.

Test IDs:
- [T-co-30-01] Scalarization fallback deterministic selection with tie-breakers
- [T-co-30-02] Repro bundle reproduces identical results
- [T-co-30-03] Failure floor integrates repair penalty deterministically
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ggnes.evolution.selection import nsga2_select
from ggnes.evolution.coevolution import scalarize_objectives, select_by_scalarization, create_repro_bundle, ReplayBundle
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
        pen = calculate_repair_penalty({'repair_impact_score': self.repair_impact})
        return [self.loss + pen, self.params, self.latency]


def test_co_30_01_scalarization_fallback_deterministic():
    rng = RNGManager(seed=101)
    pop = [
        Ind('A', 0.20, 1.0, 10.0, 0.0),
        Ind('B', 0.19, 1.3, 10.0, 0.0),
        Ind('C', 0.19, 1.3, 10.0, 0.0),
    ]
    weights = [0.8, 0.1, 0.1]
    scores = [scalarize_objectives(ind.objectives(), weights) for ind in pop]
    sel = select_by_scalarization(pop, scores, k=2, rng_manager=rng)
    # B and C tied on score; tie-breaker by stable order
    assert [i.name for i in sel] == ['B', 'C']


def test_co_30_02_repro_bundle_roundtrip():
    rng = RNGManager(seed=202)
    pop = [Ind('A', 0.2, 1.0, 10.0), Ind('B', 0.18, 1.3, 10.0)]
    weights = [0.8, 0.1, 0.1]
    bundle = create_repro_bundle(population=pop, weights=weights, rng_manager=rng, metadata={'preset':'STANDARD'})
    # Replay using bundle produces identical winners
    replay = ReplayBundle(bundle)
    sel1 = replay.select(k=1)
    replay2 = ReplayBundle(bundle)
    sel2 = replay2.select(k=1)
    assert [i.name for i in sel1] == [i.name for i in sel2]


def test_co_30_03_failure_floor_with_penalty():
    # Two candidates equal loss, one has high repair impact -> worse scalar
    rng = RNGManager(seed=303)
    pop = [Ind('A', 0.2, 1.0, 10.0, repair_impact=0.6), Ind('B', 0.2, 1.0, 10.0, repair_impact=0.0)]
    weights = [1.0, 0.0, 0.0]
    scores = [scalarize_objectives(ind.objectives(), weights) for ind in pop]
    sel = select_by_scalarization(pop, scores, k=1, rng_manager=rng)
    assert [i.name for i in sel] == ['B']


