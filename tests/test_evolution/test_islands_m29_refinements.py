"""M29 Refinements: topology/policies/RNG/checkpoint/budgets tests (TDD first).

Test IDs:
- [T-co-29R-01] Star topology migration determinism
- [T-co-29R-02] best_k migration policy with deterministic tie-breakers
- [T-co-29R-03] Per-island RNG namespaces isolation
- [T-co-29R-04] Versioned checkpoint includes iteration counter
- [T-co-29R-05] Time-budget batch planning deterministic with metrics
"""

from __future__ import annotations

import time

import pytest

from ggnes.evolution.islands import IslandScheduler, MigrationConfig
from ggnes.utils.rng_manager import RNGManager


def test_co_29R_01_star_topology_migration():
    rng = RNGManager(seed=1)
    cfg = MigrationConfig(migration_size=1, topology='star')
    sched = IslandScheduler(rng, cfg)
    pops = [["A0","A1"],["B0"],["C0","C1","C2"],["D0"]]
    new_pops = sched.migrate(pops)
    # Star: island 0 is hub; others send first elem to hub; hub sends first to island 1 (deterministic policy)
    assert new_pops[0] == ["A1","B0","C0","D0"]
    assert new_pops[1] == ["A0"]


def test_co_29R_02_best_k_policy_ties():
    rng = RNGManager(seed=2)
    cfg = MigrationConfig(migration_size=2, topology='ring')
    sched = IslandScheduler(rng, cfg)
    # Attach fitness values via tuples (value, id) where lower is better; ties by id
    pops = [[(0.5,'a'),(0.5,'b'),(0.7,'c')],[(0.4,'x')]]
    new_pops = sched.migrate_best_k(pops, key=lambda ind: ind[0])
    # best_k should pick (0.5,'a') then (0.5,'b') deterministically by id tie-break
    assert new_pops[1][0:2] == [(0.5,'a'),(0.5,'b')]


def test_co_29R_03_per_island_rng_namespaces():
    rng = RNGManager(seed=3)
    sched = IslandScheduler(rng)
    r0 = sched.get_island_rng(0).random()
    r1 = sched.get_island_rng(1).random()
    # Different namespaces produce different sequences
    assert abs(r0 - r1) > 1e-12
    # Same island namespace deterministic across schedulers with same seed
    rng2 = RNGManager(seed=3)
    sched2 = IslandScheduler(rng2)
    assert abs(sched.get_island_rng(0).random() - sched2.get_island_rng(0).random()) < 1e-12


def test_co_29R_04_versioned_checkpoint_includes_iteration():
    rng = RNGManager(seed=4)
    sched = IslandScheduler(rng)
    pops = [[1],[2]]
    # Simulate two iterations
    sched.iteration = 2
    ckpt = sched.checkpoint(pops)
    assert ckpt.get('_schema') == 1
    assert ckpt.get('iteration') == 2


def test_co_29R_05_time_budget_planning_is_deterministic(monkeypatch):
    rng = RNGManager(seed=5)
    sched = IslandScheduler(rng)
    items = list(range(6))
    # Fake a timer to simulate per-item cost deterministically
    t = [0.0]
    def fake_time():
        # each call advances by 0.011s
        t[0] += 0.011
        return t[0]
    monkeypatch.setattr(time, 'time', fake_time)
    batches, metrics = sched.plan_batches_by_time(items, time_budget_ms=25)
    # With ~0.011s per item, budget 25ms → fits 2 items per batch deterministically
    assert batches == [[0,1],[2,3],[4,5]]
    assert metrics['avg_item_ms'] == pytest.approx(11.0, rel=0.05)


