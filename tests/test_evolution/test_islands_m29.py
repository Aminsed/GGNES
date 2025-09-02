"""M29: Scheduling & Islands tests.

Test IDs:
- [T-co-29-01] Migration determinism across runs; ring topology; checksum stable
- [T-co-29-02] Budget-aware batching with deterministic backoff (chunking)
- [T-co-29-03] Checkpoint/restore reproduces populations and RNG state
"""

from __future__ import annotations

from ggnes.evolution.islands import IslandScheduler, MigrationConfig
from ggnes.utils.rng_manager import RNGManager


def test_co_29_01_migration_determinism_and_checksum():
    rng = RNGManager(seed=123)
    sched = IslandScheduler(rng, MigrationConfig(migration_size=2, topology='ring'))
    pops = [["A0", "A1", "A2"], ["B0", "B1"], ["C0", "C1", "C2", "C3"]]
    csum_before = sched.determinism_checksum(pops)
    new_pops = sched.migrate(pops)
    # Ring migration with size=2: first two from island i go to i+1
    assert new_pops[0] == ["A2", "C0", "C1"]
    assert new_pops[1] == ["A0", "A1"]
    assert new_pops[2] == ["C2", "C3", "B0", "B1"]
    # Checksum changes deterministically
    csum_after = sched.determinism_checksum(new_pops)
    assert csum_before != csum_after
    # Re-running with the same inputs yields the same outputs
    rng2 = RNGManager(seed=123)
    sched2 = IslandScheduler(rng2, MigrationConfig(migration_size=2, topology='ring'))
    assert sched2.migrate(pops) == new_pops


def test_co_29_02_budget_aware_batching():
    rng = RNGManager(seed=7)
    sched = IslandScheduler(rng)
    items = list(range(10))
    batches_3 = sched.plan_batches(items, max_items_per_batch=3)
    assert batches_3 == [[0,1,2],[3,4,5],[6,7,8],[9]]
    # Edge cases
    assert sched.plan_batches(items, max_items_per_batch=0) == [[]]
    assert sched.plan_batches([], max_items_per_batch=3) == []


def test_co_29_03_checkpoint_and_restore():
    rng = RNGManager(seed=42)
    sched = IslandScheduler(rng)
    pops = [[1,2],[3]]
    ckpt = sched.checkpoint(pops)
    # Advance RNG state deterministically
    _ = rng.get_context_rng('selection').random()
    restored_pops = sched.restore(ckpt)
    assert restored_pops == pops
    # RNG state restored; next random should match as if uninterrupted
    r_after = rng.get_context_rng('selection').random()
    rng2 = RNGManager(seed=42)
    r_expected = rng2.get_context_rng('selection').random()
    assert abs(r_after - r_expected) < 1e-12


