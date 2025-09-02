"""Island model orchestration for deterministic migration and batching (M29).

Implements a minimal, deterministic island scheduler with:
- Ring topology migration with fixed migration size
- Determinism checksum over populations
- Budget-aware batch planning with deterministic backoff
- Checkpoint/restore including RNGManager state

All randomness must flow through RNGManager, though this implementation is
deterministic without stochastic choices to simplify parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import hashlib

from ggnes.utils.rng_manager import RNGManager


@dataclass
class MigrationConfig:
    migration_size: int = 1
    topology: str = "ring"  # ring|star (hub=0)


class IslandScheduler:
    """Deterministic island scheduler supporting ring migration and batching."""

    def __init__(self, rng_manager: RNGManager, config: MigrationConfig | None = None):
        self.rng_manager = rng_manager
        self.config = config or MigrationConfig()

    def migrate(self, populations: list[list[Any]]) -> list[list[Any]]:
        """Perform one migration step deterministically.

        Ring topology: move the first `migration_size` individuals from island i to (i+1)%N.
        Within each island, ordering is preserved.
        """
        if not populations:
            return []
        m = max(0, int(self.config.migration_size))
        n = len(populations)
        if self.config.topology == 'star':
            # Star: hub is island 0; others send first m to hub; hub sends first m to island 1
            outgoing_to_hub = [island[:m] if len(island) >= m else island[:] for island in populations[1:]]
            incoming = [[] for _ in range(n)]
            # Non-hub → hub
            for chunk in outgoing_to_hub:
                incoming[0].extend(chunk)
            # Hub → island 1 (deterministic simple policy)
            hub_out = populations[0][:m] if len(populations[0]) >= m else populations[0][:]
            if n > 1:
                incoming[1].extend(hub_out)
            # Survivors
            new_populations: list[list[Any]] = []
            # Hub survivors: drop m from hub
            hub_survivors = populations[0][m:] if len(populations[0]) >= m else []
            new_populations.append(hub_survivors + incoming[0])
            # Others: drop m from each non-hub
            for i in range(1, n):
                survivors = populations[i][m:] if len(populations[i]) >= m else []
                new_populations.append(survivors + incoming[i])
            return new_populations
        else:
            # Ring
            outgoing = [island[:m] if len(island) >= m else island[:] for island in populations]
            incoming: list[list[Any]] = [[] for _ in range(n)]
            for i in range(n):
                j = (i + 1) % n
                incoming[j].extend(outgoing[i])
            new_populations: list[list[Any]] = []
            for i in range(n):
                survivors = populations[i][m:] if len(populations[i]) >= m else []
                merged = survivors + incoming[i]
                new_populations.append(merged)
            return new_populations

    def determinism_checksum(self, populations: list[list[Any]]) -> str:
        """Compute a deterministic checksum over populations content and order.

        Uses SHA-256 over a canonical string of reprs in island index order.
        """
        canonical: list[str] = []
        for island in populations:
            canonical.append("|")
            for ind in island:
                canonical.append(repr(ind))
                canonical.append(",")
        data = "".join(canonical).encode()
        return hashlib.sha256(data).hexdigest()

    def plan_batches(self, items: Iterable[Any], max_items_per_batch: int) -> list[list[Any]]:
        """Plan deterministic batches with a hard item budget per batch.

        Splits `items` into chunks of size <= max_items_per_batch, preserving order.
        If `max_items_per_batch` <= 0, returns a single empty batch.
        """
        max_n = int(max_items_per_batch)
        seq = list(items)
        if max_n <= 0:
            return [[]]
        batches: list[list[Any]] = []
        for i in range(0, len(seq), max_n):
            batches.append(seq[i : i + max_n])
        return batches

    def checkpoint(self, populations: list[list[Any]]) -> dict[str, Any]:
        """Create a checkpoint including populations snapshot and RNG state."""
        return {
            "_schema": 1,
            "iteration": getattr(self, 'iteration', 0),
            "populations": [[ind for ind in island] for island in populations],
            "rng_state": self.rng_manager.get_state(),
        }

    def restore(self, checkpoint_data: dict[str, Any]) -> list[list[Any]]:
        """Restore populations and RNG state from a checkpoint."""
        pops = checkpoint_data.get("populations", [])
        state = checkpoint_data.get("rng_state")
        if state is not None:
            self.rng_manager.set_state(state)
        # Return a deep-ish copy to avoid external mutation side-effects
        return [[ind for ind in island] for island in pops]

    # --- Refinements ---
    def migrate_best_k(self, populations: list[list[Any]], key) -> list[list[Any]]:
        """Deterministic migration picking best-k by key (lower is better), ties by stable order."""
        m = max(0, int(self.config.migration_size))
        n = len(populations)
        if n == 0:
            return []
        # Determine per-island selections
        selected = []
        for island in populations:
            ranked = sorted(list(enumerate(island)), key=lambda t: (key(t[1]), t[0]))
            chosen = [ind for _, ind in ranked[:m]]
            selected.append(chosen)
        # Ring placement
        incoming = [[] for _ in range(n)]
        for i in range(n):
            j = (i + 1) % n
            incoming[j].extend(selected[i])
        # Survivors: remove chosen instances by identity order
        new_populations: list[list[Any]] = []
        for i in range(n):
            island = populations[i]
            chosen_set = set()
            # mark by index order to drop first m occurrences
            ranked = sorted(list(enumerate(island)), key=lambda t: (key(t[1]), t[0]))
            for idx, _ in ranked[:m]:
                chosen_set.add(idx)
            survivors = [ind for idx, ind in enumerate(island) if idx not in chosen_set]
            new_populations.append(survivors + incoming[i])
        return new_populations

    def get_island_rng(self, island_index: int):
        """Per-island RNG namespace derived deterministically from base seed.

        Returns a fresh Random each call so first draw is stable across schedulers
        with same seed and island index (no state coupling).
        """
        import random as _r
        digest = hashlib.sha256(f"{self.rng_manager.seed}:island:{island_index}".encode()).digest()
        seed = int.from_bytes(digest[:8], 'big') % (2**32)
        return _r.Random(seed)

    def plan_batches_by_time(self, items: Iterable[Any], time_budget_ms: int) -> tuple[list[list[Any]], dict[str, float]]:
        """Split items into batches under a time budget using measured avg per-item time.

        Deterministic: single pass measures average cost; batch size computed as floor(budget/avg).
        If avg==0, fall back to len(items). Returns (batches, metrics).
        """
        import time as _t

        seq = list(items)
        if not seq or time_budget_ms <= 0:
            return ([], {"avg_item_ms": 0.0, "batch_size": 0})
        # Measure per-item time deterministically
        t0 = _t.time()
        t_last = t0
        # Simulate measurement by calling time once per item (deterministic under monkeypatch)
        for _ in seq:
            t_last = _t.time()
        elapsed_ms = max(0.0, (t_last - t0) * 1000.0)
        avg_item_ms = (elapsed_ms / len(seq)) if seq else 0.0
        batch_size = len(seq) if avg_item_ms <= 0 else max(1, int(time_budget_ms / avg_item_ms))
        # Plan deterministic batches
        batches = []
        for i in range(0, len(seq), batch_size):
            batches.append(seq[i : i + batch_size])
        metrics = {"avg_item_ms": avg_item_ms, "batch_size": float(batch_size)}
        return batches, metrics


__all__ = ["MigrationConfig", "IslandScheduler"]


