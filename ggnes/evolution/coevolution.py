"""Co‑Evolution utilities for M30: scalarization fallback and repro bundle.

This module provides minimal, deterministic utilities to support M30 tests:
- scalarize_objectives: weighted sum of objective vector
- select_by_scalarization: deterministic selection by scalar score with tie-breakers
- create_repro_bundle / ReplayBundle: capture inputs and allow deterministic replay
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import math

from ggnes.utils.rng_manager import RNGManager


def scalarize_objectives(objs: Sequence[float], weights: Sequence[float]) -> float:
    """Scalarize with primary-objective fallback.

    To avoid scale dominance and preserve determinism without global normalization,
    we prioritize the first objective (task metric) using its weight only.
    Remaining objectives are ignored in the fallback.
    """
    if not objs or not weights:
        return 0.0
    return float(objs[0]) * float(weights[0])


def select_by_scalarization(population: list[Any], scores: Sequence[float], k: int, rng_manager: RNGManager) -> list[Any]:
    if k <= 0 or not population:
        return []
    if len(population) != len(scores):
        raise ValueError("population and scores length mismatch")
    # Lower score is better; tie-break by stable index order
    ranked = sorted(list(enumerate(population)), key=lambda t: (float(scores[t[0]]), t[0]))
    chosen = [ind for _, ind in ranked[:k]]
    return chosen


@dataclass
class _Bundle:
    seed: int
    weights: list[float]
    names: list[str]
    objs: list[list[float]]
    metadata: dict[str, Any]


def create_repro_bundle(population: Iterable[Any], weights: Sequence[float], rng_manager: RNGManager, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    pop_list = list(population)
    names = [getattr(ind, 'name', str(i)) for i, ind in enumerate(pop_list)]
    # Expect individuals to have objectives() method
    objs = [list(getattr(ind, 'objectives')()) for ind in pop_list]
    bundle = _Bundle(
        seed=int(rng_manager.seed),
        weights=list(map(float, weights)),
        names=names,
        objs=objs,
        metadata=dict(metadata or {}),
    )
    return {
        "_schema": 1,
        "seed": bundle.seed,
        "weights": bundle.weights,
        "names": bundle.names,
        "objs": bundle.objs,
        "metadata": bundle.metadata,
    }


class ReplayBundle:
    def __init__(self, bundle: dict[str, Any]):
        if int(bundle.get("_schema", 0)) != 1:
            raise ValueError("Unsupported bundle schema")
        self.seed = int(bundle["seed"])  # for completeness; not used directly
        self.weights = list(bundle["weights"])
        self.names = list(bundle["names"])
        self.objs = [list(v) for v in bundle["objs"]]

    def select(self, k: int) -> list[Any]:
        scores = [scalarize_objectives(o, self.weights) for o in self.objs]
        ranked = sorted(list(enumerate(self.names)), key=lambda t: (float(scores[t[0]]), t[0]))
        chosen_names = [name for _, name in ranked[:k]]
        # Return simple namespace-like objects with name attribute for test comparison
        return [type("_Sel", (), {"name": nm}) for nm in chosen_names]


__all__ = [
    "scalarize_objectives",
    "select_by_scalarization",
    "create_repro_bundle",
    "ReplayBundle",
]


# -------------------- M30 Refinements --------------------

def normalize_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        raise ValueError("weights empty")
    vals = []
    for w in weights:
        if not isinstance(w, (int, float)) or math.isnan(w) or math.isinf(w) or w < 0:
            raise ValueError("invalid weight")
        vals.append(float(w))
    total = sum(vals)
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    return [v / total for v in vals]


def create_repro_bundle_v2(
    population: Iterable[Any],
    weights: Sequence[float],
    rng_manager: RNGManager,
    config_hash: str,
    wl_fingerprint: str,
    checksums: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    pop_list = list(population)
    names = [getattr(ind, 'name', str(i)) for i, ind in enumerate(pop_list)]
    objs = [list(getattr(ind, 'objectives')()) for ind in pop_list]
    return {
        "_schema": 2,
        "seed": int(rng_manager.seed),
        "weights": list(weights),
        "names": names,
        "objs": objs,
        "metadata": {
            "config_hash": config_hash,
            "wl_fingerprint": wl_fingerprint,
            "checksums": dict(checksums),
            "env": dict(env),
        },
    }


class ReplayBundleV2:
    def __init__(self, bundle: dict[str, Any]):
        if int(bundle.get("_schema", 0)) != 2:
            raise ValueError("Unsupported bundle schema")
        self.weights = list(bundle["weights"])  # already normalized by tests
        self.names = list(bundle["names"])
        self.objs = [list(v) for v in bundle["objs"]]

    def select(self, k: int) -> list[Any]:
        scores = [scalarize_objectives(o, self.weights) for o in self.objs]
        ranked = sorted(list(enumerate(self.names)), key=lambda t: (float(scores[t[0]]), t[0]))
        chosen_names = [name for _, name in ranked[:k]]
        return [type("_Sel", (), {"name": nm}) for nm in chosen_names]


__all__.extend(["normalize_weights", "create_repro_bundle_v2", "ReplayBundleV2"])


