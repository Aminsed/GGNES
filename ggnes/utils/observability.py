"""Observability helpers for explainability and consolidated reports (M32).

This module provides small, deterministic utilities to:
- Explain a CompositeGenotype (canonical inputs, UUID, scheme metadata, checksums,
  diffs vs baseline, constraint/timing breakdowns)
- Compute operator efficacy metrics between two CompositeGenotypes
- Build a consolidated report combining derivation checksum, WL fingerprint,
  per-batch metrics, seed/device/dtype, and RNG context signatures
- Produce island/migration logs from the IslandScheduler
- Validate explain/report payloads against a strict schema (allowing extra fields)

The functions are intentionally lightweight to avoid coupling and keep code
readable.
"""

from __future__ import annotations

import hashlib
import time
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ggnes.core.graph import Graph
from ggnes.hierarchical.derivation import DerivationEngine, DerivationNode
from ggnes.evolution.composite_genotype import CompositeGenotype
from ggnes.evolution.islands import IslandScheduler
from ggnes.utils.rng_manager import RNGManager


def _short_sha16(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _rng_state_signature(rng) -> str:
    # Hash the textual representation of RNG state deterministically
    state_str = str(rng.getstate())
    return _short_sha16(state_str)


def explain_composite(
    geno: CompositeGenotype,
    baseline: Optional[CompositeGenotype] = None,
) -> Dict[str, Any]:
    """Return a deterministic explanation of a CompositeGenotype.

    Includes:
    - canonical inputs (as used for deterministic UUIDs)
    - uuid (string)
    - scheme metadata (namespace, version, precision, strict/freeze flags)
    - checksums over the canonical inputs and sub-parts for stability
    - diffs vs baseline (rules added/removed/changed; params changed)
    - timing breakdowns (validate_ms, uuid_ms, checksum_ms)
    - constraint reasoning result (policy validation ok)
    """
    t0 = time.perf_counter()
    # Validate policy and compute canonical inputs
    geno.validate()
    canonical = geno.as_canonical_inputs()
    t1 = time.perf_counter()
    uid = geno.uuid()
    t2 = time.perf_counter()
    scheme = geno.provider.scheme_metadata()

    # Compute simple, stable checksums for observability
    full_checksum = _short_sha16(str(canonical))
    rules_checksum = _short_sha16(str(canonical.get("g1", {})))
    params_checksum = _short_sha16(str(canonical.get("g2", {})))
    hier_checksum = _short_sha16(str(canonical.get("g3", {})))
    t3 = time.perf_counter()

    # Constraint reasoning: reuse validation result
    constraints_ok = True
    try:
        geno.validate()
    except Exception:
        constraints_ok = False

    diffs: Dict[str, Any] = {"rules_added": [], "rules_removed": [], "params_changed": {}}
    if baseline is not None:
        base_inputs = baseline.as_canonical_inputs()
        # G1 rules diff by rule_id presence
        def _rules_set(src) -> set:
            return set(r.get("rule_id") for r in src.get("g1", {}).get("rules", []) if isinstance(r, dict) and "rule_id" in r)

        added = sorted(list(_rules_set(canonical) - _rules_set(base_inputs)))
        removed = sorted(list(_rules_set(base_inputs) - _rules_set(canonical)))
        diffs["rules_added"] = added
        diffs["rules_removed"] = removed

        # G2 param diffs (numeric and boolean)
        p_before = base_inputs.get("g2", {})
        p_after = canonical.get("g2", {})
        params_changed: Dict[str, Dict[str, Any]] = {}
        for key in sorted(set(p_before.keys()) | set(p_after.keys())):
            if p_before.get(key) != p_after.get(key):
                params_changed[key] = {"before": p_before.get(key), "after": p_after.get(key)}
        diffs["params_changed"] = params_changed

    return {
        "uuid": uid,
        "scheme": scheme,
        "canonical": canonical,
        "checksums": {
            "full": full_checksum,
            "rules": rules_checksum,
            "params": params_checksum,
            "hier": hier_checksum,
        },
        "diffs": diffs,
        "timings": {
            "validate_ms": (t1 - t0) * 1000.0,
            "uuid_ms": (t2 - t1) * 1000.0,
            "checksum_ms": (t3 - t2) * 1000.0,
        },
        "constraints": {
            "policy_constraints_ok": bool(constraints_ok),
        },
    }


def compute_operator_efficacy(before: CompositeGenotype, after: CompositeGenotype) -> Dict[str, Any]:
    """Compute simple, deterministic operator efficacy metrics between genotypes.

    Metrics include deltas of key policy fields and G1 rule count. This is
    designed for explainability and testability and does not alter semantics.
    """
    return {
        "rule_count_delta": len(after.g1.rules) - len(before.g1.rules),
        "learning_rate_delta": float(after.g2.learning_rate) - float(before.g2.learning_rate),
        "batch_size_delta": int(after.g2.batch_size) - int(before.g2.batch_size),
        "training_epochs_delta": int(after.g2.training_epochs) - int(before.g2.training_epochs),
        "parallel_execution_change": bool(after.g2.parallel_execution) != bool(before.g2.parallel_execution),
    }


def compute_rng_signatures(rng_manager: RNGManager, contexts: Iterable[str]) -> Dict[str, str]:
    """Return short deterministic signatures for the given RNG contexts.

    Contexts not yet created are instantiated deterministically by RNGManager.
    """
    sigs: Dict[str, str] = {}
    for name in contexts:
        rng = rng_manager.get_context_rng(name)
        sigs[name] = _rng_state_signature(rng)
    return sigs


def consolidated_report(
    engine: DerivationEngine,
    root: DerivationNode,
    graph: Graph,
    *,
    genotype_explain: Dict[str, Any] | None = None,
    rng_manager: Optional[RNGManager] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    extra_rng_contexts: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a consolidated, deterministic report for observability.

    The report contains:
    - derivation checksum (from engine.explain(root))
    - WL fingerprint of the provided graph
    - per-batch metrics from engine.last_expand_metrics (if any)
    - optional composite genotype explanation if provided
    - a determinism checksum over the whole report
    """
    derivation_info = engine.explain(root)
    wl_fingerprint = graph.compute_fingerprint()
    batches = engine.last_expand_metrics.get("batches", [])

    report: Dict[str, Any] = {
        "schema_version": 2,
        "derivation_checksum": derivation_info.get("checksum"),
        "wl_fingerprint": wl_fingerprint,
        "batches": list(batches),
    }
    if genotype_explain is not None:
        report["genotype"] = genotype_explain

    # Environment context
    if rng_manager is not None:
        report.setdefault("env", {})
        report["env"]["seed"] = int(getattr(rng_manager, "seed", 0))
        # Include deterministic RNG signatures for requested contexts
        contexts = list(extra_rng_contexts) if extra_rng_contexts is not None else [
            "selection",
            "repair",
            "aggregation_dropout",
        ]
        report["env"]["rng_signatures"] = compute_rng_signatures(rng_manager, contexts)
    if device is not None:
        report.setdefault("env", {})
        report["env"]["device"] = str(device)
    if dtype is not None:
        report.setdefault("env", {})
        report["env"]["dtype"] = str(dtype)

    # Determinism checksum over a stable string representation
    report_checksum = _short_sha16(str(report))
    report["determinism_checksum"] = report_checksum
    return report


def island_migration_report(
    scheduler: IslandScheduler,
    before: List[List[Any]],
    after: List[List[Any]],
) -> Dict[str, Any]:
    """Create a deterministic report of an island migration step.

    Includes topology, migration size, island count, per-island sizes before/after,
    and estimates of moved_in/moved_out counts.
    """
    n = len(before)
    topology = scheduler.config.topology
    m = int(scheduler.config.migration_size)

    per_island: List[Dict[str, Any]] = []
    for i in range(n):
        before_i = before[i] if i < len(before) else []
        after_i = after[i] if i < len(after) else []
        moved_out = max(0, len(before_i) - len([ind for ind in before_i if ind in after_i]))
        moved_in = max(0, len(after_i) - len([ind for ind in after_i if ind in before_i]))
        per_island.append(
            {
                "index": i,
                "size_before": len(before_i),
                "size_after": len(after_i),
                "moved_out_estimate": moved_out,
                "moved_in_estimate": moved_in,
            }
        )

    report = {
        "topology": topology,
        "migration_size": m,
        "islands": n,
        "per_island": per_island,
        "migration_checksum": _short_sha16(str({"before": before, "after": after})),
        "rng_namespace_sig_island0": _rng_state_signature(scheduler.get_island_rng(0)) if n > 0 else None,
    }
    return report


# -------------------------- Schema validation --------------------------

def _require(obj: Mapping[str, Any], key: str, typ) -> None:
    if key not in obj:
        raise ValueError(f"missing required field: {key}")
    if typ is not None and not isinstance(obj[key], typ):
        raise ValueError(f"field {key} has wrong type: expected {typ}")


def validate_explain_payload(payload: Mapping[str, Any]) -> None:
    _require(payload, "uuid", str)
    _require(payload, "scheme", Mapping)
    _require(payload, "canonical", Mapping)
    _require(payload, "checksums", Mapping)
    _require(payload["checksums"], "full", str)
    _require(payload["checksums"], "rules", str)
    _require(payload["checksums"], "params", str)
    _require(payload["checksums"], "hier", str)
    # Optional sections
    if "diffs" in payload:
        _require(payload["diffs"], "params_changed", Mapping)
    if "timings" in payload:
        _require(payload["timings"], "validate_ms", (int, float))
        _require(payload["timings"], "uuid_ms", (int, float))
        _require(payload["timings"], "checksum_ms", (int, float))
    if "constraints" in payload:
        _require(payload["constraints"], "policy_constraints_ok", (bool,))


def validate_consolidated_report(report: Mapping[str, Any]) -> None:
    _require(report, "derivation_checksum", str)
    _require(report, "wl_fingerprint", str)
    _require(report, "batches", list)
    _require(report, "determinism_checksum", str)
    # Optional env with seed/device/dtype and rng signatures
    if "env" in report:
        env = report["env"]
        if not isinstance(env, Mapping):
            raise ValueError("env must be a mapping")
        if "rng_signatures" in env:
            if not isinstance(env["rng_signatures"], Mapping):
                raise ValueError("env.rng_signatures must be a mapping")


def validate_consolidated_report_v2(report: Mapping[str, Any]) -> None:
    # strict: require schema_version==2 and then reuse tolerant checks
    _require(report, "schema_version", int)
    if int(report["schema_version"]) != 2:
        raise ValueError("schema_version must be 2")
    validate_consolidated_report(report)


def migrate_consolidated_report_to_v2(report: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(report)
    out["schema_version"] = 2
    return out


def validate_island_report(report: Mapping[str, Any]) -> None:
    _require(report, "topology", str)
    _require(report, "migration_size", int)
    _require(report, "islands", int)
    _require(report, "per_island", list)
    _require(report, "migration_checksum", str)


__all__ = [
    "explain_composite",
    "compute_operator_efficacy",
    "consolidated_report",
    "compute_rng_signatures",
    "island_migration_report",
    "validate_explain_payload",
    "validate_consolidated_report",
    "validate_island_report",
    "determinism_signature",
    "assert_determinism_equivalence",
]


# ----------------------- Determinism Gate Utilities -----------------------

def determinism_signature(
    report: Mapping[str, Any], *, include_env: bool = True
) -> str:
    """Compute a stable signature for a consolidated report for gate checks.

    By default includes env (seed/device/dtype) so mismatches are caught.
    """
    base: Dict[str, Any] = {
        "derivation_checksum": report.get("derivation_checksum"),
        "wl_fingerprint": report.get("wl_fingerprint"),
        "batches": report.get("batches"),
    }
    if include_env and isinstance(report.get("env"), Mapping):
        env = report["env"]
        base["env"] = {
            "seed": env.get("seed"),
            "device": env.get("device"),
            "dtype": env.get("dtype"),
        }
    return _short_sha16(str(base))


def assert_determinism_equivalence(
    reports: Iterable[Mapping[str, Any]],
    *,
    include_env: bool = True,
) -> Tuple[str, List[str]]:
    """Assert all reports are equivalent under the determinism signature.

    Returns the reference signature and list of encountered signatures.
    Raises AssertionError if drift is detected.
    """
    signatures: List[str] = []
    for rep in reports:
        signatures.append(determinism_signature(rep, include_env=include_env))
    ref = signatures[0] if signatures else ""
    if any(sig != ref for sig in signatures):
        raise AssertionError(f"Determinism drift: {signatures}")
    return ref, signatures


