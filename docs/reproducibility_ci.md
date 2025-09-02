# Reproducibility and CI

This document specifies how to make GGNES runs reproducible and how to integrate determinism checks into CI. All guidance below matches the shipped APIs in `ggnes/utils/observability.py` and the tests under `tests/test_m33`.

## Artifact protocol

For every experiment run (demo, tutorial, benchmark), persist the following alongside any checkpoints or CSV logs:
- A consolidated report (schema v2) generated via `consolidated_report(...)`.
- The Weisfeiler–Lehman (WL) fingerprint of the graph (already inside the report).
- The determinism checksum (already inside the report) and the determinism signature (see below if you store separately).
- The environment block (seed/device/dtype) and RNG context signatures (already inside the report when `rng_manager` is provided).

Pin the following to maximize reproducibility:
- `RNGManager(seed=...)` value used, including any hierarchical context names.
- Graph‐level configuration that affects WL fingerprinting (e.g., `wl_iterations`).
- Backend configuration for translation (device/dtype) and cache toggles (cache parity is guaranteed; toggling cache must not change outputs).

## Determinism signatures and gates

The determinism signature is a short, stable digest over core report fields computed by `determinism_signature(report, include_env=True)`.
- `include_env=True` (default) is recommended for CI to detect device/dtype/seed divergence.
- Set `include_env=False` only when intentionally comparing across environments and you want to ignore device/dtype differences.
- RNG context signatures: by default `consolidated_report` includes signatures for `['selection', 'repair', 'aggregation_dropout']`. To include additional contexts (e.g., `'mutation'`, custom hierarchical names), pass `extra_rng_contexts` to `consolidated_report`.

Two primary gate styles are supported:
1) Pairwise check between reference and candidate run using `assert_determinism_equivalence([ref_report, cand_report], include_env=True)`.
2) Multi‐run check across N reports; raises on any drift.

### Example (Python)
```python
from ggnes.utils.observability import (
    consolidated_report,
    determinism_signature,
    assert_determinism_equivalence,
    migrate_consolidated_report_to_v2,
    validate_consolidated_report_v2,
)

# After your run, build a fresh report
after = consolidated_report(engine, root, graph, rng_manager=rng, device='cpu', dtype='float32')
validate_consolidated_report_v2(after)

# Load a stored reference (JSON) and migrate if needed
ref = load_json('artifacts/reference_report.json')
ref_v2 = migrate_consolidated_report_to_v2(ref)
validate_consolidated_report_v2(ref_v2)

# Gate (will raise AssertionError on drift)
assert_determinism_equivalence([ref_v2, after], include_env=True)
print('Determinism gate passed:', determinism_signature(after))
```

### Example (CI snippet)
Run a small Python gate after tests/build:
```bash
python - <<'PY'
import json, sys
from ggnes.utils.observability import (
  migrate_consolidated_report_to_v2, validate_consolidated_report_v2,
  assert_determinism_equivalence, determinism_signature
)
ref = json.load(open('artifacts/reference_report.json'))
cur = json.load(open('artifacts/current_report.json'))
ref = migrate_consolidated_report_to_v2(ref)
cur = migrate_consolidated_report_to_v2(cur)
validate_consolidated_report_v2(ref)
validate_consolidated_report_v2(cur)
assert_determinism_equivalence([ref, cur], include_env=True)
print('OK signature:', determinism_signature(cur))
PY
```

## Golden vectors

Golden determinism vectors are minimal JSON excerpts (or full reports) checked into version control to lock determinism across refactors. Recommended practice:
- Store the entire consolidated report (v2) when feasible; otherwise store a reduced object containing the fields used by `determinism_signature`.
- Update goldens only when intentional changes occur (e.g., schema evolution or expected behavior change) and document why.

## Migration and validation

The codebase ships both tolerant and strict validators:
- `validate_consolidated_report(report)`: tolerant (accepts extra fields) for legacy artifacts.
- `validate_consolidated_report_v2(report)`: strict; enforces `schema_version == 2`.
- `migrate_consolidated_report_to_v2(report)`: sets `schema_version` to 2 for older artifacts; use before strict validation.

CI should:
1) Migrate legacy reports to v2.
2) Run strict validation.
3) Apply determinism gate with `include_env=True`.

## Scope and caveats

- The determinism signature covers: derivation checksum, WL fingerprint, batches, and optionally the env triplet (seed/device/dtype). If your CI intentionally varies device/dtype, set `include_env=False` and document the policy.
- WL fingerprints depend on graph structure and configuration (e.g., `wl_iterations`). Pin these in configs for longitudinal comparisons.
- RNG context signatures (stored in the `env` block as short hashes) enable debugging of context‐specific drift and can be extended by passing `extra_rng_contexts` to `consolidated_report` when custom contexts are relevant.
- Cache parity: translation cache metrics may differ (hits/misses/entries), but toggling cache must not change outputs or determinism signatures.

## Practical checklist

- [ ] Use `RNGManager(seed=...)` and avoid direct `random`/`numpy.random` calls.
- [ ] Generate and persist a consolidated report (schema v2) per run.
- [ ] Migrate and strictly validate reports in CI.
- [ ] Compute determinism signatures; gate on `assert_determinism_equivalence`.
- [ ] Maintain golden reports/vectors; update with rationale when behavior changes.
- [ ] Pin graph and backend configs that influence fingerprints (e.g., WL iterations, dtype/device).
- [ ] Keep module/library/CompositeGenotype artifacts with scheme metadata for replay.

## Related tests and modules

- `tests/test_m33/test_determinism_gate.py`: gate behavior and drift detection
- `tests/test_m33/test_schema_v2_migration.py`: strict vs tolerant validation and migration
- `ggnes/utils/observability.py`: `consolidated_report`, validators/migration, `determinism_signature`, `assert_determinism_equivalence`
