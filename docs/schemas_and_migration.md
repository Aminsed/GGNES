# Schemas and Migration

This document specifies the schemas, validators, and migration helpers implemented in the current codebase. All content below directly reflects the functions in `ggnes/utils/observability.py`, `ggnes/hierarchical/module_spec.py`, and `ggnes/evolution/composite_genotype.py`.

## Consolidated report (v1 → v2)

Source: `ggnes/utils/observability.py`

### v2 (current, strict)
Required top-level fields (validated by `validate_consolidated_report_v2`):
- `schema_version: int` — MUST equal `2`
- `derivation_checksum: str`
- `wl_fingerprint: str`
- `batches: list` — per-derivation batch metrics (opaque to validator)
- `determinism_checksum: str`

Optional `env: mapping` may include:
- `seed: int`
- `device: str`
- `dtype: str`
- `rng_signatures: mapping[str, str]` — short stable hashes per RNG context (e.g., `selection`, `repair`, `aggregation_dropout`)

Notes:
- `validate_consolidated_report` is tolerant (does not require `schema_version`; accepts extra fields) and is suitable for legacy artifacts or forward-compat tolerance.
- `migrate_consolidated_report_to_v2(report)` sets `schema_version` to `2` on a copy; it does not otherwise change structure. Use it before applying strict v2 validation to v1 artifacts.

### v1 (legacy, tolerant)
Top-level fields validated by the tolerant validator:
- `derivation_checksum: str`
- `wl_fingerprint: str`
- `batches: list`
- `determinism_checksum: str`
- Optional `env` (see above). No `schema_version` required.

### Determinism signatures
- `determinism_signature(report, include_env=True) -> str` — Stable digest over core fields; with `include_env=True` (recommended for CI), incorporates `(seed, device, dtype)`.
- `assert_determinism_equivalence(reports, include_env=True)` — Raises `AssertionError` on drift; returns `(ref_signature, signatures)` on success.

## Genotype explain payload

Source: `ggnes/utils/observability.py`

`validate_explain_payload(payload)` requires:
- `uuid: str`
- `scheme: mapping` — provider scheme metadata
- `canonical: mapping` — canonical inputs used for UUID derivation
- `checksums: mapping` with keys `full`, `rules`, `params`, `hier` (all strings)

Optional sections:
- `diffs.params_changed: mapping` — param→{before, after}
- `timings.validate_ms|uuid_ms|checksum_ms: number`
- `constraints.policy_constraints_ok: bool`

## Island migration report

Source: `ggnes/utils/observability.py`

`validate_island_report(report)` requires:
- `topology: str`
- `migration_size: int`
- `islands: int`
- `per_island: list` — entries may include size/movement metrics
- `migration_checksum: str`

## ModuleSpec serialization and migration

Source: `ggnes/hierarchical/module_spec.py`

### Serialization
`ModuleSpec.serialize()` emits:
- `name: str`
- `version: int`
- `parameters: list[{name, default?, required?}]`
- `ports: list[{name, size, dtype?, is_stateful?}]`
- `attributes: mapping`
- `invariants: list[str]`

### Deserialization
- `deserialize_module_spec(data)` — tolerant; ignores unknown fields
- `deserialize_module_spec_strict(data)` — strict; raises on unknown fields

### Parameter override migration
- `migrate_param_overrides(old_spec, new_spec, overrides, rename=None, removed_with_defaults=None) -> mapping` — applies rename map; drops removed parameters only if covered by defaults.
- `signatures_equal_under_migration(old_spec, new_spec, overrides, rename=None, removed_with_defaults=None) -> bool` — binds parameters under old/new specs, normalizes version and rename-induced key changes, and compares binding signatures (JSON) for equality.

## CompositeGenotype schema

Source: `ggnes/evolution/composite_genotype.py`

### Serialize
`CompositeGenotype.serialize()` emits:
- `_schema: 1` — current version for composite genotype serialization
- `g1: {rules: list[dict]}` — rule dicts include `rule_id` (str UUID) and optional fields like `priority`, `probability`
- `g2: mapping` — policy parameters (e.g., `training_epochs`, `batch_size`, `learning_rate`, `parallel_execution`, `wl_iterations`, `failure_floor_threshold`)
- `g3: {modules: mapping[str, mapping], attributes: mapping}` — JSON-scalar values only
- `uuid_scheme: mapping` — provider scheme metadata for provenance

### Deserialize
`CompositeGenotype.deserialize(data, provider=None)`
- Requires `_schema == 1`; raises `ValueError` for other versions (future versions are not yet defined in code).
- Reconstructs typed `G1Grammar`, `G2Policy`, `G3Hierarchy`; uses provided `DeterministicUUIDProvider` or a default one.

### Identity
- `uuid()` computes a deterministic UUID over the canonicalized tuple `(g1, g2, g3, schema_version)`. Freeze/strict flags are honored by the provider.

## Accuracy and limits

- Consolidated report migration currently upgrades v1→v2 by setting `schema_version=2` only; field shapes are otherwise unchanged. This matches `migrate_consolidated_report_to_v2`.
- Composite genotype serialization supports only `_schema==1`; migration between composite genotype schema versions is not implemented.
- ModuleSpec migration helpers are designed for simple renames and removals with defaults; arbitrary structural migrations are intentionally out of scope and should be modeled as new specs with explicit overrides.

## Examples

### Consolidated report migration and strict validation
```python
from ggnes.utils.observability import migrate_consolidated_report_to_v2, validate_consolidated_report_v2
rep_v1 = {"derivation_checksum":"...","wl_fingerprint":"...","batches":[],"determinism_checksum":"..."}
rep_v2 = migrate_consolidated_report_to_v2(rep_v1)
validate_consolidated_report_v2(rep_v2)
```

### ModuleSpec strict deserialization
```python
from ggnes.hierarchical.module_spec import deserialize_module_spec_strict
spec = deserialize_module_spec_strict({"name":"Block","version":1,"parameters":[],"ports":[],"attributes":{},"invariants":[]})
```

### Parameter override migration and signature parity
```python
from ggnes.hierarchical.module_spec import migrate_param_overrides, signatures_equal_under_migration
# old_spec, new_spec, overrides prepared elsewhere
new_over = migrate_param_overrides(old_spec, new_spec, overrides, rename={"dim":"width"})
assert signatures_equal_under_migration(old_spec, new_spec, overrides, rename={"dim":"width"})
```
