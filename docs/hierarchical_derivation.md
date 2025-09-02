# Hierarchical Derivation

This chapter specifies, precisely and exhaustively, how hierarchical modules are defined, validated, bound, and expanded in GGNES. It reflects the current implementation in `ggnes/hierarchical/module_spec.py` and `ggnes/hierarchical/derivation.py`.

## ModuleSpec
A `ModuleSpec` is the unit of hierarchical composition.

### Structure
- `name: str`, `version: int`
- `parameters: List[ParameterSpec]`
  - `ParameterSpec(name: str, default: Any = None, domain: Optional[Callable[[Any], bool]] = None, required: bool = False)`
- `ports: List[PortSpec]`
  - `PortSpec(name: str, size: int, dtype: str = "float32", is_stateful: bool = False)`
- `attributes: Dict[str, Any]` (opaque key–value map)
- `invariants: List[str]` (guards over params/ports/attributes)

### Parameter binding
`validate_and_bind_params(overrides, strict=False, allow_unknown_overrides=False)`:
- Start from defaults in `ParameterSpec` entries.
- Apply `overrides`:
  - Unknown override keys raise `ValidationError("unknown_param_override", extras=...)` unless `allow_unknown_overrides=True`.
  - Missing required parameters raise `ValidationError("missing_param", param=...)`.
- Expression defaults: if a default value is a string `"=expr"`, it is treated as an expression to be evaluated once its dependencies are bound. Resolution is topological:
  - Dependencies are detected from variable names in the AST; iteration proceeds until no unresolved expressions remain; if a cycle remains, raises `ValidationError("param_cycle_detected", unresolved=...)`.
- Domains: if `spec.domain` is provided, it is called with the final value; a `False` result or exception raises `ValidationError("invalid_param_domain", param=..., value=...)`.
- Strict mode: if `strict=True`, any non‑finite floating‑point value in the bound environment raises `ValidationError("non_finite_param", param=...)`.

Returned value is a read‑only mapping view of parameters only (no ports/attributes yet).

### Safe expression evaluation
Expressions are parsed and evaluated under a guarded AST with a small whitelist of nodes (`Expression`, `BinOp` with `+ - * / // % **`, `UnaryOp`, `BoolOp`, comparisons, `Name`, `Constant`, limited `Attribute`, limited `Subscript`).
- Port attributes accessible: `.size`, `.dtype`, `.is_stateful` on sanitized port identifiers (see below).
- Attribute map subscripting is allowed only as `attributes['literal_key']` where the key is a string literal.
- Non‑identifier or reserved port names are sanitized to valid identifiers (prefixed and with non‑alnum mapped to `_`). Sanitization is applied consistently to both environment and invariant checks.

### Invariants
Each string in `invariants` is evaluated under a read‑only environment that includes:
- All bound parameters
- Port views: `{sanitized_port_name: PortView(name, size, dtype, is_stateful)}`
- `attributes`: a shallow copy of the `ModuleSpec.attributes` dict

An invariant must evaluate to `True`; otherwise `ValidationError("invariant_violation", invariant=...)` is raised. Any evaluation error is surfaced as `ValidationError("invariant_error", invariant=...)`.

### Binding signature
`binding_signature(bound_env)` returns a canonical JSON string containing:
- `module`, `version`
- `params`: canonicalized bound params (sorted keys, fixed float formatting)
- `attributes`: canonicalized attributes
- `ports`: canonicalized port metadata map `{name: {size, dtype, is_stateful}}`

Two semantically identical specs (possibly after migration/rename) can be compared via their (normalized) signatures (see migration helpers below).

### Explain params
`explain_params(overrides, graph_config=None, provider=None, strict=False, freeze_signature=None)` returns:
- `bound`: parameters dict (materialized)
- `invariants`: list of `{invariant, status: bool}` entries
- `signature`: binding signature string
- `uuid`: if `provider` is supplied (or derived from `graph_config`), a deterministic UUID for this binding (`entity_type="module"`) based on the signature
- `metrics`:
  - `bind_ms`: binding (incl. expression resolution) time
  - `inv_eval_ms`: invariant evaluation time
  - `uuid_cache_hits`: difference in provider cache hits before/after

If `freeze_signature` is provided, any change in signature raises `ValidationError("frozen_params_changed", before=..., after=...)`.

### Registry and serialization
- `ModuleRegistry.register(spec)` / `ModuleRegistry.get(name, version)` provide a simple in‑process registry; duplicate registration raises `ValidationError("duplicate_module", ...)`.
- `serialize_module_library(specs)` produces a JSON‑serializable list. `deserialize_module_library(data, strict=False)` loads it, with a strict variant rejecting unknown fields in the serialized form.

### Parameter migration helpers
- `migrate_param_overrides(old_spec, new_spec, overrides, rename=None, removed_with_defaults=None)` creates a new override dict valid for `new_spec`, applying simple key renames and dropping removed parameters when a default is provided.
- `signatures_equal_under_migration(old_spec, new_spec, overrides, rename=None, removed_with_defaults=None)` returns `True` if the binding signatures are equal after normalizations:
  - Normalize `version` field to `0` in both signatures
  - Apply rename mapping to old params; drop removed params where default applies
  - Compare only common param keys after normalization

## DerivationEngine
`DerivationEngine(graph, config=None, rng_manager=None, uuid_provider=None)` expands a `ModuleSpec` tree into the graph using nested transactions.

### Limits & budgets
- `max_derivation_depth` (default 8)
- `max_derivation_expansions` (default 1000)
- Optional `derivation_time_budget_ms`: enforces a run budget, measured from `expand` start

Violations raise typed `ValidationError`s:
- `derivation_timeout` with runtime and budget context
- `hierarchy_limit` with `reason` in `{max_depth_exceeded, max_expansions_exceeded}`

### Determinism and UUIDs
For each expanded node, a deterministic UUID is derived (`entity_type="hier_module"`) over the tuple `{module, version, path, binding_signature}`. The RNG context is advanced deterministically per expansion:
```
"expand:module:{spec.name}@{spec.version}:path:{root|i.j.k}"
```

### Structural effect
Each module expansion adds a single `HIDDEN` node (inside a `TransactionManager`) to represent the module, with attributes that include:
- `module_name`, `module_version`, `derivation_uuid`
- `output_size` fixed to `1` (minimal, non‑disruptive)

All staged changes are committed atomically; on validation failure, they are rolled back and the engine returns a structured error.

### Child ordering and guards
- Children are prepared by computing (or approximating) their binding signatures and then sorted canonically by `(name, version, signature)`.
- `max_children_per_node` (if set in `config`) guards the number of child expansions; violations yield `ValidationError("hierarchy_limit", reason="max_children_exceeded", count=..., limit=...)`.

### Parallel execution knobs (observational parity only)
Configuration keys (all optional):
- `parallel_hg_execution: bool` (default False)
- `hg_max_parallel_workers: int` (default 4)
- `hg_parallel_batch_policy: {"MAX_INDEPENDENT_SET","FIXED_SIZE","PRIORITY_CAP"}`
- `hg_fixed_batch_size: int` (used when policy is `FIXED_SIZE`)
- `hg_priority_cap: int` (used when policy is `PRIORITY_CAP`)
- `hg_parallel_conflict_strategy: {"SKIP","REQUEUE","SPECULATIVE_MERGE"}`
- `hg_parallel_max_requeues: int` (default 1)
- `parallel_memory_budget_mb: int` (default 0 → unlimited)
- `hg_child_mem_cost_mb: int` (default 1)

Execution remains serial; batching is deterministic and used to produce metrics only. Conflicts (e.g., child failures) increment counters and, depending on strategy, can requeue up to the configured limit.

### Metrics (`last_expand_metrics`)
- `batches`: list of `{size, worker_count, checksum, mis_size}`
- `batches_processed: int`
- `conflicts: int`, `requeues: int`, `rollbacks: int`
- `memory_backoff_count: int` (when memory budget splits batches)
- `worker_cap: int`, `batch_policy: str`

### Errors and recovery
- Any `ValidationError` during expansion causes a rollback and is re‑raised.
- Other exceptions are wrapped as `ValidationError("derivation_failed", module=..., module_version=..., module_path=...)` after rollback.

### Explain
`explain(node)` returns a canonical, serializable summary:
- `module`, `version`, `signature`, `uuid`, `path: List[int]`, `children: List[explain(child)]`, and a `checksum`
- `checksum` is the first 16 hex chars of SHA‑256 over the flattened list of tuples `(name, version, signature, uuid, path)` for the node and its descendants.

## Interplay with observability
- Use `explain(node)` together with the derivation checksum captured in consolidated reports to relate genotype explain payloads (G3) and the realized hierarchical structure.
- `ModuleSpec.explain_params(..., provider=...)` exposes `uuid_cache_hits` in metrics to verify deterministic UUID caching under repeated calls.

## Notes on stability
- Binding signatures are insensitive to dict key order and float formatting differences (canonicalization fixed precision).
- Child ordering uses signatures to avoid ambiguity across logically equivalent parameter environments.
- RNG contexts for hierarchical expansion are derived from stable strings and advanced without mutating unrelated contexts, ensuring reproducibility across runs and platforms.
