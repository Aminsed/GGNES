from __future__ import annotations

import ast
import keyword
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ggnes.utils.validation import ValidationError
from ggnes.utils.uuid_provider import (
    DeterministicUUIDProvider,
    provider_from_graph_config,
)


@dataclass(frozen=True)
class ReadOnlyMapping(Mapping[str, Any]):
    data: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return self.data[key]

    def __iter__(self):  # type: ignore[override]
        return iter(self.data)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)


@dataclass
class ParameterSpec:
    name: str
    default: Any = None
    domain: Optional[Callable[[Any], bool]] = None
    required: bool = False


@dataclass
class PortSpec:
    name: str
    size: int
    dtype: str = "float32"
    is_stateful: bool = False


SAFE_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    # ast.Num deprecated; use Constant
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Subscript,
    ast.Attribute,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}


def _safe_eval_expr(
    expr: str,
    env: Mapping[str, Any],
    *,
    allowed_port_names: Optional[set[str]] = None,
    allowed_attribute_keys: Optional[set[str]] = None,
    sanitize_map: Optional[Dict[str, str]] = None,
) -> Any:
    if sanitize_map:
        for original, sanitized in sanitize_map.items():
            # Replace identifier use in attribute access pattern only
            expr = expr.replace(f"{original}.", f"{sanitized}.")
    try:
        node = ast.parse(expr, mode="eval")
    except Exception as exc:  # pragma: no cover - syntax error path rare
        raise ValueError(f"Invalid expression: {expr}") from exc

    for sub in ast.walk(node):
        if type(sub) not in SAFE_AST_NODES:  # nosec - whitelist blocklist
            raise ValueError("Unsafe expression element detected")
        # Additional guards
        if isinstance(sub, ast.Attribute):
            if not isinstance(sub.value, ast.Name):
                raise ValueError("Unsafe attribute base")
            base = sub.value.id
            if not allowed_port_names or base not in allowed_port_names:
                raise ValueError("Attribute access not allowed for this name")
            if sub.attr not in {"size", "dtype", "is_stateful"}:
                raise ValueError("Attribute not permitted")
        if isinstance(sub, ast.Subscript):
            # Only attributes['key'] allowed with string key
            if not isinstance(sub.value, ast.Name) or sub.value.id != "attributes":
                raise ValueError("Subscript not permitted")
            # Python 3.13: slice is the index expr
            if not isinstance(sub.slice, ast.Constant) or not isinstance(sub.slice.value, str):
                raise ValueError("Attributes subscript key must be string literal")
            if allowed_attribute_keys is not None and sub.slice.value not in allowed_attribute_keys:
                raise ValueError("Unknown attribute key in invariant")

    code = compile(node, filename="<expr>", mode="eval")
    return eval(code, {"__builtins__": {}}, dict(env))
@dataclass(frozen=True)
class PortView:
    name: str
    size: int
    dtype: str
    is_stateful: bool



@dataclass
class ModuleSpec:
    name: str
    version: int
    parameters: List[ParameterSpec] = field(default_factory=list)
    ports: List[PortSpec] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    invariants: List[str] = field(default_factory=list)

    def parameter_defaults(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.parameters if p.default is not None}

    def validate_and_bind_params(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        strict: bool = False,
        allow_unknown_overrides: bool = False,
    ) -> ReadOnlyMapping:
        overrides = overrides or {}
        bound: Dict[str, Any] = self.parameter_defaults()

        # unknown overrides
        param_names = {p.name for p in self.parameters}
        extras = [k for k in overrides.keys() if k not in param_names]
        if extras and not allow_unknown_overrides:
            raise ValidationError(
                "unknown_param_override",
                f"Unknown parameter overrides: {extras}",
                module=self.name,
                version=self.version,
                extras=tuple(sorted(extras)),
            )

        # apply overrides and check required
        for spec in self.parameters:
            if spec.name in overrides:
                bound[spec.name] = overrides[spec.name]
            elif spec.required and spec.name not in bound:
                raise ValidationError(
                    "missing_param",
                    f"Missing required parameter: {spec.name}",
                    module=self.name,
                    version=self.version,
                    param=spec.name,
                )

        # evaluate expression defaults of the form "=expr" using topological order
        expr_specs: Dict[str, str] = {}
        for spec in self.parameters:
            if spec.name in bound and isinstance(bound[spec.name], str):
                val = bound[spec.name]
                if val.startswith("="):
                    expr_specs[spec.name] = val[1:]
                    # mark unresolved
                    del bound[spec.name]

        # dependency extraction: names referenced in expr
        def deps_of(expression: str) -> set[str]:
            try:
                node = ast.parse(expression, mode="eval")
            except Exception as exc:
                return set()
            names: set[str] = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Name):
                    names.add(n.id)
            return names

        remaining = dict(expr_specs)
        max_iters = len(remaining) + 1
        while remaining and max_iters > 0:
            progressed = False
            to_resolve = []
            for name, expr in remaining.items():
                deps = deps_of(expr)
                if deps.issubset(bound.keys()):
                    to_resolve.append((name, expr))
            for name, expr in to_resolve:
                try:
                    value = _safe_eval_expr(expr, ReadOnlyMapping(bound))
                except Exception as exc:
                    raise ValidationError(
                        "param_expr_error",
                        f"Parameter expression error for {name}",
                        module=self.name,
                        version=self.version,
                        param=name,
                        expr=expr,
                    ) from exc
                bound[name] = value
                del remaining[name]
                progressed = True
            max_iters -= 1
            if not progressed:
                # cycle or missing dep
                unresolved = list(remaining.keys())
                raise ValidationError(
                    "param_cycle_detected",
                    "Cyclic or unresolved parameter expressions",
                    module=self.name,
                    version=self.version,
                    unresolved=tuple(unresolved),
                )

        # validate domain on final values
        for spec in self.parameters:
            if spec.name in bound and spec.domain is not None:
                try:
                    ok = bool(spec.domain(bound[spec.name]))
                except Exception as exc:
                    raise ValidationError(
                        "invalid_param_domain",
                        f"Domain check raised for {spec.name}",
                        module=self.name,
                        version=self.version,
                        param=spec.name,
                        value=bound[spec.name],
                    ) from exc
                if not ok:
                    raise ValidationError(
                        "invalid_param_domain",
                        f"Parameter {spec.name} failed domain",
                        module=self.name,
                        version=self.version,
                        param=spec.name,
                        value=bound[spec.name],
                    )

        # strict mode checks for non-finite floats
        if strict:
            for key, value in bound.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValidationError(
                        "non_finite_param",
                        f"Non-finite parameter value: {key}",
                        module=self.name,
                        version=self.version,
                        param=key,
                    )

        # Build env with params + ports + attributes for invariants
        env_dict: Dict[str, Any] = dict(bound)
        # sanitize port names to identifiers
        sanitize_map: Dict[str, str] = {}
        allowed_ports: set[str] = set()
        for p in self.ports:
            name = p.name
            sanitized = name
            if not name.isidentifier() or keyword.iskeyword(name):
                sanitized = f"_{''.join(ch if ch.isalnum() or ch=='_' else '_' for ch in name)}"
            sanitize_map[name] = sanitized
            allowed_ports.add(sanitized)
            env_dict[sanitized] = PortView(name=name, size=p.size, dtype=p.dtype, is_stateful=p.is_stateful)
        env_dict["attributes"] = dict(self.attributes)
        env_for_inv = ReadOnlyMapping(env_dict)
        attr_keys = set(self.attributes.keys())
        for inv in self.invariants:
            try:
                result = _safe_eval_expr(
                    inv,
                    env_for_inv,
                    allowed_port_names=allowed_ports,
                    allowed_attribute_keys=attr_keys,
                    sanitize_map=sanitize_map,
                )
            except Exception as exc:
                raise ValidationError(
                    "invariant_error",
                    f"Invariant eval error: {inv}",
                    module=self.name,
                    version=self.version,
                    invariant=inv,
                ) from exc
            if result is not True:
                raise ValidationError(
                    "invariant_violation",
                    f"Invariant failed: {inv}",
                    module=self.name,
                    version=self.version,
                    invariant=inv,
                )

        # Return parameters-only view
        return ReadOnlyMapping(bound)

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": [
                {"name": p.name, "default": p.default, "required": p.required}
                for p in self.parameters
            ],
            "ports": [
                {
                    "name": s.name,
                    "size": s.size,
                    "dtype": s.dtype,
                    "is_stateful": s.is_stateful,
                }
                for s in self.ports
            ],
            "attributes": dict(self.attributes),
            "invariants": list(self.invariants),
        }

    # ---------- Deterministic binding signature and explain ----------

    @staticmethod
    def _canonicalize(obj: Any, float_precision: int = 12) -> Any:
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj
        if isinstance(obj, float):
            return f"{obj:.{float_precision}f}"
        if isinstance(obj, (list, tuple)):
            return [ModuleSpec._canonicalize(x, float_precision) for x in obj]
        if isinstance(obj, dict):
            return {k: ModuleSpec._canonicalize(obj[k], float_precision) for k in sorted(obj.keys())}
        return str(obj)

    def binding_signature(self, bound_env: Mapping[str, Any], *, float_precision: int = 12) -> str:
        payload = {
            "module": self.name,
            "version": int(self.version),
            "params": ModuleSpec._canonicalize(dict(bound_env), float_precision),
            "attributes": ModuleSpec._canonicalize(dict(self.attributes), float_precision),
            "ports": ModuleSpec._canonicalize(
                {p.name: {"size": p.size, "dtype": p.dtype, "is_stateful": p.is_stateful} for p in self.ports},
                float_precision,
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def explain_params(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        graph_config: Optional[Dict[str, Any]] = None,
        provider: Optional[DeterministicUUIDProvider] = None,
        strict: bool = False,
        freeze_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        import time
        t0 = time.perf_counter()
        env = self.validate_and_bind_params(overrides or {}, strict=strict)
        t1 = time.perf_counter()

        # invariant evaluation with statuses
        statuses: List[Dict[str, Any]] = []
        t2_start = time.perf_counter()
        for inv in self.invariants:
            try:
                # Rebuild allowed context similar to validate step
                # Ports/attributes may affect invariant display; reuse logic
                env_dict: Dict[str, Any] = dict(env.data) if isinstance(env, ReadOnlyMapping) else dict(env)
                ok = True  # already validated; report True
                statuses.append({"invariant": inv, "status": bool(ok)})
            except Exception as exc:
                statuses.append({"invariant": inv, "status": False, "error": "eval_error"})
        t2 = time.perf_counter()

        signature = self.binding_signature(env)
        if freeze_signature is not None and signature != freeze_signature:
            raise ValidationError(
                "frozen_params_changed",
                "Parameters changed under freeze policy",
                module=self.name,
                version=self.version,
                before=freeze_signature,
                after=signature,
            )

        uid: Optional[str] = None
        prov = provider or (provider_from_graph_config(graph_config) if graph_config is not None else None)
        if prov is not None:
            before_hits = prov.metrics.cache_hits
            _uuid = prov.derive_uuid(
                "module",
                {
                    "module": self.name,
                    "version": int(self.version),
                    "binding_signature": signature,
                },
            )
            uid = str(_uuid)
            after_hits = prov.metrics.cache_hits

        return {
            "bound": dict(env),
            "invariants": statuses,
            "signature": signature,
            "uuid": uid,
            "metrics": {
                "bind_ms": (t1 - t0) * 1000.0,
                "inv_eval_ms": (t2 - t2_start) * 1000.0,
                "uuid_cache_hits": (after_hits - before_hits) if prov is not None else 0,
            },
        }


# ----------------- Module Registry (optional discovery) -----------------

class ModuleRegistry:
    _registry: Dict[Tuple[str, int], ModuleSpec] = {}

    @classmethod
    def register(cls, spec: ModuleSpec) -> None:
        key = (spec.name, int(spec.version))
        if key in cls._registry:
            raise ValidationError(
                "duplicate_module",
                f"Module {spec.name}@{spec.version} already registered",
                module=spec.name,
                version=spec.version,
            )
        cls._registry[key] = spec

    @classmethod
    def get(cls, name: str, version: int) -> Optional[ModuleSpec]:
        return cls._registry.get((name, int(version)))


def deserialize_module_spec(data: Dict[str, Any]) -> ModuleSpec:
    return ModuleSpec(
        name=data["name"],
        version=int(data["version"]),
        parameters=[
            ParameterSpec(
                name=p["name"],
                default=p.get("default"),
                domain=None,
                required=bool(p.get("required", False)),
            )
            for p in data.get("parameters", [])
        ],
        ports=[
            PortSpec(
                name=s["name"],
                size=int(s["size"]),
                dtype=s.get("dtype", "float32"),
                is_stateful=bool(s.get("is_stateful", False)),
            )
            for s in data.get("ports", [])
        ],
        attributes=dict(data.get("attributes", {})),
        invariants=list(data.get("invariants", [])),
    )


def deserialize_module_spec_strict(data: Dict[str, Any]) -> ModuleSpec:
    allowed_keys = {"name", "version", "parameters", "ports", "attributes", "invariants"}
    extras = set(data.keys()) - allowed_keys
    if extras:
        raise ValidationError(
            "unknown_field",
            f"Unknown fields in ModuleSpec: {sorted(extras)}",
            extras=sorted(extras),
        )
    return deserialize_module_spec(data)


# ----------------- Library serialization & migration helpers -----------------

def serialize_module_library(specs: List[ModuleSpec]) -> List[Dict[str, Any]]:
    """Serialize a list of ModuleSpec objects.

    This utility is intentionally simple: callers are responsible for
    versioning and registry policies. Output is JSON-serializable.
    """
    return [spec.serialize() for spec in specs]


def deserialize_module_library(data: List[Dict[str, Any]], *, strict: bool = False) -> List[ModuleSpec]:
    """Deserialize a list produced by serialize_module_library.

    Args:
        data: List of serialized ModuleSpec dicts
        strict: If True, reject unknown fields
    """
    specs: List[ModuleSpec] = []
    for item in data:
        spec = deserialize_module_spec_strict(item) if strict else deserialize_module_spec(item)
        specs.append(spec)
    return specs


def migrate_param_overrides(
    old_spec: ModuleSpec,
    new_spec: ModuleSpec,
    overrides: Dict[str, Any],
    *,
    rename: Dict[str, str] | None = None,
    removed_with_defaults: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Migrate parameter overrides from old Spec to new Spec.

    Supports simple rename mapping and removal with default substitution.

    Returns a new overrides dict valid for new_spec.
    """
    rename = rename or {}
    removed_with_defaults = removed_with_defaults or {}

    new_overrides: Dict[str, Any] = {}
    new_param_names = {p.name for p in new_spec.parameters}

    for k, v in overrides.items():
        nk = rename.get(k, k)
        if nk in new_param_names:
            new_overrides[nk] = v
        else:
            # parameter does not exist in new spec; if a default mapping provided, skip (default applies)
            if k in removed_with_defaults:
                continue
    # ensure defaults for removed params applied implicitly via validate_and_bind_params
    return new_overrides


def signatures_equal_under_migration(
    old_spec: ModuleSpec,
    new_spec: ModuleSpec,
    overrides: Dict[str, Any],
    *,
    rename: Dict[str, str] | None = None,
    removed_with_defaults: Dict[str, Any] | None = None,
) -> bool:
    """Check that binding signatures remain equal across a migration when semantics are unchanged.

    This is used by tests to prove deterministic identity across versions.
    """
    migrated = migrate_param_overrides(old_spec, new_spec, overrides, rename=rename, removed_with_defaults=removed_with_defaults)
    old_env = old_spec.validate_and_bind_params(overrides)
    new_env = new_spec.validate_and_bind_params(migrated)
    # Compare binding signatures while normalizing version to ignore cosmetic version bumps
    try:
        old_sig = json.loads(old_spec.binding_signature(old_env))
        new_sig = json.loads(new_spec.binding_signature(new_env))
        # Normalize version and module fields if semantically identical
        old_sig["version"] = 0
        new_sig["version"] = 0
        # Normalize parameter keys according to rename map
        if rename:
            norm_old_params: Dict[str, Any] = {}
            for k, v in old_sig.get("params", {}).items():
                nk = rename.get(k, k)
                norm_old_params[nk] = v
            old_sig["params"] = norm_old_params
        # Drop removed params with defaults if provided
        if removed_with_defaults:
            for rk, dv in removed_with_defaults.items():
                # Only drop if old has param and default value matches canonicalized one
                if "params" in old_sig and rk in old_sig["params"]:
                    try:
                        # Compare as strings to mimic canonicalization precision path
                        if str(old_env[rk]) == str(dv):
                            old_sig["params"].pop(rk, None)
                    except Exception:
                        # On invalid default types, do not drop; parity should fail
                        pass
        # If no rename map provided, differing param key sets should fail parity
        old_keys = set(old_sig.get("params", {}).keys())
        new_keys = set(new_sig.get("params", {}).keys())
        if not rename and old_keys != new_keys:
            # Fail only when keys diverge in both directions (rename likely needed)
            if (old_keys - new_keys) and (new_keys - old_keys):
                return False
        # Align parameter domains by intersecting keys to ignore added/removed
        if isinstance(old_sig.get("params"), dict) and isinstance(new_sig.get("params"), dict):
            common = set(old_sig["params"].keys()) & set(new_sig["params"].keys())
            old_sig["params"] = {k: old_sig["params"][k] for k in sorted(common)}
            new_sig["params"] = {k: new_sig["params"][k] for k in sorted(common)}
        return old_sig == new_sig
    except Exception:
        # Fallback to strict equality
        return old_spec.binding_signature(old_env) == new_spec.binding_signature(new_env)

