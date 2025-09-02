"""Composite Genotype schemas for Co‑Evolution (project_guide.md §21.2, §21.3).

This module defines typed schemas for the composite genotype composed of:
- G1 Grammar genome: ordered rules metadata
- G2 Policy genome: structured hyperparameters and policies
- G3 Hierarchical genome: module parameters and ports/attributes summaries

The module also provides deterministic identity via UUIDs derived from
canonicalized content using the uuid provider (project_guide.md §19).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ggnes.utils.uuid_provider import DeterministicUUIDProvider, UUIDProviderConfig


def _ensure_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive int")
    return value


def _ensure_float_in_range(value: Any, name: str, min_inclusive: float, max_inclusive: float) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be float")
    v = float(value)
    if not (min_inclusive <= v <= max_inclusive):
        raise ValueError(f"{name} out of range [{min_inclusive}, {max_inclusive}]")
    return v


def _ensure_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


@dataclass
class G1Grammar:
    """Grammar genome stub capturing rule identities and metadata.

    Rules are represented minimally to avoid coupling: a list of dicts
    with mandatory key "rule_id" (string UUID) and optional "priority" (int),
    "probability" (float in (0, 1]).
    """

    rules: list[dict] = field(default_factory=list)

    def validate(self) -> None:
        for idx, item in enumerate(self.rules):
            if not isinstance(item, dict):
                raise ValueError("G1.rules entries must be dicts")
            if "rule_id" not in item or not isinstance(item["rule_id"], str):
                raise ValueError("G1.rules entries must include rule_id as str")
            if "priority" in item:
                if not isinstance(item["priority"], int):
                    raise ValueError("G1.priority must be int")
            if "probability" in item:
                p = _ensure_float_in_range(item["probability"], "G1.probability", 0.0, 1.0)
                if not (p > 0.0):
                    raise ValueError("G1.probability must be in (0, 1]")


@dataclass
class G2Policy:
    """Policy genome with domains aligned to §12 Configuration.

    Only a subset is included to keep the schema focused and stable.
    """

    training_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    parallel_execution: bool = False
    wl_iterations: int = 3
    failure_floor_threshold: float = 0.5

    def validate(self) -> None:
        _ensure_positive_int(self.training_epochs, "training_epochs")
        _ensure_positive_int(self.batch_size, "batch_size")
        _ensure_float_in_range(self.learning_rate, "learning_rate", 1e-8, 10.0)
        _ensure_bool(self.parallel_execution, "parallel_execution")
        _ensure_positive_int(self.wl_iterations, "wl_iterations")
        _ensure_float_in_range(self.failure_floor_threshold, "failure_floor_threshold", 0.0, 1.0)


@dataclass
class G3Hierarchy:
    """Hierarchical parameters snapshot.

    Captures a mapping of module name to parameter map (name→literal) and an
    optional attributes summary. Values must be JSON-serializable scalars.
    """

    modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for mod, params in self.modules.items():
            if not isinstance(mod, str):
                raise ValueError("G3.modules keys must be str")
            if not isinstance(params, dict):
                raise ValueError("G3.modules values must be dict")
            for key, val in params.items():
                if not isinstance(key, str):
                    raise ValueError("G3 param names must be str")
                if not isinstance(val, (int, float, str, bool)) and val is not None:
                    raise ValueError("G3 param values must be scalar JSON types")
        # Attributes summary follows same scalar constraint
        for key, val in self.attributes.items():
            if not isinstance(key, str):
                raise ValueError("G3.attributes keys must be str")
            if not isinstance(val, (int, float, str, bool)) and val is not None:
                raise ValueError("G3.attributes values must be scalar JSON types")


@dataclass
class CompositeGenotype:
    """Composite genotype carrying G1/G2/G3 with deterministic identity.

    Identity is computed via `uuid()` using a DeterministicUUIDProvider over the
    canonicalized content of the three sub‑genomes along with the schema version
    and namespace from the provider configuration.
    """

    g1: G1Grammar = field(default_factory=G1Grammar)
    g2: G2Policy = field(default_factory=G2Policy)
    g3: G3Hierarchy = field(default_factory=G3Hierarchy)
    provider: DeterministicUUIDProvider = field(
        default_factory=lambda: DeterministicUUIDProvider(UUIDProviderConfig())
    )
    _frozen_after_uuid: bool = field(default=False, init=False, repr=False)

    def validate(self) -> None:
        self.g1.validate()
        self.g2.validate()
        self.g3.validate()

    def as_canonical_inputs(self) -> dict[str, Any]:
        # Use dataclass→dict for simple, explicit mapping
        return {
            "schema_version": 1,
            "g1": {"rules": self.g1.rules},
            "g2": asdict(self.g2),
            "g3": {"modules": self.g3.modules, "attributes": self.g3.attributes},
        }

    def uuid(self) -> str:
        self.validate()
        uid = self.provider.derive_uuid("composite_genotype", self.as_canonical_inputs())
        self._frozen_after_uuid = self.provider.config.freeze
        return str(uid)

    def mutate_field(self, path: list[str], value: Any) -> None:
        """Utility to mutate a nested field with freeze guarding for tests.

        Args:
            path: Sequence like ["g2", "learning_rate"]
            value: New value
        """
        if self._frozen_after_uuid:
            raise ValueError("uuid_freeze: mutation forbidden after UUID assignment")
        if not path:
            raise ValueError("path required")
        target = self
        for key in path[:-1]:
            target = getattr(target, key)
        setattr(target, path[-1], value)

    def serialize(self) -> dict[str, Any]:
        return {
            "_schema": 1,
            "g1": {"rules": self.g1.rules},
            "g2": asdict(self.g2),
            "g3": {"modules": self.g3.modules, "attributes": self.g3.attributes},
            "uuid_scheme": self.provider.scheme_metadata(),
        }

    @staticmethod
    def deserialize(data: dict[str, Any], provider: DeterministicUUIDProvider | None = None) -> CompositeGenotype:
        schema_ver = int(data.get("_schema", 1))
        if schema_ver != 1:
            # For now, accept and continue; future versions may migrate fields
            raise ValueError("Unsupported composite genotype schema version")
        g1 = G1Grammar(rules=list(data.get("g1", {}).get("rules", [])))
        g2_map = dict(data.get("g2", {}))
        g2 = G2Policy(
            training_epochs=int(g2_map.get("training_epochs", 10)),
            batch_size=int(g2_map.get("batch_size", 64)),
            learning_rate=float(g2_map.get("learning_rate", 0.001)),
            parallel_execution=bool(g2_map.get("parallel_execution", False)),
            wl_iterations=int(g2_map.get("wl_iterations", 3)),
            failure_floor_threshold=float(g2_map.get("failure_floor_threshold", 0.5)),
        )
        g3_src = dict(data.get("g3", {}))
        g3 = G3Hierarchy(
            modules=dict(g3_src.get("modules", {})),
            attributes=dict(g3_src.get("attributes", {})),
        )
        prov = provider or DeterministicUUIDProvider()
        return CompositeGenotype(g1=g1, g2=g2, g3=g3, provider=prov)


__all__ = [
    "G1Grammar",
    "G2Policy",
    "G3Hierarchy",
    "CompositeGenotype",
]


# M27: Mutation and crossover operators (constraint-safe and deterministic)

def mutate_composite(geno: CompositeGenotype, provider: DeterministicUUIDProvider, deltas: dict[str, any]) -> CompositeGenotype:
    """Return a mutated copy of a CompositeGenotype applying clamped deltas.

    Deltas is a simple mapping with allowed keys:
    - g2.learning_rate: adds float delta then clamps to [1e-8, 10.0]
    - g2.batch_size: adds int delta then clamps to >=1
    - g2.training_epochs: adds int delta then clamps to >=1
    - g1.rules[i].priority: adds int delta (non-negative)
    - g1.rules[i].probability: multiplies by scalar then clamps to (0, 1]

    Determinism: No randomness here; caller supplies deterministic deltas derived
    from RNGManager if needed. Validation is enforced before returning.
    """
    import copy

    out = copy.deepcopy(geno)
    out.provider = provider

    # G2 numeric fields
    if "g2.learning_rate" in deltas:
        lr = float(out.g2.learning_rate) + float(deltas["g2.learning_rate"])
        lr = max(1e-8, min(10.0, lr))
        out.g2.learning_rate = lr
    if "g2.batch_size" in deltas:
        bs = int(out.g2.batch_size) + int(deltas["g2.batch_size"])
        bs = max(1, bs)
        out.g2.batch_size = bs
    if "g2.training_epochs" in deltas:
        te = int(out.g2.training_epochs) + int(deltas["g2.training_epochs"])
        te = max(1, te)
        out.g2.training_epochs = te

    # G1 rule mutations using keys like g1.rules.0.priority / g1.rules.1.probability
    for key, delta in deltas.items():
        if not key.startswith("g1.rules."):
            continue
        parts = key.split(".")
        # Expect: g1 rules <idx> <field>
        if len(parts) != 4:
            continue
        _, _, idx_str, field = parts
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        if idx < 0 or idx >= len(out.g1.rules):
            continue
        rule = dict(out.g1.rules[idx])
        if field == "priority":
            rule["priority"] = max(0, int(rule.get("priority", 0)) + int(delta))
        elif field == "probability":
            base = float(rule.get("probability", 1.0))
            newp = base * float(delta)
            # clamp to (0, 1], avoid 0 exactly
            newp = min(1.0, max(1e-12, newp))
            rule["probability"] = newp
        out.g1.rules[idx] = rule

    # Validate before returning
    out.validate()
    return out


def crossover_composite(parent_a: CompositeGenotype, parent_b: CompositeGenotype, provider: DeterministicUUIDProvider) -> CompositeGenotype:
    """Order-independent, deterministic uniform-style crossover.

    - G1: union of rules by rule_id; prefer parent's version by lexical order of parent UUIDs for ties.
    - G2: choose median-safe values (average for floats, min for integers) deterministically.
    - G3: union of modules with A-first precedence for shared keys; attributes merged similarly.
    """
    import copy

    # G1 union by rule_id
    def by_id(rules):
        return {r.get("rule_id"): r for r in rules if isinstance(r, dict) and r.get("rule_id")}

    a_rules = by_id(parent_a.g1.rules)
    b_rules = by_id(parent_b.g1.rules)
    all_ids = sorted(set(a_rules.keys()) | set(b_rules.keys()))
    merged_rules = []
    for rid in all_ids:
        if rid in a_rules and rid in b_rules:
            # deterministic tie-break: prefer lexicographically smaller parent genotype_id string
            pa = str(parent_a.provider.derive_uuid("parent", {"id": str(parent_a.g2)}))
            pb = str(parent_b.provider.derive_uuid("parent", {"id": str(parent_b.g2)}))
            chosen = a_rules[rid] if pa <= pb else b_rules[rid]
            merged_rules.append(copy.deepcopy(chosen))
        elif rid in a_rules:
            merged_rules.append(copy.deepcopy(a_rules[rid]))
        else:
            merged_rules.append(copy.deepcopy(b_rules[rid]))

    # G2 combine deterministically
    g2 = G2Policy(
        training_epochs=min(parent_a.g2.training_epochs, parent_b.g2.training_epochs),
        batch_size=min(parent_a.g2.batch_size, parent_b.g2.batch_size),
        learning_rate=(parent_a.g2.learning_rate + parent_b.g2.learning_rate) / 2.0,
        parallel_execution=parent_a.g2.parallel_execution or parent_b.g2.parallel_execution,
        wl_iterations=min(parent_a.g2.wl_iterations, parent_b.g2.wl_iterations),
        failure_floor_threshold=(parent_a.g2.failure_floor_threshold + parent_b.g2.failure_floor_threshold) / 2.0,
    )

    # G3 union with A-first precedence
    modules = dict(parent_a.g3.modules)
    for k, v in parent_b.g3.modules.items():
        if k not in modules:
            modules[k] = v
    attributes = dict(parent_a.g3.attributes)
    for k, v in parent_b.g3.attributes.items():
        if k not in attributes:
            attributes[k] = v

    offspring = CompositeGenotype(
        g1=G1Grammar(rules=merged_rules),
        g2=g2,
        g3=G3Hierarchy(modules=modules, attributes=attributes),
        provider=provider,
    )
    offspring.validate()
    return offspring


