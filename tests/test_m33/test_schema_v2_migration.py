from ggnes.core.graph import Graph
from ggnes.hierarchical.derivation import DerivationEngine
from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
from ggnes.utils.observability import (
    consolidated_report,
    validate_consolidated_report,
    validate_consolidated_report as validate_tolerant,
)
from ggnes.utils.observability import (
    validate_consolidated_report_v2 as validate_strict,
    migrate_consolidated_report_to_v2,
)


def _derive_once():
    g = Graph()
    spec = ModuleSpec(
        name="S",
        version=1,
        parameters=[ParameterSpec("d", default=2)],
        ports=[PortSpec("in", 2), PortSpec("out", 2)],
        invariants=["out.size == d"],
    )
    engine = DerivationEngine(g)
    node = engine.expand(spec, {"d": 2})
    return g, engine, node


def test_co_33_01_schema_v2_present_and_migration_helper():
    g, engine, node = _derive_once()
    rep = consolidated_report(engine, node, g)
    # v2 schema present by default
    assert rep.get("schema_version") == 2
    # Drop and migrate back to v2
    rep_v1 = dict(rep)
    rep_v1.pop("schema_version", None)
    migrated = migrate_consolidated_report_to_v2(rep_v1)
    assert migrated.get("schema_version") == 2
    # Strict validator accepts migrated
    validate_strict(migrated)


def test_co_33_01b_strict_vs_tolerant_validation_and_unknown_fields():
    g, engine, node = _derive_once()
    rep = consolidated_report(engine, node, g)
    # Tolerant validation passes regardless
    validate_tolerant(rep)
    # Strict requires schema_version=2
    validate_strict(rep)

    # Create v1-like (no schema_version)
    rep_v1 = dict(rep)
    rep_v1.pop("schema_version", None)
    # Tolerant passes
    validate_tolerant(rep_v1)
    # Strict fails on missing
    try:
        validate_strict(rep_v1)
        assert False, "strict validator should fail when schema_version missing"
    except Exception:
        pass

    # Unknown fields tolerated in both
    rep_extra = dict(rep)
    rep_extra["extra"] = {"k": "v"}
    validate_tolerant(rep_extra)
    validate_strict(rep_extra)


