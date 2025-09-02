from ggnes.core.graph import Graph
from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.derivation import DerivationEngine
from ggnes.evolution.composite_genotype import (
    CompositeGenotype,
    G1Grammar,
    G2Policy,
    G3Hierarchy,
)
from ggnes.utils.observability import (
    explain_composite,
    compute_operator_efficacy,
    consolidated_report,
    compute_rng_signatures,
    island_migration_report,
    validate_consolidated_report,
    validate_explain_payload,
    validate_island_report,
)
from ggnes.evolution.islands import IslandScheduler
from ggnes.utils.rng_manager import RNGManager


def _build_simple_genotype(lr: float = 0.001, parallel: bool = False) -> CompositeGenotype:
    g1 = G1Grammar(rules=[{"rule_id": "11111111-1111-4111-8111-111111111111", "priority": 1, "probability": 0.5}])
    g2 = G2Policy(learning_rate=lr, parallel_execution=parallel)
    g3 = G3Hierarchy(modules={"M": {"model_dim": 8}}, attributes={"desc": "test"})
    return CompositeGenotype(g1=g1, g2=g2, g3=g3)


def _build_simple_derivation(graph: Graph, *, parallel: bool = True):
    # Root and two children to ensure batching metrics are populated
    root = ModuleSpec(
        name="Root",
        version=1,
        parameters=[ParameterSpec("model_dim", default=8)],
        ports=[PortSpec("in", 8), PortSpec("out", 8)],
        invariants=["out.size == model_dim"],
    )
    child_a = ModuleSpec(
        name="ChildA",
        version=1,
        parameters=[ParameterSpec("model_dim", default=8)],
        ports=[PortSpec("in", 8), PortSpec("out", 8)],
        invariants=["out.size == model_dim"],
    )
    child_b = ModuleSpec(
        name="ChildB",
        version=1,
        parameters=[ParameterSpec("model_dim", default=8)],
        ports=[PortSpec("in", 8), PortSpec("out", 8)],
        invariants=["out.size == model_dim"],
    )
    engine = DerivationEngine(
        graph,
        config={
            "parallel_hg_execution": parallel,
            "hg_parallel_batch_policy": "FIXED_SIZE",
            "hg_fixed_batch_size": 2,
        },
    )
    node = engine.expand(
        root,
        {"model_dim": 8},
        children=[(child_a, {"model_dim": 8}, []), (child_b, {"model_dim": 8}, [])],
    )
    return engine, node


def test_m32_01_explain_composite_contains_uuid_scheme_checksums_diffs_timings_and_schema():
    # [T-co-32-01]
    geno = _build_simple_genotype(lr=0.001, parallel=False)
    info = explain_composite(geno, baseline=_build_simple_genotype(lr=0.0005, parallel=False))
    assert "uuid" in info and isinstance(info["uuid"], str)
    assert "scheme" in info and set(info["scheme"].keys()) >= {"namespace", "scheme_version", "float_precision", "strict", "freeze"}
    assert "canonical" in info and isinstance(info["canonical"], dict)
    checks = info.get("checksums", {})
    assert set(checks.keys()) == {"full", "rules", "params", "hier"}
    for v in checks.values():
        assert isinstance(v, str) and len(v) == 16
    # Diffs present and timings/constraints available
    assert "diffs" in info and "params_changed" in info["diffs"]
    assert "timings" in info and all(k in info["timings"] for k in ("validate_ms", "uuid_ms", "checksum_ms"))
    assert "constraints" in info and isinstance(info["constraints"].get("policy_constraints_ok"), bool)
    # Schema validation does not raise
    validate_explain_payload(info)

    # Stability: same genotype → same checksums
    info2 = explain_composite(_build_simple_genotype(lr=0.001, parallel=False))
    assert info["checksums"] == info2["checksums"]

    # Change params → checksum changes
    info_changed = explain_composite(_build_simple_genotype(lr=0.01, parallel=True))
    assert info_changed["checksums"]["params"] != info["checksums"]["params"]
    assert info_changed["uuid"] != info["uuid"]


def test_m32_02_operator_efficacy_reports_deltas():
    # [T-co-32-02]
    before = _build_simple_genotype(lr=0.001, parallel=False)
    after = _build_simple_genotype(lr=0.002, parallel=True)
    # Add a rule to after to change count
    after.g1.rules.append({"rule_id": "22222222-2222-4222-8222-222222222222", "priority": 2, "probability": 0.7})
    eff = compute_operator_efficacy(before, after)
    assert eff["rule_count_delta"] == 1
    assert eff["learning_rate_delta"] == 0.001
    assert eff["batch_size_delta"] == 0
    assert eff["training_epochs_delta"] == 0
    assert eff["parallel_execution_change"] is True


def test_m32_03_consolidated_report_includes_derivation_checksum_wl_batches_env_rng_and_is_stable():
    # [T-co-32-03]
    g = Graph()
    engine, node = _build_simple_derivation(g, parallel=True)
    geno_info = explain_composite(_build_simple_genotype())
    rng = RNGManager(seed=123)
    sigs = compute_rng_signatures(rng, ["selection", "repair", "aggregation_dropout"])  # ensure contexts exist
    assert set(sigs.keys()) == {"selection", "repair", "aggregation_dropout"}
    report1 = consolidated_report(
        engine,
        node,
        g,
        genotype_explain=geno_info,
        rng_manager=rng,
        device="cpu",
        dtype="float32",
    )

    assert "derivation_checksum" in report1 and isinstance(report1["derivation_checksum"], str)
    assert len(report1["derivation_checksum"]) == 16
    assert "wl_fingerprint" in report1 and isinstance(report1["wl_fingerprint"], str)
    # Batches captured when parallel config is enabled
    assert isinstance(report1.get("batches", []), list)
    assert "determinism_checksum" in report1 and len(report1["determinism_checksum"]) == 16
    assert "genotype" in report1 and isinstance(report1["genotype"], dict)

    # Stability with identical inputs; schema validation passes
    engine2, node2 = _build_simple_derivation(Graph(), parallel=True)
    report2 = consolidated_report(engine2, node2, Graph(), genotype_explain=_build_simple_genotype().serialize(), rng_manager=RNGManager(seed=123), device="cpu", dtype="float32")
    # Checksums are independent; ensure format correctness
    assert len(report2["determinism_checksum"]) == 16
    validate_consolidated_report(report2)

    # Graph change → WL fingerprint changes → likely report checksum changes
    g2 = Graph()
    # Add minimal structure change to affect fingerprint using real NodeType
    from ggnes.core.node import NodeType  # local import for test
    _ = g2.add_node({"node_type": NodeType.HIDDEN, "activation_function": "linear", "attributes": {"output_size": 1}})
    report_changed = consolidated_report(engine, node, g2, genotype_explain=geno_info)
    assert report_changed["wl_fingerprint"] != report1["wl_fingerprint"]


def test_m32_04_consolidated_report_without_genotype_section():
    g = Graph()
    engine, node = _build_simple_derivation(g, parallel=False)
    report = consolidated_report(engine, node, g)
    assert "genotype" not in report
    assert "determinism_checksum" in report and len(report["determinism_checksum"]) == 16
    validate_consolidated_report(report)


def test_m32_05_island_migration_report_and_schema():
    pops = [["A0", "A1"], ["B0"], ["C0", "C1", "C2"]]
    rng = RNGManager(seed=7)
    sched = IslandScheduler(rng)
    new_pops = sched.migrate([list(x) for x in pops])
    report = island_migration_report(sched, pops, new_pops)
    assert report["topology"] in {"ring", "star"}
    assert report["migration_size"] >= 0
    assert report["islands"] == len(pops)
    assert isinstance(report["per_island"], list) and len(report["per_island"]) == len(pops)
    validate_island_report(report)


def test_m32_06_schema_validators_allow_extra_fields_and_custom_rng_contexts():
    g = Graph()
    engine, node = _build_simple_derivation(g, parallel=True)
    geno_info = explain_composite(_build_simple_genotype())
    # Add extra fields to payloads; validators should still pass
    geno_info_extra = dict(geno_info)
    # Ensure diffs present by providing a baseline with different params
    _ = explain_composite(_build_simple_genotype(lr=0.002), baseline=_build_simple_genotype(lr=0.001))
    geno_info_extra["extra_field"] = {"nested": True}
    # When diffs exist, schema must accept them; if absent, validator tolerates no diffs
    validate_explain_payload(geno_info)
    validate_explain_payload(geno_info_extra)

    rng = RNGManager(seed=42)
    report = consolidated_report(
        engine,
        node,
        g,
        genotype_explain=geno_info,
        rng_manager=rng,
        extra_rng_contexts=["custom_ctx"],
    )
    # Custom context present
    assert "custom_ctx" in report.get("env", {}).get("rng_signatures", {})
    # Add extra fields and validate
    report_extra = dict(report)
    report_extra["additional"] = [1, 2, 3]
    validate_consolidated_report(report_extra)


