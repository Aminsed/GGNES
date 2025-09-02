from ggnes.utils.observability import consolidated_report, assert_determinism_equivalence
from ggnes.hierarchical.derivation import DerivationEngine
from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
from ggnes.core.graph import Graph
from ggnes.utils.rng_manager import RNGManager


def test_co_33_02_determinism_gate_fails_on_checksum_drift():
    # Build a small derivation and consolidated report twice; checksums match
    g = Graph()
    spec = ModuleSpec(
        name="Gate",
        version=1,
        parameters=[ParameterSpec("model_dim", default=4)],
        ports=[PortSpec("in", 4), PortSpec("out", 4)],
        invariants=["out.size == model_dim"],
    )
    engine = DerivationEngine(g, config={"parallel_hg_execution": False})
    node = engine.expand(spec, {"model_dim": 4})
    rng = RNGManager(seed=5)
    rep1 = consolidated_report(engine, node, g, rng_manager=rng, device="cpu", dtype="float32")
    rep2 = consolidated_report(engine, node, g, rng_manager=RNGManager(seed=5), device="cpu", dtype="float32")
    # Gate: reports equivalent across identical conditions
    assert_determinism_equivalence([rep1, rep2])

    # Introduce deterministic change (graph structure) → WL fingerprint changes ⇒ checksum should drift
    g2 = Graph()
    from ggnes.core.node import NodeType
    # Create a simple Input->Output graph to guarantee WL fingerprint difference
    input_id = g2.add_node({"node_type": NodeType.INPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    output_id = g2.add_node({"node_type": NodeType.OUTPUT, "activation_function": "linear", "attributes": {"output_size": 1}})
    _ = g2.add_edge(input_id, output_id, {"weight": 0.1})
    rep_changed = consolidated_report(engine, node, g2, rng_manager=rng, device="cpu", dtype="float32")
    assert rep_changed["wl_fingerprint"] != rep1["wl_fingerprint"]
    assert rep_changed["determinism_checksum"] != rep1["determinism_checksum"]


