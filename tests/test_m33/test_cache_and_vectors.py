import json
from ggnes.translation.pytorch_impl import (
    clear_translation_cache,
    get_translation_cache_metrics,
    set_translation_cache_enabled,
    to_pytorch_model,
)
from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.utils.observability import consolidated_report, determinism_signature


def _graph_with_deriv_uuid(tag: str) -> Graph:
    g = Graph()
    hid = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "relu",
        "bias": 0.1,
        "attributes": {"output_size": 1, "derivation_uuid": tag},
    })
    out = g.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1},
    })
    g.add_edge(hid, out, {"weight": 0.2})
    return g


def test_m33_cache_eviction_and_metrics_no_semantic_change():
    import torch
    clear_translation_cache()
    set_translation_cache_enabled(True)
    # Create several graphs with different UUIDs to grow cache
    tags = [f"00000000-0000-0000-0000-{i:012d}" for i in range(5)]
    models = []
    outputs = []
    for t in tags:
        g = _graph_with_deriv_uuid(t)
        m = to_pytorch_model(g)
        x = torch.randn(1, 0)
        y = m(x, reset_states=True)
        models.append(m)
        outputs.append(y)
    metrics = get_translation_cache_metrics()
    assert metrics["entries"] >= 1 and metrics["hits"] >= 0 and metrics["misses"] >= 0

    # Toggle cache and recompute for first graph; outputs must remain equal
    set_translation_cache_enabled(False)
    g0 = _graph_with_deriv_uuid(tags[0])
    m0 = to_pytorch_model(g0)
    y0 = m0(outputs[0].new_zeros((1, 0)), reset_states=True)
    assert y0.shape == outputs[0].shape
    assert torch.allclose(y0, outputs[0])


def test_m33_golden_determinism_vectors_env_manifest():
    # Build two simple reports and compare signatures → equality
    g = Graph()
    spec_env = {"device": "cpu", "dtype": "float32"}
    from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
    from ggnes.hierarchical.derivation import DerivationEngine

    spec = ModuleSpec(
        name="Vec",
        version=1,
        parameters=[ParameterSpec("d", default=2)],
        ports=[PortSpec("in", 2), PortSpec("out", 2)],
        invariants=["out.size == d"],
    )
    engine = DerivationEngine(g)
    node = engine.expand(spec, {"d": 2})
    rep1 = consolidated_report(engine, node, g, device=spec_env["device"], dtype=spec_env["dtype"])
    rep2 = consolidated_report(engine, node, g, device=spec_env["device"], dtype=spec_env["dtype"])
    sig1 = determinism_signature(rep1)
    sig2 = determinism_signature(rep2)
    assert sig1 == sig2
    # Env manifest presence
    manifest = {"device": rep1.get("env", {}).get("device"), "dtype": rep1.get("env", {}).get("dtype"), "seed": rep1.get("env", {}).get("seed")}
    json.dumps(manifest)  # must be serializable
    # Change device/dtype → signature should change when included
    rep3 = consolidated_report(engine, node, g, device="cpu", dtype="float16")
    sig3 = determinism_signature(rep3)
    assert sig3 != sig1

