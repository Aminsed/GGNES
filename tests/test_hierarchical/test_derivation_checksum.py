from ggnes.core.graph import Graph
from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.derivation import DerivationEngine


def test_derivation_explain_contains_determinism_checksum():
    g = Graph()
    spec = ModuleSpec(
        name="Chk",
        version=1,
        parameters=[ParameterSpec("model_dim", default=8)],
        ports=[PortSpec("in", 8), PortSpec("out", 8)],
        invariants=["out.size == model_dim"],
    )
    root = DerivationEngine(g).expand(spec, {"model_dim": 8})
    info = DerivationEngine(g).explain(root)
    assert "checksum" in info and isinstance(info["checksum"], str) and len(info["checksum"]) == 16


