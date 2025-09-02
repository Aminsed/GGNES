from ggnes.core.graph import Graph
from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.derivation import DerivationEngine


def test_derivation_checksum_changes_with_children():
    g = Graph()
    parent = ModuleSpec(name="P", version=1, parameters=[ParameterSpec("d", default=8)], ports=[PortSpec("out", 8)], invariants=["out.size==d"])  # type: ignore[operator]
    child = ModuleSpec(name="C", version=1, parameters=[ParameterSpec("d", default=8)], ports=[PortSpec("out", 8)], invariants=["out.size==d"])  # type: ignore[operator]

    eng = DerivationEngine(g)
    root_no_child = eng.expand(parent, {"d": 8}, children=[])
    info_no_child = eng.explain(root_no_child)

    # Two-level nested children to exercise flatten recursion
    root_with_child = eng.expand(parent, {"d": 8}, children=[(child, {"d": 8}, [(child, {"d": 8}, [])])])
    info_with_child = eng.explain(root_with_child)

    assert info_no_child["checksum"] != info_with_child["checksum"]


