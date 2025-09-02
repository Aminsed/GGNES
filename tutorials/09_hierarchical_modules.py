"""
Hierarchical Modules Tutorial

Goals:
- Define a ModuleSpec with parameters, ports, and invariants
- Expand with DerivationEngine and obtain an explain checksum
"""

from ggnes.core.graph import Graph
from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.derivation import DerivationEngine


def main():
    g = Graph()
    spec = ModuleSpec(
        name="Block",
        version=1,
        parameters=[ParameterSpec("dim", default=8)],
        ports=[PortSpec("in", 8), PortSpec("out", 8)],
        invariants=["out.size == dim"],
    )

    engine = DerivationEngine(g)
    root = engine.expand(spec, {"dim": 8})
    info = engine.explain(root)
    print("derivation_checksum:", info["checksum"]) 


if __name__ == "__main__":
    main()


