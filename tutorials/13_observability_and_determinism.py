"""
Observability & Determinism Tutorial

Goals:
- Build consolidated report from a trivial derivation
- Compute determinism signature and show stability
"""

from ggnes.core.graph import Graph
from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.derivation import DerivationEngine
from ggnes.utils.observability import consolidated_report, determinism_signature


def main():
    g = Graph()
    spec = ModuleSpec(
        name="Unit",
        version=1,
        parameters=[ParameterSpec("dim", default=1)],
        ports=[PortSpec("in", 1), PortSpec("out", 1)],
        invariants=["out.size == dim"],
    )
    engine = DerivationEngine(g, {"parallel_hg_execution": False})
    root = engine.expand(spec, {"dim": 1})
    rep = consolidated_report(engine, root, g, device='cpu', dtype='float32')
    sig = determinism_signature(rep)
    print('schema_version:', rep.get('schema_version'))
    print('determinism_sig:', sig)


if __name__ == '__main__':
    main()


