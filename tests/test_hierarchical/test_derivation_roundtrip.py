import json

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec
from ggnes.hierarchical.derivation import DerivationEngine


def _build_graph():
    g = Graph(config={
        'deterministic_uuids': True,
        'uuid_namespace': 'ggnes://uuid/v1',
        'uuid_scheme_version': 1,
    })
    # Minimal IO to stabilize WL fingerprint and reachability
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 1}})
    g.add_edge(i, o, {'weight': 0.1})
    g.input_node_ids = [i]
    g.output_node_ids = [o]
    return g


def _serialize_tree(explained):
    return json.dumps(explained, sort_keys=True)


def _deserialize_tree(s):
    return json.loads(s)


def test_derivation_explain_roundtrip_parity():
    g = _build_graph()
    engine = DerivationEngine(g, config={'max_children_per_node': 4})

    # Build module hierarchy
    root = ModuleSpec(name='Root', version=1, parameters=[ParameterSpec('d', default=1)])
    child_a = ModuleSpec(name='A', version=1, parameters=[ParameterSpec('x', default=2)])
    child_b = ModuleSpec(name='B', version=1, parameters=[ParameterSpec('y', default=3)])

    tree = engine.expand(root, children=[(child_a, {}, []), (child_b, {}, [])])
    exp1 = engine.explain(tree)
    s = _serialize_tree(exp1)
    roundtrip = _deserialize_tree(s)

    # Rebuild with same specs and order
    g2 = _build_graph()
    engine2 = DerivationEngine(g2, config={'max_children_per_node': 4})
    tree2 = engine2.expand(root, children=[(child_a, {}, []), (child_b, {}, [])])
    exp2 = engine2.explain(tree2)

    # Parity: UUIDs, signatures, and WL fingerprint (graph compute_fingerprint)
    assert roundtrip['uuid'] == exp2['uuid']
    assert roundtrip['signature'] == exp2['signature']
    assert g.compute_fingerprint() == g2.compute_fingerprint()

