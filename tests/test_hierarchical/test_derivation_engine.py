import pytest

from ggnes.core.graph import Graph
from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec, DerivationEngine
from ggnes.utils.validation import ValidationError


def _simple_spec(name: str, version: int) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        version=version,
        parameters=[
            ParameterSpec(name="model_dim", required=True, domain=lambda v: isinstance(v, int) and v >= 1),
            ParameterSpec(name="num_heads", default=1, domain=lambda v: isinstance(v, int) and v >= 1),
            ParameterSpec(name="head_dim", default="=model_dim // num_heads", domain=lambda v: isinstance(v, int) and v >= 1),
        ],
        ports=[
            PortSpec(name="in", size=8, dtype="float32", is_stateful=False),
            PortSpec(name="out", size=8, dtype="float32", is_stateful=False),
        ],
        invariants=["num_heads * head_dim == model_dim"],
    )


def test_derivation_root_only_deterministic_uuid_and_graph_side_effects():
    graph = Graph(config={
        "uuid_namespace": "ggnes://uuid/v1",
        "uuid_scheme_version": 1,
        "uuid_float_precision": 8,
    })
    spec = _simple_spec("Block", 1)

    engine = DerivationEngine(graph)
    node = engine.expand(spec, {"model_dim": 8})

    # UUID present and stable under repeat expansion (fresh engine/graph copy preserves determinism)
    graph2 = Graph(config=graph.config)
    node2 = DerivationEngine(graph2).expand(spec, {"model_dim": 8})
    assert node.uuid == node2.uuid

    # Graph received exactly one node with derivation metadata
    assert len(graph.nodes) == 1
    only_node = next(iter(graph.nodes.values()))
    assert only_node.attributes.get("module_name") == "Block"
    assert only_node.attributes.get("module_version") == 1
    assert only_node.attributes.get("derivation_uuid") == node.uuid


def test_derivation_children_canonical_order_and_uuids_include_path_and_signature():
    graph = Graph()
    parent = _simple_spec("Parent", 1)
    childA = _simple_spec("Child", 2)
    childB = _simple_spec("Child", 1)

    # Prepare children in non-canonical order to ensure sorting by (name, version, signature)
    children = [
        (childA, {"model_dim": 8}, []),
        (childB, {"model_dim": 8}, []),
    ]
    engine = DerivationEngine(graph)
    root = engine.expand(parent, {"model_dim": 8}, children=children)

    # Two children created with deterministic order: Child v1 then Child v2
    assert len(root.children) == 2
    assert root.children[0].module.version == 1
    assert root.children[1].module.version == 2

    # UUIDs differ due to path and/or signature
    assert root.children[0].uuid != root.children[1].uuid


def test_derivation_limits_depth_and_expansions_raise_structured_errors():
    graph = Graph()
    spec = _simple_spec("Deep", 1)

    # depth limit
    engine = DerivationEngine(graph, config={"max_derivation_depth": 0})
    with pytest.raises(ValidationError) as ei:
        engine.expand(spec, {"model_dim": 8}, children=[(spec, {"model_dim": 8}, [])])
    assert ei.value.error_type == "hierarchy_limit"
    assert ei.value.details.get("reason") == "max_depth_exceeded"

    # expansions limit (root counts as 1 after commit)
    graph2 = Graph()
    engine2 = DerivationEngine(graph2, config={"max_derivation_expansions": 0})
    with pytest.raises(ValidationError) as ei2:
        engine2.expand(spec, {"model_dim": 8})
    assert ei2.value.error_type == "hierarchy_limit"
    assert ei2.value.details.get("reason") == "max_expansions_exceeded"


def test_nested_transaction_rollback_on_error_does_not_mutate_graph():
    graph = Graph()
    parent = _simple_spec("Parent", 1)

    # Create a child that violates invariant (model_dim not divisible by num_heads)
    bad_child = _simple_spec("ChildBad", 1)
    # Overrides cause head_dim expr to fail invariant (7 // 2 = 3 -> 2*3 != 7)
    children = [(bad_child, {"model_dim": 7, "num_heads": 2}, [])]

    engine = DerivationEngine(graph)
    with pytest.raises(ValidationError):
        engine.expand(parent, {"model_dim": 8}, children=children)

    # No nodes should have been committed due to rollback
    assert len(graph.nodes) == 0


def test_whole_tree_determinism_same_seed_and_config():
    graph1 = Graph()
    graph2 = Graph()
    parent = _simple_spec("P", 1)
    child = _simple_spec("C", 1)
    children = [(child, {"model_dim": 8}, [])]

    e1 = DerivationEngine(graph1)
    e2 = DerivationEngine(graph2)
    t1 = e1.expand(parent, {"model_dim": 8}, children=children)
    t2 = e2.expand(parent, {"model_dim": 8}, children=children)

    def flatten(node):
        out = [(node.module.name, node.module.version, node.signature, node.uuid, node.path)]
        for ch in node.children:
            out.extend(flatten(ch))
        return out

    assert flatten(t1) == flatten(t2)


def test_tiebreak_stability_on_same_name_version_children_by_signature():
    graph = Graph()
    parent = _simple_spec("P", 1)
    # Two children share name/version but different signatures (model_dim 8 vs 16)
    child = _simple_spec("C", 1)
    children = [
        (child, {"model_dim": 16}, []),
        (child, {"model_dim": 8}, []),
    ]
    root = DerivationEngine(graph).expand(parent, {"model_dim": 8}, children=children)
    # Canonical sort key includes signature (stable string), so order is deterministic
    sigs = [c.signature for c in root.children]
    assert sigs == sorted(sigs)


def test_uuid_config_sensitivity_affects_derivation_uuid():
    # Different namespaces produce distinct UUIDs deterministically
    g1 = Graph(config={"uuid_namespace": "ggnes://uuid/v1"})
    g2 = Graph(config={"uuid_namespace": "ggnes://uuid/v2"})
    spec = _simple_spec("X", 1)
    n1 = DerivationEngine(g1).expand(spec, {"model_dim": 8})
    n2 = DerivationEngine(g2).expand(spec, {"model_dim": 8})
    assert n1.uuid != n2.uuid


def test_rng_rollback_restored_on_nested_failure():
    # Force a failure in child so that parent transaction rolls back RNG and no nodes persist
    graph = Graph()
    parent = _simple_spec("Par", 1)
    bad_child = _simple_spec("Bad", 1)
    children = [(bad_child, {"model_dim": 7, "num_heads": 2}, [])]
    eng = DerivationEngine(graph)
    with pytest.raises(ValidationError):
        eng.expand(parent, {"model_dim": 8}, children=children)
    # No nodes and RNG still usable: do a succeeding expansion
    ok = eng.expand(parent, {"model_dim": 8})
    assert len(graph.nodes) == 1
    assert ok.uuid


def test_derivation_time_budget_enforced_with_structured_error():
    graph = Graph()
    spec = _simple_spec("Timed", 1)
    # Configure a zero time budget to force timeout error
    eng = DerivationEngine(graph, config={"derivation_time_budget_ms": 0})
    with pytest.raises(ValidationError) as ei:
        eng.expand(spec, {"model_dim": 8})
    assert ei.value.error_type in {"derivation_timeout", "hierarchy_limit"}


def test_max_children_guard_raises_structured_error():
    graph = Graph()
    parent = _simple_spec("Guard", 1)
    child = _simple_spec("Kid", 1)
    # Two children but allow only max 1
    children = [
        (child, {"model_dim": 8}, []),
        (child, {"model_dim": 16}, []),
    ]
    eng = DerivationEngine(graph, config={"max_children_per_node": 1})
    with pytest.raises(ValidationError) as ei:
        eng.expand(parent, {"model_dim": 8}, children=children)
    assert ei.value.error_type == "hierarchy_limit"
    assert ei.value.details.get("reason") == "max_children_exceeded"


def test_derivation_explain_returns_canonical_summary():
    graph = Graph()
    spec = _simple_spec("Explain", 1)
    engine = DerivationEngine(graph)
    root = engine.expand(spec, {"model_dim": 8})
    info = engine.explain(root)
    assert info["module"] == "Explain"
    assert isinstance(info["uuid"], str)
    assert isinstance(info["children"], list)


