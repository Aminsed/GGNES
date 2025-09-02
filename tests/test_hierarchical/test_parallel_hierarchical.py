from __future__ import annotations

import pytest

from ggnes.core.graph import Graph
from ggnes.hierarchical.derivation import DerivationEngine
from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
from ggnes.utils.rng_manager import RNGManager


def _simple_spec(name: str, version: int = 1) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        version=version,
        parameters=[ParameterSpec(name="p", default=1)],
        ports=[PortSpec(name="in", size=4), PortSpec(name="out", size=2)],
        attributes={"alpha": 0.1},
        invariants=["in.size >= out.size or out.size > 0"],
    )


def _build_children(n: int) -> list:
    specs = []
    for i in range(n):
        specs.append((_simple_spec(f"child{i}"), {"p": i}, []))
    return specs


def _fingerprint_tree(node):
    # Hashable tuple to compare structure deterministically
    return (
        node.module.name,
        int(node.module.version),
        node.signature,
        node.uuid,
        tuple(_fingerprint_tree(c) for c in node.children),
    )


def test_hg_parallel_vs_serial_parity_fixed_size():
    rng = RNGManager(seed=123)
    g1 = Graph()
    root = _simple_spec("root")
    children = _build_children(7)

    # Serial
    e_serial = DerivationEngine(g1, config={"derivation_time_budget_ms": 10_000}, rng_manager=rng)
    t1 = e_serial.expand(root, children=children)
    fp1 = _fingerprint_tree(t1)

    # Parallel semantics (deterministic batches, sequential execution)
    g2 = Graph()
    e_par = DerivationEngine(
        g2,
        config={
            "parallel_hg_execution": True,
            "hg_max_parallel_workers": 3,
            "hg_parallel_batch_policy": "FIXED_SIZE",
            "hg_fixed_batch_size": 3,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=RNGManager(seed=123),
    )
    t2 = e_par.expand(root, children=children)
    fp2 = _fingerprint_tree(t2)

    # Parity of structure, UUIDs, signatures
    assert fp1 == fp2

    # Metrics surfaced with deterministic batch checksums
    m = e_par.last_expand_metrics
    assert m["batches_processed"] == 3  # 7 items in batches of 3 -> 3 batches
    assert m["batch_policy"].upper() == "FIXED_SIZE"
    assert len(m["batches"]) == 3
    for b in m["batches"]:
        assert 1 <= b["worker_count"] <= 3
        assert isinstance(b["checksum"], str) and len(b["checksum"]) == 16


def test_hg_parallel_single_batch_max_independent_set():
    rng = RNGManager(seed=999)
    g = Graph()
    root = _simple_spec("root")
    children = _build_children(4)
    e = DerivationEngine(
        g,
        config={
            "parallel_hg_execution": True,
            "hg_max_parallel_workers": 8,
            "hg_parallel_batch_policy": "MAX_INDEPENDENT_SET",
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=rng,
    )
    t = e.expand(root, children=children)
    # One batch expected
    m = e.last_expand_metrics
    assert m["batches_processed"] == 1
    assert len(m["batches"]) == 1
    assert m["batches"][0]["size"] == 4
    # explain checksum present and stable length
    from ggnes.hierarchical.derivation import DerivationEngine as _DE
    ex = e.explain(t)
    assert isinstance(ex["checksum"], str) and len(ex["checksum"]) == 16


def test_hg_time_budget_enforced_pre_batch():
    rng = RNGManager(seed=42)
    g = Graph()
    root = _simple_spec("root")
    # Many children to hit budget fast; tiny budget
    children = _build_children(100)
    e = DerivationEngine(
        g,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_batch_policy": "FIXED_SIZE",
            "hg_fixed_batch_size": 10,
            "derivation_time_budget_ms": 0,
        },
        rng_manager=rng,
    )
    with pytest.raises(Exception) as ei:
        _ = e.expand(root, children=children)
    msg = str(ei.value)
    assert "Derivation time budget exceeded" in msg


def test_hg_priority_cap_policy_and_mis_metrics():
    rng = RNGManager(seed=7)
    g = Graph()
    root = _simple_spec("root")
    children = _build_children(9)
    e = DerivationEngine(
        g,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_batch_policy": "PRIORITY_CAP",
            "hg_priority_cap": 4,
            "parallel_memory_budget_mb": 0,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=rng,
    )
    _ = e.expand(root, children=children)
    m = e.last_expand_metrics
    # Expect ceil(9/4)=3 batches; mis_size equals batch size, worker_count bounded by cap
    assert m["batches_processed"] == 3
    sizes = [b["size"] for b in m["batches"]]
    assert sizes[0] == 4 and sizes[1] == 4 and sizes[2] == 1
    for b in m["batches"]:
        assert b["mis_size"] == b["size"]
        assert 1 <= b["worker_count"] <= 4


def test_hg_memory_budget_backoff_splits_batches_and_records_metrics():
    rng = RNGManager(seed=11)
    g = Graph()
    root = _simple_spec("root")
    children = _build_children(6)
    e = DerivationEngine(
        g,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_batch_policy": "MAX_INDEPENDENT_SET",
            "parallel_memory_budget_mb": 2,  # cost per child defaults to 1 -> batches of 2
            "hg_child_mem_cost_mb": 1,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=rng,
    )
    _ = e.expand(root, children=children)
    m = e.last_expand_metrics
    # Expect batches of size 2 -> 3 batches; memory_backoff_count > 0
    assert m["batches_processed"] == 3
    assert any(b["size"] == 2 for b in m["batches"])
    assert m.get("memory_backoff_count", 0) >= 2


def test_hg_conflict_strategy_skip_requeue_speculative_merge_metrics():
    # Build children with 2 failing invariants and 2 passing
    def failing_spec(idx: int) -> ModuleSpec:
        return ModuleSpec(
            name=f"fail{idx}",
            version=1,
            parameters=[ParameterSpec(name="p", default=1)],
            ports=[],
            attributes={},
            invariants=["1 == 0"],  # always fails
        )

    ok = _simple_spec("ok")
    children = [
        (failing_spec(0), {}, []),
        (ok, {}, []),
        (failing_spec(1), {}, []),
        (ok, {}, []),
    ]

    # SKIP: conflicts counted, no requeues, rollbacks zero
    g1 = Graph()
    e1 = DerivationEngine(
        g1,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_batch_policy": "MAX_INDEPENDENT_SET",
            "hg_parallel_conflict_strategy": "SKIP",
            "hg_parallel_max_requeues": 2,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=RNGManager(seed=1),
    )
    t1 = e1.expand(_simple_spec("root"), children=children)
    m1 = e1.last_expand_metrics
    # Two failing
    assert int(m1.get("conflicts", 0)) >= 2
    assert int(m1.get("requeues", 0)) == 0
    assert int(m1.get("rollbacks", 0)) == 0
    # Only two passing remain as children at root
    assert len(t1.children) == 2

    # REQUEUE: conflicts + requeues increase, still drop after retries
    g2 = Graph()
    e2 = DerivationEngine(
        g2,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_conflict_strategy": "REQUEUE",
            "hg_parallel_max_requeues": 2,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=RNGManager(seed=1),
    )
    t2 = e2.expand(_simple_spec("root"), children=children)
    m2 = e2.last_expand_metrics
    assert int(m2.get("conflicts", 0)) >= 2
    # Each failing child requeued up to 2 times deterministically
    assert int(m2.get("requeues", 0)) >= 2
    assert len(t2.children) == 2

    # SPECULATIVE_MERGE: conflicts + rollbacks increase, drop failures
    g3 = Graph()
    e3 = DerivationEngine(
        g3,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_conflict_strategy": "SPECULATIVE_MERGE",
            "hg_parallel_max_requeues": 1,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=RNGManager(seed=1),
    )
    t3 = e3.expand(_simple_spec("root"), children=children)
    m3 = e3.last_expand_metrics
    assert int(m3.get("conflicts", 0)) >= 2
    assert int(m3.get("rollbacks", 0)) >= 2
    assert len(t3.children) == 2


def test_hg_parity_property_across_seeds_and_workers():
    seeds = [1, 2, 3]
    workers = [1, 2, 4, 8]
    root = _simple_spec("root")
    children = _build_children(6)
    for s in seeds:
        # Serial baseline
        gx = Graph()
        ts = DerivationEngine(gx, config={"derivation_time_budget_ms": 10_000}, rng_manager=RNGManager(seed=s)).expand(root, children=children)
        fps = _fingerprint_tree(ts)
        for w in workers:
            gp = Graph()
            tp = DerivationEngine(
                gp,
                config={
                    "parallel_hg_execution": True,
                    "hg_max_parallel_workers": w,
                    "hg_parallel_batch_policy": "FIXED_SIZE",
                    "hg_fixed_batch_size": 3,
                    "derivation_time_budget_ms": 10_000,
                },
                rng_manager=RNGManager(seed=s),
            ).expand(root, children=children)
            fpp = _fingerprint_tree(tp)
            assert fpp == fps


def test_hg_translation_parity_serial_vs_parallel():
    # Build minimal graph with IO to ensure translation runs
    def build_base_graph() -> Graph:
        g = Graph()
        from ggnes.core.node import NodeType
        in_id = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 8}})
        out_id = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
        _ = g.add_edge(in_id, out_id, {'weight': 0.1})
        return g

    root = _simple_spec("root")
    children = _build_children(5)

    gs = build_base_graph()
    ts = DerivationEngine(gs, config={"derivation_time_budget_ms": 10_000}, rng_manager=RNGManager(seed=5)).expand(root, children=children)

    gp = build_base_graph()
    tp = DerivationEngine(
        gp,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_batch_policy": "FIXED_SIZE",
            "hg_fixed_batch_size": 2,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=RNGManager(seed=5),
    ).expand(root, children=children)

    # Translate both and compare outputs
    import torch
    from ggnes.translation.pytorch import to_pytorch_model

    xs = torch.randn(4, 8)
    # Seed torch RNG to make Linear initializations deterministic across models
    torch.manual_seed(0)
    ys = to_pytorch_model(gs) (xs, reset_states=True)
    torch.manual_seed(0)
    yp = to_pytorch_model(gp) (xs, reset_states=True)
    assert torch.allclose(ys, yp)


def test_hg_derivation_uuid_parity_serial_vs_parallel():
    root = _simple_spec("root")
    children = _build_children(5)

    g1 = Graph()
    t1 = DerivationEngine(g1, config={"derivation_time_budget_ms": 10_000}, rng_manager=RNGManager(seed=9)).expand(root, children=children)

    g2 = Graph()
    t2 = DerivationEngine(
        g2,
        config={
            "parallel_hg_execution": True,
            "hg_parallel_batch_policy": "PRIORITY_CAP",
            "hg_priority_cap": 3,
            "derivation_time_budget_ms": 10_000,
        },
        rng_manager=RNGManager(seed=9),
    ).expand(root, children=children)

    def collect_derivation_uuids(graph: Graph):
        uuids = set()
        for node in graph.nodes.values():
            du = node.attributes.get('derivation_uuid') if hasattr(node, 'attributes') else None
            if du is not None:
                uuids.add(str(du))
        return uuids

    uuids1 = collect_derivation_uuids(g1)
    uuids2 = collect_derivation_uuids(g2)
    assert uuids1 == uuids2


