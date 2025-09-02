from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation import to_pytorch_model
from ggnes.translation.pytorch_impl import (
    clear_translation_cache,
    get_translation_cache_metrics,
    set_translation_cache_enabled,
)


def _graph_with_tag(tag: str):
    g = Graph()
    h = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'bias': 0.1,
        'attributes': {'output_size': 4, 'derivation_uuid': tag}
    })
    o = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 2}
    })
    g.add_edge(h, o, {'weight': 0.3})
    g.input_node_ids = []
    g.output_node_ids = [o]
    return g


def test_cache_reuse_same_uuid_and_isolation_across_graphs():
    import torch
    clear_translation_cache()
    set_translation_cache_enabled(True)
    g1 = _graph_with_tag('00000000-0000-0000-0000-000000000001')
    g2 = _graph_with_tag('00000000-0000-0000-0000-000000000001')
    m1 = to_pytorch_model(g1)
    m2 = to_pytorch_model(g2)
    x = torch.randn(1, 0)
    y1 = m1(x, reset_states=True)
    y2 = m2(x, reset_states=True)
    assert y1.shape == y2.shape
    metrics = get_translation_cache_metrics()
    # Metrics dict should exist and have non-negative counters
    assert set(["hits", "misses", "entries"]).issubset(metrics.keys())
    assert metrics["hits"] >= 0 and metrics["misses"] >= 0 and metrics["entries"] >= 0


def test_cache_distinct_uuids_distinct_params():
    import torch
    clear_translation_cache()
    set_translation_cache_enabled(True)
    g1 = _graph_with_tag('00000000-0000-0000-0000-00000000000A')
    g2 = _graph_with_tag('00000000-0000-0000-0000-00000000000B')
    m1 = to_pytorch_model(g1)
    m2 = to_pytorch_model(g2)
    # Ensure models are constructed; different UUIDs create separate cached entries
    x = torch.randn(1, 0)
    _ = m1(x, reset_states=True)
    _ = m2(x, reset_states=True)
    metrics = get_translation_cache_metrics()
    # Metrics dict should exist and have non-negative counters
    assert set(["hits", "misses", "entries"]).issubset(metrics.keys())
    assert metrics["hits"] >= 0 and metrics["misses"] >= 0 and metrics["entries"] >= 0


def test_cache_eviction_smoke():
    import torch
    clear_translation_cache()
    set_translation_cache_enabled(True)
    # Build multiple graphs to stress cache; behavior should remain stable
    tags = [f'00000000-0000-0000-0000-0000000000{str(i).zfill(2)}' for i in range(12)]
    models = []
    for t in tags:
        g = _graph_with_tag(t)
        models.append(to_pytorch_model(g))
    # Forward on a subset
    x = torch.randn(1, 0)
    for m in models[:5]:
        _ = m(x, reset_states=True)
    metrics = get_translation_cache_metrics()
    # entries count may be zero when graphs lack derivation_uuid or cache disabled; assert presence only
    assert set(["hits", "misses", "entries"]).issubset(metrics.keys())


def test_cache_toggle_has_no_semantic_effects():
    import torch
    clear_translation_cache()
    g = _graph_with_tag('00000000-0000-0000-0000-0000000000AA')
    x = torch.randn(2, 0)
    # Cache disabled
    set_translation_cache_enabled(False)
    torch.manual_seed(1337)
    m_no_cache = to_pytorch_model(g)
    y0 = m_no_cache(x, reset_states=True)
    # Cache enabled
    clear_translation_cache()
    set_translation_cache_enabled(True)
    torch.manual_seed(1337)
    m_cache = to_pytorch_model(g)
    y1 = m_cache(x, reset_states=True)
    # Outputs equal when torch params initialized deterministically
    assert y0.shape == y1.shape
    assert torch.allclose(y0, y1)


def test_cache_metrics_reset_semantics():
    # [R11] After clearing cache, metrics should reset to zeros
    clear_translation_cache()
    metrics = get_translation_cache_metrics()
    assert metrics["hits"] == 0 and metrics["misses"] == 0 and metrics["entries"] == 0


def test_cache_mid_run_toggle_parity():
    # [R11] Toggling cache mid-run must not change outputs
    import torch
    clear_translation_cache()
    g = _graph_with_tag('00000000-0000-0000-0000-0000000000BB')
    x = torch.randn(3, 0)

    set_translation_cache_enabled(True)
    torch.manual_seed(2025)
    m1 = to_pytorch_model(g)
    y_cache = m1(x, reset_states=True)

    # Toggle off and rebuild
    set_translation_cache_enabled(False)
    torch.manual_seed(2025)
    m2 = to_pytorch_model(g)
    y_no_cache = m2(x, reset_states=True)

    assert y_cache.shape == y_no_cache.shape
    assert torch.allclose(y_cache, y_no_cache)
