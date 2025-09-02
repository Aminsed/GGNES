"""
Cache & Performance Safety Tutorial (M33)

Goals:
- Inspect translation cache metrics
- Toggle cache and assert semantic parity (output equality) on a small graph
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
from ggnes.translation.pytorch_impl import (
    clear_translation_cache,
    get_translation_cache_metrics,
    set_translation_cache_enabled,
)
import torch


def build_graph():
    g = Graph()
    # Use matching dimensions to avoid randomized Linear projections
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 3}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 3}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 3}})
    g.add_edge(i, h, {'weight': 0.2})
    g.add_edge(h, o, {'weight': 0.1})
    return g


def main():
    clear_translation_cache()
    g = build_graph()
    x = torch.randn(2, 3)

    set_translation_cache_enabled(True)
    m_cache = to_pytorch_model(g, {'device': 'cpu', 'dtype': torch.float32, 'enable_translation_cache': True})
    y_cache = m_cache(x, reset_states=True).detach().clone()
    metrics_after = get_translation_cache_metrics()
    print('cache_metrics_after_build:', metrics_after)

    set_translation_cache_enabled(False)
    m_nocache = to_pytorch_model(g, {'device': 'cpu', 'dtype': torch.float32, 'enable_translation_cache': False})
    y_nocache = m_nocache(x, reset_states=True).detach().clone()

    # Semantic parity assertion
    equal = torch.allclose(y_cache, y_nocache, atol=1e-6)
    print('semantic_parity:', bool(equal))


if __name__ == '__main__':
    main()


