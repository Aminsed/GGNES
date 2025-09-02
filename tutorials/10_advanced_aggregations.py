"""
Advanced Aggregations Tutorial

Goals:
- Configure nodes to use attention/multi_head_attention/gated_sum/MoE/attn_pool
- Translate and run a forward pass to verify shapes
"""

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
import torch


def build_graph_with_attention(kind: str) -> Graph:
    g = Graph()
    i1 = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 4},
    })
    i2 = g.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 4},
    })
    h = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'linear',
        'attributes': {'output_size': 4, 'aggregation': kind, 'num_heads': 2, 'head_dim': 4, 'temperature': 1.0},
    })
    o = g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 4},
    })
    g.add_edge(i1, h, {'weight': 0.5})
    g.add_edge(i2, h, {'weight': 0.25})
    g.add_edge(h, o, {'weight': 0.2})
    return g


def main():
    for kind in ["attention", "multi_head_attention", "gated_sum", "moe", "attn_pool"]:
        g = build_graph_with_attention(kind)
        model = to_pytorch_model(g, {'device': 'cpu', 'dtype': torch.float32})
        x = torch.randn(2, 8)
        y = model(x, reset_states=True)
        print(f"kind={kind} output_shape={tuple(y.shape)}")


if __name__ == '__main__':
    main()


