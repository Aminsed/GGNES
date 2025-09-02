from __future__ import annotations

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def test_graph_bias_and_attributes_float_precision_changes_uuid_when_precision_changes():
    # Same inputs different precision → different UUIDs
    g1 = Graph(config={'id_strategy': 'HYBRID', 'deterministic_uuids': True, 'uuid_float_precision': 4})
    g2 = Graph(config={'id_strategy': 'HYBRID', 'deterministic_uuids': True, 'uuid_float_precision': 12})
    n1 = g1.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'bias': 0.1234567, 'attributes': {'output_size': 8}})
    n2 = g2.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'bias': 0.1234567, 'attributes': {'output_size': 8}})
    assert g1.nodes[n1].global_id != g2.nodes[n2].global_id


