from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def _make_graph(det_uuids: bool = True) -> Graph:
    g = Graph(config={
        'id_strategy': 'HYBRID',
        'deterministic_uuids': det_uuids,
        'uuid_scheme_version': 1,
        'uuid_namespace': 'ggnes://uuid/v1',
        'graph_provenance_uuid': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
    })
    return g


def test_graph_node_edge_uuids_deterministic():
    g1 = _make_graph(True)
    # Add nodes
    n_in = g1.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    n_h = g1.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    # Edge
    e = g1.add_edge(n_in, n_h, {'weight': 0.5})
    assert isinstance(e, uuid.UUID)
    node_gid = g1.nodes[n_h].global_id
    assert isinstance(node_gid, uuid.UUID)

    # Recreate identical graph → UUIDs equal
    g2 = _make_graph(True)
    n_in2 = g2.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    n_h2 = g2.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    e2 = g2.add_edge(n_in2, n_h2, {'weight': 0.5})
    assert g2.nodes[n_h2].global_id == node_gid
    assert e2 == e


def test_uuid_context_changes_ids():
    g = _make_graph(True)
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    g.set_uuid_context({'rule_id': 'r1', 'binding_signature': 'x=1'})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    g.set_uuid_context({'rule_id': 'r2', 'binding_signature': 'x=1'})
    c = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    assert g.nodes[a].global_id != g.nodes[b].global_id != g.nodes[c].global_id


def test_non_deterministic_mode_uses_random_uuids():
    g = _make_graph(False)
    n1 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    n2 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    assert g.nodes[n1].global_id != g.nodes[n2].global_id


