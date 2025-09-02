from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.utils.uuid_provider import DeterministicUUIDProvider, UUIDProviderConfig


def test_strict_mode_rejects_non_finite():
    cfg = UUIDProviderConfig(strict=True)
    prov = DeterministicUUIDProvider(cfg)
    # Place non-finite in inputs – should raise
    try:
        prov.derive_uuid('node', {'x': float('inf')})
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_scheme_metadata_contains_contract_fields():
    prov = DeterministicUUIDProvider(UUIDProviderConfig(namespace='ns', scheme_version=2, float_precision=8, strict=True, freeze=True))
    meta = prov.scheme_metadata()
    assert meta['namespace'] == 'ns'
    assert meta['scheme_version'] == 2
    assert meta['float_precision'] == 8
    assert meta['strict'] is True
    assert meta['freeze'] is True


def test_graph_config_defaults_do_not_break():
    # deterministic_uuids defaults to False; graph should still construct and use random UUIDs
    g = Graph()
    n = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})
    assert isinstance(g.nodes[n].global_id, uuid.UUID)


