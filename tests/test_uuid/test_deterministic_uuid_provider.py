from __future__ import annotations

import uuid

from ggnes.utils.uuid_provider import (
    DeterministicUUIDProvider,
    UUIDProviderConfig,
)


def test_canonicalization_stability_dict_order_and_lists():
    cfg = UUIDProviderConfig()
    prov = DeterministicUUIDProvider(cfg)
    a = prov.derive_uuid('node', {'b': 2, 'a': 1, 'c': [3, 2, 1]})
    b = prov.derive_uuid('node', {'c': [3, 2, 1], 'a': 1, 'b': 2})
    assert a == b


def test_uuid_determinism_same_inputs_same_uuid():
    cfg = UUIDProviderConfig(namespace='ggnes://uuid/v1', scheme_version=1)
    prov = DeterministicUUIDProvider(cfg)
    inputs = {
        'graph_provenance_uuid': '11111111-1111-1111-1111-111111111111',
        'local_node_id': 42,
        'node_type': 'HIDDEN',
        'activation_function': 'relu',
        'bias': 0.0,
        'attributes': {'output_size': 16},
    }
    u1 = prov.derive_uuid('node', inputs)
    u2 = prov.derive_uuid('node', inputs)
    assert u1 == u2


def test_cache_hits_and_misses_metrics():
    prov = DeterministicUUIDProvider()
    x = prov.derive_uuid('rule', {'x': 1})
    y = prov.derive_uuid('rule', {'x': 1})
    assert x == y
    assert prov.metrics.cache_misses >= 1
    assert prov.metrics.cache_hits >= 1


def test_explain_returns_canonical_and_uuid():
    prov = DeterministicUUIDProvider()
    res = prov.explain('edge', {'k': 'v'})
    assert 'canonical' in res and 'uuid' in res
    # UUID string must be parseable
    uuid.UUID(res['uuid'])


