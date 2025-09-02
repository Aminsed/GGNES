from __future__ import annotations

from ggnes.utils.uuid_provider import (
    DeterministicUUIDProvider,
    UUIDProviderConfig,
    provider_from_graph_config,
)


def test_float_precision_rounding_changes_uuid():
    cfg1 = UUIDProviderConfig(float_precision=4)
    cfg2 = UUIDProviderConfig(float_precision=12)
    p1 = DeterministicUUIDProvider(cfg1)
    p2 = DeterministicUUIDProvider(cfg2)
    inputs = {'value': 0.123456789}
    u1 = p1.derive_uuid('node', inputs)
    u2 = p2.derive_uuid('node', inputs)
    assert u1 != u2


def test_namespace_and_version_affect_uuid():
    p_ns1 = DeterministicUUIDProvider(UUIDProviderConfig(namespace='ns1', scheme_version=1))
    p_ns2 = DeterministicUUIDProvider(UUIDProviderConfig(namespace='ns2', scheme_version=1))
    p_v2 = DeterministicUUIDProvider(UUIDProviderConfig(namespace='ns1', scheme_version=2))
    inputs = {'k': 'v'}
    u_ns1 = p_ns1.derive_uuid('rule', inputs)
    u_ns2 = p_ns2.derive_uuid('rule', inputs)
    u_v2 = p_v2.derive_uuid('rule', inputs)
    assert u_ns1 != u_ns2
    assert u_ns1 != u_v2


def test_salt_changes_uuid():
    p_s0 = DeterministicUUIDProvider(UUIDProviderConfig(salt=None))
    p_s1 = DeterministicUUIDProvider(UUIDProviderConfig(salt='tenantA'))
    inputs = {'id': 1}
    u0 = p_s0.derive_uuid('edge', inputs)
    u1 = p_s1.derive_uuid('edge', inputs)
    assert u0 != u1


def test_provider_from_graph_config_factory():
    graph_cfg = {
        'uuid_namespace': 'ggnes://uuid/v9',
        'uuid_scheme_version': 9,
        'uuid_float_precision': 7,
        'uuid_strict': True,
        'uuid_freeze': False,
        'uuid_cache_size': 3,
        'uuid_salt': 'tenantZ',
    }
    prov = provider_from_graph_config(graph_cfg)
    meta = prov.scheme_metadata()
    assert meta['namespace'] == 'ggnes://uuid/v9'
    assert meta['scheme_version'] == 9
    assert meta['float_precision'] == 7
    assert meta['strict'] is True

