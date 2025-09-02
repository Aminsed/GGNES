from __future__ import annotations

from ggnes.utils.uuid_provider import DeterministicUUIDProvider, UUIDProviderConfig


def test_cache_eviction_behaviour_bounded():
    cfg = UUIDProviderConfig(cache_size=2)
    p = DeterministicUUIDProvider(cfg)
    # Miss 1
    p.derive_uuid('x', {'i': 1})
    # Hit 1
    p.derive_uuid('x', {'i': 1})
    # Miss 2
    p.derive_uuid('x', {'i': 2})
    # Miss 3 (evict oldest)
    p.derive_uuid('x', {'i': 3})
    # Now first should be evicted, deriving again is a miss
    misses_before = p.metrics.cache_misses
    p.derive_uuid('x', {'i': 1})
    assert p.metrics.cache_misses == misses_before + 1

