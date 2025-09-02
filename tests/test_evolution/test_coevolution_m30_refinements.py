"""M30 Refinements: weight normalization/validation and repro bundle v2."""

from __future__ import annotations

import json

import pytest

from ggnes.evolution.coevolution import normalize_weights, create_repro_bundle_v2, ReplayBundleV2
from ggnes.utils.rng_manager import RNGManager


def test_m30r_01_normalize_weights_validation_and_sum_to_one():
    # Valid: scales to sum=1
    w = normalize_weights([2, 2, 1])
    assert pytest.approx(sum(w), rel=1e-9) == 1.0
    assert all(x >= 0 for x in w)
    # Invalid: all zeros
    with pytest.raises(ValueError):
        normalize_weights([0, 0])
    # Invalid: NaN/Inf
    with pytest.raises(ValueError):
        normalize_weights([1.0, float('nan')])
    with pytest.raises(ValueError):
        normalize_weights([1.0, float('inf')])


def test_m30r_02_repro_bundle_v2_contains_metadata_and_replays():
    class Ind:
        def __init__(self, name, objs):
            self.name = name
            self._objs = objs
        def objectives(self):
            return self._objs

    pop = [Ind('A', [0.2, 1.0, 10.0]), Ind('B', [0.19, 1.3, 10.0])]
    rng = RNGManager(seed=77)
    weights = normalize_weights([0.8, 0.1, 0.1])
    bundle = create_repro_bundle_v2(
        population=pop,
        weights=weights,
        rng_manager=rng,
        config_hash='cfg123',
        wl_fingerprint='wlabc',
        checksums={'batch': 'abc123'},
        env={'py': '3.13'}
    )
    # Schema and fields
    assert bundle.get('_schema') == 2
    assert bundle['weights'] == weights
    assert bundle['metadata']['config_hash'] == 'cfg123'
    assert bundle['metadata']['wl_fingerprint'] == 'wlabc'
    # Replay parity
    rb1 = ReplayBundleV2(bundle)
    rb2 = ReplayBundleV2(json.loads(json.dumps(bundle)))
    assert [x.name for x in rb1.select(k=1)] == [x.name for x in rb2.select(k=1)]


