"""M31: Backend parity harness (skeleton parity vs PyTorch)."""

from __future__ import annotations

import pytest

from ggnes.translation.backend_parity import supported_backends, run_backend, parity_compare, rng_parity_signature, outputs_signature, backend_available, rng_context_signature
from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def _tiny_graph():
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'bias': 0.0, 'attributes': {'output_size': 4}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'bias': 0.0, 'attributes': {'output_size': 2}})
    g.add_edge(i, h, {'weight': 0.5})
    g.add_edge(h, o, {'weight': 0.3})
    return g


def test_m31_01_supported_backends_includes_pytorch():
    bk = supported_backends()
    assert 'pytorch' in bk and 'js_shim' in bk
    assert backend_available('pytorch') is True
    assert backend_available('js_shim') is True


def test_m31_02_parity_compare_pytorch_self():
    g = _tiny_graph()
    result = parity_compare(g, backends=['pytorch', 'pytorch'], seed=123)
    assert result['equal'] is True
    assert result['details']['ref'] == result['details']['cmp']


def test_m31_03_run_backend_returns_outputs_and_metadata():
    g = _tiny_graph()
    out = run_backend('pytorch', g, seed=321)
    assert 'outputs' in out and 'device' in out and 'dtype' in out
    assert hasattr(out['outputs'], 'shape')


def test_m31_04_rng_parity_signature_deterministic():
    sig1 = rng_parity_signature(seed=7)
    sig2 = rng_parity_signature(seed=7)
    assert sig1 == sig2
    # Context-specific signature differs across contexts but is stable per context
    c1 = rng_context_signature(seed=7, context='selection')
    c2 = rng_context_signature(seed=7, context='selection')
    c3 = rng_context_signature(seed=7, context='repair')
    assert c1 == c2 and c1 != c3


def test_m31_05_js_shim_parity_with_pytorch_self():
    g = _tiny_graph()
    # js_shim mirrors pytorch path deterministically for now
    result = parity_compare(g, backends=['pytorch', 'js_shim'], seed=42)
    assert result['equal'] is True


def test_m31_06_outputs_signature_deterministic():
    g = _tiny_graph()
    out = run_backend('pytorch', g, seed=11)
    sig1 = outputs_signature(out['outputs'])
    sig2 = outputs_signature(out['outputs'])
    assert sig1 == sig2


def test_m31_07_parity_compare_with_dtype_option():
    g = _tiny_graph()
    # Ensure kwargs are forwarded and parity still holds
    result = parity_compare(g, backends=['pytorch', 'js_shim'], seed=9, dtype='float32')
    assert result['equal'] is True


def test_m31_08_backend_availability_webgpu_false():
    assert backend_available('webgpu') is False


