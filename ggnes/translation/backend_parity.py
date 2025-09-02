"""Backend parity harness (M31 skeleton).

Provides a minimal parity API over the existing PyTorch translator as a baseline.
Future backends (WebGPU/WebNN) can plug into the same interface.
"""

from __future__ import annotations

from typing import Any, List
import hashlib

import torch

from ggnes.translation.pytorch import to_pytorch_model


def supported_backends() -> List[str]:
    return ['pytorch', 'js_shim']


def run_backend(backend: str, graph, seed: int = 0, device: str | None = None, dtype: Any | None = None) -> dict[str, Any]:
    if backend != 'pytorch':
        if backend == 'js_shim':
            # Mirror pytorch path deterministically to simulate browser backend
            return run_backend('pytorch', graph, seed=seed, device=device, dtype=dtype)
        raise ValueError(f"Unsupported backend: {backend}")
    torch.manual_seed(seed)
    # Normalize dtype arg
    resolved_dtype = dtype
    if isinstance(dtype, str):
        resolved_dtype = getattr(torch, dtype, torch.float32)
    if resolved_dtype is None:
        resolved_dtype = torch.float32
    config = {
        'device': device or 'cpu',
        'dtype': resolved_dtype,
    }
    model = to_pytorch_model(graph, config)
    # Create a deterministic input of correct width
    input_size = sum(graph.nodes[nid].attributes['output_size'] for nid in graph.input_node_ids)
    x = torch.randn(4, input_size, device=model.device, dtype=model.dtype)
    y = model(x, reset_states=True)
    return {
        'outputs': y.detach().cpu(),
        'device': str(model.device),
        'dtype': str(model.dtype),
    }


def parity_compare(graph, backends: list[str], seed: int = 0, **kwargs) -> dict[str, Any]:
    if not backends or len(backends) < 2:
        raise ValueError("Provide at least two backends for parity comparison")
    ref = run_backend(backends[0], graph, seed=seed, **kwargs)
    cmp = run_backend(backends[1], graph, seed=seed, **kwargs)
    ref_out = ref['outputs']
    cmp_out = cmp['outputs']
    equal = torch.allclose(ref_out, cmp_out, atol=1e-6, rtol=1e-6)
    return {
        'equal': bool(equal),
        'details': {
            'ref': ref_out.tolist(),
            'cmp': cmp_out.tolist(),
        }
    }


__all__ = ["supported_backends", "run_backend", "parity_compare"]


def rng_parity_signature(seed: int) -> str:
    """Return a deterministic signature string for RNG parity checks across backends."""
    h = hashlib.sha256(f"rng:{seed}".encode()).hexdigest()
    return h

__all__.append("rng_parity_signature")


def outputs_signature(tensor: torch.Tensor) -> str:
    """Produce a deterministic digest of outputs for golden-like parity checks."""
    arr = tensor.detach().cpu().contiguous().view(-1).numpy()
    # Use hex digest of SHA256 over bytes; simple and deterministic
    import numpy as _np
    buf = _np.ascontiguousarray(arr).tobytes()
    return hashlib.sha256(buf).hexdigest()

__all__.append("outputs_signature")


def backend_available(name: str) -> bool:
    return name in supported_backends()


def rng_context_signature(seed: int, context: str) -> str:
    return hashlib.sha256(f"rng:{seed}:{context}".encode()).hexdigest()

__all__.extend(["backend_available", "rng_context_signature"])


