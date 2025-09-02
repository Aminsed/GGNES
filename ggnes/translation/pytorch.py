"""PyTorch model translation.

Implements `to_pytorch_model(graph, config)` per project_guide.md §8.3.

Note: Torch is optional in requirements. This module defers to an implementation
that imports torch only when to_pytorch_model is invoked.
"""

from __future__ import annotations

from ..core.node import NodeType  # noqa: F401
from .state_manager import StateManager  # noqa: F401


def to_pytorch_model(graph, config: dict | None = None):
    """Translate GGNES graph to a PyTorch nn.Module.

    Args:
        graph: GGNES Graph
        config: Optional translation configuration

    Returns:
        PyTorch nn.Module

    Raises:
        ImportError: If torch is not installed.
        ValueError: For invalid activations or input width mismatches.
    """
    # Defer to implementation file to keep this module importable without torch
    from .pytorch_impl import to_pytorch_model as _impl
    return _impl(graph, config)  # noqa: W391
