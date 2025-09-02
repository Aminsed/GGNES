"""Hierarchical grammar support (ModuleSpec & Parameter System).

This package provides minimal functionality for M19:
- ModuleSpec: name/version/params/ports/attributes/invariants
- Parameter validation with domains and defaults
- Pure, deterministic expression evaluation for computed params and invariants
- Serialization/deserialization helpers
"""

from .module_spec import (
    ModuleSpec,
    ParameterSpec,
    PortSpec,
    ReadOnlyMapping,
    deserialize_module_spec,
)
from .derivation import DerivationEngine, DerivationNode

__all__ = [
    "ModuleSpec",
    "ParameterSpec",
    "PortSpec",
    "ReadOnlyMapping",
    "deserialize_module_spec",
    "DerivationEngine",
    "DerivationNode",
]


