"""Deterministic UUID provider per project_guide.md §19.

This module implements deterministic UUID derivation using SHA-256 over
canonicalized inputs, with optional strict and freeze policies, a small
memoization cache, and observability helpers.

All randomness is avoided; outputs are deterministic across runs/machines.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from typing import Any


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int | float)) and math.isfinite(value)


def _format_float(value: float, precision: int) -> str:
    # Fixed precision formatting, no scientific notation for determinism
    return f"{value:.{precision}f}"


def _canonicalize(obj: Any, float_precision: int) -> Any:
    """Canonicalize Python data into JSON-serializable structure with:
    - Sorted dict keys
    - Fixed float precision
    - Tuples normalized to lists
    """
    if obj is None or isinstance(obj, (str | bool | int)):
        return obj
    if isinstance(obj, float):
        return _format_float(obj, float_precision)
    if isinstance(obj, (list | tuple)):
        return [_canonicalize(x, float_precision) for x in obj]
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k], float_precision) for k in sorted(obj.keys())}
    # Fallback to string for enums/UUIDs and other simple types
    if isinstance(obj, (uuid.UUID)):
        return str(obj)
    return str(obj)


@dataclass
class UUIDProviderConfig:
    namespace: str = "ggnes://uuid/v1"
    scheme_version: int = 1
    strict: bool = False
    freeze: bool = False
    cache_size: int = 1024
    float_precision: int = 12
    salt: str | None = None
    simulate_collision: bool = False  # testing aid


@dataclass
class UUIDProviderMetrics:
    cache_hits: int = 0
    cache_misses: int = 0
    uuid_collisions: int = 0


class DeterministicUUIDProvider:
    """Derives deterministic UUIDs from canonical inputs.

    The provider maintains a bounded memoization cache keyed by the canonical
    JSON string.
    """

    def __init__(self, config: UUIDProviderConfig | None = None) -> None:
        self.config = config or UUIDProviderConfig()
        self._cache: dict[str, uuid.UUID] = {}
        self._cache_keys: list[str] = []  # maintain insertion order for eviction
        self.metrics = UUIDProviderMetrics()

    def _validate_inputs(self, canonical_data: Any) -> None:
        if not self.config.strict:
            return
        # Walk the structure to reject NaN/Inf

        def _walk(x: Any) -> None:
            if isinstance(x, dict):
                for v in x.values():
                    _walk(v)
            elif isinstance(x, list):
                for v in x:
                    _walk(v)
            elif isinstance(x, float):
                if not math.isfinite(x):
                    raise ValueError("uuid_strict: non-finite float in canonical inputs")
        _walk(canonical_data)

    def _validate_raw_inputs(self, raw_inputs: Any) -> None:
        if not self.config.strict:
            return

        def _walk(x: Any) -> None:
            if isinstance(x, dict):
                for v in x.values():
                    _walk(v)
            elif isinstance(x, list):
                for v in x:
                    _walk(v)
            elif isinstance(x, float):
                if not math.isfinite(x):
                    raise ValueError("uuid_strict: non-finite float in inputs")
        _walk(raw_inputs)

    def _hash_to_uuid(self, data_bytes: bytes) -> uuid.UUID:
        digest = hashlib.sha256(data_bytes).digest()
        # Truncate to 16 bytes for UUID
        truncated = digest[:16]
        # Conform to RFC 4122 variant and version 4 bits (deterministic source)
        as_bytearray = bytearray(truncated)
        as_bytearray[6] = (as_bytearray[6] & 0x0F) | (4 << 4)
        as_bytearray[8] = (as_bytearray[8] & 0x3F) | 0x80
        return uuid.UUID(bytes=bytes(as_bytearray))

    def _evict_if_needed(self) -> None:
        if len(self._cache_keys) > self.config.cache_size:
            # Evict oldest
            oldest = self._cache_keys.pop(0)
            self._cache.pop(oldest, None)

    def derive_uuid(self, entity_type: str, inputs: dict[str, Any]) -> uuid.UUID:
        """Derive a deterministic UUID for an entity.

        Args:
            entity_type: Logical entity type (e.g., 'node', 'edge', 'rule')
            inputs: Arbitrary mapping used to form the canonical JSON

        Returns:
            uuid.UUID
        """
        # Compose canonical payload
        # Validate raw inputs first for strict mode
        self._validate_raw_inputs(inputs)

        payload = {
            "namespace": self.config.namespace,
            "scheme_version": self.config.scheme_version,
            "entity_type": entity_type,
            "salt": self.config.salt or "",
            "inputs": _canonicalize(inputs, self.config.float_precision),
        }
        self._validate_inputs(payload)
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        # Cache lookup
        cached = self._cache.get(canonical_str)
        if cached is not None:
            self.metrics.cache_hits += 1
            return cached
        self.metrics.cache_misses += 1
        uid = self._hash_to_uuid(canonical_str.encode("utf-8"))
        # Simulate collision for tests
        if self.config.simulate_collision:
            if any(v == uid for v in self._cache.values()):
                self.metrics.uuid_collisions += 1
                raise RuntimeError(
                    "UUID collision detected",
                )
        # Insert into cache
        self._cache[canonical_str] = uid
        self._cache_keys.append(canonical_str)
        self._evict_if_needed()
        return uid

    def explain(self, entity_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Explain canonical tuple and resulting UUID for observability."""
        payload = {
            "namespace": self.config.namespace,
            "scheme_version": self.config.scheme_version,
            "entity_type": entity_type,
            "salt": self.config.salt or "",
            "inputs": _canonicalize(inputs, self.config.float_precision),
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        uid = self._hash_to_uuid(canonical_str.encode("utf-8"))
        return {
            "canonical": canonical_str,
            "uuid": str(uid),
        }

    def scheme_metadata(self) -> dict[str, Any]:
        """Return scheme metadata useful for serialization or logs."""
        return {
            "namespace": self.config.namespace,
            "scheme_version": self.config.scheme_version,
            "float_precision": self.config.float_precision,
            "strict": self.config.strict,
            "freeze": self.config.freeze,
        }


def provider_from_graph_config(graph_config: dict) -> DeterministicUUIDProvider:
    """Factory to create a provider from a Graph.config mapping."""
    return DeterministicUUIDProvider(
        UUIDProviderConfig(
            namespace=str(graph_config.get("uuid_namespace", "ggnes://uuid/v1")),
            scheme_version=int(graph_config.get("uuid_scheme_version", 1)),
            strict=bool(graph_config.get("uuid_strict", False)),
            freeze=bool(graph_config.get("uuid_freeze", False)),
            cache_size=int(graph_config.get("uuid_cache_size", 1024)),
            float_precision=int(graph_config.get("uuid_float_precision", 12)),
            salt=(graph_config.get("uuid_salt") if graph_config.get("uuid_salt") is not None else None),
        )
    )


