"""Hierarchical derivation engine (project_guide.md §20).

Implements a minimal, deterministic derivation engine with:
- Canonical expansion order
- Nested TransactionManager usage per expansion
- RNGManager hierarchical contexts
- Deterministic UUIDs per derivation node (module name/version + path + signature)
- Depth/expansion limits with structured errors

This module is deliberately simple and focused to keep code small and readable
while satisfying the test requirements for M20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ggnes.core.graph import Graph
from ggnes.generation.rule_engine import TransactionManager
from ggnes.utils.validation import ValidationError
from ggnes.utils.uuid_provider import (
    DeterministicUUIDProvider,
    provider_from_graph_config,
)
from ggnes.utils.rng_manager import RNGManager
from ggnes.hierarchical.module_spec import ModuleSpec


@dataclass(frozen=True)
class DerivationNode:
    """Node in a derivation tree.

    Attributes:
        module: ModuleSpec used at this node
        signature: Binding signature from ModuleSpec.explain_params()
        path: Tuple indices from root to this node (e.g., (0, 1))
        uuid: Deterministic UUID derived from module + path + signature
        children: Child derivations
    """

    module: ModuleSpec
    signature: str
    path: Tuple[int, ...]
    uuid: str
    children: Tuple["DerivationNode", ...] = field(default_factory=tuple)


class DerivationEngine:
    """Performs hierarchical derivations with nested transactions.

    Usage pattern (simple):
        engine = DerivationEngine(graph, config, rng_manager)
        root = engine.expand(root_spec, overrides={...}, children=[...])

    Children are expanded recursively in canonical order. Each expansion
    occurs inside its own TransactionManager. If an error occurs, the
    transaction is rolled back and a structured ValidationError is raised.
    """

    def __init__(
        self,
        graph: Graph,
        config: Optional[Dict[str, Any]] = None,
        rng_manager: Optional[RNGManager] = None,
        *,
        uuid_provider: Optional[DeterministicUUIDProvider] = None,
    ) -> None:
        self.graph = graph
        self.config = config or {}
        self.rng_manager = rng_manager or RNGManager()
        self.uuid_provider = uuid_provider or provider_from_graph_config(graph.config)

        # Limits with safe defaults
        self.max_depth: int = int(self.config.get("max_derivation_depth", 8))
        self.max_expansions: int = int(self.config.get("max_derivation_expansions", 1000))
        self._budget_ms: Optional[int] = (
            int(self.config.get("derivation_time_budget_ms"))
            if "derivation_time_budget_ms" in self.config
            else None
        )

        # M24 parallel hierarchical expansion knobs (observational equivalence)
        self._parallel: bool = bool(self.config.get("parallel_hg_execution", False))
        self._max_workers: int = max(1, int(self.config.get("hg_max_parallel_workers", 4)))
        self._batch_policy: str = str(self.config.get("hg_parallel_batch_policy", "MAX_INDEPENDENT_SET"))
        self._fixed_batch_size: int = max(1, int(self.config.get("hg_fixed_batch_size", 0) or 0))
        self._priority_cap: int = max(1, int(self.config.get("hg_priority_cap", 0) or 0))
        self._conflict_strategy: str = str(self.config.get("hg_parallel_conflict_strategy", "SKIP")).upper()
        self._max_requeues: int = max(0, int(self.config.get("hg_parallel_max_requeues", 1)))
        self._memory_budget_mb: int = int(self.config.get("parallel_memory_budget_mb", 0) or 0)
        self._child_mem_cost_mb: int = max(1, int(self.config.get("hg_child_mem_cost_mb", 1)))

        self._expansion_count = 0
        self._start_time: Optional[float] = None
        self._last_metrics: Dict[str, Any] = {}

    # -------------------------- Public API --------------------------

    def expand(
        self,
        root: ModuleSpec,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        children: Optional[List[Tuple[ModuleSpec, Optional[Dict[str, Any]], Optional[list]]]] = None,
    ) -> DerivationNode:
        """Expand a root module and return the derivation tree.

        Args:
            root: Root ModuleSpec
            overrides: Parameter overrides for root
            children: Optional list of tuples (child_spec, child_overrides, child_children)

        Returns:
            DerivationNode
        """
        # Initialize budget timer for this derivation
        import time
        self._start_time = time.perf_counter()
        # Reset run metrics
        self._last_metrics = {
            "batches": [],
            "batches_processed": 0,
            "worker_cap": int(self._max_workers),
            "batch_policy": self._batch_policy,
        }
        return self._expand_recursive(root, overrides or {}, path=(), children=children or [])

    def explain(self, node: DerivationNode) -> Dict[str, Any]:
        """Return a canonical, serializable summary of a derivation node.

        Includes module name/version, signature, uuid, path, and children summaries.
        """
        checksum = self._determinism_checksum(node)
        return {
            "module": node.module.name,
            "version": int(node.module.version),
            "signature": node.signature,
            "uuid": node.uuid,
            "path": list(node.path),
            "children": [self.explain(ch) for ch in node.children],
            "checksum": checksum,
        }

    # ------------------------ Internal Logic ------------------------

    def _expand_recursive(
        self,
        spec: ModuleSpec,
        overrides: Dict[str, Any],
        *,
        path: Tuple[int, ...],
        children: List[Tuple[ModuleSpec, Optional[Dict[str, Any]], Optional[list]]],
    ) -> DerivationNode:
        # Time budget check
        if self._budget_ms is not None and self._start_time is not None:
            import time
            elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
            if elapsed_ms >= self._budget_ms:
                raise ValidationError(
                    "derivation_timeout",
                    "Derivation time budget exceeded",
                    module=spec.name,
                    module_version=int(spec.version),
                    module_path=path,
                    elapsed_ms=int(elapsed_ms),
                    budget_ms=int(self._budget_ms),
                )
        # Limits
        if len(path) > self.max_depth:
            raise ValidationError(
                "hierarchy_limit",
                "Exceeded derivation depth",
                module_path=path,
                reason="max_depth_exceeded",
            )
        if self._expansion_count >= self.max_expansions:
            raise ValidationError(
                "hierarchy_limit",
                "Exceeded max expansions",
                module_path=path,
                reason="max_expansions_exceeded",
            )

        # Compute binding signature (validates params/invariants)
        info = spec.explain_params(overrides, graph_config=self.graph.config)
        signature = info["signature"]

        # Derive deterministic UUID for this derivation node
        uid = str(
            self.uuid_provider.derive_uuid(
                "hier_module",
                {
                    "module": spec.name,
                    "version": int(spec.version),
                    "path": list(path),
                    "binding_signature": signature,
                },
            )
        )

        # Begin nested transaction for this module expansion
        tx = TransactionManager(self.graph, self.rng_manager)
        try:
            tx.begin()

            # Advance RNG in hierarchical context to preserve determinism
            ctx_name = f"expand:module:{spec.name}@{spec.version}:path:{'.'.join(map(str, path)) or 'root'}"
            _ = self.rng_manager.get_context_rng(ctx_name).random()

            # Minimal structural effect: add a hidden node to represent the expansion
            node_handle = tx.buffer.add_node(
                {
                    "node_type": self._hidden_node_type(),
                    "activation_function": "linear",
                    "bias": 0.0,
                    "attributes": {
                        "output_size": 1,
                        "module_name": spec.name,
                        "module_version": int(spec.version),
                        "derivation_uuid": uid,
                    },
                }
            )

            # Enforce children guard if configured
            max_children = self.config.get("max_children_per_node")
            if isinstance(max_children, int) and max_children >= 0 and len(children) > max_children:
                raise ValidationError(
                    "hierarchy_limit",
                    "Exceeded max children for derivation node",
                    module=spec.name,
                    module_version=int(spec.version),
                    module_path=path,
                    reason="max_children_exceeded",
                    count=len(children),
                    limit=int(max_children),
                )

            # Canonical child order: by (name, version, signature)
            prepared_children: List[Tuple[ModuleSpec, Dict[str, Any], Optional[list]]] = []
            for child_spec, child_over, child_children in children:
                child_over = child_over or {}
                # Precompute signature for ordering without mutating graph
                try:
                    child_sig = child_spec.explain_params(child_over, graph_config=self.graph.config)[
                        "signature"
                    ]
                except ValidationError:
                    # Invariant or binding errors: use deterministic placeholder signature
                    child_sig = f"invalid:{child_spec.name}@{int(child_spec.version)}"
                prepared_children.append((child_spec, child_over, child_children, child_sig))
            prepared_children.sort(key=lambda t: (t[0].name, int(t[0].version), t[3]))

            # Recurse (serial or deterministic batches depending on config)
            child_nodes: List[DerivationNode] = []
            if not self._parallel:
                for idx, (child_spec, child_over, child_children, _sig) in enumerate(prepared_children):
                    child_path = (*path, idx)
                    child_node = self._expand_recursive(
                        child_spec,
                        child_over,
                        path=child_path,
                        children=child_children or [],
                    )
                    child_nodes.append(child_node)
            else:
                # Deterministic batch construction; execution remains sequential but metrics
                # allow parity assertions across worker counts/policies.
                indexed = [(i, c[0], c[1], c[2], c[3]) for i, c in enumerate(prepared_children)]
                batches: List[List[Tuple[int, ModuleSpec, Dict[str, Any], Optional[list], str]]] = []
                # Memory-aware batching: if memory budget configured, split batches
                def _memory_split(items: list[tuple[int, ModuleSpec, Dict[str, Any], Optional[list], str]]):
                    if self._memory_budget_mb <= 0:
                        return [items]
                    batches_local: List[List[Tuple[int, ModuleSpec, Dict[str, Any], Optional[list], str]]] = []
                    current: List[Tuple[int, ModuleSpec, Dict[str, Any], Optional[list], str]] = []
                    used = 0
                    backoffs = 0
                    for entry in items:
                        if used + self._child_mem_cost_mb > self._memory_budget_mb and current:
                            batches_local.append(current)
                            current = []
                            used = 0
                            backoffs += 1
                        current.append(entry)
                        used += self._child_mem_cost_mb
                    if current:
                        batches_local.append(current)
                    if backoffs:
                        self._last_metrics["memory_backoff_count"] = int(self._last_metrics.get("memory_backoff_count", 0)) + backoffs
                    return batches_local

                if self._batch_policy == "FIXED_SIZE" and self._fixed_batch_size > 0:
                    size = self._fixed_batch_size
                    for i in range(0, len(indexed), size):
                        for sub in _memory_split(indexed[i : i + size]):
                            batches.append(sub)
                elif self._batch_policy == "PRIORITY_CAP" and self._priority_cap > 0:
                    cap = self._priority_cap
                    for i in range(0, len(indexed), cap):
                        for sub in _memory_split(indexed[i : i + cap]):
                            batches.append(sub)
                else:
                    # MAX_INDEPENDENT_SET/PRIORITY_CAP collapse to single canonical batch here
                    if indexed:
                        for sub in _memory_split(indexed):
                            batches.append(sub)

                import hashlib as _hashlib
                import time as _time
                for batch in batches:
                    # Budget check before batch
                    if self._budget_ms is not None and self._start_time is not None:
                        elapsed_ms = (_time.perf_counter() - self._start_time) * 1000.0
                        if elapsed_ms >= self._budget_ms:
                            raise ValidationError(
                                "derivation_timeout",
                                "Derivation time budget exceeded",
                                module=spec.name,
                                module_version=int(spec.version),
                                module_path=path,
                                elapsed_ms=int(elapsed_ms),
                                budget_ms=int(self._budget_ms),
                            )
                    # Stable checksum over (index, signature)
                    batch_sig = str([(i, s) for (i, _sp, _ov, _ch, s) in batch]).encode()
                    checksum = _hashlib.sha256(batch_sig).hexdigest()[:16]
                    self._last_metrics.setdefault("batches", []).append(
                        {
                            "size": len(batch),
                            "worker_count": min(self._max_workers, len(batch)),
                            "checksum": checksum,
                            "mis_size": len(batch),
                        }
                    )
                    self._last_metrics["batches_processed"] = int(self._last_metrics.get("batches_processed", 0)) + 1

                    # Execute in canonical order within batch
                    conflicts = 0
                    requeues = 0
                    rollbacks = 0
                    for i, child_spec, child_over, child_children, _sig in batch:
                        attempts = 0
                        while True:
                            try:
                                child_path = (*path, i)
                                child_node = self._expand_recursive(
                                    child_spec,
                                    child_over,
                                    path=child_path,
                                    children=child_children or [],
                                )
                                child_nodes.append(child_node)
                                break
                            except ValidationError:
                                conflicts += 1
                                if self._conflict_strategy == "REQUEUE" and attempts < self._max_requeues:
                                    requeues += 1
                                    attempts += 1
                                    continue
                                # SPECULATIVE_MERGE and SKIP both drop the child; record rollback for SPECULATIVE_MERGE
                                if self._conflict_strategy == "SPECULATIVE_MERGE":
                                    rollbacks += 1
                                break
                    # Record per-batch conflict metrics
                    self._last_metrics.setdefault("conflicts", 0)
                    self._last_metrics.setdefault("requeues", 0)
                    self._last_metrics.setdefault("rollbacks", 0)
                    self._last_metrics["conflicts"] += conflicts
                    self._last_metrics["requeues"] += requeues
                    self._last_metrics["rollbacks"] += rollbacks

            # Validate transaction consistency and commit
            # No separate validate API in this TransactionManager. Commit will validate and raise.
            _ = tx.commit()

            self._expansion_count += 1
            return DerivationNode(
                module=spec,
                signature=signature,
                path=path,
                uuid=uid,
                children=tuple(child_nodes),
            )

        except ValidationError:
            # Re-raise structured errors after rollback
            tx.rollback()
            raise
        except Exception as exc:
            # Rollback and wrap as structured error
            tx.rollback()
            raise ValidationError(
                "derivation_failed",
                f"Expansion failed: {exc}",
                module=spec.name,
                module_version=int(spec.version),
                module_path=path,
            ) from exc

    # ------------------------ Small utilities -----------------------

    @staticmethod
    def _hidden_node_type():
        # Local import to avoid circulars; NodeType is in ggnes.core.node
        from ggnes.core.node import NodeType

        return NodeType.HIDDEN

    def _determinism_checksum(self, node: DerivationNode) -> str:
        import hashlib
        def flatten(n: DerivationNode) -> List[Tuple[str, int, str, str, Tuple[int, ...]]]:
            out: List[Tuple[str, int, str, str, Tuple[int, ...]]] = [(n.module.name, int(n.module.version), n.signature, n.uuid, n.path)]
            for ch in n.children:
                out.extend(flatten(ch))
            return out
        payload = str(flatten(node)).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def last_expand_metrics(self) -> Dict[str, Any]:
        return dict(self._last_metrics)


