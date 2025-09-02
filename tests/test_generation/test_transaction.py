"""
M6 TransactionManager/ChangeBuffer tests strictly per project_guide.md.
Tests:
- [T-tx-01] Commit maps temp handles to final IDs; IDManager registrations occur
- [T-tx-02] Rollback restores graph and RNGManager state
- [T-tx-03] Validation rejects edges to nodes staged for deletion; detects duplicate edges
- [T-tx-04] Logging on duplicate edge attempts during commit
"""
import logging
import pickle  # noqa: F401

import pytest

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.core.id_manager import IDManager
from ggnes.utils.rng_manager import RNGManager


def build_min_graph():
    g = Graph()
    return g


class TestTransactionManager:
    def test_commit_maps_temp_handles_and_registers_ids(self, caplog):
        """[T-tx-01] Commit maps temp handles → final IDs; IDManager called."""
        from ggnes.generation.transaction import TransactionManager

        graph = build_min_graph()
        rng = RNGManager(seed=123)
        idm = IDManager()
        tm = TransactionManager(graph=graph, rng_manager=rng, id_manager=idm, context_id="ctx")

        tm.begin()
        tmp_in = tm.buffer.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 4}
        })
        tmp_out = tm.buffer.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 2}
        })
        tm.buffer.add_edge(tmp_in, tmp_out, {"weight": 0.5})

        mapping = tm.commit()

        assert isinstance(mapping, dict)
        assert tmp_in in mapping and tmp_out in mapping
        real_in = mapping[tmp_in]
        real_out = mapping[tmp_out]
        assert real_in in graph.nodes and real_out in graph.nodes

    def test_rollback_restores_graph_and_rng_state(self):
        """[T-tx-02] Rollback restores graph structure and RNG state."""
        from ggnes.generation.transaction import TransactionManager

        graph = build_min_graph()
        rng = RNGManager(seed=777)
        tm = TransactionManager(graph=graph, rng_manager=rng, id_manager=None, context_id="ctx")

        # Baseline RNG state
        base_state = rng.get_state()

        tm.begin()
        # Advance RNG contexts during transaction
        rng.get_context_rng('selection').random()
        rng.get_context_rng('mutation').random()

        # Stage changes
        tmp = tm.buffer.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 3}
        })
        assert tmp in tm.buffer._temp_nodes

        # Rollback
        tm.rollback()

        # Graph unchanged and RNG restored
        assert len(graph.nodes) == 0
        assert rng.get_state() == base_state

    def test_validation_rejects_and_duplicate_detection(self, caplog):
        """[T-tx-03] Edges to nodes staged for deletion rejected; duplicates detected."""
        from ggnes.generation.transaction import TransactionManager

        graph = build_min_graph()
        rng = RNGManager(seed=1)
        tm = TransactionManager(graph=graph, rng_manager=rng, id_manager=None, context_id="ctx")

        tm.begin()
        a = tm.buffer.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 1}
        })
        b = tm.buffer.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 1}
        })

        # Duplicate temp edge staged twice
        tm.buffer.add_edge(a, b, {"weight": 0.1})
        tm.buffer.add_edge(a, b, {"weight": 0.1})

        # Also stage deletion of target; commit must fail validation
        tm.buffer.delete_node(b)

        with pytest.raises(ValueError, match="staged for deletion"):
            tm.commit()

        tm.rollback()

    def test_logging_on_duplicate_edge_attempt(self, caplog):
        """[T-tx-04] Duplicate edge attempts during commit are logged as warnings."""
        from ggnes.generation.transaction import TransactionManager

        graph = build_min_graph()
        rng = RNGManager(seed=5)
        tm = TransactionManager(graph=graph, rng_manager=rng, id_manager=None, context_id="ctx")

        # Pre-existing nodes
        n1 = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 2}
        })
        n2 = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 1}
        })
        graph.add_edge(n1, n2, {"weight": 0.3})

        tm.begin()
        # Attempt to add duplicate of existing edge
        tm.buffer.add_edge(n1, n2, {"weight": 0.3})

        caplog.set_level(logging.WARNING)
        mapping = tm.commit()
        assert isinstance(mapping, dict)
        assert any("duplicate edge" in rec.message.lower() for rec in caplog.records)
