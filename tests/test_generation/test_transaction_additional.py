"""
Additional tests to reach 100% coverage for TransactionManager and ChangeBuffer.
"""
import logging

import pytest

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.utils.rng_manager import RNGManager


def new_graph():
    return Graph()


def test_begin_resets_buffer_and_snapshots_rng(monkeypatch):
    from ggnes.generation.transaction import TransactionManager

    g = new_graph()
    rng = RNGManager(seed=42)
    tm = TransactionManager(graph=g, rng_manager=rng, id_manager=None, context_id="ctx")

    # Poison buffer to verify reset
    tm.buffer._temp_nodes['tmp:x'] = {}
    tm.buffer._temp_edges.append(('tmp:x', 'tmp:y', {}))
    tm.buffer._delete_nodes.add('tmp:z')

    # Spy on get_state called
    called = {'v': False}

    orig_get_state = rng.get_state

    def spy_get_state():
        called['v'] = True
        return orig_get_state()

    monkeypatch.setattr(tm.rng_manager, 'get_state', spy_get_state)

    tm.begin()

    assert called['v'] is True
    assert tm.buffer._temp_nodes == {}
    assert tm.buffer._temp_edges == []
    assert tm.buffer._delete_nodes == set()


def test_commit_invalid_endpoint_reference_raises():
    from ggnes.generation.transaction import TransactionManager

    g = new_graph()
    rng = RNGManager(seed=7)
    tm = TransactionManager(graph=g, rng_manager=rng, id_manager=None, context_id="ctx")

    tm.begin()
    a = tm.buffer.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1}
    })
    # Edge references unknown type for target to hit fallback None path
    tm.buffer.add_edge(a, 3.14159, {})

    with pytest.raises(ValueError, match="Invalid edge endpoint reference"):
        tm.commit()


def test_register_edge_called_when_id_manager_present(monkeypatch):
    from ggnes.generation.transaction import TransactionManager

    class DummyIDM:
        def __init__(self):
            self.registered_edges = 0
            self.registered_nodes = 0

        def register_node(self, node, ctx):
            self.registered_nodes += 1

        def register_edge(self, edge, ctx):
            self.registered_edges += 1

    g = new_graph()
    rng = RNGManager(seed=9)
    idm = DummyIDM()
    tm = TransactionManager(graph=g, rng_manager=rng, id_manager=idm, context_id="ctx")

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
    tm.buffer.add_edge(a, b, {"weight": 0.2})

    mapping = tm.commit()
    assert idm.registered_nodes == 2
    # One edge registered
    assert idm.registered_edges == 1
    assert isinstance(mapping, dict)


def test_delete_existing_node_applied_on_commit():
    from ggnes.generation.transaction import TransactionManager

    g = new_graph()
    rng = RNGManager(seed=12)
    tm = TransactionManager(graph=g, rng_manager=rng, id_manager=None, context_id="ctx")

    # Create a real node and then stage deletion
    node_id = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "relu",
        "attributes": {"output_size": 3}
    })

    tm.begin()
    tm.buffer.delete_node(node_id)
    mapping = tm.commit()
    assert isinstance(mapping, dict)
    assert node_id not in g.nodes


def test_add_edge_returns_none_path_logged(monkeypatch, caplog):
    from ggnes.generation.transaction import TransactionManager

    g = new_graph()
    rng = RNGManager(seed=13)
    tm = TransactionManager(graph=g, rng_manager=rng, id_manager=None, context_id="ctx")

    # Create real nodes
    n1 = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1}
    })
    n2 = g.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1}
    })

    # Monkeypatch graph to simulate add_edge returning None while no existing edge
    def fake_find_edge_by_endpoints(src, dst):
        return None

    def fake_add_edge(src, dst, props):
        return None

    monkeypatch.setattr(g, 'find_edge_by_endpoints', fake_find_edge_by_endpoints)
    monkeypatch.setattr(g, 'add_edge', fake_add_edge)

    tm.begin()
    tm.buffer.add_edge(n1, n2, {"weight": 0.4})

    caplog.set_level(logging.WARNING)
    mapping = tm.commit()
    assert isinstance(mapping, dict)
    assert any("duplicate edge attempt" in rec.message.lower() for rec in caplog.records)


def test_duplicate_staged_edges_logged_and_skipped(caplog):
    from ggnes.generation.transaction import TransactionManager

    g = new_graph()
    rng = RNGManager(seed=11)
    tm = TransactionManager(graph=g, rng_manager=rng, id_manager=None, context_id="ctx")

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
    tm.buffer.add_edge(a, b, {"weight": 0.1})
    tm.buffer.add_edge(a, b, {"weight": 0.1})

    caplog.set_level(logging.WARNING)
    mapping = tm.commit()
    assert isinstance(mapping, dict)
    # Only one edge is applied; duplicate should log
    assert any("duplicate edge" in rec.message.lower() for rec in caplog.records)
