# ruff: noqa: I001
import math
import pytest


def _make_invalid_graph_nonfinite():
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType

    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    hid = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                      'bias': math.inf, 'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    g.add_edge(i, hid, {'weight': 0.1})
    e = g.add_edge(hid, o, {'weight': math.nan})
    g.input_node_ids = [i]
    g.output_node_ids = [o]
    return g, e


def _make_invalid_graph_missing_output_size():
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType

    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    hid = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                      'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    g.add_edge(i, hid, {'weight': 0.1})
    g.add_edge(hid, o, {'weight': 0.1})
    g.input_node_ids = [i]
    g.output_node_ids = [o]
    # Corrupt the graph post-creation to remove required attribute
    g.nodes[hid].attributes.pop('output_size', None)
    return g


def _make_unreachable_output_graph():
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType

    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    # No edges, output unreachable
    g.input_node_ids = [i]
    g.output_node_ids = [o]
    return g


def test_repair_minimal_change_fixes_nonfinite_values():
    from ggnes.repair.repair import repair
    from ggnes.utils.rng_manager import RNGManager

    g, e = _make_invalid_graph_nonfinite()
    is_valid_before = g.validate()
    assert not is_valid_before

    ok, metrics = repair(g, {'strategy': 'MINIMAL_CHANGE'}, RNGManager(123))
    assert ok
    assert metrics['strategy'] == 'MINIMAL_CHANGE'
    assert 'fix_non_finite_bias' in metrics['repairs_attempted']
    assert 'fix_non_finite_weight' in metrics['repairs_attempted']
    assert metrics['repair_impact_score'] > 0.0
    assert g.validate()


def test_repair_add_missing_output_size():
    from ggnes.repair.repair import repair
    from ggnes.utils.rng_manager import RNGManager

    g = _make_invalid_graph_missing_output_size()
    assert not g.validate()

    ok, metrics = repair(g, {'strategy': 'MINIMAL_CHANGE'}, RNGManager(1))
    assert ok
    # ensure attribute added on hidden node(s)
    assert any(
        n.node_type.name == 'HIDDEN' and isinstance(n.attributes.get('output_size'), int)
        for n in g.nodes.values()
    )
    assert g.validate()


def test_repair_disable_does_nothing_and_fails():
    from ggnes.repair.repair import repair

    g = _make_invalid_graph_missing_output_size()
    ok, metrics = repair(g, {'strategy': 'DISABLE'})
    assert not ok
    assert metrics['repairs_attempted'] == []
    assert metrics['repairs_succeeded'] == []


def test_aggressive_connects_unreachable_output_deterministically():
    from ggnes.core.node import NodeType
    from ggnes.repair.repair import repair
    from ggnes.utils.rng_manager import RNGManager

    g = _make_unreachable_output_graph()
    # Add an extra hidden to choose from
    g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                'attributes': {'output_size': 2}})

    ok, metrics = repair(g, {'strategy': 'AGGRESSIVE'}, RNGManager(42))
    # Graph may still be invalid (no input path),
    # but unreachable output should have an incoming edge
    # Find output node and check incoming
    out_id = g.output_node_ids[0]
    out_node = g.nodes[out_id]
    assert any(True for _ in out_node.edges_in)
    assert 'connect_unreachable' in metrics['repairs_attempted']


def test_calculate_repair_penalty_breakpoints():
    from ggnes.repair.repair import calculate_repair_penalty

    assert calculate_repair_penalty(None) == 0.0
    assert calculate_repair_penalty({'repair_impact_score': 0.0}) == 0.0
    assert calculate_repair_penalty({'repair_impact_score': 0.1}) == pytest.approx(0.005, abs=1e-12)
    assert calculate_repair_penalty({'repair_impact_score': 0.5}) == pytest.approx(0.045, abs=1e-12)
    assert calculate_repair_penalty({'repair_impact_score': 1.0}) == pytest.approx(0.145, abs=1e-12)
