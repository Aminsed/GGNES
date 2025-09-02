# ruff: noqa: I001
import math


def _valid_graph():
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType

    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                    'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    g.add_edge(i, h, {'weight': 0.1})
    g.add_edge(h, o, {'weight': 0.1})
    g.input_node_ids = [i]
    g.output_node_ids = [o]
    return g


def test_repair_no_errors_returns_true_and_no_attempts():
    from ggnes.repair.repair import repair

    g = _valid_graph()
    ok, metrics = repair(g)  # config None path
    assert ok is True
    assert metrics['repairs_attempted'] == []
    assert metrics['repairs_succeeded'] == []


def test_aggressive_without_rng_uses_default_values_for_random_paths():
    from ggnes.repair.repair import repair

    g = _valid_graph()
    # Corrupt graph to include all three error types fixed with RNG default branches
    # 1) non_finite_bias on hidden
    hidden_id = [nid for nid in g.nodes if g.nodes[nid].node_type.name == 'HIDDEN'][0]
    g.nodes[hidden_id].bias = math.inf
    # 2) non_finite_weight on edge h->o
    out_id = g.output_node_ids[0]
    edge = g.nodes[hidden_id].edges_out[out_id]
    edge.weight = math.nan
    # 3) missing_output_size on hidden (remove attribute)
    g.nodes[hidden_id].attributes.pop('output_size', None)

    ok, metrics = repair(g, {'strategy': 'AGGRESSIVE'})  # rng_manager None
    assert ok is True
    # Defaults applied without RNG: bias -> 0.0, weight -> 0.1, output_size -> 16
    assert g.nodes[hidden_id].bias == 0.0
    assert g.nodes[hidden_id].edges_out[out_id].weight == 0.1
    assert g.nodes[hidden_id].attributes.get('output_size') == 16


def test_allowed_repairs_restrictions_respected():
    from ggnes.repair.repair import repair

    g = _valid_graph()
    # Introduce non-finite weight error only
    hidden_id = [nid for nid in g.nodes if g.nodes[nid].node_type.name == 'HIDDEN'][0]
    out_id = g.output_node_ids[0]
    g.nodes[hidden_id].edges_out[out_id].weight = math.nan

    # Disallow fix_weights
    ok, _ = repair(
        g,
        {
            'strategy': 'MINIMAL_CHANGE',
            'allowed_repairs': ['add_missing_attributes'],
        },
    )
    assert ok is False
    # Ensure weight unchanged (still NaN)
    assert math.isnan(g.nodes[hidden_id].edges_out[out_id].weight)


def test_impact_capped_at_one():
    from ggnes.repair.repair import repair
    from ggnes.core.node import NodeType

    g = _valid_graph()
    hidden_id = [nid for nid in g.nodes if g.nodes[nid].node_type.name == 'HIDDEN'][0]
    out_id = g.output_node_ids[0]
    # Create many errors to push impact above 1.0
    g.nodes[hidden_id].bias = math.inf  # +0.05
    g.nodes[hidden_id].edges_out[out_id].weight = math.inf  # +0.05
    g.nodes[hidden_id].attributes.pop('output_size', None)  # +0.1
    # Duplicate errors multiple times by adding a new hidden node with same issues
    h2 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                     'attributes': {'output_size': 2}})
    g.nodes[h2].bias = math.inf
    g.nodes[h2].attributes.pop('output_size', None)

    ok, metrics = repair(g, {'strategy': 'AGGRESSIVE'})
    assert metrics['repair_impact_score'] <= 1.0


def test_minimal_change_does_not_connect_unreachable_outputs():
    from ggnes.repair.repair import repair

    # Build graph with unreachable output only (no other errors)
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                    'attributes': {'output_size': 2}})
    g.input_node_ids = [i]
    g.output_node_ids = [o]
    ok, _ = repair(g, {'strategy': 'MINIMAL_CHANGE'})
    assert ok is False  # unreachable remains


def test_aggressive_connection_determinism_with_rng_manager():
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.repair.repair import repair
    from ggnes.utils.rng_manager import RNGManager

    def build_two_hidden_graph():
        g = Graph()
        i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear',
                        'attributes': {'output_size': 2}})
        g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                    'attributes': {'output_size': 2}})
        g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu',
                    'attributes': {'output_size': 2}})
        o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear',
                        'attributes': {'output_size': 2}})
        g.input_node_ids = [i]
        g.output_node_ids = [o]
        return g

    g1 = build_two_hidden_graph()
    g2 = build_two_hidden_graph()

    repair(g1, {'strategy': 'AGGRESSIVE'}, RNGManager(999))
    repair(g2, {'strategy': 'AGGRESSIVE'}, RNGManager(999))
    out_id1 = g1.output_node_ids[0]
    out_id2 = g2.output_node_ids[0]
    src1 = next(iter(g1.nodes[out_id1].edges_in))
    src2 = next(iter(g2.nodes[out_id2].edges_in))
    assert src1 == src2
