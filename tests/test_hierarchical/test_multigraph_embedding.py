import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import LHSPattern, RHSAction, EmbeddingLogic, Rule, Direction, Distribution


def _make_multigraph_triplet():
    g = Graph(config={'multigraph': True})
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    x = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    # Add two parallel edges A->X and X->B
    g.add_edge(a, x, {'weight': 0.1})
    g.add_edge(a, x, {'weight': 0.2})
    g.add_edge(x, b, {'weight': 0.3})
    g.add_edge(x, b, {'weight': 0.4})
    return g, a, x, b


def test_multigraph_embedding_copy_all_and_connect_single_instance_aware():
    g, a, x, b = _make_multigraph_triplet()

    lhs = LHSPattern(
        nodes=[{'label': 'X', 'match_criteria': {'node_type': NodeType.HIDDEN}}],
        edges=[],
        boundary_nodes=['X']
    )
    rhs = RHSAction(
        add_nodes=[{
            'label': 'P',
            'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'linear',
                'attributes': {'output_size': 4}
            }
        }],
        delete_nodes=['X']
    )
    emb = EmbeddingLogic(
        connection_map={
            ('X', Direction.IN): [('P', Distribution.COPY_ALL)],
            ('X', Direction.OUT): [('P', Distribution.CONNECT_SINGLE)],
        },
        excess_connection_handling='WARNING',
        unknown_direction_handling='WARNING',
        boundary_handling='PROCESS_FIRST'
    )
    rule = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={'priority': 1})

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, {'X': x})
    assert ok

    # EXPECTATIONS under multigraph:
    # - COPY_ALL for IN replicates both parallel A->X edges as A->P (two edges)
    in_edges = g.find_edges_by_endpoints(a, next(n for n in g.nodes if n not in [a, b, x]))
    assert len(in_edges) == 2
    # - CONNECT_SINGLE for OUT picks one deterministic X->B target and adds P->B single edge
    p = next(n for n in g.nodes if n not in [a, b])
    out_edges = g.find_edges_by_endpoints(p, b)
    assert len(out_edges) == 1


def test_multigraph_embedding_numeric_distributions_and_tie_breaking():
    g = Graph(config={'multigraph': True})
    # Build star: P will connect to two of the B targets deterministically
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    x = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    b1 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    b2 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    b3 = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    g.add_edge(a, x, {'weight': 0.1})
    g.add_edge(x, b1, {'weight': 0.3})
    g.add_edge(x, b2, {'weight': 0.4})
    g.add_edge(x, b3, {'weight': 0.5})

    lhs = LHSPattern(nodes=[{'label': 'X', 'match_criteria': {'node_type': NodeType.HIDDEN}}], edges=[], boundary_nodes=['X'])
    rhs = RHSAction(add_nodes=[{'label': 'P', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}}}], delete_nodes=['X'])
    emb = EmbeddingLogic(
        connection_map={
            ('X', Direction.IN): [('P', 1)],  # numeric: take first 1 incoming source deterministically
            ('X', Direction.OUT): [('P', 2)],  # numeric: take first 2 outgoing targets deterministically
        },
        excess_connection_handling='WARNING',
        unknown_direction_handling='WARNING',
        boundary_handling='PROCESS_FIRST'
    )
    rule = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={'priority': 1})
    ok = RuleEngine(g, rng_manager=None, id_manager=None).apply_rule(rule, {'X': x})
    assert ok

    p = next(n for n in g.nodes if n not in [a, b1, b2, b3])
    # IN numeric=1 → A->P only one edge
    assert len(g.find_edges_by_endpoints(a, p)) == 1
    # OUT numeric=2 → P connects to two smallest target_ids deterministically (b1, b2)
    outs = [(e.target_node_id) for e in g.find_edges_by_endpoints(p, b1)] + [(e.target_node_id) for e in g.find_edges_by_endpoints(p, b2)] + [(e.target_node_id) for e in g.find_edges_by_endpoints(p, b3)]
    assert set(outs) == {b1, b2}


def test_multigraph_connect_single_tie_break_among_parallel_edges_by_edge_id_weights():
    g = Graph(config={'multigraph': True})
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    x = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})

    # Create two parallel edges with distinct weights A->X and X->B
    eid_ax1 = g.add_edge(a, x, {'weight': 0.91})
    eid_ax2 = g.add_edge(a, x, {'weight': 0.11})
    eid_xb1 = g.add_edge(x, b, {'weight': 0.31})
    eid_xb2 = g.add_edge(x, b, {'weight': 0.32})

    # Determine expected chosen weights by edge_id string ordering
    ax_edges = g.find_edges_by_endpoints(a, x)
    ax_sorted = sorted(ax_edges, key=lambda e: (e.source_node_id, str(e.edge_id)))
    expected_in_weight = ax_sorted[0].weight
    xb_edges = g.find_edges_by_endpoints(x, b)
    xb_sorted = sorted(xb_edges, key=lambda e: (e.target_node_id, str(e.edge_id)))
    expected_out_weight = xb_sorted[0].weight

    lhs = LHSPattern(nodes=[{'label': 'X', 'match_criteria': {'node_type': NodeType.HIDDEN}}], edges=[], boundary_nodes=['X'])
    rhs = RHSAction(add_nodes=[{'label': 'P', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}}}], delete_nodes=['X'])
    emb = EmbeddingLogic(
        connection_map={
            ('X', Direction.IN): [('P', Distribution.CONNECT_SINGLE)],
            ('X', Direction.OUT): [('P', Distribution.CONNECT_SINGLE)],
        },
        excess_connection_handling='WARNING',
        unknown_direction_handling='WARNING',
        boundary_handling='PROCESS_FIRST'
    )
    rule = Rule(uuid.uuid4(), lhs, rhs, emb, metadata={'priority': 1})
    ok = RuleEngine(g, rng_manager=None, id_manager=None).apply_rule(rule, {'X': x})
    assert ok

    p = next(n for n in g.nodes if n not in [a, b])
    # Check chosen weights transferred from selected parallel edge instances
    edge_ap = g.find_edge_by_endpoints(a, p)
    edge_pb = g.find_edge_by_endpoints(p, b)
    assert edge_ap is not None and abs(edge_ap.weight - expected_in_weight) < 1e-9
    assert edge_pb is not None and abs(edge_pb.weight - expected_out_weight) < 1e-9


