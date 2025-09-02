import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import LHSPattern, RHSAction, EmbeddingLogic, Rule, Direction, Distribution


def _graph_with_boundary_triplet():
    g = Graph()
    a = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'linear',
        'attributes': {'output_size': 4},
    })
    x = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'linear',
        'attributes': {'output_size': 4},
    })
    b = g.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'linear',
        'attributes': {'output_size': 4},
    })
    g.add_edge(a, x, {'weight': 0.5})
    g.add_edge(x, b, {'weight': 0.7})
    return g, a, x, b


def test_rule_engine_applies_hierarchical_embedding_before_deletion():
    g, a, x, b = _graph_with_boundary_triplet()

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

    # Bindings for X
    bindings = {'X': x}

    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    ok = engine.apply_rule(rule, bindings)
    assert ok

    # X is deleted; P exists and should have edges from A and to B
    # Find P by scanning nodes (new node has attributes with no module tag)
    new_nodes = [nid for nid in g.nodes if nid != a and nid != b and nid != x]
    assert len(new_nodes) == 1
    p = new_nodes[0]
    assert g.find_edge_by_endpoints(a, p) is not None
    assert g.find_edge_by_endpoints(p, b) is not None
    assert x not in g.nodes


