import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import Direction, Distribution, EmbeddingLogic, LHSPattern, RHSAction, Rule


def build_graph():
    g = Graph()
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    c = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}})
    g.add_edge(a, b, {'weight': 0.1})
    g.add_edge(b, c, {'weight': 0.2})
    return g, a, b, c


def test_embedding_copy_all_and_connect_single_determinism():
    g, a, b, c = build_graph()
    lhs = LHSPattern(nodes=[{'label': 'X', 'match_criteria': {}}], edges=[], boundary_nodes=['X'])
    rhs = RHSAction(
        delete_nodes=['X'],
        add_nodes=[{'label': 'P', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}},
                   {'label': 'Q', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 4}}}],
    )
    emb = EmbeddingLogic(
        connection_map={
            ('X', Direction.IN): [('P', Distribution.COPY_ALL)],
            ('X', Direction.OUT): [('Q', Distribution.CONNECT_SINGLE)],
        },
        excess_connection_handling='WARNING',
        unknown_direction_handling='WARNING',
        boundary_handling='PROCESS_LAST',
    )
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={'priority': 1})
    engine = RuleEngine(g, rng_manager=None, id_manager=None)
    bindings = {'X': b}
    ok = engine.apply_rule(rule, bindings)
    assert ok
    # Ensure reconnections exist deterministically
    # a -> P (copied), Q -> c (single)
    existing = {a, b, c}
    new_nodes = sorted(set(g.nodes.keys()) - existing)
    assert len(new_nodes) == 2
    p = new_nodes[0]  # P added first gets lower id
    q = new_nodes[1]
    assert g.find_edge_by_endpoints(a, p) is not None
    assert g.find_edge_by_endpoints(q, c) is not None


