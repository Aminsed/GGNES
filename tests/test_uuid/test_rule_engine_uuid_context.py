from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.rule_engine import RuleEngine
from ggnes.rules.rule import EmbeddingLogic, EmbeddingStrategy, LHSPattern, RHSAction
from ggnes.utils.rng_manager import RNGManager


def test_rule_engine_sets_uuid_context_and_ids_change():
    g = Graph(config={
        'id_strategy': 'HYBRID',
        'deterministic_uuids': True,
        'uuid_namespace': 'ggnes://uuid/v1',
        'graph_provenance_uuid': 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
    })
    # seed nodes
    g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}})

    # Define a rule that adds a node labeled 'N'
    rule = type('R', (), {})()
    rule.rule_id = uuid.uuid4()
    rule.lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
    rule.rhs = RHSAction(add_nodes=[{'label': 'N', 'properties': {
        'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 8}
    }}])
    rule.embedding = EmbeddingLogic(strategy=EmbeddingStrategy.MAP_BOUNDARY_CONNECTIONS, connection_map={})
    rule.condition = None

    rng = RNGManager(123)
    re = RuleEngine(g, rng_manager=rng, id_manager=None, context_id='ctx', cooldown_steps=0)

    # First apply with bindings x=1
    ok1 = re.apply_rule(rule, {'x': 1})
    assert ok1
    gids_after_first = {g.nodes[n].global_id for n in g.nodes}

    # Apply again with different bindings → new node IDs must differ due to context
    ok2 = re.apply_rule(rule, {'x': 2})
    assert ok2
    gids_after_second = {g.nodes[n].global_id for n in g.nodes}

    assert gids_after_first != gids_after_second


