"""
Additional tests to reach 100% coverage for RuleEngine.
"""

import pytest  # noqa: F401

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def gbase():
    g = Graph()
    i = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 1}
    })
    return g, i


def test_condition_prevents_application():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=1)
    engine = RuleEngine(graph=g, rng_manager=rng)

    lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
    rhs = RHSAction(
        add_nodes=[
            {"label": "H",
             "properties": {
                 "node_type": NodeType.HIDDEN,
                 "activation_function": "relu",
                 "attributes": {"output_size": 2}
             }}
        ]
    )

    def never(_g, _b, _c):
        return False

    r = Rule(rule_id="cond", lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(), condition=never)
    assert engine.apply_rule(r, bindings={}) is False


def test_tick_cooldown_clears_entries():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=3)
    engine = RuleEngine(graph=g, rng_manager=rng, cooldown_steps=1)

    # Manually set cooldown
    engine._cooldowns['x'] = 1
    engine.tick_cooldown()
    assert 'x' not in engine._cooldowns


def test_tick_cooldown_decrements_but_keeps():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=4)
    engine = RuleEngine(graph=g, rng_manager=rng, cooldown_steps=2)
    engine._cooldowns['y'] = 2
    engine.tick_cooldown()
    assert engine._cooldowns.get('y') == 1


def test_embedding_skips_when_boundary_binding_missing():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=5)
    engine = RuleEngine(graph=g, rng_manager=rng)

    lhs = LHSPattern(nodes=[{"label": "B"}], edges=[], boundary_nodes=["B"])
    rhs = RHSAction(
        add_nodes=[
            {"label": "N1",
             "properties": {
                 "node_type": NodeType.HIDDEN,
                 "activation_function": "relu",
                 "attributes": {"output_size": 1}
             }}
        ]
    )
    emb = EmbeddingLogic()
    emb.connection_map = {("B", "in"): [("N1", 1.0)]}
    r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=emb)

    # Missing 'B' in bindings triggers continue path inside embedding
    assert engine.apply_rule(r, bindings={}) is True


def test_embedding_skips_missing_target_label():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=6)
    engine = RuleEngine(graph=g, rng_manager=rng)

    lhs = LHSPattern(nodes=[{"label": "B"}], edges=[], boundary_nodes=["B"])
    rhs = RHSAction(
        add_nodes=[
            {"label": "N1",
             "properties": {
                 "node_type": NodeType.HIDDEN,
                 "activation_function": "relu",
                 "attributes": {"output_size": 1}
             }}
        ]
    )
    emb = EmbeddingLogic()
    # Map to non-existent RHS label 'NX' to hit target_handle None path
    emb.connection_map = {("B", "in"): [("NX", 1.0)]}
    r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=emb)

    assert engine.apply_rule(r, bindings={"B": i}) is True


def test_stage_add_edges_missing_refs_skipped():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=7)
    engine = RuleEngine(graph=g, rng_manager=rng)

    lhs = LHSPattern(nodes=[{"label": "B"}], edges=[], boundary_nodes=["B"])
    rhs = RHSAction(
        add_nodes=[
            {"label": "N1",
             "properties": {
                 "node_type": NodeType.HIDDEN,
                 "activation_function": "relu",
                 "attributes": {"output_size": 1}
             }}
        ],
        add_edges=[
            {"source_label": "X", "target_label": "Y", "properties": {"weight": 0.1}}
        ]
    )
    r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=EmbeddingLogic())

    # Since X/Y not bound or created, edge staging is skipped
    assert engine.apply_rule(r, bindings={}) is True


def test_stage_add_edges_success_path():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=8)
    engine = RuleEngine(graph=g, rng_manager=rng)

    lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
    rhs = RHSAction(
        add_nodes=[
            {"label": "A",
             "properties": {
                 "node_type": NodeType.HIDDEN,
                 "activation_function": "relu",
                 "attributes": {"output_size": 1}
             }},
            {"label": "B",
             "properties": {
                 "node_type": NodeType.HIDDEN,
                 "activation_function": "relu",
                 "attributes": {"output_size": 1}
             }}
        ],
        add_edges=[
            {"source_label": "A", "target_label": "B", "properties": {"weight": 0.2}}
        ]
    )
    r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=EmbeddingLogic())
    assert engine.apply_rule(r, bindings={}) is True
    # Edge A->B should exist between the two new nodes
    new_nodes = [nid for nid in g.nodes]
    assert len(new_nodes) == 3
    # Find two non-input nodes
    non_inputs = [nid for nid in g.nodes if g.nodes[nid].node_type != NodeType.INPUT]
    assert (
        g.find_edge_by_endpoints(non_inputs[0], non_inputs[1]) is not None
        or g.find_edge_by_endpoints(non_inputs[1], non_inputs[0]) is not None
    )


def test_validation_fails_after_commit_triggers_undo(monkeypatch):
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=10)
    engine = RuleEngine(graph=g, rng_manager=rng)

    lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
    rhs = RHSAction(add_nodes=[
        {
            "label": "H",
            "properties": {
                "node_type": NodeType.HIDDEN,
                "activation_function": "relu",
                "attributes": {"output_size": 1},
            },
        }
    ])
    r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=EmbeddingLogic())

    # Monkeypatch validate to return False (after commit succeeds)
    def fake_validate(collect_errors=None, collect_warnings=None):
        if collect_errors is not None:
            collect_errors.append(object())
        return False

    # Track nodes count before and after
    before = len(g.nodes)
    monkeypatch.setattr(g, 'validate', fake_validate)
    applied = engine.apply_rule(r, bindings={})
    after = len(g.nodes)
    assert applied is False
    assert before == after


def test_embedding_out_direction_creates_reverse_edges():
    from ggnes.generation.rule_engine import RuleEngine

    g, i = gbase()
    rng = RNGManager(seed=2)
    engine = RuleEngine(graph=g, rng_manager=rng)

    lhs = LHSPattern(nodes=[{"label": "B"}], edges=[], boundary_nodes=["B"])
    rhs = RHSAction(add_nodes=[
        {
            "label": "N1",
            "properties": {
                "node_type": NodeType.HIDDEN,
                "activation_function": "relu",
                "attributes": {"output_size": 1},
            },
        }
    ])
    emb = EmbeddingLogic()
    emb.connection_map = {("B", "out"): [("N1", 1.0)]}
    r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=emb)

    assert engine.apply_rule(r, bindings={"B": i}) is True
    # Expect edge from new node to boundary i (reverse direction)
    new_ids = [nid for nid in g.nodes if nid != i]
    assert len(new_ids) == 1
    assert g.find_edge_by_endpoints(new_ids[0], i) is not None
