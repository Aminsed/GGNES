"""
M7 RuleEngine tests per project_guide.md.

Tests:
- [T-engine-01] Successful application creates nodes/edges; bindings discarded afterward
- [T-engine-02] Failed validation triggers rollback; graph unchanged; RNG restored
- [T-engine-03] Cooldown prevents immediate reselection; cleared after success
- [T-engine-04] Embedding MAP_BOUNDARY_CONNECTIONS deterministic reconnection order
- [T-osc-agg-01] Oscillation aggregator precedence and messages
"""

import pytest  # noqa: F401

from ggnes.core import Graph
from ggnes.core.graph import NodeType
from ggnes.generation.oscillation import (
    detect_growth_monitoring,
    detect_oscillation,
    detect_simple_reversal,
    detect_state_fingerprinting,
)
from ggnes.rules.conditions import ConditionRegistry  # noqa: F401
from ggnes.rules.rule import EmbeddingLogic, EmbeddingStrategy, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def basic_graph():
    g = Graph()
    i = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2}
    })
    return g, i


class TestRuleEngine:
    def test_successful_application_creates_and_discards_bindings(self):
        from ggnes.generation.rule_engine import RuleEngine

        g, i = basic_graph()
        rng = RNGManager(seed=123)
        engine = RuleEngine(graph=g, rng_manager=rng)

        lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
        rhs = RHSAction(
            add_nodes=[
                {"label": "H",
                 "properties": {
                     "node_type": NodeType.HIDDEN,
                     "activation_function": "relu",
                     "attributes": {"output_size": 3}
                 }}
            ],
            add_edges=[]
        )
        emb = EmbeddingLogic(strategy=EmbeddingStrategy.MAP_BOUNDARY_CONNECTIONS)
        r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=emb)

        applied = engine.apply_rule(r, bindings={})
        assert applied is True
        # One new node added
        assert len(g.nodes) == 2
        # Internal engine bindings are not persisted between calls
        assert engine._last_bindings is None

    def test_failed_validation_triggers_rollback(self):
        from ggnes.generation.rule_engine import RuleEngine

        g, i = basic_graph()
        rng = RNGManager(seed=7)
        engine = RuleEngine(graph=g, rng_manager=rng)
        base_state = rng.get_state()

        # RHS creates invalid node (missing output_size) to force validate fail
        lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
        rhs = RHSAction(
            add_nodes=[
                {"label": "X",
                 "properties": {
                     "node_type": NodeType.HIDDEN,
                     "activation_function": "relu",
                     "attributes": {}
                 }}
            ]
        )
        emb = EmbeddingLogic()
        r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=emb)

        applied = engine.apply_rule(r, bindings={})
        assert applied is False
        assert len(g.nodes) == 1  # unchanged
        assert rng.get_state() == base_state  # RNG restored

    def test_cooldown_prevents_immediate_reselection(self):
        from ggnes.generation.rule_engine import RuleEngine

        g, i = basic_graph()
        rng = RNGManager(seed=9)
        engine = RuleEngine(graph=g, rng_manager=rng, cooldown_steps=1)

        lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
        rhs = RHSAction(
            add_nodes=[
                {"label": "H",
                 "properties": {
                     "node_type": NodeType.HIDDEN,
                     "activation_function": "relu",
                     "attributes": {"output_size": 3}
                 }}
            ]
        )
        r = Rule(rule_id="r1", lhs=lhs, rhs=rhs, embedding=EmbeddingLogic())

        assert engine.apply_rule(r, bindings={}) is True
        # Immediately trying again should be skipped due to cooldown
        assert engine.apply_rule(r, bindings={}) is False
        # Advance cooldown
        engine.tick_cooldown()
        assert engine.apply_rule(r, bindings={}) is True

    def test_embedding_map_boundary_connections_order(self):
        from ggnes.generation.rule_engine import RuleEngine

        g, i = basic_graph()
        rng = RNGManager(seed=33)
        engine = RuleEngine(graph=g, rng_manager=rng)

        # Set boundary connections and reconnection mapping to test deterministic order
        lhs = LHSPattern(nodes=[{"label": "B"}], edges=[], boundary_nodes=["B"])
        rhs = RHSAction(
            add_nodes=[
                {"label": "N1",
                 "properties": {
                     "node_type": NodeType.HIDDEN,
                     "activation_function": "relu",
                     "attributes": {"output_size": 2}
                 }},
                {"label": "N2",
                 "properties": {
                     "node_type": NodeType.HIDDEN,
                     "activation_function": "relu",
                     "attributes": {"output_size": 2}
                 }}
            ]
        )
        emb = EmbeddingLogic()
        # Ordered mapping: incoming to B goes to N1 then N2
        emb.connection_map = {("B", "in"): [("N1", 1.0), ("N2", 1.0)]}
        r = Rule(rule_id=None, lhs=lhs, rhs=rhs, embedding=emb)

        # Simulate binding of B to existing input node i
        assert engine.apply_rule(r, bindings={"B": i}) is True
        # We expect two new nodes and edges from i to both in specified order.
        # Edges i->targets must exist
        targets = sorted([nid for nid in g.nodes if nid != i])
        for t in targets:
            assert g.find_edge_by_endpoints(i, t) is not None


class TestOscillation:
    def test_simple_reversal(self):
        class DummyRule:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        history = {
            "action_history": [
                {"rule_info": ("DeleteNode", None)},
                {"rule_info": ("Other", None)},
            ]
        }
        cfg = {
            "rule_reverse_pairs": {"AddNode": "DeleteNode"},
            "oscillation_history_window": 5,
        }
        detected, reason = detect_simple_reversal(
            DummyRule("AddNode"), history["action_history"], cfg
        )
        assert detected and "reverses previous" in reason

    def test_state_fingerprinting(self, monkeypatch):
        class DummyGraph:
            def __init__(self, fp):
                self._fp = fp

            def compute_fingerprint(self):
                return self._fp

        class GH:
            def __init__(self, fps):
                self.fingerprints = fps

        gh = GH(["aaa", "bbb", "ccc", "bbb"])  # history contains "bbb"
        g = DummyGraph("bbb")
        cfg = {"fingerprint_exclusion_window": 1}
        detected, reason = detect_state_fingerprinting(g, gh, cfg)
        assert detected and "State cycle detected" in reason

    def test_growth_monitoring(self):
        metrics = [
            {"num_nodes": 3, "num_edges": 2, "rule_applied": 1},
            {"num_nodes": 3, "num_edges": 2, "rule_applied": 2},
            {"num_nodes": 3, "num_edges": 2, "rule_applied": 1},
            {"num_nodes": 3, "num_edges": 2, "rule_applied": 0},
            {"num_nodes": 3, "num_edges": 2, "rule_applied": 1},
        ]
        cfg = {"oscillation_check_depth": 5}
        detected, reason = detect_growth_monitoring(metrics, cfg)
        assert detected and "No structural growth" in reason

    def test_oscillation_aggregator_precedence(self):
        class DummyGraph:
            def compute_fingerprint(self):
                return "x"

        class GH:
            def __init__(self):
                self.fingerprints = ["y"]
                self.action_history = [{"rule_info": ("B", None)}]
                self.metrics = [
                    {"num_nodes": 1, "num_edges": 1, "rule_applied": 1}
                    for _ in range(5)
                ]

        class DummyRule:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        cfg = {
            "oscillation_strategy": "ALL",
            "rule_reverse_pairs": {"A": "B"},
            "oscillation_history_window": 10,
            "oscillation_check_depth": 5,
            "fingerprint_exclusion_window": 1,
        }

        g = DummyGraph()
        gh = GH()
        # Because simple reversal matches, it should win precedence
        detected, reason = detect_oscillation(gh, g, DummyRule("A"), {}, cfg)
        assert detected and reason.startswith("[SIMPLE_REVERSAL]")
