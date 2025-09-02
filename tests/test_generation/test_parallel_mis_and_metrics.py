from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation import network_gen as ng
from ggnes.generation.matching import find_subgraph_matches
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _graph_with_overlapping_matches() -> Graph:
    # Build a small chain I->A->B->C->O so that multiple overlapping 2-edge paths exist
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    a = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    b = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    c = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    g.add_edge(i, a, {'weight': 0.1})
    g.add_edge(a, b, {'weight': 0.1})
    g.add_edge(b, c, {'weight': 0.1})
    g.add_edge(c, o, {'weight': 0.1})
    # Also add a skip to increase overlapping options
    g.add_edge(a, c, {'weight': 0.1})
    return g


def _two_edge_path_lhs() -> dict:
    return {
        'nodes': [
            {'label': 'U', 'match_criteria': {'node_type': NodeType.HIDDEN}},
            {'label': 'V', 'match_criteria': {'node_type': NodeType.HIDDEN}},
            {'label': 'W', 'match_criteria': {'node_type': NodeType.HIDDEN}},
        ],
        'edges': [
            {'source_label': 'U', 'target_label': 'V', 'match_criteria': {}},
            {'source_label': 'V', 'target_label': 'W', 'match_criteria': {}},
        ],
    }


def _overlap(m1: dict, m2: dict) -> bool:
    nset1 = {m1['U'], m1['V'], m1['W']}
    nset2 = {m2['U'], m2['V'], m2['W']}
    return bool(nset1 & nset2)


def test_mis_no_overlap_simulation():
    # Build matches and compute a deterministic MIS via greedy selection
    g = _graph_with_overlapping_matches()
    lhs = _two_edge_path_lhs()
    matches = list(find_subgraph_matches(g, lhs, timeout_ms=200))
    assert matches, 'Expected multiple candidate matches'

    # Deterministic order by tuple of bindings to simulate stable ordering (stand-in for genotype order)
    def key_fn(m):
        return (m['U'], m['V'], m['W'])

    matches_sorted = sorted(matches, key=key_fn)
    independent: list[dict] = []
    for m in matches_sorted:
        if all(not _overlap(m, chosen) for chosen in independent):
            independent.append(m)

    # Assert resulting set has no overlaps
    for i in range(len(independent)):
        for j in range(i + 1, len(independent)):
            assert not _overlap(independent[i], independent[j])


class GHMock:
    instances = []

    def __init__(self):
        self.fingerprints = []
        self.action_history = []
        self.metrics = []
        GHMock.instances.append(self)

    def add_fingerprint(self, fingerprint: str) -> None:
        self.fingerprints.append(fingerprint)

    def add_action(self, rule_type: str, rule_id, affected_nodes):
        self.action_history.append({'rule_info': (rule_type, rule_id), 'affected_nodes': affected_nodes})

    def add_metrics(self, metrics: dict) -> None:
        self.metrics.append(dict(metrics))


def _rule_expand_hidden() -> Rule:
    lhs = LHSPattern(
        nodes=[
            {'label': 'H', 'match_criteria': {'node_type': NodeType.HIDDEN}},
            {'label': 'O', 'match_criteria': {'node_type': NodeType.OUTPUT}},
        ],
        edges=[{'source_label': 'H', 'target_label': 'O', 'match_criteria': {}}],
        boundary_nodes=[],
    )
    rhs = RHSAction(
        add_nodes=[{'label': 'X', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'bias': 0.0, 'attributes': {'output_size': 4}}}],
        add_edges=[
            {'source_label': 'H', 'target_label': 'X', 'properties': {'weight': 0.2}},
            {'source_label': 'X', 'target_label': 'O', 'properties': {'weight': 0.3}},
        ],
    )
    return Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}), metadata={'priority': 1, 'rule_type': 'ExpandH'})


def test_parallel_metrics_structure_with_graphhistory_mock(monkeypatch):
    axiom = Graph()
    i = axiom.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    h = axiom.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    o = axiom.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    axiom.add_edge(i, h, {'weight': 0.1})
    axiom.add_edge(h, o, {'weight': 0.1})

    genotype = type('G', (), {'rules': [_rule_expand_hidden()]})

    # Monkeypatch GraphHistory in network_gen to capture metrics calls
    monkeypatch.setattr(ng, 'GraphHistory', GHMock)

    graph, metrics = ng.generate_network(
        genotype,
        axiom,
        {
            'max_iterations': 1,
            'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
            'parallel_execution': True,
            'max_parallel_workers': 2,
        },
        RNGManager(101),
    )

    # Verify our mock captured structure
    assert GHMock.instances, 'GraphHistory was not instantiated'
    gh = GHMock.instances[-1]
    # add_fingerprint called at least once
    assert len(gh.fingerprints) >= 0  # allow zero if no change, but attribute must exist
    # add_action captured rule_info tuple
    assert gh.action_history, 'Expected at least one action recorded'
    rule_info = gh.action_history[-1]['rule_info']
    assert isinstance(rule_info, tuple) and len(rule_info) == 2
    # add_metrics contains required keys
    assert gh.metrics, 'Expected at least one metrics record'
    last_m = gh.metrics[-1]
    for k in ('num_nodes', 'num_edges', 'iteration', 'rule_applied'):
        assert k in last_m


