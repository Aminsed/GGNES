from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.matching import find_subgraph_matches
from ggnes.generation.network_gen import generate_network
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _build_mg_axiom() -> Graph:
    g = Graph(config={'multigraph': True})
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 2}})
    # Two parallel edges I->H
    g.add_edge(i, h, {'weight': 0.1})
    g.add_edge(i, h, {'weight': 0.2})
    # Connect H->O
    g.add_edge(h, o, {'weight': 0.3})
    return g


def test_parallel_multigraph_edge_label_binds_specific_instance():
    # [T-par-06] Multigraph: edge_label binds to a concrete edge instance among parallels
    g = _build_mg_axiom()
    lhs = {
        'nodes': [
            {'label': 'I', 'match_criteria': {'node_type': NodeType.INPUT}},
            {'label': 'H', 'match_criteria': {'node_type': NodeType.HIDDEN}},
        ],
        'edges': [
            {'source_label': 'I', 'target_label': 'H', 'edge_label': 'E', 'match_criteria': {}},
        ],
    }
    matches = list(find_subgraph_matches(g, lhs, timeout_ms=200))
    assert matches, "Must produce at least one binding"
    e = matches[0]['E']
    # Edge must be one of the parallel instances
    i_id = next(nid for nid, n in g.nodes.items() if n.node_type == NodeType.INPUT)
    h_id = next(nid for nid, n in g.nodes.items() if n.node_type == NodeType.HIDDEN)
    mg_edge_ids = {edge.edge_id for edge in g.find_edges_by_endpoints(i_id, h_id)}
    assert e.edge_id in mg_edge_ids


def _expand_rule() -> Rule:
    lhs = LHSPattern(
        nodes=[
            {'label': 'I', 'match_criteria': {'node_type': NodeType.INPUT}},
            {'label': 'H', 'match_criteria': {'node_type': NodeType.HIDDEN}},
            {'label': 'O', 'match_criteria': {'node_type': NodeType.OUTPUT}},
        ],
        edges=[
            {'source_label': 'I', 'target_label': 'H', 'match_criteria': {}},
            {'source_label': 'H', 'target_label': 'O', 'match_criteria': {}},
        ],
        boundary_nodes=[],
    )
    rhs = RHSAction(
        add_nodes=[
            {'label': 'X', 'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'relu',
                'bias': 0.0,
                'attributes': {'output_size': 4},
            }},
        ],
        add_edges=[
            {'source_label': 'I', 'target_label': 'X', 'properties': {'weight': 0.4}},
            {'source_label': 'X', 'target_label': 'O', 'properties': {'weight': 0.5}},
        ],
    )
    return Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}), metadata={'priority': 1, 'rule_type': 'Expand'})


def test_parallel_multigraph_equivalence_serial_vs_parallel():
    # [T-par-06] Under parallel flag, multigraph behavior matches serial fingerprint deterministically
    axiom = _build_mg_axiom()
    rule = _expand_rule()
    genotype = type('G', (), {'rules': [rule]})

    serial_fp = generate_network(genotype, axiom, {'max_iterations': 1, 'parallel_execution': False}, RNGManager(321))[0].compute_fingerprint()
    parallel_fp = generate_network(
        genotype, axiom,
        {'max_iterations': 1, 'parallel_execution': True, 'max_parallel_workers': 4},
        RNGManager(321)
    )[0].compute_fingerprint()
    assert parallel_fp == serial_fp


