from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.network_gen import generate_network
from ggnes.rules.rule import Direction, Distribution, EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _build_two_hidden_axiom() -> Graph:
    graph = Graph()
    i = graph.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 8},
    })
    h1 = graph.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'relu',
        'bias': 0.0,
        'attributes': {'output_size': 6},
    })
    h2 = graph.add_node({
        'node_type': NodeType.HIDDEN,
        'activation_function': 'tanh',
        'bias': 0.0,
        'attributes': {'output_size': 6},
    })
    o = graph.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 4},
    })
    assert graph.add_edge(i, h1, {'weight': 0.1}) is not None
    assert graph.add_edge(h1, o, {'weight': 0.2}) is not None
    assert graph.add_edge(i, h2, {'weight': 0.1}) is not None
    assert graph.add_edge(h2, o, {'weight': 0.2}) is not None
    return graph


def _expand_hidden_rule() -> Rule:
    lhs = LHSPattern(
        nodes=[
            {'label': 'I', 'match_criteria': {'node_type': NodeType.INPUT}},
            {'label': 'X', 'match_criteria': {'node_type': NodeType.HIDDEN}},
            {'label': 'O', 'match_criteria': {'node_type': NodeType.OUTPUT}},
        ],
        edges=[
            {'source_label': 'I', 'target_label': 'X', 'match_criteria': {}},
            {'source_label': 'X', 'target_label': 'O', 'match_criteria': {}},
        ],
        boundary_nodes=[],
    )
    rhs = RHSAction(
        add_nodes=[
            {
                'label': 'N',
                'properties': {
                    'node_type': NodeType.HIDDEN,
                    'activation_function': 'relu',
                    'bias': 0.0,
                    'attributes': {'output_size': 5},
                },
            }
        ],
        add_edges=[
            {'source_label': 'I', 'target_label': 'N', 'properties': {'weight': 0.33}},
            {'source_label': 'N', 'target_label': 'O', 'properties': {'weight': 0.44}},
        ],
    )
    return Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}), metadata={'priority': 2, 'rule_type': 'ExpandHidden'})


def _embedding_delete_hidden_rule() -> Rule:
    lhs = LHSPattern(
        nodes=[{'label': 'X', 'match_criteria': {'node_type': NodeType.HIDDEN}}],
        edges=[],
        boundary_nodes=['X'],
    )
    rhs = RHSAction(
        delete_nodes=['X'],
        add_nodes=[
            {'label': 'P', 'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'relu',
                'bias': 0.0,
                'attributes': {'output_size': 6},
            }},
            {'label': 'Q', 'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'tanh',
                'bias': 0.0,
                'attributes': {'output_size': 6},
            }},
        ],
    )
    emb = EmbeddingLogic(
        connection_map={
            ('X', Direction.IN): [('P', Distribution.COPY_ALL)],
            ('X', Direction.OUT): [('P', Distribution.CONNECT_SINGLE)],
        },
        excess_connection_handling='WARNING',
        unknown_direction_handling='ERROR',
    )
    return Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb, metadata={'priority': 1, 'rule_type': 'EmbedDelete'})


def _cfg(overrides: dict | None = None) -> dict:
    cfg = {
        'max_iterations': 2,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_action': 'TERMINATE',
        'cooldown_iterations': 3,
        'max_match_time_ms': 250,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def test_parallel_multiple_non_overlapping_matches_across_iterations_equivalence():
    # [T-par-02]/[T-par-03] In fallback, even with batch policies, results equal serial after two iterations
    axiom = _build_two_hidden_axiom()
    rule = _expand_hidden_rule()
    genotype = type('G', (), {'rules': [rule]})

    serial_rng = RNGManager(123)
    serial_cfg = _cfg({'parallel_execution': False})
    g1, m1 = generate_network(genotype, axiom, serial_cfg, serial_rng)

    for policy in ('MAX_INDEPENDENT_SET', 'FIXED_SIZE', 'PRIORITY_CAP'):
        par_rng = RNGManager(123)
        par_cfg = _cfg({'parallel_execution': True, 'max_parallel_workers': 4, 'parallel_batch_policy': policy})
        g2, m2 = generate_network(genotype, axiom, par_cfg, par_rng)
        assert g1.compute_fingerprint() == g2.compute_fingerprint()
        assert m1['iterations'] == m2['iterations']


def test_parallel_conflict_strategies_acceptance_equivalence():
    # [T-par-04] SKIP, REQUEUE, SPECULATIVE_MERGE configs do not alter serial-equivalent outcome in fallback
    axiom = _build_two_hidden_axiom()
    rule = _expand_hidden_rule()
    genotype = type('G', (), {'rules': [rule]})

    baseline = generate_network(genotype, axiom, _cfg({'parallel_execution': False}), RNGManager(7))[0].compute_fingerprint()

    for strategy in ('SKIP', 'REQUEUE', 'SPECULATIVE_MERGE'):
        fp = generate_network(
            genotype,
            axiom,
            _cfg({'parallel_execution': True, 'max_parallel_workers': 2, 'parallel_conflict_strategy': strategy, 'parallel_max_requeues': 2}),
            RNGManager(7),
        )[0].compute_fingerprint()
        assert fp == baseline


def test_parallel_embedding_determinism_matches_serial():
    # [T-par-07] Embedding behavior deterministic with parallel flag enabled (fallback semantics)
    axiom = _build_two_hidden_axiom()
    # Ensure at least one hidden exists with external connections
    rule = _embedding_delete_hidden_rule()
    genotype = type('G', (), {'rules': [rule]})

    fp_serial = generate_network(genotype, axiom, _cfg({'parallel_execution': False}), RNGManager(55))[0].compute_fingerprint()
    fp_parallel = generate_network(genotype, axiom, _cfg({'parallel_execution': True, 'max_parallel_workers': 3}), RNGManager(55))[0].compute_fingerprint()
    assert fp_parallel == fp_serial


def test_parallel_oscillation_and_cooldown_respected_under_flag():
    # [T-par-05] With oscillation configured, parallel flag should not bypass cooldown/termination
    axiom = _build_two_hidden_axiom()
    # Use a rule that re-adds same structure to potentially oscillate under repeated selection
    rule = _expand_hidden_rule()
    genotype = type('G', (), {'rules': [rule]})

    cfg = _cfg({'parallel_execution': True, 'max_parallel_workers': 2, 'oscillation_action': 'TERMINATE', 'cooldown_iterations': 1})
    graph, metrics = generate_network(genotype, axiom, cfg, RNGManager(99))
    assert metrics['iterations'] <= 2


