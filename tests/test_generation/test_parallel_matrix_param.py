from __future__ import annotations

import uuid

import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.network_gen import generate_network
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _axiom() -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 6}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': 'relu', 'attributes': {'output_size': 6}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 3}})
    g.add_edge(i, h, {'weight': 0.2})
    g.add_edge(h, o, {'weight': 0.4})
    return g


def _rule() -> Rule:
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
        add_nodes=[{'label': 'X', 'properties': {'node_type': NodeType.HIDDEN, 'activation_function': 'tanh', 'bias': 0.0, 'attributes': {'output_size': 6}}}],
        add_edges=[
            {'source_label': 'I', 'target_label': 'X', 'properties': {'weight': 0.3}},
            {'source_label': 'X', 'target_label': 'O', 'properties': {'weight': 0.5}},
        ],
    )
    return Rule(
        rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}),
        metadata={'priority': 1, 'probability': 0.5, 'rule_type': 'Expand'}
    )


@pytest.mark.parametrize('policy', ['MAX_INDEPENDENT_SET', 'FIXED_SIZE', 'PRIORITY_CAP'])
@pytest.mark.parametrize('strategy', ['SKIP', 'REQUEUE', 'SPECULATIVE_MERGE'])
@pytest.mark.parametrize('workers', [1, 2, 4])
@pytest.mark.parametrize('seed', [3, 77])
def test_parallel_matrix_equivalence(policy, strategy, workers, seed):
    axiom = _axiom()
    genotype = type('G', (), {'rules': [_rule()]})

    serial_cfg = {
        'max_iterations': 2,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'parallel_execution': False,
    }
    parallel_cfg = {
        'max_iterations': 2,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'parallel_execution': True,
        'max_parallel_workers': workers,
        'parallel_batch_policy': policy,
        'parallel_conflict_strategy': strategy,
        'parallel_time_budget_ms': 10,
        'parallel_memory_budget_mb': 16,
        'parallel_max_requeues': 2,
    }

    fp_serial = generate_network(genotype, axiom, serial_cfg, RNGManager(seed))[0].compute_fingerprint()
    fp_parallel = generate_network(genotype, axiom, parallel_cfg, RNGManager(seed))[0].compute_fingerprint()
    assert fp_parallel == fp_serial


