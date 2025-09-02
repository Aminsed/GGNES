from __future__ import annotations

import time
import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.network_gen import generate_network
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _axiom() -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': 8}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 4}})
    g.add_edge(i, o, {'weight': 0.1})
    return g


def _expansion_rule() -> Rule:
    lhs = LHSPattern(
        nodes=[
            {'label': 'I', 'match_criteria': {'node_type': NodeType.INPUT}},
            {'label': 'O', 'match_criteria': {'node_type': NodeType.OUTPUT}},
        ],
        edges=[{'source_label': 'I', 'target_label': 'O', 'match_criteria': {}}],
        boundary_nodes=[],
    )
    rhs = RHSAction(
        add_nodes=[
            {'label': 'H1', 'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'relu',
                'bias': 0.0,
                'attributes': {'output_size': 6},
            }},
            {'label': 'H2', 'properties': {
                'node_type': NodeType.HIDDEN,
                'activation_function': 'tanh',
                'bias': 0.0,
                'attributes': {'output_size': 6},
            }},
        ],
        add_edges=[
            {'source_label': 'I', 'target_label': 'H1', 'properties': {'weight': 0.2}},
            {'source_label': 'H1', 'target_label': 'H2', 'properties': {'weight': 0.3}},
            {'source_label': 'H2', 'target_label': 'O', 'properties': {'weight': 0.4}},
        ],
    )
    return Rule(
        rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=EmbeddingLogic(connection_map={}),
        metadata={'priority': 1, 'probability': 1.0, 'rule_type': 'ExpandIO'}
    )


def _cfg(overrides: dict | None = None) -> dict:
    cfg = {
        'max_iterations': 3,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_action': 'TERMINATE',
        'cooldown_iterations': 1,
        'max_match_time_ms': 50,
        'parallel_execution': True,
        'max_parallel_workers': 4,
        'parallel_batch_policy': 'MAX_INDEPENDENT_SET',
        'parallel_conflict_strategy': 'SKIP',
        'parallel_time_budget_ms': 5,  # intentionally small
        'parallel_memory_budget_mb': 8,
        'parallel_max_requeues': 1,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def test_parallel_budgets_acceptance_and_wallclock_smoke():
    # [T-par-08] Budgets accepted; execution completes quickly under small workload
    axiom = _axiom()
    rule = _expansion_rule()
    genotype = type('G', (), {'rules': [rule]})

    start = time.monotonic()
    graph, metrics = generate_network(genotype, axiom, _cfg(), RNGManager(8080))
    elapsed = (time.monotonic() - start)

    # Smoke wall-clock bound to catch accidental runaway; generous upper bound
    assert elapsed < 0.5

    # Metrics present and sensible
    assert metrics['iterations'] <= 3
    assert metrics['final_nodes'] >= 2
    assert metrics['final_edges'] >= 1


def test_parallel_determinism_with_budgets_and_workers():
    # [T-par-08] Determinism across worker counts and budget settings
    axiom = _axiom()
    rule = _expansion_rule()
    genotype = type('G', (), {'rules': [rule]})

    base_cfg = _cfg({'max_parallel_workers': 1, 'parallel_time_budget_ms': 10})
    fp_base = generate_network(genotype, axiom, base_cfg, RNGManager(999))[0].compute_fingerprint()

    for workers in (2, 4, 8):
        cfg = _cfg({'max_parallel_workers': workers, 'parallel_time_budget_ms': 10, 'parallel_memory_budget_mb': 16})
        fp = generate_network(genotype, axiom, cfg, RNGManager(999))[0].compute_fingerprint()
        assert fp == fp_base


