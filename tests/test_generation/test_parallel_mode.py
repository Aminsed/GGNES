from __future__ import annotations

import uuid

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.network_gen import generate_network
from ggnes.rules.rule import EmbeddingLogic, LHSPattern, RHSAction, Rule
from ggnes.utils.rng_manager import RNGManager


def _build_axiom() -> Graph:
    graph = Graph()
    input_id = graph.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 8},
    })
    output_id = graph.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 4},
    })
    assert graph.add_edge(input_id, output_id, {'weight': 0.1}) is not None
    return graph


def _build_simple_expand_rule() -> Rule:
    # Match the Input->Output edge
    lhs = LHSPattern(
        nodes=[
            {'label': 'I', 'match_criteria': {'node_type': NodeType.INPUT}},
            {'label': 'O', 'match_criteria': {'node_type': NodeType.OUTPUT}},
        ],
        edges=[
            {'source_label': 'I', 'target_label': 'O', 'match_criteria': {}},
        ],
        boundary_nodes=[],
    )

    # Add a hidden node H and connect I->H and H->O
    rhs = RHSAction(
        add_nodes=[
            {
                'label': 'H',
                'properties': {
                    'node_type': NodeType.HIDDEN,
                    'activation_function': 'relu',
                    'bias': 0.0,
                    'attributes': {'output_size': 6},
                },
            }
        ],
        add_edges=[
            {
                'source_label': 'I',
                'target_label': 'H',
                'properties': {'weight': 0.3, 'attributes': {}},
            },
            {
                'source_label': 'H',
                'target_label': 'O',
                'properties': {'weight': 0.5, 'attributes': {}},
            },
        ],
    )

    embedding = EmbeddingLogic(connection_map={})
    return Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=embedding, metadata={'priority': 1, 'rule_type': 'Expand'})


def _gen_config(overrides: dict | None = None) -> dict:
    cfg = {
        'max_iterations': 1,
        'selection_strategy': 'PRIORITY_THEN_PROBABILITY_THEN_ORDER',
        'oscillation_action': 'TERMINATE',
        'cooldown_iterations': 3,
        'max_match_time_ms': 250,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _fingerprint_with_cfg(genotype, axiom: Graph, cfg: dict, seed: int = 1234) -> str:
    rng = RNGManager(seed)
    graph, _ = generate_network(genotype, axiom, cfg, rng)
    return graph.compute_fingerprint()


def test_parallel_equivalence_across_workers():
    # [T-par-01] Determinism: parallel vs serial equality for same seed
    axiom = _build_axiom()
    rule = _build_simple_expand_rule()
    genotype = type('G', (), {'rules': [rule]})

    serial_cfg = _gen_config({'parallel_execution': False})
    baseline = _fingerprint_with_cfg(genotype, axiom, serial_cfg, seed=42)

    for workers in (1, 2, 4):
        par_cfg = _gen_config({'parallel_execution': True, 'max_parallel_workers': workers})
        fp = _fingerprint_with_cfg(genotype, axiom, par_cfg, seed=42)
        assert fp == baseline


def test_parallel_policy_keys_are_accepted_and_noop_but_deterministic():
    # [T-par-03] Batch policies provided; behavior equals serial (feature-flagged noop)
    # [T-par-04] Conflict strategies accepted without altering results in current serial fallback
    axiom = _build_axiom()
    rule = _build_simple_expand_rule()
    genotype = type('G', (), {'rules': [rule]})

    serial_cfg = _gen_config({'parallel_execution': False})
    baseline = _fingerprint_with_cfg(genotype, axiom, serial_cfg, seed=7)

    par_cfg = _gen_config({
        'parallel_execution': True,
        'max_parallel_workers': 4,
        'parallel_batch_policy': 'MAX_INDEPENDENT_SET',
        'parallel_conflict_strategy': 'SKIP',
        'parallel_time_budget_ms': 10_000,
        'parallel_memory_budget_mb': 1024,
        'parallel_max_requeues': 1,
    })
    fp = _fingerprint_with_cfg(genotype, axiom, par_cfg, seed=7)
    assert fp == baseline


def test_parallel_metrics_and_iterations_match_serial():
    # [T-par-09] With parallel off vs on, metrics equality is preserved in fallback
    axiom = _build_axiom()
    rule = _build_simple_expand_rule()
    genotype = type('G', (), {'rules': [rule]})

    rng_serial = RNGManager(100)
    serial_cfg = _gen_config({'parallel_execution': False})
    g1, m1 = generate_network(genotype, axiom, serial_cfg, rng_serial)

    rng_par = RNGManager(100)
    par_cfg = _gen_config({'parallel_execution': True, 'max_parallel_workers': 2})
    g2, m2 = generate_network(genotype, axiom, par_cfg, rng_par)

    assert g1.compute_fingerprint() == g2.compute_fingerprint()
    assert m1['iterations'] == m2['iterations']
    assert m1['final_nodes'] == m2['final_nodes']
    assert m1['final_edges'] == m2['final_edges']


def test_parallel_regression_guard_serial_behavior_unaffected():
    # [T-par-08] With parallel_execution=False, ensure behavior unchanged when other parallel keys present
    axiom = _build_axiom()
    rule = _build_simple_expand_rule()
    genotype = type('G', (), {'rules': [rule]})

    cfg_base = _gen_config({'parallel_execution': False})
    fp_base = _fingerprint_with_cfg(genotype, axiom, cfg_base, seed=99)

    cfg_with_noise = _gen_config({
        'parallel_execution': False,
        'max_parallel_workers': 8,
        'parallel_batch_policy': 'FIXED_SIZE',
        'parallel_conflict_strategy': 'REQUEUE',
        'parallel_max_requeues': 3,
    })
    fp_noise = _fingerprint_with_cfg(genotype, axiom, cfg_with_noise, seed=99)
    assert fp_base == fp_noise


def test_parallel_flag_works_with_no_rules_and_quiescence():
    # [T-par-05] Cooldown/oscillation respected under parallel batches (no rules → quiescence)
    # Here we just ensure the flag does not break quiescence semantics.
    axiom = _build_axiom()
    genotype = type('G', (), {'rules': []})
    par_cfg = _gen_config({'parallel_execution': True, 'max_parallel_workers': 2})
    rng = RNGManager(11)
    graph, metrics = generate_network(genotype, axiom, par_cfg, rng)
    # Quiescent: no change
    assert metrics['iterations'] == 0
    assert metrics['final_nodes'] == len(axiom.nodes)
    assert metrics['final_edges'] == sum(len(n.edges_out) for n in axiom.nodes.values())


