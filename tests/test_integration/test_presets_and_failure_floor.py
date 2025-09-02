import pytest


def test_presets_exist_and_have_expected_keys():
    # Import from top-level package for convenience as in guide examples
    import ggnes as g

    assert isinstance(g.PRESET_MINIMAL, dict)
    assert isinstance(g.PRESET_STANDARD, dict)
    assert isinstance(g.PRESET_RESEARCH, dict)

    # Spot-check required keys per guide §12.2
    for preset in (g.PRESET_MINIMAL, g.PRESET_STANDARD, g.PRESET_RESEARCH):
        assert 'population_size' in preset
        assert 'max_iterations' in preset
        assert 'max_rules_per_genotype' in preset
        assert 'repair_strategy' in preset
        assert 'training_epochs' in preset

    # Research preset enables parallel_execution True per spec
    assert g.PRESET_RESEARCH.get('parallel_execution') is True


def test_failure_floor_policy_relaxes_config_when_threshold_exceeded():
    from ggnes.utils.failure_floor import apply_failure_floor_policy

    # Config with explicit threshold and iterations
    config = {
        'max_iterations': 50,
        'repair_strategy': 'MINIMAL_CHANGE',
        'failure_floor_threshold': 0.5,
    }

    # 3/4 failures -> 75% > 50% threshold
    results = [
        {'fitness': float('-inf')},
        {'fitness': float('-inf')},
        {'fitness': float('-inf')},
        {'fitness': 0.0},
    ]

    apply_failure_floor_policy(results, config)

    # Max iterations reduced by 20%, floored to int
    assert config['max_iterations'] == 40
    # Strategy set to AGGRESSIVE per spec
    assert config['repair_strategy'] == 'AGGRESSIVE'


def test_failure_floor_policy_noop_when_no_results_or_below_threshold():
    from ggnes.utils.failure_floor import apply_failure_floor_policy

    base_config = {
        'max_iterations': 100,
        'repair_strategy': 'MINIMAL_CHANGE',
        'failure_floor_threshold': 0.75,
    }

    # No results -> noop
    config = dict(base_config)
    apply_failure_floor_policy([], config)
    assert config == base_config

    # Below threshold (2/8 = 25% < 75%) -> noop
    config = dict(base_config)
    results = [{'fitness': 1.0}] * 6 + [{'fitness': float('-inf')}] * 2
    apply_failure_floor_policy(results, config)
    assert config == base_config


def test_minimal_axiom_translation_roundtrip():
    # Use the minimal axiom from project guide example (§13.1)
    # We reconstruct it using public APIs to avoid depending on guide code directly.
    from ggnes.core.graph import Graph
    from ggnes.core.node import NodeType
    from ggnes.translation import to_pytorch_model

    graph = Graph()
    input_id = graph.add_node({
        'node_type': NodeType.INPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 10},
    })
    output_id = graph.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'bias': 0.0,
        'attributes': {'output_size': 5},
    })
    edge_id = graph.add_edge(input_id, output_id, {'weight': 0.1, 'enabled': True, 'attributes': {}})
    assert edge_id is not None

    # Translate and perform a forward pass
    torch = pytest.importorskip('torch')
    model = to_pytorch_model(graph)
    x = torch.randn(4, 10)
    y = model(x, reset_states=True)
    assert y.shape == (4, 5)


