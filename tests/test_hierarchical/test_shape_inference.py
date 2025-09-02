from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.shape_inference import plan_projections


def test_shape_inference_projection_plan_deterministic_and_correct():
    spec = ModuleSpec(
        name="S",
        version=1,
        parameters=[ParameterSpec("model_dim", default=16)],
        ports=[PortSpec("in", 16), PortSpec("out", 16)],
        invariants=["out.size == model_dim"],
    )
    env = spec.validate_and_bind_params({})
    plan1 = plan_projections(spec, env, incoming_edges=[(1, 8), (2, 32)], aggregation='sum')
    plan2 = plan_projections(spec, env, incoming_edges=[(1, 8), (2, 32)], aggregation='sum')
    assert plan1 == plan2
    assert all(sz == 16 for _, sz in plan1.inputs)
    assert plan1.post_aggregation_size == 16

    # concat grows linearly with fan-in deterministically
    plan3 = plan_projections(spec, env, incoming_edges=[(1, 8), (2, 32)], aggregation='concat')
    assert plan3.post_aggregation_size == 32


