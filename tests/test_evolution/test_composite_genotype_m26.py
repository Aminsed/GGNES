import pytest

from ggnes.evolution.composite_genotype import (
    CompositeGenotype,
    G1Grammar,
    G2Policy,
    G3Hierarchy,
)
from ggnes.utils.uuid_provider import DeterministicUUIDProvider, UUIDProviderConfig


def make_provider(strict=False, freeze=False, float_precision=12, salt=None):
    return DeterministicUUIDProvider(
        UUIDProviderConfig(
            namespace="ggnes://uuid/v1",
            scheme_version=1,
            strict=strict,
            freeze=freeze,
            float_precision=float_precision,
            salt=salt,
        )
    )


def test_m26_uuid_determinism_same_inputs_same_uuid():
    provider = make_provider()
    g = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "11111111-1111-4111-8111-111111111111", "priority": 3, "probability": 0.5}]),
        g2=G2Policy(training_epochs=5, batch_size=32, learning_rate=0.01, parallel_execution=False, wl_iterations=2, failure_floor_threshold=0.4),
        g3=G3Hierarchy(modules={"Block": {"width": 64}}, attributes={"depth": 3}),
        provider=provider,
    )
    u1 = g.uuid()
    # Recreate with same content → identical UUID
    g2c = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "11111111-1111-4111-8111-111111111111", "probability": 0.5, "priority": 3}]),
        g2=G2Policy(training_epochs=5, batch_size=32, learning_rate=0.01, parallel_execution=False, wl_iterations=2, failure_floor_threshold=0.4),
        g3=G3Hierarchy(modules={"Block": {"width": 64}}, attributes={"depth": 3}),
        provider=provider,
    )
    u2 = g2c.uuid()
    assert u1 == u2


def test_m26_uuid_changes_with_content_changes():
    provider = make_provider()
    base = CompositeGenotype(provider=provider)
    u1 = base.uuid()
    base.mutate_field(["g2", "learning_rate"], 0.002)
    u2 = base.uuid()
    assert u1 != u2


def test_m26_freeze_forbids_mutation_after_uuid_when_enabled():
    provider = make_provider(freeze=True)
    cg = CompositeGenotype(provider=provider)
    _ = cg.uuid()
    with pytest.raises(ValueError):
        cg.mutate_field(["g2", "learning_rate"], 0.5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_epochs", 0),  # invalid
        ("batch_size", -1),
        ("learning_rate", 0.0),
        ("failure_floor_threshold", 2.0),
    ],
)
def test_m26_policy_validation(field, value):
    g2 = G2Policy()
    setattr(g2, field, value)
    cg = CompositeGenotype(g2=g2)
    with pytest.raises(ValueError):
        cg.validate()


def test_m26_g1_rules_validation():
    g1 = G1Grammar(rules=[{"priority": 1}])  # missing rule_id
    cg = CompositeGenotype(g1=g1)
    with pytest.raises(ValueError):
        cg.validate()

    # Non-dict entry
    g1b = G1Grammar(rules=[123])
    cgb = CompositeGenotype(g1=g1b)
    with pytest.raises(ValueError):
        cgb.validate()

    # Priority not int
    g1c = G1Grammar(rules=[{"rule_id": "11111111-1111-4111-8111-111111111111", "priority": "high"}])
    cgc = CompositeGenotype(g1=g1c)
    with pytest.raises(ValueError):
        cgc.validate()

    # probability must be in (0, 1]
    g1d = G1Grammar(rules=[{"rule_id": "11111111-1111-4111-8111-111111111111", "probability": 0.0}])
    cgd = CompositeGenotype(g1=g1d)
    with pytest.raises(ValueError):
        cgd.validate()

    # boundary ok = 1.0
    g1e = G1Grammar(rules=[{"rule_id": "11111111-1111-4111-8111-111111111111", "probability": 1.0}])
    cge = CompositeGenotype(g1=g1e)
    cge.validate()


def test_m26_g3_scalar_constraints():
    g3 = G3Hierarchy(modules={"M": {"p": object()}}, attributes={"k": object()})
    cg = CompositeGenotype(g3=g3)
    with pytest.raises(ValueError):
        cg.validate()

    # Non-str module key
    g3b = G3Hierarchy(modules={123: {"p": 1}})
    cgb = CompositeGenotype(g3=g3b)
    with pytest.raises(ValueError):
        cgb.validate()

    # Module params not dict
    g3c = G3Hierarchy(modules={"M": 5})
    cgc = CompositeGenotype(g3=g3c)
    with pytest.raises(ValueError):
        cgc.validate()

    # Param name not str
    g3d = G3Hierarchy(modules={"M": {123: 1}})
    cgd = CompositeGenotype(g3=g3d)
    with pytest.raises(ValueError):
        cgd.validate()

    # Attribute key not str
    g3e = G3Hierarchy(attributes={123: True})
    cge = CompositeGenotype(g3=g3e)
    with pytest.raises(ValueError):
        cge.validate()

    # Attribute value not scalar
    g3f = G3Hierarchy(attributes={"k": object()})
    cgf = CompositeGenotype(g3=g3f)
    with pytest.raises(ValueError):
        cgf.validate()


def test_m26_serialization_roundtrip_and_uuid_parity():
    provider = make_provider(float_precision=8)
    cg = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "22222222-2222-4222-8222-222222222222"}]),
        g2=G2Policy(training_epochs=7, batch_size=16, learning_rate=0.02, parallel_execution=True, wl_iterations=4, failure_floor_threshold=0.25),
        g3=G3Hierarchy(modules={"Attn": {"heads": 2, "dim": 32}}, attributes={"use_bias": True}),
        provider=provider,
    )
    uid = cg.uuid()
    data = cg.serialize()
    cg_r = CompositeGenotype.deserialize(data, provider=provider)
    uid_r = cg_r.uuid()
    assert uid == uid_r


def test_m26_provider_namespace_and_precision_affect_uuid():
    cg = CompositeGenotype()
    u_default = cg.uuid()
    cg2 = CompositeGenotype(provider=DeterministicUUIDProvider(UUIDProviderConfig(namespace="ggnes://uuid/v2")))
    u_ns = cg2.uuid()
    assert u_default != u_ns

    cg3 = CompositeGenotype(provider=DeterministicUUIDProvider(UUIDProviderConfig(float_precision=6)))
    u_prec = cg3.uuid()
    # Different precision on defaults should still differ deterministically
    assert u_default != u_prec


def test_m26_strict_mode_rejects_non_finite_inputs():
    provider = make_provider(strict=True)
    cg = CompositeGenotype(provider=provider)
    cg.g3.modules = {"M": {"p": float("inf")}}
    with pytest.raises(ValueError):
        cg.uuid()


def test_m26_policy_validation_types_and_bool():
    # learning_rate wrong type
    g2 = G2Policy()
    g2.learning_rate = "0.1"
    cg = CompositeGenotype(g2=g2)
    with pytest.raises(ValueError):
        cg.validate()

    # parallel_execution must be bool
    g2b = G2Policy()
    g2b.parallel_execution = "yes"
    cgb = CompositeGenotype(g2=g2b)
    with pytest.raises(ValueError):
        cgb.validate()


def test_m26_mutate_field_path_required():
    cg = CompositeGenotype()
    with pytest.raises(ValueError):
        cg.mutate_field([], 1)


def test_m26_deserialize_unsupported_schema_version():
    provider = make_provider()
    with pytest.raises(ValueError):
        CompositeGenotype.deserialize({"_schema": 2}, provider=provider)


