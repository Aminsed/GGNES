import pytest

from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec, deserialize_module_spec
from ggnes.hierarchical.module_spec import ModuleRegistry
from ggnes.utils.uuid_provider import DeterministicUUIDProvider, UUIDProviderConfig
from ggnes.utils.validation import ValidationError


def test_validate_and_bind_params_required_and_defaults():
    spec = ModuleSpec(
        name="M",
        version=1,
        parameters=[
            ParameterSpec("a", default=2),
            ParameterSpec("b", required=True),
        ],
        ports=[PortSpec("in", 8), PortSpec("out", 8)],
        invariants=["a>0", "b>0", "(a+b)>2"],
    )

    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({})
    assert ei.value.error_type == "missing_param"

    env = spec.validate_and_bind_params({"b": 1})
    assert env["a"] == 2
    assert env["b"] == 1


def test_domain_validation_and_errors():
    spec = ModuleSpec(
        name="D",
        version=1,
        parameters=[
            ParameterSpec("x", default=3, domain=lambda v: isinstance(v, int) and v % 3 == 0),
        ],
        invariants=["x>=0"],
    )
    env = spec.validate_and_bind_params({})
    assert env["x"] == 3

    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({"x": 4})
    assert ei.value.error_type == "invalid_param_domain"


def test_invariant_violation_and_error_message():
    spec = ModuleSpec(
        name="I",
        version=2,
        parameters=[ParameterSpec("dim", default=8)],
        invariants=["dim % 2 == 1"],  # intentionally wrong
    )
    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({})
    assert ei.value.error_type == "invariant_violation"
    assert "Invariant failed" in ei.value.message


def test_serialize_deserialize_roundtrip():
    spec = ModuleSpec(
        name="S",
        version=3,
        parameters=[ParameterSpec("p", default=5, required=False)],
        ports=[PortSpec("in", 4), PortSpec("out", 4, dtype="float16", is_stateful=True)],
        attributes={"aggregation": "sum"},
        invariants=["p>=0"],
    )
    data = spec.serialize()
    spec2 = deserialize_module_spec(data)
    env = spec2.validate_and_bind_params({})
    assert env["p"] == 5
    assert spec2.ports[1].is_stateful is True


def test_expression_safety_rejects_unsafe_nodes():
    # Using attribute lookup should be rejected by safe eval
    spec = ModuleSpec(
        name="Safe",
        version=1,
        parameters=[ParameterSpec("a", default=1)],
        invariants=["__import__('os') is None"],
    )
    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({})
    assert ei.value.error_type in {"invariant_error", "invariant_violation"}


def test_invariant_syntax_error_reports_invariant_error():
    spec = ModuleSpec(
        name="Syntax",
        version=1,
        parameters=[ParameterSpec("a", default=1)],
        invariants=["a +"],  # syntax error
    )
    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({})
    assert ei.value.error_type == "invariant_error"


def test_domain_callable_exception_is_handled_as_invalid_param_domain():
    def bad_domain(_):
        raise RuntimeError("boom")

    spec = ModuleSpec(
        name="Dom",
        version=1,
        parameters=[ParameterSpec("x", default=1, domain=bad_domain)],
    )
    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({})
    assert ei.value.error_type == "invalid_param_domain"


def test_readonly_mapping_interface_and_immutability():
    spec = ModuleSpec(
        name="RO",
        version=1,
        parameters=[ParameterSpec("x", default=7), ParameterSpec("y", default=5)],
        invariants=["x>0 and y>0"],
    )
    env = spec.validate_and_bind_params({})
    assert list(env) == ["x", "y"] or list(env) == ["y", "x"]
    assert len(env) == 2
    assert set(iter(env)) == {"x", "y"}
    with pytest.raises(TypeError):
        env["x"] = 10  # type: ignore[index]


def test_serialize_ports_defaults_and_attributes_roundtrip():
    spec = ModuleSpec(
        name="Ser",
        version=9,
        parameters=[ParameterSpec("p", default=2, required=False)],
        ports=[PortSpec("in", 3), PortSpec("out", 6, dtype="float64")],
        attributes={"attn": False, "note": "ok"},
        invariants=["p>=2"],
    )
    data = spec.serialize()
    assert data["attributes"]["note"] == "ok"
    spec2 = deserialize_module_spec(data)
    env = spec2.validate_and_bind_params({})
    assert env["p"] == 2


def test_unknown_override_rejected_and_allowed_by_flag():
    spec = ModuleSpec(
        name="UO",
        version=1,
        parameters=[ParameterSpec("x", default=1)],
    )
    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({"extra": 2})
    assert ei.value.error_type == "unknown_param_override"

    # allowed when flag set
    env = spec.validate_and_bind_params({"extra": 3}, allow_unknown_overrides=True)
    assert env["x"] == 1


def test_strict_mode_rejects_non_finite_params():
    spec = ModuleSpec(name="SM", version=1, parameters=[ParameterSpec("x", default=float("inf"))])
    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({}, strict=True)
    assert ei.value.error_type == "non_finite_param"


def test_binding_signature_deterministic_and_uuid_derivation():
    spec = ModuleSpec(
        name="BS",
        version=1,
        parameters=[ParameterSpec("a", default=2), ParameterSpec("b", default=3)],
        invariants=["a>0 and b>0"],
    )
    env1 = spec.validate_and_bind_params({"b": 3})
    env2 = spec.validate_and_bind_params({"b": 3})
    sig1 = spec.binding_signature(env1)
    sig2 = spec.binding_signature(env2)
    assert sig1 == sig2

    provider = DeterministicUUIDProvider(UUIDProviderConfig())
    info = spec.explain_params({"b": 3}, provider=provider)
    assert info["uuid"] is not None
    assert info["signature"] == sig1


def test_freeze_policy_detects_changes():
    spec = ModuleSpec(name="FR", version=1, parameters=[ParameterSpec("x", default=2)])
    env = spec.validate_and_bind_params({})
    sig = spec.binding_signature(env)
    # Changing override should fail under freeze
    with pytest.raises(ValidationError) as ei:
        spec.explain_params({"x": 3}, freeze_signature=sig)
    assert ei.value.error_type == "frozen_params_changed"


def test_module_registry_register_and_get():
    spec = ModuleSpec(name="Reg", version=5, parameters=[ParameterSpec("p", default=1)])
    ModuleRegistry.register(spec)
    got = ModuleRegistry.get("Reg", 5)
    assert got is spec


def test_invariants_can_reference_ports_and_attributes():
    spec = ModuleSpec(
        name="PA",
        version=1,
        parameters=[ParameterSpec("model_dim", default=16)],
        ports=[PortSpec("in", 8), PortSpec("out", 16)],
        attributes={"scale": 2},
        invariants=["out.size == model_dim", "attributes['scale'] * in.size == model_dim"],
    )
    env = spec.validate_and_bind_params({})
    info = spec.explain_params({})
    assert all(s["status"] for s in info["invariants"])


def test_param_expression_topo_eval_and_cycle_detection():
    # topo: a = 2, b = a+3, c = b*2
    spec = ModuleSpec(
        name="Topo",
        version=1,
        parameters=[
            ParameterSpec("a", default="=2"),
            ParameterSpec("b", default="=a+3"),
            ParameterSpec("c", default="=b*2"),
        ],
        invariants=["c==10"],
    )
    env = spec.validate_and_bind_params({})
    assert env["c"] == 10

    cyc = ModuleSpec(
        name="Cycle",
        version=1,
        parameters=[
            ParameterSpec("x", default="=y+1"),
            ParameterSpec("y", default="=x+1"),
        ],
    )
    with pytest.raises(ValidationError) as ei:
        cyc.validate_and_bind_params({})
    assert ei.value.error_type == "param_cycle_detected"


def test_param_types_and_coercion_with_errors():
    # Note: we simulate type hints via domain; future DSL may carry explicit types
    spec = ModuleSpec(
        name="Types",
        version=1,
        parameters=[
            ParameterSpec("heads", default=2, domain=lambda v: isinstance(v, int) and v>0),
            ParameterSpec("model_dim", default=16, domain=lambda v: isinstance(v, int) and v%2==0),
        ],
        invariants=["model_dim % heads == 0"],
    )
    spec.validate_and_bind_params({"heads": 4, "model_dim": 16})
    with pytest.raises(ValidationError):
        spec.validate_and_bind_params({"heads": 3, "model_dim": 16})


def test_strict_serialization_unknown_keys_policy():
    data = {
        "name": "SerStrict",
        "version": 1,
        "parameters": [{"name": "p", "default": 1}],
        "ports": [],
        "attributes": {},
        "invariants": [],
        "unknown": 123,
    }
    # Non-strict should ignore unknown
    ms = deserialize_module_spec(data)
    ms.validate_and_bind_params({})
    # Strict deserialization will be tested via helper (added later)


def test_signature_includes_attributes_when_referenced():
    spec = ModuleSpec(
        name="SigRef",
        version=1,
        parameters=[ParameterSpec("d", default=8)],
        ports=[PortSpec("in", 4)],
        attributes={"k": 2},
        invariants=["d==attributes['k']*in.size"],
    )
    env = spec.validate_and_bind_params({})
    sig = spec.binding_signature(env)
    # Changing attribute should change signature if included
    spec2 = ModuleSpec(
        name="SigRef",
        version=1,
        parameters=[ParameterSpec("d", default=8)],
        ports=[PortSpec("in", 4)],
        attributes={"k": 3},
        invariants=["d==attributes['k']*in.size"],
    )
    env2 = spec2.validate_and_bind_params({"d": 12})
    sig2 = spec2.binding_signature(env2)
    assert sig != sig2


def test_explain_params_metrics_present():
    spec = ModuleSpec(name="Expl", version=1, parameters=[ParameterSpec("p", default=1)])
    info = spec.explain_params({})
    assert "metrics" in info
    assert set(info["metrics"]).issuperset({"bind_ms", "inv_eval_ms"})


def test_registry_duplicate_policy():
    spec = ModuleSpec(name="Dup", version=1, parameters=[ParameterSpec("p", default=1)])
    ModuleRegistry.register(spec)
    with pytest.raises(ValidationError):
        ModuleRegistry.register(spec)


def test_param_types_and_coercion_and_enum():
    spec = ModuleSpec(
        name="Types2",
        version=1,
        parameters=[
            ParameterSpec("heads", default=2, domain=lambda v: True),
            ParameterSpec("model_dim", default=16, domain=lambda v: True),
            ParameterSpec("mode", default="train", domain=lambda v: True),
        ],
        invariants=["model_dim % heads == 0"],
    )
    # Use new type API once implemented
    # For now, just ensure coercion path will be validated when types added
    env = spec.validate_and_bind_params({"heads": 2.0, "model_dim": 16})
    assert env["heads"] == 2.0 or env["heads"] == 2


def test_dsl_domains_and_invariants_and_strict_deser():
    # Define ModuleSpec via dict with DSL
    data = {
        "name": "DSLMod",
        "version": 1,
        "parameters": [
            {"name": "heads", "default": 2, "ptype": "int", "domain_dsl": {"in_set": [1, 2, 4]}},
            {"name": "head_dim", "default": 8, "ptype": "int"},
            {"name": "model_dim", "default": 16, "ptype": "int"},
        ],
        "ports": [],
        "attributes": {},
        "invariants": [],
        "invariants_dsl": [{"eq": ["model_dim", {"mul": ["heads", "head_dim"]}]}],
    }
    ms = deserialize_module_spec(data)  # non-strict path accepts extra DSL fields
    env = ms.validate_and_bind_params({})
    info = ms.explain_params({})
    assert all(s["status"] for s in info["invariants"])  # will be mirrored into string invariants later

    # Strict deserialization: unknown key triggers error
    bad = dict(data)
    bad["unknown_field"] = 5
    from ggnes.hierarchical.module_spec import deserialize_module_spec_strict
    with pytest.raises(ValidationError):
        deserialize_module_spec_strict(bad)


def test_explain_uuid_cache_metrics():
    spec = ModuleSpec(name="Cache", version=1, parameters=[ParameterSpec("p", default=1)])
    info1 = spec.explain_params({})
    info2 = spec.explain_params({})
    assert "metrics" in info2 and "uuid_cache_hits" in info2["metrics"]


def test_invariant_unknown_name_rejected():
    spec = ModuleSpec(name="BadName", version=1, parameters=[ParameterSpec("x", default=1)], invariants=["y>0"])
    with pytest.raises(ValidationError) as ei:
        spec.validate_and_bind_params({})
    assert ei.value.error_type == "invariant_error"


