from ggnes.rules.rule import EmbeddingLogic, Direction
from ggnes.utils.serialization import deserialize_embedding
from ggnes.hierarchical import ModuleSpec, ParameterSpec
from ggnes.hierarchical.module_spec import (
    serialize_module_library,
    deserialize_module_library,
    migrate_param_overrides,
    signatures_equal_under_migration,
)


def test_embedding_logic_serialization_roundtrip_policies_and_numeric():
    emb = EmbeddingLogic(
        connection_map={
            ('X', Direction.IN): [('P', 'CONNECT_SINGLE'), ('Q', 2)],
            ('X', Direction.OUT): [('P', 'COPY_ALL')],
        },
        excess_connection_handling='ERROR',
        unknown_direction_handling='WARNING',
        boundary_handling='PROCESS_LAST',
    )
    data = {
        'strategy': emb.strategy.name,
        'connection_map': {
            f"{k[0]}:{k[1].name}": [{'rhs_label': v[0], 'distribution': (v[1].name if hasattr(v[1], 'name') else v[1])} for v in vals]
            for k, vals in emb.connection_map.items()
        },
        'excess_connection_handling': emb.excess_connection_handling,
        'unknown_direction_handling': emb.unknown_direction_handling,
        'boundary_handling': emb.boundary_handling,
    }

    # Roundtrip using code serializer/deserializer
    emb2 = deserialize_embedding(data)
    ser2 = {
        'strategy': emb2.strategy.name,
        'connection_map': {
            f"{k[0]}:{k[1].name}": [{'rhs_label': v[0], 'distribution': (v[1].name if hasattr(v[1], 'name') else v[1])} for v in vals]
            for k, vals in emb2.connection_map.items()
        },
        'excess_connection_handling': emb2.excess_connection_handling,
        'unknown_direction_handling': emb2.unknown_direction_handling,
        'boundary_handling': emb2.boundary_handling,
    }
    assert ser2 == data


def test_module_library_serialize_deserialize_roundtrip():
    specs = [
        ModuleSpec(name="A", version=1, parameters=[ParameterSpec("p", default=1)]),
        ModuleSpec(name="B", version=2, parameters=[ParameterSpec("q", default=2)]),
    ]
    data = serialize_module_library(specs)
    loaded = deserialize_module_library(data, strict=True)
    assert len(loaded) == 2
    assert loaded[0].name == "A" and loaded[1].version == 2


def test_migration_helpers_and_signature_parity():
    old = ModuleSpec(
        name="M",
        version=1,
        parameters=[ParameterSpec("alpha", default=2), ParameterSpec("beta", default=3)],
    )
    new = ModuleSpec(
        name="M",
        version=2,
        parameters=[ParameterSpec("a", default=2), ParameterSpec("beta", default=3)],
    )
    overrides = {"alpha": 4, "beta": 5}
    migrated = migrate_param_overrides(old, new, overrides, rename={"alpha": "a"})
    assert migrated == {"a": 4, "beta": 5}
    assert signatures_equal_under_migration(old, new, overrides, rename={"alpha": "a"})


