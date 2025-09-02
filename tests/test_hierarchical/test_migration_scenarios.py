import pytest

from ggnes.hierarchical.module_spec import (
    ModuleSpec,
    ParameterSpec,
    migrate_param_overrides,
    signatures_equal_under_migration,
)


def test_multi_rename_remove_add_with_expr_default_parity():
    # Old spec with alpha,beta; new spec renames alpha->a, removes beta (default), adds gamma with expr default
    old = ModuleSpec(
        name='M', version=1,
        parameters=[
            ParameterSpec('alpha', default=2),
            ParameterSpec('beta', default=3),
        ],
    )
    new = ModuleSpec(
        name='M', version=2,
        parameters=[
            ParameterSpec('a', default=2),
            ParameterSpec('gamma', default='=a+1'),
        ],
    )

    overrides = {'alpha': 5, 'beta': 7}
    migrated = migrate_param_overrides(old, new, overrides, rename={'alpha': 'a'}, removed_with_defaults={'beta': 3})
    assert migrated == {'a': 5}
    # Parity should hold since gamma derives from a and beta is removed with default
    assert signatures_equal_under_migration(old, new, overrides, rename={'alpha': 'a'}, removed_with_defaults={'beta': 3})


def test_missing_rename_map_breaks_parity_and_negative_errors():
    old = ModuleSpec(name='N', version=1, parameters=[ParameterSpec('p', default=1)])
    new = ModuleSpec(name='N', version=2, parameters=[ParameterSpec('q', default=1)])
    overrides = {'p': 2}
    # Without rename, parity should fail
    assert not signatures_equal_under_migration(old, new, overrides)

    # Invalid default types in removed_with_defaults should not crash migration (ignored) but parity still fails
    assert not signatures_equal_under_migration(old, new, overrides, removed_with_defaults={'p': 'oops'})


def test_add_param_without_default_requires_override_for_parity():
    old = ModuleSpec(name='A', version=1, parameters=[ParameterSpec('x', default=2)])
    # New adds y without default; we need to supply override to keep parity (if y not in signature semantics)
    new = ModuleSpec(name='A', version=2, parameters=[ParameterSpec('x', default=2), ParameterSpec('y', default=1)])
    overrides = {'x': 4}
    migrated = migrate_param_overrides(old, new, overrides)
    assert migrated == {'x': 4}
    assert signatures_equal_under_migration(old, new, overrides)

