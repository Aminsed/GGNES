"""Configuration presets for GGNES (project_guide.md §12.2).

These presets are simple dictionaries intended to seed a configuration
for experiments or tests. They can be extended or overridden by callers.
"""

from __future__ import annotations

# Minimal preset - for testing and simple problems
PRESET_MINIMAL: dict = {
    'population_size': 20,
    'max_iterations': 20,
    'max_rules_per_genotype': 10,
    'repair_strategy': 'DISABLE',
    'training_epochs': 5,
    # Deterministic UUIDs defaults (can be overridden by callers)
    'deterministic_uuids': False,
    'uuid_scheme_version': 1,
    'uuid_namespace': 'ggnes://uuid/v1',
}


# Standard preset - balanced for most use cases
PRESET_STANDARD: dict = {
    'population_size': 100,
    'max_iterations': 50,
    'max_rules_per_genotype': 20,
    'repair_strategy': 'MINIMAL_CHANGE',
    'training_epochs': 10,
    'deterministic_uuids': False,
    'uuid_scheme_version': 1,
    'uuid_namespace': 'ggnes://uuid/v1',
}


# Research preset - for extensive exploration
PRESET_RESEARCH: dict = {
    'population_size': 500,
    'max_iterations': 100,
    'max_rules_per_genotype': 50,
    'repair_strategy': 'AGGRESSIVE',
    'training_epochs': 20,
    'parallel_execution': True,
    'deterministic_uuids': False,
    'uuid_scheme_version': 1,
    'uuid_namespace': 'ggnes://uuid/v1',
}


__all__ = [
    'PRESET_MINIMAL',
    'PRESET_STANDARD',
    'PRESET_RESEARCH',
]


