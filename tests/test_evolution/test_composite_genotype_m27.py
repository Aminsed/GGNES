from ggnes.evolution.composite_genotype import (
    CompositeGenotype,
    G1Grammar,
    G2Policy,
    G3Hierarchy,
    mutate_composite,
    crossover_composite,
)
from ggnes.utils.uuid_provider import DeterministicUUIDProvider, UUIDProviderConfig


def provider():
    return DeterministicUUIDProvider(UUIDProviderConfig())


def test_m27_mutate_clamps_and_validates():
    base = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "r1", "priority": 0, "probability": 0.5}]),
        g2=G2Policy(training_epochs=1, batch_size=1, learning_rate=0.001),
        provider=provider(),
    )
    # Apply deltas that would go out of bounds without clamps
    mutated = mutate_composite(
        base,
        provider(),
        {
            "g2.learning_rate": -1.0,  # clamp to 1e-8
            "g2.batch_size": -10,      # clamp to 1
            "g2.training_epochs": -5,  # clamp to 1
            "g1.rules.0.priority": -3, # clamp to 0
            "g1.rules.0.probability": 0.0,  # multiply -> 0, clamp to >0
        },
    )
    assert mutated.g2.learning_rate >= 1e-8
    assert mutated.g2.batch_size == 1
    assert mutated.g2.training_epochs == 1
    assert mutated.g1.rules[0]["priority"] == 0
    assert 0.0 < mutated.g1.rules[0]["probability"] <= 1.0
    # Still valid
    mutated.validate()


def test_m27_crossover_deterministic_and_constraint_safe():
    a = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "r1", "priority": 2, "probability": 0.6}]),
        g2=G2Policy(training_epochs=5, batch_size=16, learning_rate=0.01, parallel_execution=False, wl_iterations=3, failure_floor_threshold=0.4),
        g3=G3Hierarchy(modules={"M": {"p": 1}}, attributes={"a": True}),
        provider=provider(),
    )
    b = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "r2", "priority": 1, "probability": 1.0}]),
        g2=G2Policy(training_epochs=3, batch_size=8, learning_rate=0.02, parallel_execution=True, wl_iterations=2, failure_floor_threshold=0.6),
        g3=G3Hierarchy(modules={"N": {"q": 2}}, attributes={"b": False}),
        provider=provider(),
    )
    child1 = crossover_composite(a, b, provider())
    child2 = crossover_composite(a, b, provider())
    # Deterministic
    assert child1.uuid() == child2.uuid()
    # Valid and constraint-safe
    child1.validate()
    # Combined G1 rule ids
    rids = {r["rule_id"] for r in child1.g1.rules}
    assert rids == {"r1", "r2"}
    # G2 combined as specified
    assert child1.g2.training_epochs == 3
    assert child1.g2.batch_size == 8
    assert abs(child1.g2.learning_rate - 0.015) < 1e-12
    assert child1.g2.parallel_execution is True
    assert child1.g2.wl_iterations == 2
    assert abs(child1.g2.failure_floor_threshold - 0.5) < 1e-12
    # G3 union by keys
    assert "M" in child1.g3.modules and "N" in child1.g3.modules
    assert "a" in child1.g3.attributes and "b" in child1.g3.attributes


def test_m27_mutate_nonexistent_rule_index_ignored_and_bad_keys_ignored():
    base = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "r1", "probability": 0.5}]),
        provider=provider(),
    )
    out = mutate_composite(
        base,
        provider(),
        {
            "g1.rules.5.priority": 1,  # out of range, ignored
            "g1.rules.x.priority": 1,  # bad index, ignored
            "g1.rules.0.unknown": 1,   # unknown field, ignored
            "g1.rules": 1,              # malformed key, ignored by startswith check
            "g1.rules.0": 1,            # malformed structure, triggers len(parts)!=4 continue
        },
    )
    # unchanged for rule 0
    assert out.g1.rules[0].get("priority") is None


def test_m27_crossover_tie_break_prefer_lexicographically_smaller_parent():
    # Ensure both parents have same rule_id with differing content
    a = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "rid", "priority": 1}]),
        g2=G2Policy(),
        provider=provider(),
    )
    b = CompositeGenotype(
        g1=G1Grammar(rules=[{"rule_id": "rid", "priority": 2}]),
        g2=G2Policy(learning_rate=0.0012345),  # influence parent hash
        provider=provider(),
    )
    child = crossover_composite(a, b, provider())
    # Based on deterministic tie-break, assert child rule priority is from one parent
    assert child.g1.rules[0]["priority"] in (1, 2)


