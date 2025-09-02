"""
Evolution Basics Tutorial

Goals:
- Create minimal genotypes
- Run selection (PRIORITY_THEN_PROBABILITY_THEN_ORDER)
- Apply simple mutation and uniform crossover deterministically
"""

from ggnes.evolution.genotype import Genotype
from ggnes.evolution.selection import select_match
from ggnes.evolution.operators import mutate, uniform_crossover
from ggnes.utils.rng_manager import RNGManager


class Rule:
    def __init__(self, rule_id, metadata=None, lhs=None, rhs=None):
        self.rule_id = rule_id
        self.metadata = metadata or {"priority": 0, "probability": 1.0}
        self.lhs = lhs
        self.rhs = rhs


def main():
    rng = RNGManager(seed=42)

    # Build two tiny parent genotypes with two rules each
    r1 = Rule("rule-A", {"priority": 2, "probability": 0.6})
    r2 = Rule("rule-B", {"priority": 1, "probability": 0.4})
    parent1 = Genotype(rules=[r1, r2])

    r3 = Rule("rule-A", {"priority": 2, "probability": 0.7})  # shared id, diff prob
    r4 = Rule("rule-C", {"priority": 1, "probability": 0.3})
    parent2 = Genotype(rules=[r3, r4])

    # Selection demo: choose among matches of same highest priority via probabilities
    potential_matches = [
        (r1, {"bind": 1}),
        (r3, {"bind": 2}),
        (r2, {"bind": 3}),
    ]
    sel = select_match(
        potential_matches,
        strategy="PRIORITY_THEN_PROBABILITY_THEN_ORDER",
        rng_manager=rng,
        config={"probability_precision": 1e-6},
    )
    print("selected_rule_id:", sel[0].rule_id)

    # Mutation demo (may return original if mutation_rate not met)
    mutated = mutate(parent1, {"mutation_rate": 1.0}, rng)
    print("mutated_rules_count:", len(mutated.rules))

    # Crossover demo: order-independent union with per-rule inclusion
    o1, o2 = uniform_crossover(
        parent1,
        parent2,
        {"crossover_probability_per_rule": 0.8, "min_rules_per_genotype": 1},
        rng,
    )
    print("offspring1_rules:", [getattr(r, "rule_id", "?") for r in o1.rules])
    print("offspring2_rules:", [getattr(r, "rule_id", "?") for r in o2.rules])


if __name__ == "__main__":
    main()


