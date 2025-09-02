"""
Composite Genotype and Selection Tutorial

Goals:
- Construct a CompositeGenotype (G1/G2/G3)
- Compute its deterministic UUID
- Run NSGA-II selection on a toy population
"""

from ggnes.evolution.composite_genotype import (
    CompositeGenotype, G1Grammar, G2Policy, G3Hierarchy
)
from ggnes.evolution.selection import nsga2_select
from ggnes.utils.rng_manager import RNGManager


def main():
    g1 = G1Grammar(rules=[{"rule_id": "r-1", "priority": 1, "probability": 0.8}])
    g2 = G2Policy(training_epochs=5, batch_size=16, learning_rate=0.01)
    g3 = G3Hierarchy(modules={"Block": {"dim": 8}}, attributes={"note": "demo"})
    geno = CompositeGenotype(g1=g1, g2=g2, g3=g3)
    uid = geno.uuid()
    print("genotype_uuid:", uid)

    # NSGA-II selection on a toy population of 5 individuals with 2 objectives
    population = list(range(5))
    objectives = [
        [5.0, 1.0],
        [4.0, 2.0],
        [3.0, 3.0],
        [2.0, 4.0],
        [1.0, 5.0],
    ]
    rng = RNGManager(seed=123)
    selected = nsga2_select(population, objectives, k=3, rng_manager=rng)
    print("nsga2_selected:", selected)


if __name__ == "__main__":
    main()


