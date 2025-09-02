"""
Parallel Rule Application Tutorial

Goals:
- Configure generation with parallel flags and compare basic metrics
  (This demo uses empty rules; metrics demonstrate configuration wiring.)
"""

from ggnes.core.graph import Graph
from ggnes.generation.network_gen import generate_network
from ggnes.utils.rng_manager import RNGManager


class DummyGenotype:
    def __init__(self):
        self.rules = []


def main():
    g = Graph()
    geno = DummyGenotype()
    rng = RNGManager(seed=7)

    serial_cfg = {
        'max_iterations': 3,
        'parallel_execution': False,
    }
    parallel_cfg = {
        'max_iterations': 3,
        'parallel_execution': True,
        'hg_max_parallel_workers': 4,
        'hg_parallel_batch_policy': 'MAX_INDEPENDENT_SET',
    }

    g1, m1 = generate_network(geno, g, serial_cfg, rng)
    g2, m2 = generate_network(geno, g, parallel_cfg, rng)
    print('serial_iters:', m1['iterations'], 'parallel_iters:', m2['iterations'])


if __name__ == '__main__':
    main()


