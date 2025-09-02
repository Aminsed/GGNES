"""RNGManager for deterministic randomness."""

import hashlib
import random
from typing import Any


class RNGManager:
    """Manages deterministic random number generation.

    Provides context-specific RNGs for different parts of the system,
    ensuring reproducibility and independence between contexts.
    """

    def __init__(self, seed: int = None):
        """Initialize RNGManager with a seed.

        Args:
            seed: Random seed. If None, generates a random seed.
        """
        self.seed = seed or random.randint(0, 2**32 - 1)
        self.contexts = {}
        self._initialize_contexts()

    def _initialize_contexts(self):
        """Initialize RNG contexts with derived seeds."""
        # Create context-specific RNGs
        base_rng = random.Random(self.seed)

        self.contexts = {
            'selection': random.Random(base_rng.randint(0, 2**32 - 1)),
            'mutation': random.Random(base_rng.randint(0, 2**32 - 1)),
            'crossover': random.Random(base_rng.randint(0, 2**32 - 1)),
            'repair': random.Random(base_rng.randint(0, 2**32 - 1)),
            'application': random.Random(base_rng.randint(0, 2**32 - 1))
        }

    def get_context_rng(self, context: str) -> random.Random:
        """Get RNG for a specific context.

        Args:
            context: Context name

        Returns:
            Random instance for the context
        """
        if context not in self.contexts:
            # Create new context with derived seed using stable hashing
            # Use SHA-256 for stable cross-run determinism
            context_bytes = f"{self.seed}:{context}".encode()
            hash_digest = hashlib.sha256(context_bytes).digest()
            # Convert first 8 bytes to integer
            context_seed = int.from_bytes(hash_digest[:8], 'big') % (2**32)
            self.contexts[context] = random.Random(context_seed)
        return self.contexts[context]

    def get_rng_for_mutation(self, genotype_id) -> random.Random:
        """Get RNG for mutating a specific genotype.

        Args:
            genotype_id: Identifier for the genotype

        Returns:
            Random instance for this mutation
        """
        # Use mutation context RNG to derive a seed
        mutation_rng = self.get_context_rng('mutation')

        # Create deterministic seed from genotype_id
        # Use SHA-256 for stable hashing
        seed_bytes = f"mutation:{genotype_id}".encode()
        hash_digest = hashlib.sha256(seed_bytes).digest()
        base_seed = int.from_bytes(hash_digest[:8], 'big') % (2**32)

        # Mix with mutation context state for determinism
        # Save and restore state to avoid affecting subsequent calls
        saved_state = mutation_rng.getstate()
        mutation_rng.seed(base_seed)
        derived_seed = mutation_rng.randint(0, 2**32 - 1)
        mutation_rng.setstate(saved_state)

        return random.Random(derived_seed)

    def get_rng_for_crossover(self, parent1_id, parent2_id) -> random.Random:
        """Get RNG for crossover between two parents.

        Order-independent: (A, B) produces same RNG as (B, A).

        Args:
            parent1_id: First parent identifier
            parent2_id: Second parent identifier

        Returns:
            Random instance for this crossover
        """
        # Sort parent IDs to ensure order independence
        sorted_parents = sorted([str(parent1_id), str(parent2_id)])

        # Use crossover context RNG
        crossover_rng = self.get_context_rng('crossover')

        # Create deterministic seed from sorted parent IDs
        seed_bytes = f"crossover:{sorted_parents[0]}:{sorted_parents[1]}".encode()
        hash_digest = hashlib.sha256(seed_bytes).digest()
        base_seed = int.from_bytes(hash_digest[:8], 'big') % (2**32)

        # Mix with crossover context state
        saved_state = crossover_rng.getstate()
        crossover_rng.seed(base_seed)
        derived_seed = crossover_rng.randint(0, 2**32 - 1)
        crossover_rng.setstate(saved_state)

        return random.Random(derived_seed)

    def get_state(self) -> dict[str, Any]:
        """Get current state of all RNG contexts.

        Returns:
            Dict containing seed and all context states
        """
        return {
            'seed': self.seed,
            'contexts': {
                name: rng.getstate()
                for name, rng in self.contexts.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state of all RNG contexts.

        Args:
            state: State dict from get_state()
        """
        self.seed = state['seed']

        # Restore existing contexts
        for context_name, context_state in state['contexts'].items():
            if context_name not in self.contexts:
                # Create context if it doesn't exist
                self.get_context_rng(context_name)
            self.contexts[context_name].setstate(context_state)
