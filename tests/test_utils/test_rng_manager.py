"""
Comprehensive tests for RNGManager to ensure strict adherence to project_guide.md.
"""

import hashlib
import pickle

from ggnes.utils.rng_manager import RNGManager


class TestRNGManagerBasics:
    """Basic tests for RNGManager initialization and context management."""

    def test_rng_manager_initialization_with_seed(self):
        """Test RNGManager initializes with provided seed."""
        rng_manager = RNGManager(seed=12345)
        assert rng_manager.seed == 12345

    def test_rng_manager_initialization_without_seed(self):
        """Test RNGManager generates random seed when none provided."""
        rng_manager = RNGManager()
        assert isinstance(rng_manager.seed, int)
        assert 0 <= rng_manager.seed < 2**32

    def test_standard_contexts_exist(self):
        """Test standard contexts are initialized."""
        rng_manager = RNGManager(seed=42)

        expected_contexts = ['selection', 'mutation', 'crossover', 'repair', 'application']
        for context in expected_contexts:
            assert context in rng_manager.contexts
            assert rng_manager.contexts[context] is not None


class TestRNGManagerDeterminism:
    """Tests for deterministic behavior per [T-rng-01]."""

    def test_context_rngs_repeat_with_same_seed(self):
        """[T-rng-01] Context RNGs repeat given same seed."""
        # Create two managers with same seed
        rng1 = RNGManager(seed=100)
        rng2 = RNGManager(seed=100)

        # Test each standard context
        contexts = ['selection', 'mutation', 'crossover', 'repair', 'application']

        for context in contexts:
            # Generate sequences from both
            seq1 = [rng1.get_context_rng(context).random() for _ in range(10)]
            seq2 = [rng2.get_context_rng(context).random() for _ in range(10)]

            # Should be identical
            assert seq1 == seq2

    def test_different_contexts_produce_different_sequences(self):
        """[T-rng-01] Different contexts produce different sequences."""
        rng_manager = RNGManager(seed=200)

        # Get sequences from different contexts
        selection_seq = [rng_manager.get_context_rng('selection').random() for _ in range(10)]
        mutation_seq = [rng_manager.get_context_rng('mutation').random() for _ in range(10)]
        crossover_seq = [rng_manager.get_context_rng('crossover').random() for _ in range(10)]

        # All should be different
        assert selection_seq != mutation_seq
        assert mutation_seq != crossover_seq
        assert selection_seq != crossover_seq

    def test_new_context_creation_deterministic(self):
        """Test that new contexts are created deterministically."""
        rng1 = RNGManager(seed=300)
        rng2 = RNGManager(seed=300)

        # Create new context in both
        custom_seq1 = [rng1.get_context_rng('custom_context').random() for _ in range(10)]
        custom_seq2 = [rng2.get_context_rng('custom_context').random() for _ in range(10)]

        assert custom_seq1 == custom_seq2

    def test_new_context_uses_sha256_hashing(self):
        """Test new contexts use SHA-256 hashing as specified."""
        rng_manager = RNGManager(seed=400)

        # Access new context
        custom_rng = rng_manager.get_context_rng('test_context')

        # Verify the seed was derived correctly
        context_bytes = f"{rng_manager.seed}:test_context".encode()
        hash_digest = hashlib.sha256(context_bytes).digest()
        expected_seed = int.from_bytes(hash_digest[:8], 'big') % (2**32)

        # Create a new RNG with expected seed and compare sequences
        import random
        expected_rng = random.Random(expected_seed)

        for _ in range(10):
            assert custom_rng.random() == expected_rng.random()


class TestRNGManagerMutationCrossover:
    """Tests for mutation/crossover RNG generation per [T-rng-02]."""

    def test_get_rng_for_mutation_deterministic(self):
        """[T-rng-02] get_rng_for_mutation is deterministic."""
        rng_manager = RNGManager(seed=500)

        # Same genotype ID should produce same sequence
        genotype_id = "genotype_123"

        rng1 = rng_manager.get_rng_for_mutation(genotype_id)
        seq1 = [rng1.random() for _ in range(10)]

        rng2 = rng_manager.get_rng_for_mutation(genotype_id)
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2

    def test_get_rng_for_mutation_different_genotypes(self):
        """[T-rng-02] Different genotypes get different mutation RNGs."""
        rng_manager = RNGManager(seed=600)

        rng1 = rng_manager.get_rng_for_mutation("genotype_1")
        rng2 = rng_manager.get_rng_for_mutation("genotype_2")

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 != seq2

    def test_get_rng_for_crossover_order_independent(self):
        """[T-rng-02] get_rng_for_crossover is order-independent."""
        rng_manager = RNGManager(seed=700)

        parent1_id = "parent_A"
        parent2_id = "parent_B"

        # Get RNG with parents in different orders
        rng1 = rng_manager.get_rng_for_crossover(parent1_id, parent2_id)
        seq1 = [rng1.random() for _ in range(10)]

        rng2 = rng_manager.get_rng_for_crossover(parent2_id, parent1_id)
        seq2 = [rng2.random() for _ in range(10)]

        # Should be identical regardless of order
        assert seq1 == seq2

    def test_get_rng_for_crossover_deterministic(self):
        """[T-rng-02] get_rng_for_crossover is deterministic."""
        rng_manager = RNGManager(seed=800)

        parent1_id = "parent_X"
        parent2_id = "parent_Y"

        # Multiple calls should produce same sequence
        rng1 = rng_manager.get_rng_for_crossover(parent1_id, parent2_id)
        seq1 = [rng1.random() for _ in range(10)]

        rng2 = rng_manager.get_rng_for_crossover(parent1_id, parent2_id)
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2

    def test_mutation_and_crossover_use_correct_contexts(self):
        """Test that mutation and crossover use their respective contexts."""
        rng_manager = RNGManager(seed=900)

        # Test that mutation uses mutation context
        mutation_rng1 = rng_manager.get_rng_for_mutation("genotype1")
        mutation_rng2 = rng_manager.get_rng_for_mutation("genotype2")

        # Different genotypes should produce different RNGs
        seq1 = [mutation_rng1.random() for _ in range(5)]
        seq2 = [mutation_rng2.random() for _ in range(5)]
        assert seq1 != seq2

        # Test that crossover uses crossover context
        crossover_rng1 = rng_manager.get_rng_for_crossover("parent1", "parent2")
        crossover_rng2 = rng_manager.get_rng_for_crossover("parent3", "parent4")

        # Different parent pairs should produce different RNGs
        seq3 = [crossover_rng1.random() for _ in range(5)]
        seq4 = [crossover_rng2.random() for _ in range(5)]
        assert seq3 != seq4


class TestRNGManagerStatePersistence:
    """Tests for state save/restore per [T-rng-03]."""

    def test_get_state_captures_all_contexts(self):
        """[T-rng-03] get_state captures state of all contexts."""
        rng_manager = RNGManager(seed=1000)

        # Use various contexts
        rng_manager.get_context_rng('selection').random()
        rng_manager.get_context_rng('mutation').random()
        rng_manager.get_context_rng('custom').random()

        state = rng_manager.get_state()

        # Should have seed and all contexts
        assert 'seed' in state
        assert 'contexts' in state
        assert 'selection' in state['contexts']
        assert 'mutation' in state['contexts']
        assert 'custom' in state['contexts']

    def test_set_state_restores_sequences(self):
        """[T-rng-03] set_state/get_state roundtrip restores sequences."""
        rng_manager = RNGManager(seed=1100)

        # Generate some values from different contexts
        [rng_manager.get_context_rng('selection').random() for _ in range(5)]
        [rng_manager.get_context_rng('mutation').random() for _ in range(5)]

        # Save state
        saved_state = rng_manager.get_state()

        # Generate more values (advancing the state)
        more_selection = [rng_manager.get_context_rng('selection').random() for _ in range(5)]
        more_mutation = [rng_manager.get_context_rng('mutation').random() for _ in range(5)]

        # Create new manager and restore state
        new_manager = RNGManager(seed=9999)  # Different seed
        new_manager.set_state(saved_state)

        # Should generate the same "more" values
        restored_selection = [new_manager.get_context_rng('selection').random() for _ in range(5)]
        restored_mutation = [new_manager.get_context_rng('mutation').random() for _ in range(5)]

        assert restored_selection == more_selection
        assert restored_mutation == more_mutation

    def test_state_is_pickleable(self):
        """[T-rng-03] State can be pickled for transaction rollback."""
        rng_manager = RNGManager(seed=1200)

        # Use some contexts
        rng_manager.get_context_rng('selection').randint(1, 100)
        rng_manager.get_context_rng('repair').choice(['a', 'b', 'c'])

        # Get state
        state = rng_manager.get_state()

        # Should be pickleable
        pickled = pickle.dumps(state)
        unpickled = pickle.loads(pickled)

        # Restore to new manager
        new_manager = RNGManager()
        new_manager.set_state(unpickled)

        # Should have same seed
        assert new_manager.seed == rng_manager.seed

    def test_state_preserves_custom_contexts(self):
        """Test that custom contexts are preserved in state."""
        rng_manager = RNGManager(seed=1300)

        # Create custom contexts
        custom1 = rng_manager.get_context_rng('custom1')
        custom2 = rng_manager.get_context_rng('custom2')

        # Generate values
        vals1 = [custom1.random() for _ in range(3)]
        vals2 = [custom2.random() for _ in range(3)]

        # Save and restore
        state = rng_manager.get_state()
        new_manager = RNGManager()
        new_manager.set_state(state)

        # Custom contexts should exist and continue sequences
        next_vals1 = [new_manager.get_context_rng('custom1').random() for _ in range(3)]
        next_vals2 = [new_manager.get_context_rng('custom2').random() for _ in range(3)]

        # Verify continuity by creating reference
        ref_manager = RNGManager(seed=1300)
        ref_custom1 = ref_manager.get_context_rng('custom1')
        ref_custom2 = ref_manager.get_context_rng('custom2')
        ref_vals1 = [ref_custom1.random() for _ in range(6)]
        ref_vals2 = [ref_custom2.random() for _ in range(6)]

        assert vals1 + next_vals1 == ref_vals1
        assert vals2 + next_vals2 == ref_vals2


class TestRNGManagerAdvanced:
    """Advanced tests for RNGManager edge cases and integration."""

    def test_rng_manager_thread_safety_note(self):
        """Test that RNGManager is designed for single-threaded use."""
        # This is a documentation test - RNGManager should be used
        # in single-threaded context as per spec
        rng_manager = RNGManager(seed=1400)

        # Each context should be a separate Random instance
        selection = rng_manager.get_context_rng('selection')
        mutation = rng_manager.get_context_rng('mutation')

        # Verify they are different instances
        assert selection is not mutation

    def test_mutation_rng_derives_from_mutation_context(self):
        """Test mutation RNG properly derives from mutation context."""
        rng_manager = RNGManager(seed=1500)

        genotype_id = "test_genotype"

        # Implementation should hash genotype_id with mutation context seed
        mutation_rng = rng_manager.get_rng_for_mutation(genotype_id)

        # Generate some values
        vals = [mutation_rng.random() for _ in range(5)]

        # Verify determinism
        mutation_rng2 = rng_manager.get_rng_for_mutation(genotype_id)
        vals2 = [mutation_rng2.random() for _ in range(5)]

        assert vals == vals2

    def test_crossover_rng_sorted_parent_ids(self):
        """Test crossover RNG uses sorted parent IDs for order independence."""
        rng_manager = RNGManager(seed=1600)

        # These should produce identical RNGs
        rng1 = rng_manager.get_rng_for_crossover("parent_Z", "parent_A")
        rng2 = rng_manager.get_rng_for_crossover("parent_A", "parent_Z")

        # Verify by comparing generated sequences
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2

    def test_rng_methods_available(self):
        """Test that context RNGs have standard Random methods."""
        rng_manager = RNGManager(seed=1700)

        rng = rng_manager.get_context_rng('selection')

        # Test various Random methods
        assert hasattr(rng, 'random')
        assert hasattr(rng, 'randint')
        assert hasattr(rng, 'choice')
        assert hasattr(rng, 'shuffle')
        assert hasattr(rng, 'gauss')
        assert hasattr(rng, 'getstate')
        assert hasattr(rng, 'setstate')

        # Test they work
        assert 0 <= rng.random() <= 1
        assert 1 <= rng.randint(1, 10) <= 10
        assert rng.choice(['a', 'b', 'c']) in ['a', 'b', 'c']
