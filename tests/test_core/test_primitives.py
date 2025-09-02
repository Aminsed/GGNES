"""
Test module for PrimitivesLibrary.
Tests activation and aggregation function registration per M2 milestone.
"""

from ggnes.core.primitives import PrimitivesLibrary


class TestPrimitivesLibrary:
    """Tests for PrimitivesLibrary."""

    def test_standard_activations_exist(self):
        """[T-core-07] PrimitivesLibrary has standard activations."""
        expected_activations = {
            'linear', 'relu', 'sigmoid', 'tanh', 'softmax',
            'leaky_relu', 'elu', 'gelu', 'lstm', 'gru'
        }

        for activation in expected_activations:
            assert PrimitivesLibrary.is_valid_activation(activation)

    def test_standard_aggregations_exist(self):
        """[T-core-07] PrimitivesLibrary has standard aggregations."""
        expected_aggregations = {
            'sum', 'mean', 'max', 'min', 'concat', 'matrix_product'
        }

        for aggregation in expected_aggregations:
            assert PrimitivesLibrary.is_valid_aggregation(aggregation)

    def test_register_custom_activation(self):
        """[T-core-07] PrimitivesLibrary.register_activation works."""
        # Register a custom activation
        PrimitivesLibrary.register_activation("custom_relu")

        # Verify it's registered
        assert PrimitivesLibrary.is_valid_activation("custom_relu")
        assert "custom_relu" in PrimitivesLibrary.ACTIVATIONS

        # Can also register with a function
        def my_activation(x):
            return x * 2

        PrimitivesLibrary.register_activation("double", my_activation)
        assert PrimitivesLibrary.is_valid_activation("double")

    def test_register_custom_aggregation(self):
        """[T-core-07] PrimitivesLibrary.register_aggregation works."""
        # Register a custom aggregation
        PrimitivesLibrary.register_aggregation("custom_sum")

        # Verify it's registered
        assert PrimitivesLibrary.is_valid_aggregation("custom_sum")
        assert "custom_sum" in PrimitivesLibrary.AGGREGATIONS

        # Can also register with a function
        def my_aggregation(inputs):
            return sum(inputs) * 2

        PrimitivesLibrary.register_aggregation("double_sum", my_aggregation)
        assert PrimitivesLibrary.is_valid_aggregation("double_sum")

    def test_invalid_activation_check(self):
        """[T-core-07] PrimitivesLibrary.is_valid_activation returns False for invalid."""
        assert not PrimitivesLibrary.is_valid_activation("nonexistent_activation")
        assert not PrimitivesLibrary.is_valid_activation("")
        assert not PrimitivesLibrary.is_valid_activation(None)

    def test_invalid_aggregation_check(self):
        """[T-core-07] PrimitivesLibrary.is_valid_aggregation returns False for invalid."""
        assert not PrimitivesLibrary.is_valid_aggregation("nonexistent_aggregation")
        assert not PrimitivesLibrary.is_valid_aggregation("")
        assert not PrimitivesLibrary.is_valid_aggregation(None)
