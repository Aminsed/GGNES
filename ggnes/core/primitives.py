"""PrimitivesLibrary for activation and aggregation functions."""


class PrimitivesLibrary:
    """Registry of activation and aggregation functions."""

    # Standard activation functions
    ACTIVATIONS = {
        'linear', 'relu', 'sigmoid', 'tanh', 'softmax',
        'leaky_relu', 'elu', 'gelu', 'lstm', 'gru'
    }

    # Standard aggregation functions
    AGGREGATIONS = {
        'sum', 'mean', 'max', 'min', 'concat',
        'matrix_product',  # Note: concatenates and flattens inputs
        # Advanced aggregations (implemented in translation runtime per project_guide.md §8.4)
        'attention', 'multi_head_attention', 'gated_sum', 'topk_weighted_sum', 'moe', 'attn_pool'
    }

    # Custom registries
    _custom_activations = {}
    _custom_aggregations = {}

    @classmethod
    def register_activation(cls, name: str, function=None):
        """Register a custom activation function.

        Args:
            name: Activation function name
            function: Optional function for validation
        """
        cls._custom_activations[name] = function
        cls.ACTIVATIONS.add(name)

    @classmethod
    def register_aggregation(cls, name: str, function=None):
        """Register a custom aggregation function.

        Args:
            name: Aggregation function name
            function: Optional function for validation
        """
        cls._custom_aggregations[name] = function
        cls.AGGREGATIONS.add(name)

    @classmethod
    def is_valid_activation(cls, name: str) -> bool:
        """Check if activation function is registered."""
        return name in cls.ACTIVATIONS if name else False

    @classmethod
    def is_valid_aggregation(cls, name: str) -> bool:
        """Check if aggregation function is registered."""
        return name in cls.AGGREGATIONS if name else False
