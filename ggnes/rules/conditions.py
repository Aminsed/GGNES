"""ConditionRegistry and standard conditions."""

from collections.abc import Callable
from typing import Any


class ConditionRegistry:
    """Registry for condition functions.

    Manages registration and creation of condition functions used in rules.
    Conditions are callables with signature: (graph_view, bindings, graph_context) -> bool
    """

    _registry: dict[str, Callable] = {}
    _standard_registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a condition function.

        Args:
            name: Unique name for the condition

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def register_standard(cls, name: str) -> Callable:
        """Register a parameterizable standard condition.

        Standard conditions accept parameters before the standard
        (graph_view, bindings, graph_context) arguments.

        Args:
            name: Unique name for the standard condition

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            cls._standard_registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable | None:
        """Get a registered condition by name.

        Args:
            name: Condition name

        Returns:
            Condition function or None
        """
        return cls._registry.get(name)

    @classmethod
    def create_parameterized(cls, name: str, **params) -> Callable:
        """Create a parameterized version of a standard condition.

        Args:
            name: Standard condition name
            **params: Parameters to bind

        Returns:
            Parameterized condition function

        Raises:
            ValueError: If standard condition not found
        """
        func = cls._standard_registry.get(name)
        if not func:
            raise ValueError(f"No standard condition named {name}")

        # Create wrapper that reorders arguments
        def param_func(graph_view, bindings, graph_context):
            return func(graph_view=graph_view, bindings=bindings,
                        graph_context=graph_context, **params)

        # Add metadata attributes
        param_func.standard_name = name
        param_func.params = params

        return param_func

    @classmethod
    def compose_and(cls, *conditions: Callable) -> Callable:
        """Create AND composition of conditions.

        All conditions must return True for the composition to return True.

        Args:
            *conditions: Condition functions to compose

        Returns:
            Composed condition function
        """
        def composed(graph_view: Any, bindings: dict, graph_context: dict) -> bool:
            for condition in conditions:
                if not condition(graph_view, bindings, graph_context):
                    return False
            return True
        return composed

    @classmethod
    def compose_or(cls, *conditions: Callable) -> Callable:
        """Create OR composition of conditions.

        At least one condition must return True for the composition to return True.

        Args:
            *conditions: Condition functions to compose

        Returns:
            Composed condition function
        """
        def composed(graph_view: Any, bindings: dict, graph_context: dict) -> bool:
            for condition in conditions:
                if condition(graph_view, bindings, graph_context):
                    return True
            return False
        return composed

    @classmethod
    def compose_not(cls, condition: Callable) -> Callable:
        """Create NOT composition of condition.

        Negates the result of the condition.

        Args:
            condition: Condition function to negate

        Returns:
            Negated condition function
        """
        def composed(graph_view: Any, bindings: dict, graph_context: dict) -> bool:
            return not condition(graph_view, bindings, graph_context)
        return composed
