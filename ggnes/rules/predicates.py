"""PredicateRegistry and predicate factories."""

from collections.abc import Callable


class PredicateRegistry:
    """Registry for match criteria predicates.

    Manages predicates used in LHS pattern matching. Predicates are
    functions that test whether a graph element matches certain criteria.
    """

    _registry: dict[str, Callable] = {}
    _factories: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a predicate function.

        Args:
            name: Unique name for the predicate

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            func._predicate_name = name
            return func
        return decorator

    @classmethod
    def register_factory(cls, name: str) -> Callable:
        """Register a predicate factory.

        Factories are functions that create parameterized predicates.

        Args:
            name: Factory name

        Returns:
            Decorator function
        """
        def decorator(factory: Callable) -> Callable:
            cls._factories[name] = factory
            return factory
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable | None:
        """Get a registered predicate by name.

        Args:
            name: Predicate name

        Returns:
            Predicate function or None
        """
        return cls._registry.get(name)

    @classmethod
    def create(cls, factory_name: str, **params) -> Callable | None:
        """Create a predicate using a factory.

        Args:
            factory_name: Name of the factory
            **params: Parameters for the factory

        Returns:
            Created predicate function or None if factory not found
        """
        factory = cls._factories.get(factory_name)
        if factory:
            return factory(**params)
        return None
