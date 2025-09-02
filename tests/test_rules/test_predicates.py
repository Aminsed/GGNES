"""
Comprehensive tests for PredicateRegistry to ensure strict adherence to project_guide.md.
"""

import re

from ggnes.rules.predicates import PredicateRegistry


class TestPredicateRegistry:
    """Tests for PredicateRegistry [T-rule-03]."""

    def setup_method(self):
        """Clear registry before each test."""
        PredicateRegistry._registry = {}
        PredicateRegistry._factories = {}

    def test_register_and_get_predicate(self):
        """[T-rule-03] PredicateRegistry register basic functionality."""
        @PredicateRegistry.register('is_hidden')
        def is_hidden(value):
            return value == 'HIDDEN'

        # Should be retrievable
        pred = PredicateRegistry.get('is_hidden')
        assert pred is is_hidden
        assert pred('HIDDEN') is True
        assert pred('INPUT') is False

        # Should have name attribute
        assert hasattr(is_hidden, '_predicate_name')
        assert is_hidden._predicate_name == 'is_hidden'

    def test_get_non_existent_predicate(self):
        """Test get returns None for non-existent predicates."""
        result = PredicateRegistry.get('non_existent')
        assert result is None

    def test_register_factory(self):
        """[T-rule-03] PredicateRegistry factory create functionality."""
        @PredicateRegistry.register_factory('greater_than')
        def greater_than_factory(threshold):
            def predicate(value):
                return value > threshold
            predicate._factory_name = 'greater_than'
            predicate._factory_params = {'threshold': threshold}
            return predicate

        # Factory should be registered
        assert 'greater_than' in PredicateRegistry._factories

        # Create predicates using factory
        gt_5 = PredicateRegistry.create('greater_than', threshold=5)
        gt_10 = PredicateRegistry.create('greater_than', threshold=10)

        # Test they work correctly
        assert gt_5(6) is True
        assert gt_5(4) is False
        assert gt_10(11) is True
        assert gt_10(9) is False

        # Check attributes
        assert hasattr(gt_5, '_factory_name')
        assert gt_5._factory_name == 'greater_than'
        assert hasattr(gt_5, '_factory_params')
        assert gt_5._factory_params == {'threshold': 5}

    def test_create_non_existent_factory(self):
        """Test create returns None for non-existent factory."""
        result = PredicateRegistry.create('non_existent', param=1)
        assert result is None

    def test_in_set_factory(self):
        """[T-rule-03] Test example factory: in_set."""
        @PredicateRegistry.register_factory('in_set')
        def in_set_factory(valid_values):
            def predicate(value):
                return value in valid_values
            predicate._factory_name = 'in_set'
            predicate._factory_params = {'valid_values': valid_values}
            return predicate

        # Create predicate for specific set
        valid_types = PredicateRegistry.create('in_set', valid_values={'A', 'B', 'C'})

        assert valid_types('A') is True
        assert valid_types('B') is True
        assert valid_types('D') is False
        assert valid_types('') is False

    def test_between_factory(self):
        """[T-rule-03] Test example factory: between."""
        @PredicateRegistry.register_factory('between')
        def between_factory(min_val, max_val, inclusive=True):
            def predicate(value):
                if inclusive:
                    return min_val <= value <= max_val
                else:
                    return min_val < value < max_val
            predicate._factory_name = 'between'
            predicate._factory_params = {
                'min_val': min_val,
                'max_val': max_val,
                'inclusive': inclusive
            }
            return predicate

        # Test inclusive
        between_1_10 = PredicateRegistry.create('between', min_val=1, max_val=10)
        assert between_1_10(1) is True
        assert between_1_10(5) is True
        assert between_1_10(10) is True
        assert between_1_10(0) is False
        assert between_1_10(11) is False

        # Test exclusive
        between_1_10_exc = PredicateRegistry.create(
            'between',
            min_val=1,
            max_val=10,
            inclusive=False
        )
        assert between_1_10_exc(1) is False
        assert between_1_10_exc(5) is True
        assert between_1_10_exc(10) is False

    def test_regex_predicate(self):
        """Test regex-based predicate for match criteria."""
        @PredicateRegistry.register('matches_pattern')
        def matches_pattern(value, pattern):
            return bool(re.match(pattern, str(value)))

        pred = PredicateRegistry.get('matches_pattern')

        # Test with activation function pattern
        assert pred('relu', r'^relu') is True
        assert pred('leaky_relu', r'^relu') is False
        assert pred('conv2d_relu', r'.*relu$') is True

    def test_complex_predicate(self):
        """Test complex predicate with multiple checks."""
        @PredicateRegistry.register('valid_weight')
        def valid_weight(value):
            return (isinstance(value, int | float)
                    and -1.0 <= value <= 1.0
                    and value != 0)  # Non-zero constraint

        pred = PredicateRegistry.get('valid_weight')

        assert pred(0.5) is True
        assert pred(-0.5) is True
        assert pred(1.0) is True
        assert pred(-1.0) is True
        assert pred(0) is False  # Zero not allowed
        assert pred(1.1) is False  # Out of range
        assert pred('0.5') is False  # Wrong type

    def test_predicate_for_node_matching(self):
        """Test predicates designed for node match criteria."""
        @PredicateRegistry.register_factory('has_attribute')
        def has_attribute_factory(attr_name, expected_value=None):
            def predicate(node_attrs):
                if attr_name not in node_attrs:
                    return False
                if expected_value is not None:
                    return node_attrs[attr_name] == expected_value
                return True
            predicate._factory_name = 'has_attribute'
            predicate._factory_params = {
                'attr_name': attr_name,
                'expected_value': expected_value
            }
            return predicate

        # Check if node has specific attribute
        has_output_size = PredicateRegistry.create('has_attribute', attr_name='output_size')
        assert has_output_size({'output_size': 10}) is True
        assert has_output_size({'other_attr': 5}) is False

        # Check attribute value
        output_size_10 = PredicateRegistry.create(
            'has_attribute',
            attr_name='output_size',
            expected_value=10
        )
        assert output_size_10({'output_size': 10}) is True
        assert output_size_10({'output_size': 5}) is False

    def test_predicate_composition(self):
        """Test composing predicates for complex matching."""
        @PredicateRegistry.register('is_positive')
        def is_positive(value):
            return value > 0

        @PredicateRegistry.register('is_even')
        def is_even(value):
            return value % 2 == 0

        # Manual composition (AND)
        def positive_even(value):
            return (PredicateRegistry.get('is_positive')(value)
                    and PredicateRegistry.get('is_even')(value))

        assert positive_even(4) is True
        assert positive_even(3) is False  # Not even
        assert positive_even(-2) is False  # Not positive
        assert positive_even(0) is False  # Not positive

    def test_edge_match_predicates(self):
        """Test predicates for edge matching criteria."""
        @PredicateRegistry.register_factory('weight_near')
        def weight_near_factory(target, tolerance=0.1):
            def predicate(weight):
                return abs(weight - target) <= tolerance
            predicate._factory_name = 'weight_near'
            predicate._factory_params = {
                'target': target,
                'tolerance': tolerance
            }
            return predicate

        # Match edges with weight close to 0.5
        near_half = PredicateRegistry.create('weight_near', target=0.5, tolerance=0.1)
        assert near_half(0.5) is True
        assert near_half(0.45) is True
        assert near_half(0.55) is True
        assert near_half(0.4) is True
        assert near_half(0.6) is True
        assert near_half(0.39) is False
        assert near_half(0.61) is False

    def test_callable_as_predicate(self):
        """Test that any callable can be used as a predicate."""
        # Lambda
        def is_zero(x):
            return x == 0
        assert is_zero(0) is True
        assert is_zero(1) is False

        # Regular function
        def custom_check(value):
            return isinstance(value, str) and len(value) > 5

        assert custom_check("hello world") is True
        assert custom_check("hi") is False
        assert custom_check(12345) is False

        # Class with __call__
        class RangePredicate:
            def __init__(self, min_val, max_val):
                self.min_val = min_val
                self.max_val = max_val

            def __call__(self, value):
                return self.min_val <= value <= self.max_val

        in_range = RangePredicate(10, 20)
        assert in_range(15) is True
        assert in_range(5) is False
