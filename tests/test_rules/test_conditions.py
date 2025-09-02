"""
Comprehensive tests for ConditionRegistry to ensure strict adherence to project_guide.md.
"""

from unittest.mock import Mock

import pytest

from ggnes.rules.conditions import ConditionRegistry


class TestConditionRegistry:
    """Tests for ConditionRegistry [T-rule-02]."""

    def setup_method(self):
        """Clear registry before each test."""
        # Clear both registries to ensure test isolation
        ConditionRegistry._registry = {}
        ConditionRegistry._standard_registry = {}

    def test_register_and_get_condition(self):
        """[T-rule-02] ConditionRegistry register/get basic functionality."""
        # Define a test condition
        @ConditionRegistry.register('test_condition')
        def test_cond(graph_view, bindings, graph_context):
            return graph_context.get('num_nodes', 0) > 5

        # Should be retrievable
        retrieved = ConditionRegistry.get('test_condition')
        assert retrieved is test_cond

        # Test it works
        mock_graph = Mock()
        bindings = {}
        context = {'num_nodes': 10}
        assert retrieved(mock_graph, bindings, context) is True

        context = {'num_nodes': 3}
        assert retrieved(mock_graph, bindings, context) is False

    def test_get_non_existent_condition(self):
        """Test get returns None for non-existent conditions."""
        result = ConditionRegistry.get('non_existent')
        assert result is None

    def test_register_multiple_conditions(self):
        """Test registering multiple conditions."""
        @ConditionRegistry.register('cond1')
        def cond1(g, b, c):
            return True

        @ConditionRegistry.register('cond2')
        def cond2(g, b, c):
            return False

        assert ConditionRegistry.get('cond1') is cond1
        assert ConditionRegistry.get('cond2') is cond2

    def test_register_standard_condition(self):
        """Test registering parameterizable standard conditions."""
        @ConditionRegistry.register_standard('node_count_greater')
        def node_count_greater(threshold, graph_view, bindings, graph_context):
            return graph_context.get('num_nodes', 0) > threshold

        # Should be in standard registry, not regular
        assert ConditionRegistry.get('node_count_greater') is None
        assert ConditionRegistry._standard_registry['node_count_greater'] is node_count_greater

    def test_create_parameterized_condition(self):
        """[T-rule-02] ConditionRegistry parameterized creation."""
        # Register a standard condition
        @ConditionRegistry.register_standard('depth_check')
        def depth_check(min_depth, max_depth, graph_view, bindings, graph_context):
            depth = graph_context.get('depth', 0)
            return min_depth <= depth <= max_depth

        # Create parameterized version
        depth_5_to_10 = ConditionRegistry.create_parameterized(
            'depth_check',
            min_depth=5,
            max_depth=10
        )

        # Test it works with bound parameters
        mock_graph = Mock()
        bindings = {}

        assert depth_5_to_10(mock_graph, bindings, {'depth': 7}) is True
        assert depth_5_to_10(mock_graph, bindings, {'depth': 3}) is False
        assert depth_5_to_10(mock_graph, bindings, {'depth': 12}) is False

        # Check attributes
        assert hasattr(depth_5_to_10, 'standard_name')
        assert depth_5_to_10.standard_name == 'depth_check'
        assert hasattr(depth_5_to_10, 'params')
        assert depth_5_to_10.params == {'min_depth': 5, 'max_depth': 10}

    def test_create_parameterized_non_existent(self):
        """Test create_parameterized raises for non-existent standard condition."""
        with pytest.raises(ValueError, match="No standard condition named fake_condition"):
            ConditionRegistry.create_parameterized('fake_condition', param=1)

    def test_compose_and(self):
        """[T-rule-02] ConditionRegistry compose with AND logic."""
        def cond1(g, b, c):
            return c.get('test1', False)

        def cond2(g, b, c):
            return c.get('test2', False)

        def cond3(g, b, c):
            return c.get('test3', False)

        and_condition = ConditionRegistry.compose_and(cond1, cond2, cond3)

        mock_graph = Mock()
        bindings = {}

        # All true -> True
        context = {'test1': True, 'test2': True, 'test3': True}
        assert and_condition(mock_graph, bindings, context) is True

        # One false -> False
        context = {'test1': True, 'test2': False, 'test3': True}
        assert and_condition(mock_graph, bindings, context) is False

        # All false -> False
        context = {'test1': False, 'test2': False, 'test3': False}
        assert and_condition(mock_graph, bindings, context) is False

    def test_compose_or(self):
        """[T-rule-02] ConditionRegistry compose with OR logic."""
        def cond1(g, b, c):
            return c.get('test1', False)

        def cond2(g, b, c):
            return c.get('test2', False)

        def cond3(g, b, c):
            return c.get('test3', False)

        or_condition = ConditionRegistry.compose_or(cond1, cond2, cond3)

        mock_graph = Mock()
        bindings = {}

        # All true -> True
        context = {'test1': True, 'test2': True, 'test3': True}
        assert or_condition(mock_graph, bindings, context) is True

        # One true -> True
        context = {'test1': False, 'test2': True, 'test3': False}
        assert or_condition(mock_graph, bindings, context) is True

        # All false -> False
        context = {'test1': False, 'test2': False, 'test3': False}
        assert or_condition(mock_graph, bindings, context) is False

    def test_compose_not(self):
        """[T-rule-02] ConditionRegistry compose with NOT logic."""
        def cond(g, b, c):
            return c.get('test', False)

        not_condition = ConditionRegistry.compose_not(cond)

        mock_graph = Mock()
        bindings = {}

        # True -> False
        assert not_condition(mock_graph, bindings, {'test': True}) is False

        # False -> True
        assert not_condition(mock_graph, bindings, {'test': False}) is True

    def test_compose_complex(self):
        """Test complex composition of conditions."""
        @ConditionRegistry.register('has_inputs')
        def has_inputs(g, b, c):
            return c.get('num_inputs', 0) > 0

        @ConditionRegistry.register('has_outputs')
        def has_outputs(g, b, c):
            return c.get('num_outputs', 0) > 0

        @ConditionRegistry.register('is_large')
        def is_large(g, b, c):
            return c.get('num_nodes', 0) > 10

        # (has_inputs AND has_outputs) OR (NOT is_large)
        complex_condition = ConditionRegistry.compose_or(
            ConditionRegistry.compose_and(
                ConditionRegistry.get('has_inputs'),
                ConditionRegistry.get('has_outputs')
            ),
            ConditionRegistry.compose_not(
                ConditionRegistry.get('is_large')
            )
        )

        mock_graph = Mock()
        bindings = {}

        # Has inputs and outputs -> True (first part true)
        context = {'num_inputs': 2, 'num_outputs': 1, 'num_nodes': 20}
        assert complex_condition(mock_graph, bindings, context) is True

        # Small graph -> True (second part true)
        context = {'num_inputs': 0, 'num_outputs': 0, 'num_nodes': 5}
        assert complex_condition(mock_graph, bindings, context) is True

        # Large graph without I/O -> False (both parts false)
        context = {'num_inputs': 0, 'num_outputs': 0, 'num_nodes': 15}
        assert complex_condition(mock_graph, bindings, context) is False

    def test_condition_signature(self):
        """Test conditions receive correct arguments."""
        call_tracker = Mock()

        @ConditionRegistry.register('tracking_condition')
        def tracking_cond(graph_view, bindings, graph_context):
            call_tracker(graph_view, bindings, graph_context)
            return True

        condition = ConditionRegistry.get('tracking_condition')

        mock_graph = Mock()
        test_bindings = {'A': 1, 'B': 2}
        test_context = {'num_nodes': 5, 'custom_metric': 42}

        condition(mock_graph, test_bindings, test_context)

        call_tracker.assert_called_once_with(mock_graph, test_bindings, test_context)

    def test_empty_composition(self):
        """Test compose methods with no arguments."""
        # AND with no conditions should return True (vacuous truth)
        and_empty = ConditionRegistry.compose_and()
        assert and_empty(Mock(), {}, {}) is True

        # OR with no conditions should return False
        or_empty = ConditionRegistry.compose_or()
        assert or_empty(Mock(), {}, {}) is False
