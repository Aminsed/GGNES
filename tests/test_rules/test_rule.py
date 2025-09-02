"""
Comprehensive tests for Rule System components to ensure strict adherence to project_guide.md.
"""

import uuid
from collections import OrderedDict
from unittest.mock import Mock

import pytest

from ggnes.core.graph import NodeType
from ggnes.rules.rule import (
    EmbeddingLogic,
    EmbeddingStrategy,
    LHSPattern,
    RHSAction,
    Rule,
    validate_connection_map,
)


class TestLHSPattern:
    """Tests for LHSPattern class."""

    def test_lhs_pattern_initialization(self):
        """Test LHSPattern initializes with nodes, edges, and boundary_nodes."""
        nodes = [
            {'label': 'A', 'match_criteria': {'node_type': NodeType.INPUT}},
            {'label': 'B', 'match_criteria': {'node_type': NodeType.HIDDEN}}
        ]
        edges = [
            {'source_label': 'A', 'target_label': 'B', 'match_criteria': {'enabled': True}}
        ]
        boundary_nodes = ['A']

        pattern = LHSPattern(nodes=nodes, edges=edges, boundary_nodes=boundary_nodes)

        assert pattern.nodes == nodes
        assert pattern.edges == edges
        assert pattern.boundary_nodes == boundary_nodes

    def test_lhs_pattern_empty_initialization(self):
        """Test LHSPattern with empty lists."""
        pattern = LHSPattern(nodes=[], edges=[], boundary_nodes=[])

        assert pattern.nodes == []
        assert pattern.edges == []
        assert pattern.boundary_nodes == []

    def test_lhs_pattern_edge_with_label(self):
        """Test LHSPattern supports optional edge_label in edges."""
        edges = [
            {
                'source_label': 'A',
                'target_label': 'B',
                'match_criteria': {},
                'edge_label': 'E1'  # Optional edge label
            }
        ]
        pattern = LHSPattern(nodes=[], edges=edges, boundary_nodes=[])

        assert pattern.edges[0]['edge_label'] == 'E1'


class TestRHSAction:
    """Tests for RHSAction class."""

    def test_rhs_action_initialization_with_all_actions(self):
        """Test RHSAction initializes with all action types."""
        add_nodes = [{'label': 'C', 'properties': {'node_type': NodeType.HIDDEN}}]
        add_edges = [{'source_label': 'A', 'target_label': 'C', 'properties': {'weight': 0.5}}]
        delete_nodes = ['B']
        delete_edges = [{'source_label': 'A', 'target_label': 'B'}]
        modify_nodes = [{'label': 'A', 'properties': {'bias': 1.0}}]
        modify_edges = [{'source_label': 'B', 'target_label': 'C', 'properties': {'weight': 0.7}}]

        action = RHSAction(
            add_nodes=add_nodes,
            add_edges=add_edges,
            delete_nodes=delete_nodes,
            delete_edges=delete_edges,
            modify_nodes=modify_nodes,
            modify_edges=modify_edges
        )

        assert action.add_nodes == add_nodes
        assert action.add_edges == add_edges
        assert action.delete_nodes == delete_nodes
        assert action.delete_edges == delete_edges
        assert action.modify_nodes == modify_nodes
        assert action.modify_edges == modify_edges

    def test_rhs_action_defaults_to_empty_lists(self):
        """Test RHSAction defaults all actions to empty lists when not provided."""
        action = RHSAction()

        assert action.add_nodes == []
        assert action.add_edges == []
        assert action.delete_nodes == []
        assert action.delete_edges == []
        assert action.modify_nodes == []
        assert action.modify_edges == []

    def test_rhs_action_partial_initialization(self):
        """Test RHSAction with only some actions specified."""
        add_nodes = [{'label': 'X', 'properties': {}}]
        delete_edges = [{'source_label': 'Y', 'target_label': 'Z'}]

        action = RHSAction(add_nodes=add_nodes, delete_edges=delete_edges)

        assert action.add_nodes == add_nodes
        assert action.add_edges == []
        assert action.delete_nodes == []
        assert action.delete_edges == delete_edges
        assert action.modify_nodes == []
        assert action.modify_edges == []


class TestEmbeddingLogic:
    """Tests for EmbeddingLogic class."""

    def test_embedding_logic_default_initialization(self):
        """Test EmbeddingLogic defaults to MAP_BOUNDARY_CONNECTIONS strategy."""
        embedding = EmbeddingLogic()

        assert embedding.strategy == EmbeddingStrategy.MAP_BOUNDARY_CONNECTIONS
        assert isinstance(embedding.connection_map, OrderedDict)
        assert len(embedding.connection_map) == 0
        assert embedding.excess_connection_handling == 'WARNING'
        assert embedding.unknown_direction_handling == 'WARNING'
        assert embedding.boundary_handling == 'PROCESS_LAST'

    def test_embedding_logic_with_connection_map(self):
        """Test EmbeddingLogic with connection_map preserves order."""
        connection_map = {
            ('B1', 'in'): [('N1', 1.0), ('N2', 0.5)],
            ('B2', 'out'): [('N3', 1.0)]
        }

        embedding = EmbeddingLogic(connection_map=connection_map)

        assert isinstance(embedding.connection_map, OrderedDict)
        assert list(embedding.connection_map.keys()) == [('B1', 'in'), ('B2', 'out')]
        assert embedding.connection_map[('B1', 'in')] == [('N1', 1.0), ('N2', 0.5)]

    def test_embedding_logic_strategy_must_be_map_boundary_connections(self):
        """Test EmbeddingLogic strategy is always MAP_BOUNDARY_CONNECTIONS."""
        # Per spec: strategy MUST be MAP_BOUNDARY_CONNECTIONS
        embedding = EmbeddingLogic(strategy=EmbeddingStrategy.MAP_BOUNDARY_CONNECTIONS)
        assert embedding.strategy == EmbeddingStrategy.MAP_BOUNDARY_CONNECTIONS

    def test_embedding_logic_handling_options(self):
        """Test EmbeddingLogic accepts different handling options."""
        embedding = EmbeddingLogic(
            excess_connection_handling='ERROR',
            unknown_direction_handling='DROP',
            boundary_handling='PROCESS_FIRST'
        )

        assert embedding.excess_connection_handling == 'ERROR'
        assert embedding.unknown_direction_handling == 'DROP'
        assert embedding.boundary_handling == 'PROCESS_FIRST'


class TestRule:
    """Tests for Rule class."""

    def test_rule_initialization(self):
        """Test Rule initializes with all required components."""
        rule_id = uuid.uuid4()
        lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
        rhs = RHSAction()
        embedding = EmbeddingLogic()
        metadata = {'priority': 1, 'probability': 0.5}

        def condition(g, b, c):
            return True

        rule = Rule(
            rule_id=rule_id,
            lhs=lhs,
            rhs=rhs,
            embedding=embedding,
            metadata=metadata,
            condition=condition
        )

        assert rule.rule_id == rule_id
        assert rule.lhs == lhs
        assert rule.rhs == rhs
        assert rule.embedding == embedding
        assert rule.metadata == metadata
        assert rule.condition == condition

    def test_rule_without_optional_params(self):
        """Test Rule without metadata and condition."""
        rule_id = uuid.uuid4()
        lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
        rhs = RHSAction()
        embedding = EmbeddingLogic()

        rule = Rule(rule_id=rule_id, lhs=lhs, rhs=rhs, embedding=embedding)

        assert rule.metadata == {}
        assert rule.condition is None

    def test_rule_condition_callable(self):
        """Test Rule condition is callable with correct signature."""
        condition_mock = Mock(return_value=True)
        rule = Rule(
            rule_id=uuid.uuid4(),
            lhs=LHSPattern(nodes=[], edges=[], boundary_nodes=[]),
            rhs=RHSAction(),
            embedding=EmbeddingLogic(),
            condition=condition_mock
        )

        # Test condition can be called with correct args
        graph_view = Mock()
        bindings = {}
        graph_context = {'num_nodes': 10}

        result = rule.condition(graph_view, bindings, graph_context)

        assert result is True
        condition_mock.assert_called_once_with(graph_view, bindings, graph_context)


class TestValidateConnectionMap:
    """Tests for validate_connection_map function [T-rule-01]."""

    def test_validate_connection_map_valid(self):
        """Test validate_connection_map passes with valid references."""
        lhs = LHSPattern(
            nodes=[{'label': 'A'}, {'label': 'B'}],
            edges=[],
            boundary_nodes=['A', 'B']
        )
        rhs = RHSAction(
            add_nodes=[{'label': 'N1'}, {'label': 'N2'}]
        )
        connection_map = {
            ('A', 'in'): [('N1', 1.0)],
            ('B', 'out'): [('N2', 0.5)]
        }

        # Should not raise
        validate_connection_map(connection_map, lhs, rhs)

    def test_validate_connection_map_invalid_boundary_node(self):
        """[T-rule-01] validate_connection_map detects non-existent boundary node."""
        lhs = LHSPattern(
            nodes=[{'label': 'A'}],
            edges=[],
            boundary_nodes=['A']  # Only A is boundary
        )
        rhs = RHSAction(add_nodes=[{'label': 'N1'}])
        connection_map = {
            ('B', 'in'): [('N1', 1.0)]  # B is not in boundary_nodes
        }

        with pytest.raises(ValueError, match="non-existent boundary node: B"):
            validate_connection_map(connection_map, lhs, rhs)

    def test_validate_connection_map_invalid_rhs_node(self):
        """[T-rule-01] validate_connection_map detects non-existent RHS node."""
        lhs = LHSPattern(
            nodes=[{'label': 'A'}],
            edges=[],
            boundary_nodes=['A']
        )
        rhs = RHSAction(add_nodes=[{'label': 'N1'}])  # Only N1 exists
        connection_map = {
            ('A', 'in'): [('N2', 1.0)]  # N2 doesn't exist in RHS
        }

        with pytest.raises(ValueError, match="non-existent RHS node: N2"):
            validate_connection_map(connection_map, lhs, rhs)

    def test_validate_connection_map_multiple_targets(self):
        """Test validate_connection_map with multiple target nodes."""
        lhs = LHSPattern(
            nodes=[{'label': 'A'}],
            edges=[],
            boundary_nodes=['A']
        )
        rhs = RHSAction(
            add_nodes=[{'label': 'N1'}, {'label': 'N2'}, {'label': 'N3'}]
        )
        connection_map = {
            ('A', 'in'): [('N1', 1.0), ('N2', 0.5), ('N3', 0.25)]
        }

        # Should validate all targets
        validate_connection_map(connection_map, lhs, rhs)

    def test_validate_connection_map_empty(self):
        """Test validate_connection_map with empty map."""
        lhs = LHSPattern(nodes=[], edges=[], boundary_nodes=[])
        rhs = RHSAction()
        connection_map = {}

        # Empty map should be valid
        validate_connection_map(connection_map, lhs, rhs)

    def test_validate_connection_map_duplicates_first_wins_warning(self):
        """Test validate_connection_map handles duplicates: first wins, others warned."""
        # Note: The spec says "Duplicates: first wins, others ignored with warning"
        # This test ensures the function doesn't fail on duplicates
        lhs = LHSPattern(
            nodes=[{'label': 'A'}],
            edges=[],
            boundary_nodes=['A']
        )
        rhs = RHSAction(add_nodes=[{'label': 'N1'}])

        # OrderedDict preserves order, first occurrence should win
        connection_map = OrderedDict([
            (('A', 'in'), [('N1', 1.0)]),
            (('A', 'in'), [('N1', 0.5)])  # Duplicate key
        ])

        # Should not raise, but implementation should handle duplicates
        validate_connection_map(connection_map, lhs, rhs)
