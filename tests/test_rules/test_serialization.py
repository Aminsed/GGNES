"""
Serialization/Deserialization tests for rules, predicates, and conditions.

Strictly adheres to project_guide.md Section 15.
"""

from __future__ import annotations

import uuid

import pytest

from ggnes.rules.conditions import ConditionRegistry
from ggnes.rules.predicates import PredicateRegistry
from ggnes.rules.rule import (
    Direction,
    Distribution,
    EmbeddingLogic,
    EmbeddingStrategy,
    LHSPattern,
    RHSAction,
    Rule,
)
from ggnes.utils.serialization import (
    deserialize_condition,
    deserialize_embedding,
    deserialize_lhs,
    deserialize_predicate,
    deserialize_rhs,
    deserialize_rule,
    serialize_condition,
    serialize_embedding,
    serialize_lhs,
    serialize_predicate,
    serialize_rhs,
    serialize_rule,
)


def setup_module(_module):
    # Clear registries to ensure deterministic tests
    ConditionRegistry._registry = {}
    ConditionRegistry._standard_registry = {}
    PredicateRegistry._registry = {}
    PredicateRegistry._factories = {}


def test_predicate_serialization_roundtrip_factories_and_registered():
    @PredicateRegistry.register_factory('greater_than')
    def greater_than_factory(threshold):
        def predicate(value):
            return value > threshold
        predicate._factory_name = 'greater_than'
        predicate._factory_params = {'threshold': threshold}
        return predicate

    @PredicateRegistry.register('is_relu')
    def is_relu(value):
        return value == 'relu'

    gt_5 = PredicateRegistry.create('greater_than', threshold=5)
    ser_gt = serialize_predicate(gt_5)
    pred_gt = deserialize_predicate(ser_gt)
    assert pred_gt(6) is True and pred_gt(4) is False

    ser_reg = serialize_predicate(is_relu)
    pred_reg = deserialize_predicate(ser_reg)
    assert pred_reg('relu') is True and pred_reg('tanh') is False


def test_predicate_serialize_unknown_raises():
    with pytest.raises(ValueError):
        serialize_predicate(lambda x: x)


def test_condition_serialization_roundtrip_parameterized_and_registered():
    @ConditionRegistry.register_standard('min_nodes')
    def min_nodes(graph_view, bindings, graph_context, threshold):
        return graph_context.get('num_nodes', 0) >= threshold

    @ConditionRegistry.register('always_true')
    def always_true(graph_view, bindings, graph_context):
        return True

    # Parameterized
    param = ConditionRegistry.create_parameterized('min_nodes', threshold=3)
    ser_param = serialize_condition(param)
    cond_param = deserialize_condition(ser_param)
    assert cond_param(None, {}, {'num_nodes': 4}) is True
    assert cond_param(None, {}, {'num_nodes': 2}) is False

    # Registered
    ser_reg = serialize_condition(ConditionRegistry.get('always_true'))
    cond_reg = deserialize_condition(ser_reg)
    assert cond_reg(None, {}, {}) is True


def test_condition_serialize_unknown_raises():
    def unregistered(g, b, c):
        return False

    with pytest.raises(ValueError):
        serialize_condition(unregistered)


def test_lhs_rhs_serialization_roundtrip_with_predicates_and_edge_labels():
    @PredicateRegistry.register_factory('in_set')
    def in_set_factory(values):
        def predicate(value):
            return value in values
        predicate._factory_name = 'in_set'
        predicate._factory_params = {'values': values}
        return predicate

    lhs = LHSPattern(
        nodes=[
            {'label': 'A', 'match_criteria': {'node_type': PredicateRegistry.create('in_set', values={'INPUT'})}},
            # Include a literal to exercise non-callable path during serialize/deserialize
            {'label': 'B', 'match_criteria': {'node_type': 'HIDDEN'}},
        ],
        edges=[
            {
                'source_label': 'A',
                'target_label': 'B',
                # Use factory predicate in edge criteria to cover deserialize_predicate for edges
                'match_criteria': {'enabled': PredicateRegistry.create('in_set', values={True})},
                'edge_label': 'E1',
            },
            {
                'source_label': 'B',
                'target_label': 'A',
                # Literal criterion exercises non-predicate branch and edge without edge_label
                'match_criteria': {'enabled': False},
            },
        ],
        boundary_nodes=['A']
    )

    rhs = RHSAction(
        add_nodes=[{'label': 'N1', 'properties': {'activation_function': 'relu'}}],
        add_edges=[{'source_label': 'A', 'target_label': 'N1', 'properties': {'weight': 0.5}}],
        delete_nodes=['B'],
        delete_edges=[{'source_label': 'A', 'target_label': 'B'}],
        modify_nodes=[{'label': 'A', 'properties': {'bias': 0.1}}],
        modify_edges=[{'source_label': 'B', 'target_label': 'N1', 'properties': {'weight': 0.7}}],
    )

    ser_lhs = serialize_lhs(lhs)
    deser_lhs = deserialize_lhs(ser_lhs)
    assert isinstance(deser_lhs, LHSPattern)
    assert deser_lhs.boundary_nodes == ['A']
    assert deser_lhs.edges[0]['edge_label'] == 'E1'
    # Edge predicate should be callable and behave as configured
    edge_pred = deser_lhs.edges[0]['match_criteria']['enabled']
    assert callable(edge_pred)
    assert edge_pred(True) is True
    assert edge_pred(False) is False

    ser_rhs = serialize_rhs(rhs)
    deser_rhs = deserialize_rhs(ser_rhs)
    assert isinstance(deser_rhs, RHSAction)
    assert deser_rhs.delete_nodes == ['B']
    assert deser_rhs.modify_nodes[0]['properties']['bias'] == 0.1


def test_embedding_serialization_roundtrip_with_enums():
    emb = EmbeddingLogic(
        strategy=EmbeddingStrategy.MAP_BOUNDARY_CONNECTIONS,
        connection_map={
            ('X', Direction.IN): [('Y', Distribution.COPY_ALL)],
            ('X', Direction.OUT): [('Z', Distribution.CONNECT_SINGLE)],
        },
        excess_connection_handling='WARNING',
        unknown_direction_handling='ERROR',
        boundary_handling='PROCESS_FIRST',
    )

    ser = serialize_embedding(emb)
    deser = deserialize_embedding(ser)
    assert isinstance(deser, EmbeddingLogic)
    assert list(deser.connection_map.keys()) == [('X', Direction.IN), ('X', Direction.OUT)]
    assert deser.excess_connection_handling == 'WARNING'
    assert deser.unknown_direction_handling == 'ERROR'
    assert deser.boundary_handling == 'PROCESS_FIRST'


def test_rule_serialization_roundtrip_with_condition_and_metadata():
    @ConditionRegistry.register('always_true')
    def always_true(graph_view, bindings, graph_context):
        return True

    lhs = LHSPattern(nodes=[{'label': 'A', 'match_criteria': {}}], edges=[], boundary_nodes=['A'])
    rhs = RHSAction(add_nodes=[{'label': 'Y', 'properties': {}}])
    emb = EmbeddingLogic(connection_map={('A', Direction.IN): [('Y', Distribution.COPY_ALL)]})
    rule = Rule(rule_id=uuid.uuid4(), lhs=lhs, rhs=rhs, embedding=emb,
                metadata={'priority': 2, 'probability': 0.4},
                condition=ConditionRegistry.get('always_true'))

    ser = serialize_rule(rule)
    deser = deserialize_rule(ser)

    assert deser.metadata == {'priority': 2, 'probability': 0.4}
    assert callable(deser.condition)
    assert deser.condition(None, {}, {}) is True


def test_deserialize_invalid_condition_type_raises():
    with pytest.raises(ValueError):
        deserialize_condition({'type': 'unknown'})


def test_deserialize_invalid_predicate_type_raises():
    with pytest.raises(ValueError):
        deserialize_predicate({'type': 'unknown'})


