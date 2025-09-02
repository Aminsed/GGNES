"""Serialization/deserialization functions.

Implements the rule, condition, and predicate (de)serialization per
project_guide.md Section 15.
"""

from __future__ import annotations

import uuid
from typing import Any

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


def serialize_predicate(predicate: Any) -> dict:
    """Serialize a predicate function.

    Supports factory-created predicates and registered predicates.
    """
    if hasattr(predicate, '_factory_name') and hasattr(predicate, '_factory_params'):
        return {
            'type': 'factory',
            'factory': getattr(predicate, '_factory_name'),
            'params': getattr(predicate, '_factory_params'),
        }

    if hasattr(predicate, '_predicate_name'):
        return {
            'type': 'registered',
            'name': getattr(predicate, '_predicate_name'),
        }

    raise ValueError(f"Cannot serialize predicate: {predicate}")


def serialize_condition(condition: Any) -> dict:
    """Serialize a condition function."""
    if hasattr(condition, 'standard_name') and hasattr(condition, 'params'):
        return {
            'type': 'parameterized',
            'name': getattr(condition, 'standard_name'),
            'params': getattr(condition, 'params'),
        }

    for name, func in ConditionRegistry._registry.items():  # type: ignore[attr-defined]
        if func == condition:
            return {'type': 'registered', 'name': name}

    raise ValueError(f"Cannot serialize condition: {condition}")


def serialize_lhs(lhs: LHSPattern) -> dict:
    """Serialize LHS pattern."""
    return {
        'nodes': [
            {
                'label': node['label'],
                'match_criteria': {
                    k: serialize_predicate(v) if callable(v) else v
                    for k, v in node.get('match_criteria', {}).items()
                },
            }
            for node in lhs.nodes
        ],
        'edges': [
            {
                'source_label': edge['source_label'],
                'target_label': edge['target_label'],
                'edge_label': edge.get('edge_label'),
                'match_criteria': {
                    k: serialize_predicate(v) if callable(v) else v
                    for k, v in edge.get('match_criteria', {}).items()
                },
            }
            for edge in lhs.edges
        ],
        'boundary_nodes': lhs.boundary_nodes,
    }


def serialize_rhs(rhs: RHSAction) -> dict:
    """Serialize RHS action."""
    return {
        'add_nodes': rhs.add_nodes,
        'add_edges': rhs.add_edges,
        'delete_nodes': rhs.delete_nodes,
        'delete_edges': rhs.delete_edges,
        'modify_nodes': rhs.modify_nodes,
        'modify_edges': rhs.modify_edges,
    }


def serialize_embedding(embedding: EmbeddingLogic) -> dict:
    """Serialize embedding logic."""
    return {
        'strategy': embedding.strategy.name,
        'connection_map': {
            f"{key[0]}:{key[1].name}": [
                {
                    'rhs_label': rhs_label,
                    'distribution': (dist.name if hasattr(dist, 'name') else dist),
                }
                for rhs_label, dist in value
            ]
            for key, value in embedding.connection_map.items()
        },
        'excess_connection_handling': embedding.excess_connection_handling,
        'unknown_direction_handling': embedding.unknown_direction_handling,
        'boundary_handling': embedding.boundary_handling,
    }


def serialize_rule(rule: Rule) -> dict:
    """Serialize a rule to dictionary."""
    return {
        'rule_id': str(rule.rule_id),
        'lhs': serialize_lhs(rule.lhs),
        'rhs': serialize_rhs(rule.rhs),
        'embedding': serialize_embedding(rule.embedding),
        'metadata': rule.metadata.copy(),
        'condition': serialize_condition(rule.condition) if rule.condition else None,
    }


def deserialize_predicate(data: dict) -> Any:
    """Deserialize a predicate function."""
    ptype = data.get('type')
    if ptype == 'factory':
        return PredicateRegistry.create(data['factory'], **data['params'])
    if ptype == 'registered':
        return PredicateRegistry.get(data['name'])
    raise ValueError(f"Unknown predicate type: {ptype}")


def deserialize_condition(data: dict) -> Any:
    """Deserialize a condition function."""
    ctype = data.get('type')
    if ctype == 'parameterized':
        return ConditionRegistry.create_parameterized(data['name'], **data['params'])
    if ctype == 'registered':
        return ConditionRegistry.get(data['name'])
    raise ValueError(f"Unknown condition type: {ctype}")


def deserialize_lhs(data: dict) -> LHSPattern:
    """Deserialize LHS pattern."""
    nodes: list[dict] = []
    for node_data in data['nodes']:
        node = {'label': node_data['label'], 'match_criteria': {}}
        for k, v in node_data.get('match_criteria', {}).items():
            if isinstance(v, dict) and 'type' in v:
                node['match_criteria'][k] = deserialize_predicate(v)
            else:
                node['match_criteria'][k] = v
        nodes.append(node)

    edges: list[dict] = []
    for edge_data in data['edges']:
        edge = {
            'source_label': edge_data['source_label'],
            'target_label': edge_data['target_label'],
            'match_criteria': {},
        }
        if edge_data.get('edge_label'):
            edge['edge_label'] = edge_data['edge_label']
        for k, v in edge_data.get('match_criteria', {}).items():
            if isinstance(v, dict) and 'type' in v:
                edge['match_criteria'][k] = deserialize_predicate(v)
            else:
                edge['match_criteria'][k] = v
        edges.append(edge)

    return LHSPattern(nodes=nodes, edges=edges, boundary_nodes=data['boundary_nodes'])


def deserialize_rhs(data: dict) -> RHSAction:
    """Deserialize RHS action."""
    return RHSAction(
        add_nodes=data.get('add_nodes', []),
        add_edges=data.get('add_edges', []),
        delete_nodes=data.get('delete_nodes', []),
        delete_edges=data.get('delete_edges', []),
        modify_nodes=data.get('modify_nodes', []),
        modify_edges=data.get('modify_edges', []),
    )


def deserialize_embedding(data: dict) -> EmbeddingLogic:
    """Deserialize embedding logic."""
    connection_map: dict = {}
    for key_str, value_list in data.get('connection_map', {}).items():
        boundary_label, direction_str = key_str.split(':')
        key = (boundary_label, Direction[direction_str])
        mapped: list[tuple[str, Any]] = []
        for item in value_list:
            dist_val = item['distribution']
            if isinstance(dist_val, str):
                # Enum by name
                try:
                    dist_parsed = Distribution[dist_val]
                except Exception:
                    dist_parsed = dist_val  # Preserve unknown as-is
            elif isinstance(dist_val, int):
                # Numeric distribution passes through as integer
                dist_parsed = int(dist_val)
            else:
                # Fallback: attempt enum by value, else keep as-is
                try:
                    dist_parsed = Distribution(dist_val)
                except Exception:
                    dist_parsed = dist_val
            mapped.append((item['rhs_label'], dist_parsed))
        connection_map[key] = mapped

    return EmbeddingLogic(
        strategy=EmbeddingStrategy[data['strategy']] if isinstance(data['strategy'], str) else EmbeddingStrategy(data['strategy']),
        connection_map=connection_map,
        excess_connection_handling=data.get('excess_connection_handling', 'WARNING'),
        unknown_direction_handling=data.get('unknown_direction_handling', 'WARNING'),
        boundary_handling=data.get('boundary_handling', 'PROCESS_LAST'),
    )


def deserialize_rule(data: dict) -> Rule:
    """Deserialize a rule from dictionary."""
    return Rule(
        rule_id=uuid.UUID(data['rule_id']),
        lhs=deserialize_lhs(data['lhs']),
        rhs=deserialize_rhs(data['rhs']),
        embedding=deserialize_embedding(data['embedding']),
        metadata=data.get('metadata', {}),
        condition=deserialize_condition(data['condition']) if data.get('condition') else None,
    )

