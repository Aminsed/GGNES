from ggnes.utils.serialization import deserialize_embedding, serialize_embedding
from ggnes.rules.rule import EmbeddingLogic, Direction


def test_deserialize_numeric_and_unknown_distribution_preserved():
    data = {
        'strategy': 'MAP_BOUNDARY_CONNECTIONS',
        'connection_map': {
            'X:IN': [
                {'rhs_label': 'P', 'distribution': 2},
                {'rhs_label': 'Q', 'distribution': 'CONNECT_SINGLE'},
                {'rhs_label': 'R', 'distribution': 'UNSUPPORTED'},
            ]
        },
        'excess_connection_handling': 'WARNING',
        'unknown_direction_handling': 'WARNING',
        'boundary_handling': 'PROCESS_LAST',
    }
    emb = deserialize_embedding(data)
    # Numeric stays int, CONNECT_SINGLE becomes enum, UNKNOWN remains string
    entries = emb.connection_map[('X', Direction.IN)]
    assert entries[0][1] == 2
    assert getattr(entries[1][1], 'name', None) == 'CONNECT_SINGLE'
    assert entries[2][1] == 'UNSUPPORTED'
    # Roundtrip retains shapes
    round_data = serialize_embedding(emb)
    assert round_data['connection_map']['X:IN'][0]['distribution'] == 2
    assert round_data['connection_map']['X:IN'][1]['distribution'] == 'CONNECT_SINGLE'
    assert round_data['connection_map']['X:IN'][2]['distribution'] == 'UNSUPPORTED'


