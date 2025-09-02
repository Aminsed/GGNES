from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.repair.repair import repair


def test_repair_disable_returns_unsuccessful():
    g = Graph()
    # Make graph invalid: missing output node reachability
    g.add_node({
        'node_type': NodeType.OUTPUT,
        'activation_function': 'linear',
        'attributes': {'output_size': 1},
    })
    ok, metrics = repair(g, {'strategy': 'DISABLE'})
    assert ok is False
    assert metrics['strategy'] == 'DISABLE'


