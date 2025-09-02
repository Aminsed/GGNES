import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType


def test_per_edge_projections_keyed_by_edge_id_in_multigraph():
    torch = pytest.importorskip("torch")
    from ggnes.translation.pytorch import to_pytorch_model

    g = Graph(config={"multigraph": True})
    inp = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 3},
    })
    hid = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 5, "aggregation": "sum"},
    })
    out = g.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 5},
    })

    # Two parallel edges need projection 3->5; each edge must get its own projection module
    e1 = g.add_edge(inp, hid, {"weight": 0.5})
    e2 = g.add_edge(inp, hid, {"weight": 1.5})
    # Connect hidden to output to produce final outputs
    g.add_edge(hid, out, {"weight": 1.0})

    model = to_pytorch_model(g)

    # Projections should be keyed by edge_id (two distinct modules for the parallels)
    proj_keys = set(model.projections.keys())
    assert f"proj_{e1}" in proj_keys
    assert f"proj_{e2}" in proj_keys

    # Quick forward smoke: shape should be [batch, 5]
    x = torch.randn(2, 3)
    y = model(x, reset_states=True)
    assert y.shape == (2, 5)


