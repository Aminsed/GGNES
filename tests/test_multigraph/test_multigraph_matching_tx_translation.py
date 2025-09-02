import pytest

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.generation.matching import find_subgraph_matches
from ggnes.generation.transaction import TransactionManager


# [T-mg-match-01]
def test_matching_binds_specific_edge_instance_with_edge_label():
    g = Graph(config={"multigraph": True})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })
    b = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })
    g.add_edge(a, b, {"weight": 0.1})
    g.add_edge(a, b, {"weight": 0.2})

    lhs = {
        "nodes": [
            {"label": "A", "match_criteria": {"node_type": NodeType.INPUT}},
            {"label": "B", "match_criteria": {"node_type": NodeType.HIDDEN}},
        ],
        "edges": [
            {
                "source_label": "A",
                "target_label": "B",
                "edge_label": "E",
                "match_criteria": {},
            }
        ],
    }
    matches = list(find_subgraph_matches(g, lhs, timeout_ms=100))
    assert matches, "Should match at least one binding"
    # Edge label E must be bound to a specific edge instance
    for m in matches:
        assert "E" in m and getattr(m["E"], "edge_id", None) is not None


# [T-mg-tx-01]
def test_transaction_allows_parallel_edges_when_multigraph():
    class DummyRNG:
        def get_state(self):
            return {"seed": 0, "contexts": {}}

        def set_state(self, s):
            pass

    g = Graph(config={"multigraph": True})
    a = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })
    b = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 2},
    })

    tm = TransactionManager(graph=g, rng_manager=DummyRNG())
    tm.begin()
    tm.buffer.add_edge(a, b, {"weight": 0.1})
    tm.buffer.add_edge(a, b, {"weight": 0.2})
    tm.commit()

    assert len(g.find_edges_by_endpoints(a, b)) == 2


# [T-mg-trans-01]
@pytest.mark.parametrize("agg", ["sum", "concat", "matrix_product"])
def test_translation_aggregates_parallel_edges_and_uses_per_edge_weights(agg):
    torch = pytest.importorskip("torch")
    from ggnes.translation.pytorch import to_pytorch_model

    g = Graph(config={"multigraph": True})
    inp = g.add_node({
        "node_type": NodeType.INPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 4},
    })
    hid = g.add_node({
        "node_type": NodeType.HIDDEN,
        "activation_function": "linear",
        "attributes": {"output_size": 4, "aggregation": agg},
    })
    out = g.add_node({
        "node_type": NodeType.OUTPUT,
        "activation_function": "linear",
        "attributes": {"output_size": 4},
    })
    # Two parallel edges with different weights; both should contribute
    g.add_edge(inp, hid, {"weight": 0.5})
    g.add_edge(inp, hid, {"weight": 1.5})
    # Connect hidden to output with unit weight to preserve scale
    g.add_edge(hid, out, {"weight": 1.0})

    model = to_pytorch_model(g)
    x = torch.ones(2, 4)
    y = model(x, reset_states=True)
    assert y.shape == (2, 4)
    # For sum aggregation without projections, expected scale ~ sum of weights (2.0)
    if agg == "sum":
        # Each edge multiplies input by its scalar weight parameter (broadcasted)
        # With two edges: (0.5 + 1.5) * ones = 2.0 * ones at hidden, unit edge to output, then + bias
        assert torch.allclose(y, torch.full((2, 4), 2.0, device=y.device), atol=1e-4)


