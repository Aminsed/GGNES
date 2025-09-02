from ggnes.hierarchical.embedding import BoundaryInfo, ExternalEdge, plan_embedding
from ggnes.core.graph import Graph
from ggnes.rules.rule import Direction


def test_planner_unknown_boundary_handling_warns():
    g = Graph()
    info = BoundaryInfo(node_id=5, external_in=(), external_out=())
    plan, warns = plan_embedding(
        g,
        {('X', Direction.IN): [('P', 'COPY_ALL')]},
        {'X': info},
        {'P': 10},
        boundary_handling='SOMETHING_UNKNOWN',
        unknown_policy='WARNING',
    )
    assert any('Unknown boundary handling' in w for w in warns)


def test_planner_excess_out_warning_emitted():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(),
        external_out=(ExternalEdge(5, 7, 0.2, True, {}), ExternalEdge(5, 8, 0.3, True, {})),
    )
    plan, warns = plan_embedding(
        g,
        {('X', Direction.OUT): [('P', 'CONNECT_SINGLE')]},
        {'X': info},
        {'P': 10},
        excess_policy='WARNING',
    )
    assert any('Excess OUT' in w for w in warns)


def test_planner_unknown_direction_unknown_branch():
    g = Graph()
    info = BoundaryInfo(node_id=5, external_in=(), external_out=())
    plan, warns = plan_embedding(
        g,
        {('X', 'DIAGONAL'): [('P', 'COPY_ALL')]},  # type: ignore[arg-type]
        {'X': info},
        {'P': 10},
        unknown_policy='WARNING',
    )
    # Planner treats unknown direction in the map as unknown; warning recorded
    assert any('Unknown direction' in w for w in warns)


