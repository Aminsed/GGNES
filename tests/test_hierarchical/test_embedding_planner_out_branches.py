from ggnes.hierarchical.embedding import BoundaryInfo, ExternalEdge, plan_embedding
from ggnes.core.graph import Graph
from ggnes.rules.rule import Direction


def _info_out_three():
    return BoundaryInfo(
        node_id=5,
        external_in=(),
        external_out=(
            ExternalEdge(5, 7, 0.2, True, {}),
            ExternalEdge(5, 8, 0.3, True, {}),
            ExternalEdge(5, 9, 0.4, True, {}),
        ),
    )


def test_out_copy_all_adds_all_targets():
    g = Graph()
    info = _info_out_three()
    plan, warns = plan_embedding(
        g,
        {('X', Direction.OUT): [('P', 'COPY_ALL')]},
        {'X': info},
        {'P': 10},
    )
    edges = {(e.source_id, e.target_id) for e in plan}
    assert edges == {(10, 7), (10, 8), (10, 9)}


def test_out_connect_single_picks_first_target():
    g = Graph()
    info = _info_out_three()
    plan, warns = plan_embedding(
        g,
        {('X', Direction.OUT): [('P', 'CONNECT_SINGLE')]},
        {'X': info},
        {'P': 10},
    )
    edges = {(e.source_id, e.target_id) for e in plan}
    assert edges == {(10, 7)}


def test_out_unknown_distribution_warns_and_skips():
    g = Graph()
    info = _info_out_three()
    plan, warns = plan_embedding(
        g,
        {('X', Direction.OUT): [('P', 'UNSUPPORTED')]},
        {'X': info},
        {'P': 10},
        unknown_policy='WARNING',
    )
    assert plan == [] and any('Unknown distribution' in w for w in warns)


