from ggnes.hierarchical.embedding import BoundaryInfo, ExternalEdge, plan_embedding
from ggnes.core.graph import Graph
from ggnes.rules.rule import Direction


def test_embedding_unknown_direction_warning_paths():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(ExternalEdge(1, 5, 0.1, True, {}),),
        external_out=(ExternalEdge(5, 7, 0.2, True, {}),),
    )
    # Provide only IN mapping; OUT should trigger a warning when unknown_policy is WARNING
    plan, warns = plan_embedding(
        g,
        {('X', Direction.IN): [('P', 'COPY_ALL')]},
        {'X': info},
        {'P': 11},
        unknown_policy='WARNING'
    )
    # Missing OUT mapping does not produce a warning in planner (only embedding in RuleEngine does)
    assert warns == []


def test_embedding_unknown_direction_key_warns():
    g = Graph()
    info = BoundaryInfo(node_id=5, external_in=(), external_out=())
    # Provide an invalid direction key to trigger unknown direction handling path
    plan, warns = plan_embedding(
        g,
        {('X', 'SIDEWAYS'): [('P', 'COPY_ALL')]},  # type: ignore[arg-type]
        {'X': info},
        {'P': 10},
        unknown_policy='WARNING'
    )
    assert any('Unknown direction' in w for w in warns)


