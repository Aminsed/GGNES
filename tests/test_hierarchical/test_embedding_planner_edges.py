import pytest

from ggnes.hierarchical.embedding import BoundaryInfo, ExternalEdge, plan_embedding
from ggnes.core.graph import Graph
from ggnes.rules.rule import Direction


def test_embedding_unknown_boundary_label_error():
    g = Graph()
    info = BoundaryInfo(node_id=1, external_in=(), external_out=())
    with pytest.raises(ValueError):
        plan_embedding(
            g,
            {("MISSING", Direction.IN): [("P", "COPY_ALL")]},
            {"X": info},
            {"P": 2},
            unknown_policy="ERROR",
        )


def test_embedding_unknown_boundary_handling_warning():
    g = Graph()
    info = BoundaryInfo(node_id=1, external_in=(), external_out=())
    plan, warns = plan_embedding(
        g,
        {("X", Direction.IN): [("P", "COPY_ALL")]},
        {"X": info},
        {"P": 2},
        unknown_policy="WARNING",
        boundary_handling="FOO",
    )
    # Unknown handling produces a warning but still returns a plan (empty here)
    assert isinstance(warns, list) and any("Unknown boundary handling" in w for w in warns)


def test_embedding_excess_in_error():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(ExternalEdge(1, 5, 0.1, True, {}), ExternalEdge(2, 5, 0.2, True, {})),
        external_out=(),
    )
    with pytest.raises(ValueError):
        plan_embedding(
            g,
            {("X", Direction.IN): [("P", "CONNECT_SINGLE")]},
            {"X": info},
            {"P": 10},
            excess_policy="ERROR",
        )


def test_embedding_numeric_zero_int_results_no_edges():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(ExternalEdge(1, 5, 0.1, True, {}),),
        external_out=(ExternalEdge(5, 7, 0.2, True, {}),),
    )
    plan, warns = plan_embedding(
        g,
        {("X", Direction.IN): [("P", 0)], ("X", Direction.OUT): [("P", 0)]},
        {"X": info},
        {"P": 10},
        excess_policy="DROP",
    )
    assert plan == []


def test_embedding_in_numeric_positive_selects_k_sources():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(
            ExternalEdge(1, 5, 0.1, True, {}),
            ExternalEdge(2, 5, 0.2, True, {}),
            ExternalEdge(3, 5, 0.3, True, {}),
        ),
        external_out=(),
    )
    plan, warns = plan_embedding(
        g,
        {('X', Direction.IN): [('P', 2)]},
        {'X': info},
        {'P': 10},
        excess_policy='DROP',
    )
    edges = {(e.source_id, e.target_id) for e in plan}
    assert edges == {(1, 10), (2, 10)}


def test_embedding_unknown_distribution_warns_and_skips():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(ExternalEdge(1, 5, 0.1, True, {}),),
        external_out=(),
    )
    plan, warns = plan_embedding(
        g,
        {('X', Direction.IN): [('P', 'UNSUPPORTED')]},
        {'X': info},
        {'P': 10},
        unknown_policy='WARNING',
    )
    assert plan == [] and any('Unknown distribution' in w for w in warns)


