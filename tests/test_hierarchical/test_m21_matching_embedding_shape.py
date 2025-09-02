import pytest

from ggnes.core.graph import Graph
from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.matching import (
    MatchCriteria,
    evaluate_match,
    make_param_predicate,
    make_port_predicate,
    make_attr_predicate,
    rank_module_candidates,
)
from ggnes.hierarchical.embedding import (
    BoundaryInfo,
    ExternalEdge,
    plan_embedding,
)
from ggnes.rules.rule import Direction
from ggnes.utils.validation import ValidationError


def _spec():
    return ModuleSpec(
        name="M",
        version=1,
        parameters=[
            ParameterSpec("heads", default=2, domain=lambda v: isinstance(v, int) and v > 0),
            ParameterSpec("head_dim", default=8, domain=lambda v: isinstance(v, int) and v > 0),
            ParameterSpec("model_dim", default="=heads*head_dim", domain=lambda v: isinstance(v, int) and v > 0),
        ],
        ports=[PortSpec("in", 16), PortSpec("out", 16)],
        attributes={"agg": "sum", "scale": 1},
        invariants=["model_dim % heads == 0", "out.size == model_dim"],
    )


def test_matching_param_port_attr_predicates_all_true():
    spec = _spec()
    criteria = MatchCriteria(
        param_predicates=(
            make_param_predicate(lambda env: env["heads"] == 2),
            make_param_predicate(lambda env: env["model_dim"] == 16),
        ),
        port_predicates=(make_port_predicate(lambda ports: ports["out"].size == 16),),
        attr_predicates=(make_attr_predicate(lambda attrs: attrs["agg"] == "sum"),),
    )
    assert evaluate_match(spec, {}, criteria) is True


def test_matching_fails_when_any_predicate_false():
    spec = _spec()
    criteria = MatchCriteria(
        param_predicates=(make_param_predicate(lambda env: env["heads"] == 3),),
    )
    assert evaluate_match(spec, {}, criteria) is False


def test_matching_uses_overrides_and_invariant_validation():
    spec = _spec()
    # heads=2 ⇒ model_dim becomes 16 via expression (matches out.size)
    criteria = MatchCriteria(
        param_predicates=(make_param_predicate(lambda env: env["model_dim"] == 16),),
    )
    assert evaluate_match(spec, {"heads": 2}, criteria) is True
    # Invalid override that breaks invariant (out.size is 16, model_dim becomes 24)
    with pytest.raises(ValidationError):
        evaluate_match(spec, {"heads": 3, "head_dim": 8}, criteria)


def test_embedding_plan_copy_single_numeric_and_policies():
    g = Graph()
    # boundary node id 5 has two incoming from 1 and 2, and three outgoing to 7,8,9
    info = BoundaryInfo(
        node_id=5,
        external_in=(
            ExternalEdge(1, 5, 0.1, True, {}),
            ExternalEdge(2, 5, 0.2, True, {}),
        ),
        external_out=(
            ExternalEdge(5, 7, 0.3, True, {}),
            ExternalEdge(5, 8, 0.4, True, {}),
            ExternalEdge(5, 9, 0.5, True, {}),
        ),
    )

    # RHS nodes bound: P -> 11, Q -> 12
    rhs = {"P": 11, "Q": 12}

    # Map: IN → COPY_ALL then CONNECT_SINGLE; OUT → numeric=2
    conn_map = {
        ("X", Direction.IN): [("P", "COPY_ALL"), ("Q", "CONNECT_SINGLE")],
        ("X", Direction.OUT): [("P", 2)],
    }

    plan, warns = plan_embedding(g, conn_map, {"X": info}, rhs, excess_policy="WARNING", unknown_policy="WARNING")

    # Expected edges (per spec):
    # IN COPY_ALL to P: (1->11), (2->11)
    # IN CONNECT_SINGLE to Q: none, because COPY_ALL marks all as handled
    # OUT numeric=2 from P: first two sorted OUT edges ⇒ (11->7), (11->8)
    edges = {(e.source_id, e.target_id) for e in plan}
    assert (1, 11) in edges and (2, 11) in edges and (11, 7) in edges and (11, 8) in edges
    # Excess unhandled IN: none; OUT: one unhandled ⇒ warn present
    assert any("Excess OUT" in w for w in warns)


def test_embedding_unknown_policies_raise_or_warn():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(ExternalEdge(1, 5, 0.1, True, {}),),
        external_out=(),
    )
    # Unknown RHS label triggers WARNING by default
    plan, warns = plan_embedding(g, {("X", Direction.IN): [("NOPE", "CONNECT_SINGLE")]}, {"X": info}, {"P": 11})
    assert len(plan) == 0
    assert any("Unknown RHS label" in w for w in warns)

    # With ERROR policy, it raises
    with pytest.raises(ValueError):
        plan_embedding(
            g,
            {("X", Direction.IN): [("NOPE", "CONNECT_SINGLE")]},
            {"X": info},
            {"P": 11},
            unknown_policy="ERROR",
        )


def test_embedding_boundary_handling_ignore_and_process_last():
    g = Graph()
    infoA = BoundaryInfo(
        node_id=10,
        external_in=(ExternalEdge(1, 10, 0.1, True, {}),),
        external_out=(ExternalEdge(10, 3, 0.2, True, {}),),
    )
    infoB = BoundaryInfo(
        node_id=11,
        external_in=(ExternalEdge(2, 11, 0.3, True, {}),),
        external_out=(ExternalEdge(11, 4, 0.4, True, {}),),
    )
    rhs = {"P": 20}

    conn_map = {
        ("A", Direction.IN): [("P", "CONNECT_SINGLE")],
        ("B", Direction.OUT): [("P", "COPY_ALL")],
    }
    # IGNORE: produces empty plan
    plan0, warns0 = plan_embedding(g, conn_map, {"A": infoA, "B": infoB}, rhs, boundary_handling="IGNORE")
    assert plan0 == [] and warns0 == []

    # PROCESS_LAST: reversed processing order; deterministic but we just assert presence of both effects
    plan1, _ = plan_embedding(g, conn_map, {"A": infoA, "B": infoB}, rhs, boundary_handling="PROCESS_LAST")
    edges = {(e.source_id, e.target_id) for e in plan1}
    assert (1, 20) in edges  # IN connect single from A to P
    assert (20, 4) in edges  # OUT copy all from P to B's target


def test_embedding_boundary_handling_process_first_and_policy_matrix():
    g = Graph()
    info = BoundaryInfo(
        node_id=5,
        external_in=(ExternalEdge(1, 5, 0.1, True, {}),),
        external_out=(ExternalEdge(5, 7, 0.2, True, {}),),
    )
    rhs = {"P": 10}
    conn_map = {
        ("X", Direction.IN): [("P", "CONNECT_SINGLE")],
        ("X", Direction.OUT): [("P", "COPY_ALL")],
    }
    # PROCESS_FIRST should process in original order deterministically
    plan, warns = plan_embedding(
        g,
        conn_map,
        {"X": info},
        rhs,
        excess_policy="DROP",
        unknown_policy="WARNING",
        boundary_handling="PROCESS_FIRST",
    )
    edges = {(e.source_id, e.target_id) for e in plan}
    assert (1, 10) in edges and (10, 7) in edges

    # unknown RHS with ERROR raises
    with pytest.raises(ValueError):
        plan_embedding(
            g,
            {("X", Direction.IN): [("NOPE", "CONNECT_SINGLE")]},
            {"X": info},
            rhs,
            unknown_policy="ERROR",
            boundary_handling="PROCESS_FIRST",
        )

    # With numeric=0 there is no excess because zero picks nothing; ensure no error
    plan2, warns2 = plan_embedding(
        g,
        {("X", Direction.OUT): [("P", 0)]},
        {"X": info},
        rhs,
        excess_policy="DROP",
        boundary_handling="PROCESS_FIRST",
    )
    assert plan2 == []


def test_matching_deterministic_truncation_limit():
    specs = [
        _spec(),
        _spec(),
        _spec(),
    ]
    from ggnes.hierarchical.matching import rank_module_candidates_with_limit
    # Provide invariant-satisfying overrides so that binding succeeds
    ranked = rank_module_candidates_with_limit(
        specs,
        overrides=[
            {"heads": 2},                 # model_dim=16
            {"heads": 4, "head_dim": 4}, # model_dim=16
            {"heads": 8, "head_dim": 2}, # model_dim=16
        ],
        limit=2,
    )
    assert len(ranked) == 2
    # Deterministic subset (first two by signature after sorting)
    sigs = [sig for _, sig in ranked]
    assert sigs == sorted(sigs)


def test_rank_module_candidates_deterministic_by_signature():
    a = _spec()
    b = _spec()
    # Different overrides produce different signatures and deterministic ordering
    ranked = rank_module_candidates([a, b], overrides=[{"heads": 2}, {"heads": 4, "head_dim": 4}])
    # Sorted by (name, version, signature) — same name/version, so signature decides
    sigs = [sig for _, sig in ranked]
    assert sigs == sorted(sigs)


