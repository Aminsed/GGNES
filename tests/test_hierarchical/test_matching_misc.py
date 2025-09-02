from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.matching import evaluate_match, make_port_predicate, make_attr_predicate, rank_module_candidates_with_limit, MatchCriteria, make_param_predicate


def _spec_ok():
    return ModuleSpec(
        name="MM",
        version=1,
        parameters=[ParameterSpec("x", default=2)],
        ports=[PortSpec("out", 2)],
        invariants=["out.size == x"],
        attributes={"k": 1},
    )


def test_evaluate_match_no_criteria_returns_true():
    spec = _spec_ok()
    assert evaluate_match(spec, {}, None) is True


def test_make_port_and_attr_predicate_wrappers():
    spec = _spec_ok()
    ports_ok = make_port_predicate(lambda ports: ports["out"].size == 2)
    attrs_ok = make_attr_predicate(lambda attrs: attrs.get("k") == 1)
    crit = MatchCriteria(param_predicates=(), port_predicates=(ports_ok,), attr_predicates=(attrs_ok,))
    assert evaluate_match(spec, {}, crit) is True


def test_rank_module_candidates_with_limit_none_and_large_limit():
    specs = [_spec_ok(), _spec_ok()]
    # limit None returns all
    all_ranked = rank_module_candidates_with_limit(specs, overrides=[{}, {}], limit=None)
    assert len(all_ranked) == 2
    # large limit returns all but preserves deterministic order
    ranked = rank_module_candidates_with_limit(specs, overrides=[{}, {}], limit=10)
    assert [sig for _, sig in ranked] == sorted([sig for _, sig in ranked])


def test_evaluate_match_multiple_predicates_cover_loops_and_early_returns():
    spec = _spec_ok()
    # True predicates across all lists
    crit_true = MatchCriteria(
        param_predicates=(make_param_predicate(lambda env: env['x'] == 2), make_param_predicate(lambda env: True)),
        port_predicates=(make_port_predicate(lambda ports: 'out' in ports),),
        attr_predicates=(make_attr_predicate(lambda attrs: isinstance(attrs, dict)),),
    )
    assert evaluate_match(spec, {}, crit_true) is True
    # Early return cases
    crit_param_false = MatchCriteria(param_predicates=(make_param_predicate(lambda env: False),))
    assert evaluate_match(spec, {}, crit_param_false) is False
    crit_port_false = MatchCriteria(port_predicates=(make_port_predicate(lambda ports: False),))
    assert evaluate_match(spec, {}, crit_port_false) is False
    crit_attr_false = MatchCriteria(attr_predicates=(make_attr_predicate(lambda attrs: False),))
    assert evaluate_match(spec, {}, crit_attr_false) is False


