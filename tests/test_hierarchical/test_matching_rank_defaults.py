from ggnes.hierarchical import ModuleSpec, ParameterSpec, PortSpec
from ggnes.hierarchical.matching import rank_module_candidates, MatchCriteria, evaluate_match, make_param_predicate, make_port_predicate, make_attr_predicate


def _spec(name):
    return ModuleSpec(name=name, version=1, parameters=[ParameterSpec('x', default=2)], ports=[PortSpec('out', 2)], invariants=["out.size==x"])  # type: ignore[operator]


def test_rank_module_candidates_defaults_overrides_none():
    a = _spec('A')
    b = _spec('B')
    ranked = rank_module_candidates([b, a])  # overrides None triggers default path
    # Must be sorted by (name, version, signature)
    names = [spec.name for spec, _ in ranked]
    assert names == ['A', 'B']


def test_evaluate_match_multiple_predicates_cover_loops():
    spec = _spec('C')
    crit = MatchCriteria(
        param_predicates=(make_param_predicate(lambda env: env['x'] == 2), make_param_predicate(lambda env: 1 + 1 == 2)),
        port_predicates=(make_port_predicate(lambda ports: 'out' in ports), make_port_predicate(lambda ports: ports['out'].size == 2)),
        attr_predicates=(make_attr_predicate(lambda attrs: isinstance(attrs, dict)), make_attr_predicate(lambda attrs: True)),
    )
    assert evaluate_match(spec, {}, crit) is True


