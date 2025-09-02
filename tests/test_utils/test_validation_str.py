from ggnes.utils.validation import EdgeError, NodeError, ValidationError


def test_validation_error_str_formats():
    nerr = NodeError(3, 'missing_output_size', 'Missing or invalid', output_size=None)
    s = str(nerr)
    assert s.startswith('[missing_output_size]')
    assert 'Missing or invalid' in s

    eid = 'edge-1'
    eerr = EdgeError(eid, 'non_finite_weight', 'Non-finite', weight=float('inf'))
    s2 = str(eerr)
    assert s2.startswith('[non_finite_weight]')
    assert 'Non-finite' in s2


"""Coverage test for ValidationError.__str__."""


def test_validation_error_str_contains_type_and_message():
    err = ValidationError('some_error', 'Something went wrong', code=123)
    s = str(err)
    assert '[some_error]' in s
    assert 'Something went wrong' in s


