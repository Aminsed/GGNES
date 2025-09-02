import warnings

from ggnes.translation.state_manager import StateManager

try:
    import torch  # type: ignore
    _torch_available = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _torch_available = False


def test_initialize_sets_batch_and_device():
    sm = StateManager()
    sm.initialize(batch_size=4, device="cpu")
    assert sm.batch_size == 4
    assert sm.device == "cpu"


def test_get_state_default_handles_torch_presence():
    sm = StateManager()
    sm.initialize(batch_size=2, device="cpu")
    state = sm.get_state(1, hidden_size=3, state_type="default")
    if _torch_available:
        # Torch tensor of zeros with shape [batch, hidden]
        assert hasattr(state, 'shape')
        assert list(state.shape) == [2, 3]
        assert (state == 0).all().item() is True
    else:
        assert isinstance(state, list)
        assert state == [0.0] * 3


def test_get_state_lstm_handles_torch_presence():
    sm = StateManager()
    sm.initialize(batch_size=2, device="cpu")
    h, c = sm.get_state(2, hidden_size=5, state_type="lstm")
    if _torch_available:
        assert hasattr(h, 'shape') and hasattr(c, 'shape')
        assert list(h.shape) == [2, 5]
        assert list(c.shape) == [2, 5]
        assert (h == 0).all().item() is True
        assert (c == 0).all().item() is True
    else:
        assert isinstance(h, list) and isinstance(c, list)
        assert h == [0.0] * 5 and c == [0.0] * 5


def test_update_and_get_prev_output_and_reset():
    sm = StateManager(max_sequence_length=2)
    sm.initialize(batch_size=1, device="cpu")
    sm.update_state(3, [1.0, 2.0])
    assert sm.states[3] == [1.0, 2.0]

    sm.store_output(3, [0.1, 0.2])
    assert sm.get_prev_output(3) == [0.1, 0.2]

    sm.reset()
    assert sm.get_prev_output(3) is None
    assert sm.states == {}
    assert sm.sequence_length == 0


def test_sequence_length_warning_emitted():
    sm = StateManager(max_sequence_length=1)
    sm.initialize(batch_size=1, device="cpu")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sm.store_output(1, [0.0])  # first timestep sets sequence_length to 1
        sm.store_output(1, [0.0])  # second should warn
        assert any(item.category is ResourceWarning for item in w)


def test_get_state_default_forced_no_torch_fallback(monkeypatch):
    sm = StateManager()
    sm.initialize(batch_size=2, device="cpu")

    import builtins as _b

    real_import = _b.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - harness
        if name == 'torch':
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_b, "__import__", fake_import)

    state = sm.get_state(10, hidden_size=3, state_type="default")
    assert isinstance(state, list) and state == [0.0] * 3


def test_get_state_lstm_forced_no_torch_fallback(monkeypatch):
    sm = StateManager()
    sm.initialize(batch_size=2, device="cpu")

    import builtins as _b

    real_import = _b.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - harness
        if name == 'torch':
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_b, "__import__", fake_import)

    h, c = sm.get_state(11, hidden_size=5, state_type="lstm")
    assert isinstance(h, list) and isinstance(c, list)
    assert h == [0.0] * 5 and c == [0.0] * 5
