from code_agent_harness.engine.cycle_guard import CycleGuard


def test_cycle_guard_blocks_repeated_tool_calls() -> None:
    guard = CycleGuard(max_repeats=2)
    assert guard.record("read_file", {"path": "a.py"}) is False
    assert guard.record("read_file", {"path": "a.py"}) is False
    assert guard.record("read_file", {"path": "a.py"}) is True


def test_cycle_guard_normalizes_argument_order() -> None:
    guard = CycleGuard(max_repeats=2)

    assert guard.record("search", {"b": 2, "a": 1}) is False
    assert guard.record("search", {"a": 1, "b": 2}) is False
    assert guard.record("search", {"b": 2, "a": 1}) is True
