from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.types.state import SessionState


def test_valid_transition_idle_to_running() -> None:
    machine = EngineStateMachine(SessionState.IDLE)
    machine.transition(SessionState.RUNNING)
    assert machine.state is SessionState.RUNNING


def test_invalid_transition_completed_to_awaiting_confirmation() -> None:
    machine = EngineStateMachine(SessionState.COMPLETED)
    try:
        machine.transition(SessionState.AWAITING_CONFIRMATION)
    except ValueError as exc:
        assert "Invalid transition" in str(exc)
    else:
        raise AssertionError("expected transition failure")
