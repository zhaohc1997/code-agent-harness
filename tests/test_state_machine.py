import pytest

from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.types.state import SessionState


def test_valid_transition_idle_to_running() -> None:
    machine = EngineStateMachine(SessionState.IDLE)
    machine.transition(SessionState.RUNNING)
    assert machine.state is SessionState.RUNNING


def test_invalid_transition_completed_to_awaiting_confirmation() -> None:
    machine = EngineStateMachine(SessionState.COMPLETED)
    with pytest.raises(ValueError, match="Invalid transition"):
        machine.transition(SessionState.AWAITING_CONFIRMATION)


@pytest.mark.parametrize("bad_state", [None, "running"])
def test_invalid_runtime_inputs_raise_value_error(bad_state: object) -> None:
    machine = EngineStateMachine(SessionState.IDLE)

    with pytest.raises(ValueError):
        machine.transition(bad_state)  # type: ignore[arg-type]
