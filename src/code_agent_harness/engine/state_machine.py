from code_agent_harness.types.state import SessionState

ALLOWED_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.IDLE: {SessionState.RUNNING},
    SessionState.RUNNING: {
        SessionState.AWAITING_CONFIRMATION,
        SessionState.AWAITING_USER_INPUT,
        SessionState.COMPLETED,
        SessionState.FAILED,
        SessionState.CANCELLED,
    },
    SessionState.AWAITING_CONFIRMATION: {SessionState.RUNNING, SessionState.CANCELLED},
    SessionState.AWAITING_USER_INPUT: {SessionState.RUNNING, SessionState.CANCELLED},
    SessionState.COMPLETED: {SessionState.RUNNING},
    SessionState.FAILED: {SessionState.RUNNING},
    SessionState.CANCELLED: {SessionState.RUNNING},
}


class EngineStateMachine:
    def __init__(self, state: SessionState) -> None:
        self.state = state

    def transition(self, next_state: SessionState) -> None:
        allowed_states = ALLOWED_TRANSITIONS.get(self.state, set())
        if next_state not in allowed_states:
            raise ValueError(f"Invalid transition: {self.state.value} -> {next_state.value}")
        self.state = next_state
