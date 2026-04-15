from types import MappingProxyType

from code_agent_harness.types.state import SessionState

_ALLOWED_TRANSITIONS = MappingProxyType({
    SessionState.IDLE: frozenset({SessionState.RUNNING}),
    SessionState.RUNNING: frozenset({
        SessionState.AWAITING_CONFIRMATION,
        SessionState.AWAITING_USER_INPUT,
        SessionState.COMPLETED,
        SessionState.FAILED,
        SessionState.CANCELLED,
    }),
    SessionState.AWAITING_CONFIRMATION: frozenset({SessionState.RUNNING, SessionState.CANCELLED}),
    SessionState.AWAITING_USER_INPUT: frozenset({SessionState.RUNNING, SessionState.CANCELLED}),
    SessionState.COMPLETED: frozenset({SessionState.RUNNING}),
    SessionState.FAILED: frozenset({SessionState.RUNNING}),
    SessionState.CANCELLED: frozenset({SessionState.RUNNING}),
})


class EngineStateMachine:
    def __init__(self, state: object) -> None:
        self.state = self._require_session_state(state, "current state")

    def transition(self, next_state: object) -> None:
        current_state = self._require_session_state(self.state, "current state")
        candidate_state = self._require_session_state(next_state, "next state")
        allowed_states = _ALLOWED_TRANSITIONS.get(current_state, frozenset())
        if candidate_state not in allowed_states:
            raise ValueError(f"Invalid transition: {current_state.value} -> {candidate_state.value}")
        self.state = candidate_state

    @staticmethod
    def _require_session_state(value: object, label: str) -> SessionState:
        if isinstance(value, SessionState):
            return value
        raise ValueError(f"{label} must be a SessionState, got {type(value).__name__}")
