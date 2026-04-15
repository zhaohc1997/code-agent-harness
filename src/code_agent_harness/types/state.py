from enum import Enum


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_USER_INPUT = "awaiting_user_input"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
