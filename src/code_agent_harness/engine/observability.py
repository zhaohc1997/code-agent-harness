from dataclasses import dataclass, field
from datetime import datetime, timezone

from code_agent_harness.storage.logs import StructuredLogger


@dataclass(frozen=True)
class DecisionPointEvent:
    event_name: str
    session_id: str
    turn_id: int
    component: str
    status: str
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, object]:
        return {
            "event_name": self.event_name,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "component": self.component,
            "status": self.status,
            "metadata": self.metadata,
        }


class Observability:
    def __init__(self, logger: StructuredLogger) -> None:
        self.logger = logger

    def emit(self, event: DecisionPointEvent) -> None:
        self.logger.append(event.as_dict())
