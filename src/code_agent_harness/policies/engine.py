from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class PolicyDecision:
    outcome: str
    reason: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)


class PolicyEngine(Protocol):
    def evaluate(self, tool_name: str, arguments: dict[str, object]) -> PolicyDecision:
        ...
