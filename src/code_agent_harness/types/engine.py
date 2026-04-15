from dataclasses import dataclass

from code_agent_harness.types.state import SessionState


@dataclass(frozen=True)
class LLMResponse:
    content: list[object]
    stop_reason: str
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class RuntimeResult:
    state: SessionState
    output_text: str
    messages: list[object]
