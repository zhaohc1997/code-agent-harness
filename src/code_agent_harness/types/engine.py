from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    content: list[object]
    stop_reason: str
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class RuntimeResult:
    state: str
    output_text: str
    messages: list[object]
