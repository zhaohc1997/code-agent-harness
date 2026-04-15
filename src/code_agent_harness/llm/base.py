from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    content: list[object]
    stop_reason: str
    usage: dict[str, int] | None = None


class LLMProvider(Protocol):
    def generate(
        self,
        system_prompt: str,
        messages: list[object],
        tools: list[object],
        extra: dict[str, object],
    ) -> LLMResponse:
        ...
