from typing import Protocol

from code_agent_harness.types.engine import LLMRequest
from code_agent_harness.types.engine import LLMResponse


class LLMProvider(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        ...
