from code_agent_harness.llm.base import LLMRequest
from code_agent_harness.llm.base import LLMResponse


class OpenAICompatibleProvider:
    """Phase 1 stub for an OpenAI-compatible provider."""

    def __init__(self, client: object) -> None:
        self.client = client

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("Phase 1 stub: OpenAICompatibleProvider is not implemented yet")
