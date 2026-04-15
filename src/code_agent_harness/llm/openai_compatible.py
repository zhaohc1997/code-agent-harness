from code_agent_harness.llm.base import LLMResponse


class OpenAICompatibleProvider:
    def __init__(self, client: object) -> None:
        self.client = client

    def generate(
        self,
        system_prompt: str,
        messages: list[object],
        tools: list[object],
        extra: dict[str, object],
    ) -> LLMResponse:
        raise NotImplementedError
