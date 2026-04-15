from code_agent_harness.llm.base import LLMResponse


class FakeProvider:
    def __init__(self, script: list[dict[str, object]]) -> None:
        self._script = list(script)

    def generate(
        self,
        system_prompt: str,
        messages: list[object],
        tools: list[object],
        extra: dict[str, object],
    ) -> LLMResponse:
        payload = self._script.pop(0)
        return LLMResponse(**payload)
