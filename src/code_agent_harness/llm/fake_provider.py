from code_agent_harness.llm.base import LLMRequest
from code_agent_harness.llm.base import LLMResponse


class FakeProviderScriptExhausted(RuntimeError):
    pass


class FakeProvider:
    def __init__(self, script: list[dict[str, object]]) -> None:
        self._script = list(script)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._script:
            raise FakeProviderScriptExhausted("FakeProvider script is exhausted")
        payload = self._script.pop(0)
        return LLMResponse(**payload)
