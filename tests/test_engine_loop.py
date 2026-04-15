import pytest

from code_agent_harness import __version__
import code_agent_harness.llm as llm


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_llm_package_exports_are_usable() -> None:
    request = llm.LLMRequest(system_prompt="sys", messages=[], tools=[], extra={})
    provider = llm.FakeProvider(script=[{"stop_reason": "end_turn", "content": []}])
    response = provider.generate(request)
    assert isinstance(request, llm.LLMRequest)
    assert isinstance(response, llm.LLMResponse)


def test_fake_provider_preserves_scripted_usage() -> None:
    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "end_turn",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 5},
            }
        ]
    )
    request = llm.LLMRequest(system_prompt="sys", messages=[], tools=[], extra={})
    response = provider.generate(request)
    assert response.stop_reason == "end_turn"
    assert response.usage == {"input_tokens": 3, "output_tokens": 5}


def test_fake_provider_script_exhaustion_raises_targeted_error() -> None:
    provider = llm.FakeProvider(script=[])
    request = llm.LLMRequest(system_prompt="sys", messages=[], tools=[], extra={})
    with pytest.raises(llm.FakeProviderScriptExhausted):
        provider.generate(request)
