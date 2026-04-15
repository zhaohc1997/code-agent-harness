from code_agent_harness import __version__
from code_agent_harness.llm.fake_provider import FakeProvider


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_fake_provider_returns_scripted_responses() -> None:
    provider = FakeProvider(script=[{"stop_reason": "end_turn", "content": []}])
    response = provider.generate(system_prompt="sys", messages=[], tools=[], extra={})
    assert response.stop_reason == "end_turn"
