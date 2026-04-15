import pytest

from code_agent_harness import __version__
import code_agent_harness.llm as llm
from code_agent_harness.config import RuntimePaths
from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import RegisteredTool, ToolRegistry
from code_agent_harness.types.state import SessionState
from code_agent_harness.types.tools import ToolDefinition


@pytest.fixture
def runtime_dependencies(tmp_path):
    paths = RuntimePaths(tmp_path / ".agenth")
    registry = ToolRegistry(
        lambda: [
            RegisteredTool(
                definition=ToolDefinition(name="read_file"),
                handler=lambda arguments: f"read:{arguments['path']}",
            )
        ]
    )
    return {
        "system_prompt": "You are a test agent.",
        "sessions": SessionStore(paths.sessions),
        "checkpoints": CheckpointStore(paths.checkpoints),
        "registry": registry,
        "executor": ToolExecutor(registry=registry, blob_store_root=tmp_path / ".agenth"),
        "cancellation": CancellationToken(),
        "state_machine": EngineStateMachine(SessionState.IDLE),
    }


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


def test_engine_completes_without_tool_calls(runtime_dependencies) -> None:
    from code_agent_harness.engine.loop import AgentRuntime

    provider = llm.FakeProvider(
        script=[
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            }
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    result = runtime.run(session_id="s1", user_input="Summarize status")

    assert result.state == SessionState.COMPLETED
    assert result.output_text == "done"
