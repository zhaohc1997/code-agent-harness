import pytest

from code_agent_harness.storage.blobs import BlobStore
from code_agent_harness.config import RuntimePaths
from code_agent_harness.engine.cancellation import CancellationToken
from code_agent_harness.engine.state_machine import EngineStateMachine
from code_agent_harness.storage.checkpoints import CheckpointStore
from code_agent_harness.storage.sessions import SessionStore
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import RegisteredTool, ToolRegistry, UnknownToolError
from code_agent_harness.types.state import SessionState
from code_agent_harness.types.tools import ToolDefinition
from code_agent_harness.llm.fake_provider import FakeProvider


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


def test_tool_registry_is_dynamic_per_call() -> None:
    calls = []

    def loader():
        calls.append(len(calls))
        return [
            RegisteredTool(
                definition=ToolDefinition(name=f"tool-{index}"),
                handler=lambda arguments, index=index: f"value-{index}",
            )
            for index in range(len(calls))
        ]

    registry = ToolRegistry(loader)

    assert [tool.name for tool in registry.list_tools()] == ["tool-0"]
    assert [tool.name for tool in registry.list_tools()] == ["tool-0", "tool-1"]


def test_execute_small_tool_output_stays_inline(tmp_path) -> None:
    registry = ToolRegistry(
        lambda: [
            RegisteredTool(
                definition=ToolDefinition(name="read_file"),
                handler=lambda arguments: arguments["content"],
            )
        ]
    )
    executor = ToolExecutor(registry=registry, blob_store_root=tmp_path / ".agenth")

    result = executor.execute("read_file", {"content": "short content"})

    assert result.content == "short content"
    assert result.external_blob_id is None


def test_execute_large_tool_output_externalizes_and_persists_blob(tmp_path) -> None:
    content = "x" * 25001
    registry = ToolRegistry(
        lambda: [
            RegisteredTool(
                definition=ToolDefinition(name="read_file"),
                handler=lambda arguments: arguments["content"],
            )
        ]
    )
    executor = ToolExecutor(registry=registry, blob_store_root=tmp_path / ".agenth")

    result = executor.execute("read_file", {"content": content})

    assert result.external_blob_id is not None
    assert result.content == f"[externalized:{result.external_blob_id}]"
    assert BlobStore(tmp_path / ".agenth" / "blobs").load(result.external_blob_id) == content


def test_execute_unknown_tool_fails_clearly(tmp_path) -> None:
    registry = ToolRegistry(lambda: [])
    executor = ToolExecutor(registry=registry, blob_store_root=tmp_path / ".agenth")

    with pytest.raises(UnknownToolError, match="unknown tool: missing_tool"):
        executor.execute("missing_tool", {})


def test_protocol_requires_tool_results_adjacent(runtime_dependencies) -> None:
    from code_agent_harness.engine.loop import AgentRuntime

    provider = FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "a.py"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "finished"}],
            },
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    result = runtime.run(session_id="s1", user_input="Read a file")

    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1]["content"] == [{"type": "text", "text": "finished"}]

    matching_pairs = [
        (assistant_message, user_message)
        for assistant_message, user_message in zip(result.messages, result.messages[1:])
        if assistant_message.get("role") == "assistant" and user_message.get("role") == "user"
    ]
    tool_exchange_pairs = [
        (assistant_message, user_message)
        for assistant_message, user_message in matching_pairs
        if assistant_message.get("content") == [
            {
                "type": "tool_call",
                "id": "tool-1",
                "name": "read_file",
                "arguments": {"path": "a.py"},
            }
        ]
        and user_message.get("content") == [
            {
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "tool_name": "read_file",
                "content": "read:a.py",
            }
        ]
    ]

    assert len(tool_exchange_pairs) == 1


def test_protocol_rejects_tool_use_without_tool_calls(runtime_dependencies) -> None:
    from code_agent_harness.engine.loop import AgentRuntime

    provider = FakeProvider(
        script=[
            {
                "stop_reason": "tool_use",
                "content": [{"type": "text", "text": "not actually a tool call"}],
            }
        ]
    )
    runtime = AgentRuntime(provider=provider, **runtime_dependencies)

    with pytest.raises(ValueError, match="tool_use response must include at least one tool_call"):
        runtime.run(session_id="s1", user_input="Read a file")

    session = runtime_dependencies["sessions"].load("s1")
    assert session["state"] == SessionState.FAILED.value
    assert session["messages"][-1]["role"] == "assistant"
