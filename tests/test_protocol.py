import pytest

from code_agent_harness.storage.blobs import BlobStore
from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import RegisteredTool, ToolRegistry, UnknownToolError
from code_agent_harness.types.tools import ToolDefinition


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
