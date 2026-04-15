from code_agent_harness.tools.executor import ToolExecutor
from code_agent_harness.tools.registry import ToolRegistry


def test_tool_registry_is_dynamic_per_call() -> None:
    calls = []

    def loader():
        calls.append(len(calls))
        return calls

    registry = ToolRegistry(loader)

    assert registry.list_tools() == [0]
    assert registry.list_tools() == [0, 1]


def test_small_tool_output_stays_inline(tmp_path) -> None:
    registry = ToolRegistry(lambda: [])
    executor = ToolExecutor(registry=registry, blob_store_root=tmp_path / ".agenth")

    result = executor._apply_limit("read_file", "short content")

    assert result.content == "short content"
    assert result.external_blob_id is None


def test_large_tool_output_is_externalized(tmp_path) -> None:
    registry = ToolRegistry(lambda: [])
    executor = ToolExecutor(registry=registry, blob_store_root=tmp_path / ".agenth")
    content = "x" * 25001
    result = executor._apply_limit("read_file", content)

    assert result.external_blob_id is not None
