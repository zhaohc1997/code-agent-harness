from collections.abc import Callable, Iterable
from dataclasses import dataclass

from code_agent_harness.types.tools import ToolDefinition


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: Callable[[dict[str, object]], object]


class UnknownToolError(LookupError):
    pass


class ToolRegistrySnapshot:
    def __init__(self, tools: Iterable[RegisteredTool]) -> None:
        self._tools = tuple(tools)
        self._tools_by_name = {tool.definition.name: tool for tool in self._tools}

    def list_tools(self) -> list[ToolDefinition]:
        return [registered_tool.definition for registered_tool in self._tools]

    def resolve_tool(self, tool_name: str) -> RegisteredTool:
        try:
            return self._tools_by_name[tool_name]
        except KeyError as exc:
            raise UnknownToolError(f"unknown tool: {tool_name}") from exc


class ToolRegistry:
    def __init__(self, loader: Callable[[], Iterable[RegisteredTool]]) -> None:
        self._loader = loader

    def list_tools(self) -> list[ToolDefinition]:
        return [registered_tool.definition for registered_tool in self._loader()]

    def snapshot(self) -> ToolRegistrySnapshot:
        return ToolRegistrySnapshot(self._loader())

    def resolve_tool(self, tool_name: str) -> RegisteredTool:
        for registered_tool in self._loader():
            if registered_tool.definition.name == tool_name:
                return registered_tool
        raise UnknownToolError(f"unknown tool: {tool_name}")
