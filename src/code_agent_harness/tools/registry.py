from collections.abc import Callable, Iterable
from dataclasses import dataclass

from code_agent_harness.types.tools import ToolDefinition


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: Callable[[dict[str, object]], object]


class UnknownToolError(LookupError):
    pass


class ToolRegistry:
    def __init__(self, loader: Callable[[], Iterable[RegisteredTool]]) -> None:
        self._loader = loader

    def list_tools(self) -> list[ToolDefinition]:
        return [registered_tool.definition for registered_tool in self._loader()]

    def resolve_tool(self, tool_name: str) -> RegisteredTool:
        for registered_tool in self._loader():
            if registered_tool.definition.name == tool_name:
                return registered_tool
        raise UnknownToolError(f"unknown tool: {tool_name}")
