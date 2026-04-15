from collections.abc import Callable, Iterable

from code_agent_harness.types.tools import ToolDefinition


class ToolRegistry:
    def __init__(self, loader: Callable[[], Iterable[ToolDefinition]]) -> None:
        self._loader = loader

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._loader())
