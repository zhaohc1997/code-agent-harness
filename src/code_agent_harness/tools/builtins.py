from code_agent_harness.tools.registry import RegisteredTool
from code_agent_harness.types.tools import ToolDefinition


def _not_implemented(tool_name: str):
    def handler(arguments: dict[str, object]) -> object:
        raise NotImplementedError(f"builtin tool {tool_name} is not implemented")

    return handler


BUILTIN_TOOL_DEFINITIONS = (
    RegisteredTool(
        definition=ToolDefinition(
            name="read_file",
            description="Read the contents of a file.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        handler=_not_implemented("read_file"),
    ),
    RegisteredTool(
        definition=ToolDefinition(
            name="search_text",
            description="Search for text inside repository files.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        ),
        handler=_not_implemented("search_text"),
    ),
    RegisteredTool(
        definition=ToolDefinition(
            name="shell",
            description="Run a shell command.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
        handler=_not_implemented("shell"),
    ),
)


def load_builtin_tools() -> list[RegisteredTool]:
    return list(BUILTIN_TOOL_DEFINITIONS)
