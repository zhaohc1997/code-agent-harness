from code_agent_harness.types.tools import ToolDefinition

BUILTIN_TOOL_DEFINITIONS = (
    ToolDefinition(
        name="read_file",
        description="Read the contents of a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
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
    ToolDefinition(
        name="shell",
        description="Run a shell command.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    ),
)


def load_builtin_tools() -> list[ToolDefinition]:
    return list(BUILTIN_TOOL_DEFINITIONS)
