from pathlib import Path

from code_agent_harness.tools.apply_patch_tool import create_apply_patch_handler
from code_agent_harness.tools.ask_confirmation import create_ask_confirmation_handler
from code_agent_harness.tools.git_status import create_git_status_handler
from code_agent_harness.tools.list_files import create_list_files_handler
from code_agent_harness.tools.read_file import create_read_file_handler
from code_agent_harness.tools.registry import RegisteredTool
from code_agent_harness.tools.run_tests import create_run_tests_handler
from code_agent_harness.tools.search_text import create_search_text_handler
from code_agent_harness.types.tools import ToolDefinition


def _disabled_shell_handler(arguments: dict[str, object]) -> object:
    del arguments
    raise PermissionError("builtin tool shell is disabled for the code assistant profile")


def _build_builtin_tools(workspace_root: Path) -> tuple[RegisteredTool, ...]:
    read_file_handler = create_read_file_handler(workspace_root)
    search_text_handler = create_search_text_handler(workspace_root)
    list_files_handler = create_list_files_handler(workspace_root)
    apply_patch_handler = create_apply_patch_handler(workspace_root)
    run_tests_handler = create_run_tests_handler(workspace_root)
    git_status_handler = create_git_status_handler(workspace_root)
    ask_confirmation_handler = create_ask_confirmation_handler(workspace_root)

    return (
        RegisteredTool(
            definition=ToolDefinition(
                name="read_file",
                description="Read a UTF-8 file rooted in the workspace, optionally limited to an inclusive line range.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            handler=read_file_handler,
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="search_text",
                description="Search for an exact text fragment in workspace files and return matching file, line number, and line content.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            ),
            handler=search_text_handler,
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="list_files",
                description="List files under a workspace-relative directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            ),
            handler=list_files_handler,
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="apply_patch",
                description="Apply structured text replacements to a workspace file and fail if expected old text is missing or ambiguous.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "replacements": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_text": {"type": "string", "minLength": 1},
                                    "new_text": {"type": "string", "minLength": 1},
                                    "replace_all": {"type": "boolean"},
                                },
                                "required": ["old_text", "new_text"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["path", "replacements"],
                    "additionalProperties": False,
                },
            ),
            handler=apply_patch_handler,
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="run_tests",
                description="Run pytest from the workspace root via the current Python interpreter using an argument array. Shell command strings are not accepted.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["args"],
                    "additionalProperties": False,
                },
            ),
            handler=run_tests_handler,
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="git_status",
                description="Return `git status --short` for the workspace root.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            ),
            handler=git_status_handler,
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="ask_confirmation",
                description="Record that a human confirmation step is required and echo the confirmation prompt.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "minLength": 1},
                        "default": {"type": "boolean"},
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            ),
            handler=ask_confirmation_handler,
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="shell",
                description="Compatibility-only placeholder. Arbitrary shell execution is disabled.",
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
            handler=_disabled_shell_handler,
        ),
    )


def load_builtin_tools(workspace_root: Path | str | None = None) -> list[RegisteredTool]:
    root = Path.cwd() if workspace_root is None else Path(workspace_root)
    return list(_build_builtin_tools(root))
