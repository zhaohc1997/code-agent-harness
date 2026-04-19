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
                description=(
                    "Read a UTF-8 file before editing. Fields: path: string workspace-relative "
                    "file path; start_line: integer optional inclusive first line; end_line: "
                    "integer optional inclusive last line."
                ),
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
                description=(
                    "Search workspace files for an exact text fragment, not a regex. Fields: "
                    "pattern: string exact text fragment; path: string optional "
                    "workspace-relative file or directory scope."
                ),
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
                description=(
                    "List files under a workspace-relative directory. Fields: path: string "
                    "optional directory; omit path to list the workspace root."
                ),
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
                description=(
                    "Apply structured text replacements. Fields: path: string workspace-relative "
                    "file path; replacements: array of objects with old_text: string, "
                    "new_text: string, replace_all: boolean optional. Destructive edits may "
                    "require confirmation."
                ),
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
                description=(
                    "Run pytest from the workspace root. Fields: args: array[string] required "
                    "pytest arguments, prefer targeted tests such as ['-q', "
                    "'tests/test_file.py']. Shell command strings are invalid."
                ),
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
                description=(
                    "Record a required human confirmation step. Fields: message: string "
                    "required prompt; default: boolean optional default answer. This does not "
                    "execute the risky action."
                ),
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
