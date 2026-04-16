from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from code_agent_harness.tools.read_file import _resolve_workspace_path


def create_list_files_handler(workspace_root: Path | str) -> Callable[[dict[str, object]], object]:
    root = Path(workspace_root)

    def handler(arguments: dict[str, object]) -> object:
        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str):
            raise ValueError("path must be a string")

        list_root = _resolve_workspace_path(root, raw_path)
        if list_root.is_file():
            return str(list_root.relative_to(root.resolve()))

        files = sorted(path.relative_to(root.resolve()).as_posix() for path in list_root.rglob("*") if path.is_file())
        return "\n".join(files)

    return handler
