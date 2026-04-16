from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from code_agent_harness.tools.read_file import _resolve_workspace_path


def create_search_text_handler(workspace_root: Path | str) -> Callable[[dict[str, object]], object]:
    root = Path(workspace_root)

    def handler(arguments: dict[str, object]) -> object:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern is required")

        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str):
            raise ValueError("path must be a string")

        search_root = _resolve_workspace_path(root, raw_path)
        if search_root.is_file():
            candidate_paths = [search_root]
        else:
            candidate_paths = [path for path in search_root.rglob("*") if path.is_file()]

        matches: list[str] = []
        for file_path in candidate_paths:
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            relative_path = file_path.relative_to(root.resolve())
            for line_number, line in enumerate(lines, start=1):
                if pattern in line:
                    matches.append(f"{relative_path}:{line_number}:{line}")
        return "\n".join(matches)

    return handler
