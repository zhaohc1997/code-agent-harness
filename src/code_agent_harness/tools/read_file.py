from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    root_resolved = workspace_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace root: {raw_path}") from exc
    return resolved


def create_read_file_handler(workspace_root: Path | str) -> Callable[[dict[str, object]], object]:
    root = Path(workspace_root)

    def handler(arguments: dict[str, object]) -> object:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("path is required")

        path = _resolve_workspace_path(root, raw_path)
        content = path.read_text(encoding="utf-8")

        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")
        if start_line is None and end_line is None:
            return content

        if start_line is not None and (not isinstance(start_line, int) or start_line < 1):
            raise ValueError("start_line must be a positive integer")
        if end_line is not None and (not isinstance(end_line, int) or end_line < 1):
            raise ValueError("end_line must be a positive integer")

        lines = content.splitlines(keepends=True)
        start_index = 0 if start_line is None else start_line - 1
        end_index = len(lines) if end_line is None else end_line
        if start_index > end_index:
            raise ValueError("start_line must be less than or equal to end_line")
        return "".join(lines[start_index:end_index])

    return handler
