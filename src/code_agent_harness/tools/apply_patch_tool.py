from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from code_agent_harness.tools.read_file import _resolve_workspace_path


def create_apply_patch_handler(workspace_root: Path | str) -> Callable[[dict[str, object]], object]:
    root = Path(workspace_root)

    def handler(arguments: dict[str, object]) -> object:
        raw_path = arguments.get("path")
        replacements = arguments.get("replacements")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("path is required")
        if not isinstance(replacements, list) or not replacements:
            raise ValueError("replacements is required")

        path = _resolve_workspace_path(root, raw_path)
        content = path.read_text(encoding="utf-8")
        replacement_count = 0

        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise ValueError("each replacement must be an object")
            old_text = replacement.get("old_text")
            new_text = replacement.get("new_text")
            replace_all = replacement.get("replace_all", False)
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise ValueError("replacement.old_text and replacement.new_text are required")
            if not old_text:
                raise ValueError("replacement.old_text must be a non-empty string")
            if not new_text:
                raise ValueError("replacement.new_text must be a non-empty string")
            if not isinstance(replace_all, bool):
                raise ValueError("replacement.replace_all must be a boolean")
            occurrences = content.count(old_text)
            if occurrences == 0:
                raise ValueError(f"old text not found in {raw_path}: {old_text!r}")
            if not replace_all and occurrences > 1:
                raise ValueError(
                    f"ambiguous single replacement in {raw_path}: {occurrences} matches for {old_text!r}"
                )
            content = content.replace(old_text, new_text, -1 if replace_all else 1)
            replacement_count += occurrences if replace_all else 1

        path.write_text(content, encoding="utf-8")
        noun = "replacement" if replacement_count == 1 else "replacements"
        return f"applied {replacement_count} {noun} to {raw_path}"

    return handler
