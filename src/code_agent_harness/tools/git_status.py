from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess


def create_git_status_handler(workspace_root: Path | str) -> Callable[[dict[str, object]], object]:
    root = Path(workspace_root)

    def handler(arguments: dict[str, object]) -> object:
        del arguments
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        if completed.stdout.strip():
            return completed.stdout
        if completed.stderr.strip():
            return completed.stderr
        if completed.returncode != 0:
            return f"git status failed with exit code {completed.returncode}"
        return "clean"

    return handler
