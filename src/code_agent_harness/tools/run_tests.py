from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess


def create_run_tests_handler(workspace_root: Path | str) -> Callable[[dict[str, object]], object]:
    root = Path(workspace_root)

    def handler(arguments: dict[str, object]) -> object:
        args = arguments.get("args")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError("args must be an array of strings")

        completed = subprocess.run(
            ["pytest", *args],
            cwd=root,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if output:
            return output
        return f"pytest exited with code {completed.returncode}"

    return handler
