from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def create_ask_confirmation_handler(workspace_root: Path | str) -> Callable[[dict[str, object]], object]:
    del workspace_root

    def handler(arguments: dict[str, object]) -> object:
        message = arguments.get("message")
        if not isinstance(message, str) or not message:
            raise ValueError("message is required")

        default = arguments.get("default", False)
        if not isinstance(default, bool):
            raise ValueError("default must be a boolean")

        return {
            "confirmed": default,
            "message": message,
            "requires_human_confirmation": True,
        }

    return handler
