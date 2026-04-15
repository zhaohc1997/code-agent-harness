from __future__ import annotations

import json


class CycleGuard:
    def __init__(self, max_repeats: int = 2) -> None:
        self.max_repeats = max_repeats
        self._last_key: tuple[str, str] | None = None
        self._repeat_count = 0

    def record(self, tool_name: str, arguments: dict[str, object]) -> bool:
        normalized_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        key = (tool_name, normalized_arguments)
        self._repeat_count = self._repeat_count + 1 if key == self._last_key else 0
        self._last_key = key
        return self._repeat_count >= self.max_repeats
