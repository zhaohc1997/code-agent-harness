from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def estimate_tokens(messages: Any) -> int:
    return max(1, len(str(messages)) // 4)


def micro_compact(messages: Sequence[dict[str, Any]], keep_recent: int = 1) -> list[dict[str, Any]]:
    if keep_recent <= 0:
        return []
    return list(messages[-keep_recent:])
