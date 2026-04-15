from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any


def estimate_tokens(messages: Any) -> int:
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return max(1, len(serialized) // 4)


def micro_compact(messages: Sequence[dict[str, Any]], keep_recent: int = 1) -> list[dict[str, Any]]:
    compacted = copy.deepcopy(list(messages))
    tool_result_locations = _tool_result_locations(compacted)
    preserve = set(tool_result_locations[-keep_recent:]) if keep_recent > 0 else set()

    for message_index, block_index in tool_result_locations:
        if (message_index, block_index) in preserve:
            continue
        content_blocks = compacted[message_index].get("content")
        if isinstance(content_blocks, list):
            block = content_blocks[block_index]
            content_blocks[block_index] = {
                "type": block.get("type", "tool_result"),
                "tool_name": block.get("tool_name"),
                "content": _compact_tool_result_content(block.get("content")),
            }

    return compacted


def _tool_result_locations(messages: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    locations: list[tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                locations.append((message_index, block_index))
    return locations


def _compact_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        preview = content[:32]
        return f"[compacted tool_result len={len(content)} preview={preview!r}]"
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return f"[compacted tool_result len={len(serialized)}]"
