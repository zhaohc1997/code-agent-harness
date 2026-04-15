from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any


@dataclass(frozen=True)
class AutoCompactResult:
    messages: list[dict[str, Any]]
    applied_micro_compaction: bool
    applied_auto_summary: bool
    input_token_estimate: int
    output_token_estimate: int


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


def auto_compact(
    messages: Sequence[dict[str, Any]],
    *,
    system_prompt: str,
    task_goal: str,
    max_tokens: int,
    trigger_ratio: float = 0.65,
    keep_recent: int = 4,
) -> AutoCompactResult:
    input_token_estimate = estimate_tokens(messages)
    micro_compacted = micro_compact(messages)
    threshold = max(1, int(max_tokens * trigger_ratio))
    if estimate_tokens(micro_compacted) <= threshold or len(micro_compacted) <= keep_recent:
        return AutoCompactResult(
            messages=micro_compacted,
            applied_micro_compaction=micro_compacted != list(messages),
            applied_auto_summary=False,
            input_token_estimate=input_token_estimate,
            output_token_estimate=estimate_tokens(micro_compacted),
        )

    split_index = max(1, len(micro_compacted) - keep_recent)
    prefix = micro_compacted[:split_index]
    tail = copy.deepcopy(micro_compacted[split_index:])
    summarized_messages = [{"role": "user", "content": _build_summary(prefix, system_prompt, task_goal)}, *tail]
    return AutoCompactResult(
        messages=summarized_messages,
        applied_micro_compaction=micro_compacted != list(messages),
        applied_auto_summary=True,
        input_token_estimate=input_token_estimate,
        output_token_estimate=estimate_tokens(summarized_messages),
    )


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


def _build_summary(
    messages: Sequence[dict[str, Any]],
    system_prompt: str,
    task_goal: str,
    *,
    max_items: int = 12,
    max_chars_per_item: int = 160,
) -> str:
    lines = [
        "Continuation context summary.",
        f"System identity: {system_prompt}",
        f"Task goal: {task_goal}",
        "Earlier turns:",
    ]
    for message in messages[-max_items:]:
        role = message.get("role", "unknown")
        lines.append(f"- {role}: {_summarize_content(message.get('content'), max_chars_per_item=max_chars_per_item)}")
    return "\n".join(lines)


def _summarize_content(content: Any, *, max_chars_per_item: int) -> str:
    if isinstance(content, str):
        return _truncate(content, max_chars_per_item)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(_truncate(str(block), max_chars_per_item))
                continue
            block_type = block.get("type", "unknown")
            if block_type == "tool_result":
                parts.append(
                    f"tool_result {block.get('tool_name')}: "
                    f"{_truncate(json.dumps(block.get('content'), ensure_ascii=False, default=str), max_chars_per_item)}"
                )
                continue
            if block_type == "tool_call":
                parts.append(
                    f"tool_call {block.get('name')} "
                    f"{_truncate(json.dumps(block.get('arguments'), sort_keys=True, ensure_ascii=False, default=str), max_chars_per_item)}"
                )
                continue
            parts.append(_truncate(json.dumps(block, sort_keys=True, ensure_ascii=False, default=str), max_chars_per_item))
        return "; ".join(parts)
    return _truncate(json.dumps(content, sort_keys=True, ensure_ascii=False, default=str), max_chars_per_item)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars - 3]}..."
