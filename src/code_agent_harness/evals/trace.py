from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from code_agent_harness.types.engine import RuntimeResult
from code_agent_harness.types.state import SessionState


@dataclass(frozen=True)
class TraceToolCall:
    index: int
    tool_name: str
    arguments: dict[str, object]
    tool_use_id: str
    has_result: bool
    result_status: str


@dataclass(frozen=True)
class EvalTrace:
    tool_calls: tuple[TraceToolCall, ...]
    final_output: str
    final_state: SessionState
    workspace_root: Path
    assistant_turn_count: int = 0


def extract_eval_trace(runtime_result: RuntimeResult, *, workspace_root: Path) -> EvalTrace:
    result_blocks: dict[str, dict[str, object]] = {}
    for message in runtime_result.messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if isinstance(tool_use_id, str) and tool_use_id:
                result_blocks[tool_use_id] = block

    tool_calls: list[TraceToolCall] = []
    assistant_turn_count = 0
    for message in runtime_result.messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_turn_count += 1
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_call":
                continue
            tool_use_id = str(block.get("id", ""))
            result = result_blocks.get(tool_use_id)
            result_status = "missing"
            has_result = result is not None
            if result is not None:
                result_status = "ok"
                result_content = result.get("content")
                if isinstance(result_content, dict) and isinstance(result_content.get("status"), str):
                    result_status = result_content["status"]
            arguments = block.get("arguments")
            tool_calls.append(
                TraceToolCall(
                    index=len(tool_calls),
                    tool_name=str(block.get("name", "")),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    tool_use_id=tool_use_id,
                    has_result=has_result,
                    result_status=result_status,
                )
            )

    return EvalTrace(
        tool_calls=tuple(tool_calls),
        final_output=runtime_result.output_text,
        final_state=runtime_result.state,
        workspace_root=workspace_root,
        assistant_turn_count=assistant_turn_count,
    )
