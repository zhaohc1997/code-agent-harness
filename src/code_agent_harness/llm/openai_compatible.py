from __future__ import annotations

import json
from typing import Any
import urllib.error
import urllib.request

from code_agent_harness.llm.base import LLMRequest
from code_agent_harness.llm.base import LLMResponse


class JsonHttpClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def send_json(self, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"provider request failed with HTTP {exc.code}: {error_body}") from exc
        if not isinstance(body, dict):
            raise ValueError("provider response must be a JSON object")
        return body


class OpenAICompatibleProvider:
    """Adapter for DeepSeek's OpenAI-compatible chat completions API."""

    def __init__(self, *, client: Any, model: str, base_url: str, api_key: str) -> None:
        self.client = client
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, request: LLMRequest) -> LLMResponse:
        tools_present = bool(request.tools)
        payload: dict[str, object] = {
            "model": self._api_model_name(request.extra, tools_present=tools_present),
            "messages": self._build_messages(request),
            "tools": self._normalize_tools(request.tools),
        }
        payload.update(self._normalize_extra(request.extra, tools_present=tools_present))
        raw = self.client.send_json(payload)

        if not isinstance(raw, dict):
            raise ValueError("provider response must be a JSON object")

        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("provider response must include choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("provider choice must be an object")

        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("provider choice must include a message object")

        tool_calls = message.get("tool_calls") or []
        reasoning_content = message.get("reasoning_content")
        content_blocks: list[dict[str, object]] = []
        if isinstance(reasoning_content, str) and reasoning_content:
            content_blocks.append({"type": "reasoning", "text": reasoning_content})
        if tool_calls:
            content_blocks.extend(self._normalize_tool_call(tool_call) for tool_call in tool_calls)
            return LLMResponse(
                content=content_blocks,
                stop_reason="tool_use",
                usage=self._normalize_usage(raw.get("usage")),
            )

        content_blocks.append({"type": "text", "text": self._stringify_text(message.get("content"))})
        return LLMResponse(
            content=content_blocks,
            stop_reason="end_turn",
            usage=self._normalize_usage(raw.get("usage")),
        )

    def _build_messages(self, request: LLMRequest) -> list[dict[str, object]]:
        runtime_messages = self._collapse_historical_tool_loops(request.messages) if request.tools else request.messages
        messages = [{"role": "system", "content": request.system_prompt}]
        for message in runtime_messages:
            messages.extend(self._normalize_message(message))
        return messages

    def _collapse_historical_tool_loops(self, messages: list[object]) -> list[object]:
        last_tool_call_index = -1
        for index, message in enumerate(messages):
            if self._message_contains_block_type(message, "tool_call"):
                last_tool_call_index = index

        if last_tool_call_index <= 0:
            return messages

        collapsed_messages: list[object] = []
        for index, message in enumerate(messages):
            if index >= last_tool_call_index:
                collapsed_messages.append(message)
                continue
            if self._message_contains_block_type(message, "tool_call") or self._message_contains_block_type(
                message, "tool_result"
            ):
                collapsed_messages.append(self._summarize_runtime_message(message))
                continue
            collapsed_messages.append(message)
        return collapsed_messages

    def _normalize_message(self, message: object) -> list[dict[str, object]]:
        if not isinstance(message, dict):
            raise ValueError("runtime message must be an object")

        role = message.get("role")
        content = message.get("content")
        if role == "user":
            return self._normalize_user_message(content)
        if role == "assistant":
            return [self._normalize_assistant_message(content)]
        raise ValueError(f"unsupported runtime role: {role!r}")

    def _normalize_user_message(self, content: object) -> list[dict[str, object]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        if not isinstance(content, list):
            return [{"role": "user", "content": self._encode_content(content)}]

        text_parts: list[str] = []
        messages: list[dict[str, object]] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(self._encode_content(block))
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                tool_message = {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": self._encode_content(block.get("content")),
                }
                tool_name = block.get("tool_name")
                if isinstance(tool_name, str) and tool_name:
                    tool_message["name"] = tool_name
                messages.append(tool_message)
                continue
            if block_type == "text":
                text_parts.append(self._stringify_text(block.get("text")))
                continue
            text_parts.append(self._encode_content(block))

        if text_parts:
            return [{"role": "user", "content": "\n\n".join(part for part in text_parts if part)}] + messages
        return messages

    def _normalize_assistant_message(self, content: object) -> dict[str, object]:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        if not isinstance(content, list):
            return {"role": "assistant", "content": self._encode_content(content)}

        text_parts: list[str] = []
        tool_calls: list[dict[str, object]] = []
        reasoning_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(self._encode_content(block))
                continue
            block_type = block.get("type")
            if block_type == "reasoning":
                reasoning_parts.append(self._stringify_text(block.get("text")))
                continue
            if block_type == "tool_call":
                tool_calls.append(self._tool_call_payload(block))
                continue
            if block_type == "text":
                text_parts.append(self._stringify_text(block.get("text")))
                continue
            text_parts.append(self._encode_content(block))

        normalized: dict[str, object] = {"role": "assistant"}
        if reasoning_parts:
            normalized["reasoning_content"] = "\n\n".join(part for part in reasoning_parts if part)
        if tool_calls:
            normalized["tool_calls"] = tool_calls
            normalized["content"] = "\n\n".join(part for part in text_parts if part)
            return normalized
        normalized["content"] = "\n\n".join(part for part in text_parts if part)
        return normalized

    def _summarize_runtime_message(self, message: object) -> dict[str, object]:
        if not isinstance(message, dict):
            return {"role": "user", "content": self._encode_content(message)}

        role = message.get("role")
        normalized_role = role if role in {"user", "assistant"} else "user"
        content = message.get("content")
        if isinstance(content, str):
            return {"role": normalized_role, "content": content}
        if not isinstance(content, list):
            return {"role": normalized_role, "content": self._encode_content(content)}

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(self._encode_content(block))
                continue
            block_type = block.get("type")
            if block_type == "reasoning":
                parts.append(f"Reasoning: {self._stringify_text(block.get('text'))}")
                continue
            if block_type == "tool_call":
                parts.append(
                    "Tool call "
                    f"{block.get('name')}: "
                    f"{self._truncate_summary(self._encode_content(block.get('arguments')))}"
                )
                continue
            if block_type == "tool_result":
                parts.append(
                    "Tool result "
                    f"{block.get('tool_name')}: "
                    f"{self._truncate_summary(self._encode_content(block.get('content')))}"
                )
                continue
            if block_type == "text":
                parts.append(self._stringify_text(block.get("text")))
                continue
            parts.append(self._encode_content(block))

        summary = "\n".join(part for part in parts if part) or "[empty summarized message]"
        return {"role": normalized_role, "content": summary}

    def _normalize_extra(self, extra: dict[str, object], *, tools_present: bool) -> dict[str, object]:
        normalized_extra = dict(extra)
        thinking = normalized_extra.get("thinking")
        thinking_requested = isinstance(thinking, dict) and (
            thinking.get("enabled") is True or thinking.get("type") == "enabled"
        )
        if thinking_requested and tools_present:
            normalized_extra.pop("thinking", None)
            return normalized_extra
        if thinking_requested:
            normalized_extra["thinking"] = {"type": "enabled"}
        return normalized_extra

    def _api_model_name(self, extra: dict[str, object], *, tools_present: bool) -> str:
        model_name = self.model.strip()
        if model_name in {"deepseek-chat", "deepseek-reasoner"}:
            if tools_present and model_name == "deepseek-reasoner":
                return "deepseek-chat"
            return model_name

        normalized_name = model_name.lower()
        thinking = extra.get("thinking")
        thinking_enabled = isinstance(thinking, dict) and (
            thinking.get("enabled") is True or thinking.get("type") == "enabled"
        )
        if normalized_name in {"deepseek-v3.2", "deepseek-v3", "deepseek-v3-2"}:
            if tools_present:
                return "deepseek-chat"
            return "deepseek-reasoner" if thinking_enabled else "deepseek-chat"
        return model_name

    def _normalize_tools(self, tools: list[object]) -> list[dict[str, object]]:
        normalized_tools: list[dict[str, object]] = []
        for tool in tools:
            if isinstance(tool, dict):
                normalized_tools.append(tool)
                continue

            tool_name = getattr(tool, "name", None)
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("tool definition must include a non-empty name")

            function_spec: dict[str, object] = {"name": tool_name}
            description = getattr(tool, "description", None)
            if isinstance(description, str) and description:
                function_spec["description"] = description

            parameters = getattr(tool, "input_schema", None)
            if isinstance(parameters, dict):
                function_spec["parameters"] = parameters

            normalized_tools.append({"type": "function", "function": function_spec})
        return normalized_tools

    def _normalize_tool_call(self, tool_call: object) -> dict[str, object]:
        if not isinstance(tool_call, dict):
            raise ValueError("provider tool call must be an object")

        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise ValueError("provider tool call must include a function object")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("provider tool call function must include a name")

        arguments_raw = function.get("arguments", "{}")
        if not isinstance(arguments_raw, str):
            raise ValueError("provider tool call arguments must be a JSON string")
        try:
            arguments = json.loads(arguments_raw) if arguments_raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError("provider tool call arguments must be valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("provider tool call arguments must decode to an object")

        return {
            "type": "tool_call",
            "id": str(tool_call.get("id", "")),
            "index": tool_call.get("index"),
            "name": name,
            "arguments": arguments,
        }

    def _tool_call_payload(self, block: dict[str, object]) -> dict[str, object]:
        payload = {
            "id": str(block.get("id", "")),
            "type": "function",
            "function": {
                "name": str(block.get("name", "")),
                "arguments": json.dumps(block.get("arguments", {}), ensure_ascii=True),
            },
        }
        index = block.get("index")
        if isinstance(index, int):
            payload["index"] = index
        return payload

    def _normalize_usage(self, usage: object) -> dict[str, int] | None:
        if not isinstance(usage, dict):
            return None
        normalized_usage: dict[str, int] = {}
        for key, value in usage.items():
            if isinstance(key, str) and isinstance(value, int):
                normalized_usage[key] = value
        return normalized_usage or None

    def _stringify_text(self, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return self._encode_content(content)

    def _encode_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True)

    def _message_contains_block_type(self, message: object, block_type: str) -> bool:
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        if not isinstance(content, list):
            return False
        return any(isinstance(block, dict) and block.get("type") == block_type for block in content)

    def _truncate_summary(self, value: str, limit: int = 400) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."
