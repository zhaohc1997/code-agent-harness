from code_agent_harness.engine.compaction import estimate_tokens, micro_compact


def test_micro_compact_preserves_latest_tool_results() -> None:
    messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "a" * 200}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "b" * 200}]},
    ]
    compacted = micro_compact(messages, keep_recent=1)
    assert compacted[-1]["content"][0]["content"] == "b" * 200


def test_estimate_tokens_scales_with_message_size() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) > estimate_tokens("a" * 4)
