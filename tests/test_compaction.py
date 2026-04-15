from code_agent_harness.engine.compaction import estimate_tokens, micro_compact


def test_micro_compact_preserves_latest_tool_results() -> None:
    messages = [
        {"role": "system", "content": "keep this"},
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "a" * 200}]},
        {"role": "assistant", "content": "keep that too"},
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "b" * 200}]},
    ]
    compacted = micro_compact(messages, keep_recent=1)

    assert compacted[-1]["content"][0]["content"] == "b" * 200


def test_micro_compact_compacts_older_tool_results_without_dropping_messages() -> None:
    messages = [
        {"role": "system", "content": "system context"},
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "a" * 200}]},
        {"role": "assistant", "content": "assistant context"},
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "search_text", "content": "b" * 200}]},
    ]

    compacted = micro_compact(messages, keep_recent=1)

    assert [message["role"] for message in compacted] == ["system", "user", "assistant", "user"]
    assert compacted[0]["content"] == "system context"
    assert compacted[2]["content"] == "assistant context"
    assert compacted[1]["content"][0]["content"] != "a" * 200
    assert compacted[1]["content"][0]["tool_name"] == "search_text"
    assert compacted[3]["content"][0]["content"] == "b" * 200


def test_estimate_tokens_scales_with_message_size() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) > estimate_tokens("a" * 4)


def test_estimate_tokens_is_stable_for_equivalent_structures() -> None:
    left = [{"b": 2, "a": [1, {"y": 3, "x": 4}]}]
    right = [{"a": [1, {"x": 4, "y": 3}], "b": 2}]

    assert estimate_tokens(left) == estimate_tokens(right)
