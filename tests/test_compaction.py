from code_agent_harness.engine.compaction import auto_compact, estimate_tokens, micro_compact


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


def test_auto_compact_reinjects_identity_and_goal_when_threshold_exceeded() -> None:
    messages = [
        {"role": "user", "content": "Investigate the failing workflow"},
        {"role": "assistant", "content": "I will inspect the logs and repo state."},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_name": "search_text", "content": "x" * 400}],
        },
        {"role": "assistant", "content": "The workflow is failing in deployment."},
        {"role": "user", "content": "Continue with the fix."},
    ]

    compacted = auto_compact(
        messages,
        system_prompt="You are code-agent-harness.",
        task_goal="Fix the workflow failure",
        max_tokens=40,
        trigger_ratio=0.65,
        keep_recent=3,
    )

    assert compacted.applied_auto_summary is True
    assert compacted.messages[0]["role"] == "user"
    assert "You are code-agent-harness." in compacted.messages[0]["content"]
    assert "Fix the workflow failure" in compacted.messages[0]["content"]
    assert compacted.messages[-1] == {"role": "user", "content": "Continue with the fix."}
