from code_agent_harness.engine.cancellation import CancellationToken


def test_cancellation_token_toggles_state() -> None:
    token = CancellationToken()

    assert token.is_cancelled() is False

    token.cancel()

    assert token.is_cancelled() is True
