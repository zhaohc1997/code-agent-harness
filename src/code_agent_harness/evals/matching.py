from __future__ import annotations

import re
from collections.abc import Iterable

from code_agent_harness.evals.tasks import ArgumentExpectation


_MISSING = object()


def _resolve_field(arguments: object, field_path: str) -> object:
    current = arguments
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _matches(actual: object, match_mode: str, expected: object) -> bool:
    if match_mode == "exact":
        return actual == expected
    if match_mode == "contains":
        if isinstance(actual, str):
            return str(expected) in actual
        if isinstance(actual, Iterable):
            return expected in actual
        return False
    if match_mode == "unordered_contains":
        if not isinstance(actual, Iterable) or isinstance(actual, (str, bytes)):
            return False
        if not isinstance(expected, Iterable) or isinstance(expected, (str, bytes)):
            return False
        actual_items = list(actual)
        return all(item in actual_items for item in expected)
    if match_mode == "regex":
        return re.search(str(expected), str(actual)) is not None
    if match_mode == "non_empty":
        return bool(actual)
    if match_mode == "length_at_least":
        try:
            return len(actual) >= int(expected)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    return False


def match_argument_expectations(
    arguments: dict[str, object], expectations: tuple[ArgumentExpectation, ...]
) -> tuple[bool, tuple[str, ...]]:
    evidence: list[str] = []
    for expectation in expectations:
        actual = _resolve_field(arguments, expectation.field_path)
        if actual is _MISSING:
            evidence.append(f"missing field {expectation.field_path}")
            continue
        if _matches(actual, expectation.match_mode, expectation.expected):
            continue
        evidence.append(
            f"field {expectation.field_path} expected {expectation.match_mode} "
            f"{expectation.expected!r}, got {actual!r}"
        )
    return (not evidence), tuple(evidence)
