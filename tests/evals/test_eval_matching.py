from code_agent_harness.evals.matching import match_argument_expectations
from code_agent_harness.evals.tasks import ArgumentExpectation


def test_match_argument_expectations_supports_exact_and_contains() -> None:
    matched, evidence = match_argument_expectations(
        {"path": "calc.py", "args": ["-q", "tests/test_calc.py"]},
        (
            ArgumentExpectation(field_path="path", match_mode="exact", expected="calc.py"),
            ArgumentExpectation(field_path="args", match_mode="contains", expected="tests/test_calc.py"),
        ),
    )

    assert matched is True
    assert evidence == ()


def test_match_argument_expectations_reports_missing_field() -> None:
    matched, evidence = match_argument_expectations(
        {"args": ["-q"]},
        (ArgumentExpectation(field_path="path", match_mode="exact", expected="calc.py"),),
    )

    assert matched is False
    assert evidence == ("missing field path",)
