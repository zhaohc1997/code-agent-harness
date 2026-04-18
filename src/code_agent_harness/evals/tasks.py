from __future__ import annotations

from dataclasses import dataclass


KNOWN_TASK_CLASSES = frozenset({"bugfix", "feature", "analysis"})
KNOWN_MATCH_MODES = frozenset(
    {"exact", "contains", "unordered_contains", "regex", "non_empty", "length_at_least"}
)


@dataclass(frozen=True)
class ArgumentExpectation:
    field_path: str
    match_mode: str
    expected: object

    def __post_init__(self) -> None:
        if self.match_mode not in KNOWN_MATCH_MODES:
            allowed = ", ".join(sorted(KNOWN_MATCH_MODES))
            raise ValueError(f"match_mode must be one of {allowed}, got {self.match_mode!r}")


@dataclass(frozen=True)
class ToolExpectation:
    name: str
    required: bool = True
    argument_expectations: tuple[ArgumentExpectation, ...] = ()
    must_appear_before: tuple[str, ...] = ()
    must_appear_after: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowExpectation:
    must_read_before_patch: bool = False
    must_run_tests_before_finish: bool = False
    forbid_patch: bool = False
    forbid_test_runs: bool = False
    allow_confirmation: bool = False
    require_response_summary: bool = True


@dataclass(frozen=True)
class OutcomeExpectation:
    repo_assertions: tuple[tuple[str, str], ...] = ()
    required_test_args_fragments: tuple[str, ...] = ()
    required_response_substrings: tuple[str, ...] = ()


@dataclass(frozen=True, init=False)
class EvalTask:
    task_id: str
    task_class: str
    fixture_name: str
    user_input: str
    tool_expectations: tuple[ToolExpectation, ...]
    workflow_expectations: WorkflowExpectation
    outcome_expectations: OutcomeExpectation
    live_eligible: bool

    def __init__(
        self,
        task_id: str,
        task_class: str,
        fixture_name: str,
        user_input: str,
        tool_expectations: tuple[ToolExpectation, ...] | None = None,
        workflow_expectations: WorkflowExpectation | None = None,
        outcome_expectations: OutcomeExpectation | None = None,
        live_eligible: bool = False,
        *,
        expected_tool_names: tuple[str, ...] | None = None,
        required_response_substrings: tuple[str, ...] | None = None,
        repo_assertions: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        if tool_expectations is not None and expected_tool_names is not None:
            raise ValueError("tool_expectations and expected_tool_names cannot both be provided")
        if outcome_expectations is not None and (
            repo_assertions is not None or required_response_substrings is not None
        ):
            raise ValueError(
                "outcome_expectations cannot be combined with repo_assertions or required_response_substrings"
            )

        if tool_expectations is None:
            if expected_tool_names is None:
                raise TypeError("tool_expectations or expected_tool_names must be provided")
            tool_expectations = tuple(ToolExpectation(name=name) for name in expected_tool_names)

        if workflow_expectations is None:
            workflow_expectations = WorkflowExpectation()

        if outcome_expectations is None:
            outcome_expectations = OutcomeExpectation(
                repo_assertions=() if repo_assertions is None else repo_assertions,
                required_response_substrings=(
                    () if required_response_substrings is None else required_response_substrings
                ),
            )

        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "task_class", task_class)
        object.__setattr__(self, "fixture_name", fixture_name)
        object.__setattr__(self, "user_input", user_input)
        object.__setattr__(self, "tool_expectations", tool_expectations)
        object.__setattr__(self, "workflow_expectations", workflow_expectations)
        object.__setattr__(self, "outcome_expectations", outcome_expectations)
        object.__setattr__(self, "live_eligible", live_eligible)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.task_class not in KNOWN_TASK_CLASSES:
            allowed = ", ".join(sorted(KNOWN_TASK_CLASSES))
            raise ValueError(f"task_class must be one of {allowed}, got {self.task_class!r}")
        if not self.tool_expectations:
            raise ValueError("tool_expectations must not be empty")

    @property
    def expected_tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tool_expectations)

    @property
    def required_response_substrings(self) -> tuple[str, ...]:
        return self.outcome_expectations.required_response_substrings

    @property
    def repo_assertions(self) -> tuple[tuple[str, str], ...]:
        return self.outcome_expectations.repo_assertions


def load_default_tasks() -> tuple[EvalTask, ...]:
    return (
        EvalTask(
            task_id="bugfix-basic",
            task_class="bugfix",
            fixture_name="bugfix_repo",
            user_input="Fix the add() function and run the smallest relevant test.",
            tool_expectations=(
                ToolExpectation(name="read_file"),
                ToolExpectation(name="apply_patch", must_appear_after=("read_file",)),
                ToolExpectation(
                    name="run_tests",
                    must_appear_after=("apply_patch",),
                    argument_expectations=(
                        ArgumentExpectation(field_path="args", match_mode="contains", expected="tests/test_calc.py"),
                    ),
                ),
            ),
            workflow_expectations=WorkflowExpectation(
                must_read_before_patch=True,
                must_run_tests_before_finish=True,
            ),
            outcome_expectations=OutcomeExpectation(
                repo_assertions=(("calc.py", "return a + b"),),
                required_test_args_fragments=("tests/test_calc.py",),
                required_response_substrings=("fixed", "passed"),
            ),
            live_eligible=True,
        ),
        EvalTask(
            task_id="feature-title-case",
            task_class="feature",
            fixture_name="feature_repo",
            user_input="Make title_case_words() capitalize each input word and run the targeted test.",
            tool_expectations=(
                ToolExpectation(name="read_file"),
                ToolExpectation(name="apply_patch", must_appear_after=("read_file",)),
                ToolExpectation(
                    name="run_tests",
                    must_appear_after=("apply_patch",),
                    argument_expectations=(
                        ArgumentExpectation(
                            field_path="args",
                            match_mode="contains",
                            expected="tests/test_text_utils.py",
                        ),
                    ),
                ),
            ),
            workflow_expectations=WorkflowExpectation(
                must_read_before_patch=True,
                must_run_tests_before_finish=True,
            ),
            outcome_expectations=OutcomeExpectation(
                repo_assertions=(("text_utils.py", "word.title()"),),
                required_test_args_fragments=("tests/test_text_utils.py",),
                required_response_substrings=("implemented", "passed"),
            ),
            live_eligible=True,
        ),
        EvalTask(
            task_id="analysis-timeout",
            task_class="analysis",
            fixture_name="analysis_repo",
            user_input="What is the default timeout and which features are enabled? Do not modify files.",
            tool_expectations=(ToolExpectation(name="read_file"),),
            workflow_expectations=WorkflowExpectation(forbid_patch=True),
            outcome_expectations=OutcomeExpectation(
                required_response_substrings=("45", "search", "history", "export"),
            ),
            live_eligible=True,
        ),
        EvalTask(
            task_id="bugfix-targeted-report",
            task_class="bugfix",
            fixture_name="bugfix_repo",
            user_input=(
                "Fix add(), run the targeted calc test, and explicitly report the test file you ran."
            ),
            tool_expectations=(
                ToolExpectation(
                    name="read_file",
                    argument_expectations=(
                        ArgumentExpectation(field_path="path", match_mode="exact", expected="calc.py"),
                    ),
                ),
                ToolExpectation(
                    name="apply_patch",
                    must_appear_after=("read_file",),
                    argument_expectations=(
                        ArgumentExpectation(field_path="path", match_mode="exact", expected="calc.py"),
                    ),
                ),
                ToolExpectation(
                    name="run_tests",
                    must_appear_after=("apply_patch",),
                    argument_expectations=(
                        ArgumentExpectation(
                            field_path="args",
                            match_mode="contains",
                            expected="tests/test_calc.py",
                        ),
                    ),
                ),
            ),
            workflow_expectations=WorkflowExpectation(
                must_read_before_patch=True,
                must_run_tests_before_finish=True,
            ),
            outcome_expectations=OutcomeExpectation(
                repo_assertions=(("calc.py", "return a + b"),),
                required_test_args_fragments=("tests/test_calc.py",),
                required_response_substrings=("fixed", "passed", "tests/test_calc.py"),
            ),
            live_eligible=True,
        ),
        EvalTask(
            task_id="analysis-readme-no-write",
            task_class="analysis",
            fixture_name="analysis_repo",
            user_input=(
                "Summarize the repository purpose from README.md. Do not modify files or run tests."
            ),
            tool_expectations=(
                ToolExpectation(
                    name="read_file",
                    argument_expectations=(
                        ArgumentExpectation(field_path="path", match_mode="exact", expected="README.md"),
                    ),
                ),
            ),
            workflow_expectations=WorkflowExpectation(
                forbid_patch=True,
                forbid_test_runs=True,
            ),
            outcome_expectations=OutcomeExpectation(
                required_response_substrings=("Analysis Repo", "without modifying files"),
            ),
            live_eligible=True,
        ),
    )
