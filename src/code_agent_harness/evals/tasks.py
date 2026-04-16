from dataclasses import dataclass


KNOWN_TASK_CLASSES = frozenset({"bugfix", "feature", "analysis"})


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    task_class: str
    fixture_name: str
    user_input: str
    expected_tool_names: tuple[str, ...]
    required_response_substrings: tuple[str, ...]
    repo_assertions: tuple[tuple[str, str], ...]
    live_eligible: bool

    def __post_init__(self) -> None:
        if self.task_class not in KNOWN_TASK_CLASSES:
            allowed = ", ".join(sorted(KNOWN_TASK_CLASSES))
            raise ValueError(
                f"task_class must be one of {allowed}, got {self.task_class!r}"
            )


def load_default_tasks() -> tuple[EvalTask, ...]:
    return (
        EvalTask(
            task_id="bugfix-basic",
            task_class="bugfix",
            fixture_name="bugfix_repo",
            user_input="Fix the add() function and run the smallest relevant test.",
            expected_tool_names=("read_file", "apply_patch", "run_tests"),
            required_response_substrings=("fixed", "passed"),
            repo_assertions=(("calc.py", "return a + b"),),
            live_eligible=True,
        ),
        EvalTask(
            task_id="feature-title-case",
            task_class="feature",
            fixture_name="feature_repo",
            user_input="Make title_case_words() capitalize each input word and run the targeted test.",
            expected_tool_names=("read_file", "apply_patch", "run_tests"),
            required_response_substrings=("implemented", "passed"),
            repo_assertions=(("text_utils.py", "word.title()"),),
            live_eligible=True,
        ),
        EvalTask(
            task_id="analysis-timeout",
            task_class="analysis",
            fixture_name="analysis_repo",
            user_input="What is the default timeout and which features are enabled? Do not modify files.",
            expected_tool_names=("read_file",),
            required_response_substrings=("45", "search", "history", "export"),
            repo_assertions=(),
            live_eligible=True,
        ),
    )
