from text_utils import title_case_words


def test_title_case_words() -> None:
    assert title_case_words(["hello", "world"]) == "Hello World"
