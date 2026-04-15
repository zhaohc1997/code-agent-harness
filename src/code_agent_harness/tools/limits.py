TOOL_LIMITS = {
    "read_file": 20_000,
    "search_text": 10_000,
    "shell": 15_000,
}

DEFAULT_TOOL_LIMIT = 10_000


def get_tool_limit(tool_name: str) -> int:
    return TOOL_LIMITS.get(tool_name, DEFAULT_TOOL_LIMIT)
