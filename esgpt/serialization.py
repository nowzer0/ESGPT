from pathlib import Path
from typing import Any


def json_safe_value(value: Any) -> Any:
    if callable(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value if not callable(item)]
    return str(value)


def namespace_to_json_dict(args: Any) -> dict[str, Any]:
    return {key: json_safe_value(value) for key, value in vars(args).items() if not callable(value)}
