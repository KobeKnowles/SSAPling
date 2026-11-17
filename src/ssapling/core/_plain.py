"""Plain values: what a JSON or YAML file holds, and reads back the same.

Internal: settings accept plain values only, so they never hold live objects;
provenance holds plain values only, so it can be stored whatever codec the
payloads use. Plain values are ``None``, booleans, numbers, strings, lists,
and mappings with string keys.
"""

from collections.abc import Mapping
from typing import Any

from ._text import join_path, show
from .errors import Problem

PLAIN = "a plain value (None, bool, int, float, str, or a list or str-keyed dict of these)"


def plain_copy(value: Any, path: str, problems: list[Problem]) -> Any:
    """A copy of ``value`` made of plain values, with tuples as lists.

    Appends a ``Problem`` for every part that is not plain, each with its
    path, so all of them can be reported at once.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [plain_copy(item, f"{path}[{i}]", problems) for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        copy = {}
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(Problem(path, f"keys must be strings, got {show(key)}"))
                continue
            copy[key] = plain_copy(item, join_path(path, key), problems)
        return copy
    problems.append(Problem(path, f"expected {PLAIN}, got {show(value)}"))
    return value
