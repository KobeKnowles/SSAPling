"""Matching descriptors against ``where``: the rule collections and aggregations share."""

from collections.abc import Mapping
from typing import Any

__all__ = ["matches"]


def matches(descriptors: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    """Whether ``descriptors`` hold every value that ``where`` names.

    A list and a tuple with equal items are equal, at any depth: a YAML or
    JSON file gives a tuple descriptor, such as a length pair ``(2, 1)``, as
    the list ``[2, 1]``.
    """
    return all(key in descriptors and _same(descriptors[key], v) for key, v in where.items())


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)
