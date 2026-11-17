"""Small text helpers shared by core's modules, for paths and error messages."""

import inspect
from typing import Any


def join_path(path: str, name: str) -> str:
    """Extend a dotted path: ``join_path("schedule", "warmup") == "schedule.warmup"``.

    An empty ``name`` is the whole of ``path``, as for a problem with a whole
    group of settings.
    """
    return f"{path}.{name}" if path and name else path or name


def show(value: Any) -> str:
    """Describe a value for an error message: its type and a shortened repr."""
    if value is None:
        return "None"
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{type(value).__name__} {text}"


def summary(obj: object) -> str | None:
    """The first paragraph of an object's own docstring, on one line, or None."""
    doc = getattr(obj, "__doc__", None)
    name = getattr(obj, "__name__", None)
    # Without a docstring, dataclasses generates one like "Name(a: int = 1)".
    if not isinstance(doc, str) or not doc.strip() or (name and doc.startswith(f"{name}(")):
        return None
    return " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())


def describe(obj: object) -> str:
    """Name a class or function by its module and qualified name, for error messages."""
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(obj)
