"""SSAPling teachers.

Responsibility: Requests, responses, capabilities, execution, and lifecycle for
user-defined teachers.

Not its responsibility: Teaching strategies, problem definitions, storage.

May import: core. Enforced by tests/test_layering.py.
"""

from collections.abc import Callable

from ..core import Registry
from .contracts import Teacher, check_resources, supports
from .errors import MissingContextError, MissingResourcesError, UnsupportedOperationError
from .runner import TeacherRunner
from .types import ResponseStatus, TeacherRequest, TeacherResponse

registry: Registry[Callable[..., Teacher]] = Registry("teacher")
"""Registered teachers. Also available as ``ssapling.registries.teachers``."""

__all__ = [
    "MissingContextError",
    "MissingResourcesError",
    "ResponseStatus",
    "Teacher",
    "TeacherRequest",
    "TeacherResponse",
    "TeacherRunner",
    "UnsupportedOperationError",
    "check_resources",
    "registry",
    "supports",
]
