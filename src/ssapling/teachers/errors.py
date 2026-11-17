"""Errors specific to teachers.

Each is a mistake in how a teacher is used, found before the teacher is asked
anything. Problems with the teacher's own answer are not errors: the runner
records them as a failed or cancelled response.
"""

from collections.abc import Iterable

from ..core.errors import SSAPlingError


class UnsupportedOperationError(SSAPlingError, LookupError):
    """A teacher was asked for an operation it does not list in ``operations``.

    Attributes:
        teacher: The teacher's class name.
        operation: The operation that was requested.
        supported: The operations the teacher does support, sorted.
    """

    def __init__(self, teacher: str, operation: str, supported: Iterable[str]) -> None:
        self.teacher = teacher
        self.operation = operation
        self.supported = tuple(sorted(supported))
        super().__init__(
            f"Teacher {teacher} does not support operation {operation!r}; "
            f"{_offers(self.supported, 'it supports', 'it supports no operations')}."
        )


class MissingContextError(SSAPlingError, LookupError):
    """A request lacks context that the teacher requires for its operation.

    Attributes:
        teacher: The teacher's class name.
        operation: The requested operation.
        missing: The required context names the request does not give.
        given: The context names the request does give, sorted.
    """

    def __init__(
        self, teacher: str, operation: str, missing: Iterable[str], given: Iterable[str]
    ) -> None:
        self.teacher = teacher
        self.operation = operation
        self.missing = tuple(missing)
        self.given = tuple(sorted(given))
        super().__init__(
            f"Teacher {teacher} needs context {_names(self.missing)} for operation "
            f"{operation!r}; {_offers(self.given, 'the request gives', 'the request gives none')}."
        )


class MissingResourcesError(SSAPlingError, LookupError):
    """A teacher requires teaching resources that are not available.

    Attributes:
        teacher: The teacher's class name.
        missing: The required resources that are not available.
        available: The resources that are available, sorted.
    """

    def __init__(self, teacher: str, missing: Iterable[str], available: Iterable[str]) -> None:
        self.teacher = teacher
        self.missing = tuple(missing)
        self.available = tuple(sorted(available))
        super().__init__(
            f"Teacher {teacher} needs teaching resources that are not available: "
            f"{_names(self.missing)}; "
            f"{_offers(self.available, 'available', 'none are available')}."
        )


def _names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _offers(names: tuple[str, ...], lead: str, empty: str) -> str:
    return f"{lead}: {_names(names)}" if names else empty
