"""Errors specific to tasks."""

from collections.abc import Iterable

from ..core.errors import SSAPlingError


class UnknownInstanceError(SSAPlingError, LookupError):
    """An enumerated task has no instance with the requested id.

    Attributes:
        instance_id: The id that was requested.
    """

    def __init__(self, instance_id: str, *, task: str | None = None) -> None:
        self.instance_id = instance_id
        where = f" in task {task!r}" if task else ""
        super().__init__(f"No instance {instance_id!r}{where}.")


class MissingTeachingResourceError(SSAPlingError, LookupError):
    """A teaching resource was requested that the task does not offer.

    Attributes:
        name: The requested resource.
        available: The names of the resources the task does offer.
    """

    def __init__(self, name: str, available: Iterable[str]) -> None:
        self.name = name
        self.available = tuple(sorted(available))
        offered = (
            f"it offers: {', '.join(repr(n) for n in self.available)}"
            if self.available
            else "it offers no teaching resources"
        )
        super().__init__(f"The task has no teaching resource {name!r}; {offered}.")
