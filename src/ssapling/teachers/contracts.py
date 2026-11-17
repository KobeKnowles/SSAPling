"""The teacher contract, and the checks consumers make before using a teacher.

A teacher is any object with ``operations``, the names of the operations it
supports, and ``respond(request)``, written as ``def`` or ``async def``. This
is a structural protocol: a downstream teacher implements it and inherits from
nothing.

Consumers check an operation before requesting it, and check a teacher's
teaching resources before a run::

    from ssapling import tasks, teachers

    teachers.check_resources(teacher, tasks.teaching_resources(task))
    if teachers.supports(teacher, "review_action"):
        response = await runner.request(teacher, request)
"""

from collections.abc import Awaitable, Collection, Iterable, Mapping
from typing import Protocol

from ..core._declarations import name_of, names
from .errors import MissingResourcesError
from .types import TeacherRequest, TeacherResponse

__all__ = ["Teacher", "check_resources", "supports"]


class Teacher(Protocol):
    """What every teacher provides: the operations it supports, and answers to requests.

    A teacher may also have any of these, each optional:

    - ``required_context``: for each operation, the context names a request
      must give, such as ``{"review_action": ("instance", "state")}``. The
      runner checks them before asking.
    - ``required_resources``: the names of the teaching resources it needs
      from the task, such as ``("procedure",)``. See ``check_resources``.
    - ``reset()``: forget episode state, before a new episode.
    - ``close()``: release what it holds, such as a connection.
    - ``export_state()`` and ``restore_state(state)``: save and restore its
      persistent state, for checkpoints. A human or remote teacher need not
      offer these.

    The four methods may be ``def`` or ``async def``. One owner calls them,
    so a teacher is never reset or closed twice by different layers.
    """

    @property
    def operations(self) -> Collection[str]:
        """The names of the operations this teacher supports, such as ``{"review_action"}``."""
        ...

    def respond(self, request: TeacherRequest) -> TeacherResponse | Awaitable[TeacherResponse]:
        """Respond to ``request``, whose operation is one of ``operations``.

        Return a ``TeacherResponse`` for this request: answered (with any
        contributions), abstained, failed, or cancelled.
        """
        ...


def supports(teacher: object, operation: str) -> bool:
    """Whether ``teacher`` supports ``operation``: whether its ``operations`` list it.

    Operations are an open set of names chosen downstream, so a name the
    teacher has not heard of is simply not supported.

    A teacher's ``operations`` are its only declaration. To switch an
    operation off, for example by a setting, leave it out of ``operations``;
    a ``supports`` method on the teacher itself is not consulted.

    Raises:
        TypeError: ``teacher`` has no ``operations``, or they are not a
            collection of names. A single string is rejected, because
            ``"review" in "review_action"`` would be true.
    """
    if not isinstance(operation, str):
        raise TypeError(f"an operation is a name, got {operation!r}")
    return operation in _operations(teacher)


def check_resources(teacher: object, available: Iterable[str]) -> None:
    """Check that every teaching resource ``teacher`` requires is available.

    Call it before a run, with the names of the resources the task offers::

        teachers.check_resources(teacher, tasks.teaching_resources(task))

    ``teacher`` may be a teacher or its class, so a teacher can be checked
    before it is built (``required_resources`` must then be a class
    attribute). A teacher without ``required_resources`` needs none.

    Raises:
        MissingResourcesError: A required resource is not available. The
            error lists every missing resource and every available one.
        TypeError: ``required_resources`` or ``available`` is not a
            collection of names.
    """
    required = _required_resources(teacher)
    if isinstance(available, str | bytes) or not isinstance(available, Iterable):
        raise TypeError(f"available must be a collection of resource names, got {available!r}")
    offered = set(available)
    missing = [name for name in required if name not in offered]
    if missing:
        raise MissingResourcesError(name_of(teacher), missing, offered)


# Reading a teacher's declarations. The runner uses these too.


def _operations(teacher: object) -> frozenset[str]:
    declared = getattr(teacher, "operations", None)
    if declared is None:
        raise TypeError(
            f"{name_of(teacher)} has no operations: a teacher lists the operations it "
            "supports, such as operations = {'review_action'}"
        )
    return frozenset(
        names(declared, f"{name_of(teacher)}.operations", example="('review_action',)")
    )


def _required_context(teacher: object, operation: str) -> tuple[str, ...]:
    """The context names ``teacher`` requires for ``operation``; empty if it declares none."""
    declared = getattr(teacher, "required_context", None)
    if declared is None:
        return ()
    where = f"{name_of(teacher)}.required_context"
    if not isinstance(declared, Mapping):
        raise TypeError(
            f"{where} must map operations to the context names each needs, such as "
            f"{{'review_action': ('observation',)}}, got {declared!r}"
        )
    operations = _operations(teacher)
    unknown = sorted(repr(name) for name in declared if name not in operations)
    if unknown:
        raise ValueError(f"{where} names operations missing from operations: {', '.join(unknown)}")
    return names(declared.get(operation, ()), f"{where}[{operation!r}]", example="('state',)")


def _required_resources(teacher: object) -> tuple[str, ...]:
    declared = getattr(teacher, "required_resources", None)
    if declared is None:
        return ()
    return names(declared, f"{name_of(teacher)}.required_resources", example="('procedure',)")
