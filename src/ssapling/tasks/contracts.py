"""The task contract, its optional features, and helpers for consumers.

A task is any object with a ``score`` method. It becomes a *generated* task by
also offering ``generate``, an *enumerated* task by offering ``ids`` and
``get``, and may offer ``check_partial`` and ``teaching_resources``. These are
structural protocols: a downstream task implements the methods and inherits
from nothing.

Consumers check a feature before using it::

    from ssapling import tasks

    if tasks.supports(task, "check_partial"):
        report = task.check_partial(instance, state)
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from ..core._declarations import has_feature
from .errors import MissingTeachingResourceError
from .types import Instance, PartialReport, Score

__all__ = [
    "FEATURES",
    "EnumeratedTask",
    "GeneratedTask",
    "PartialChecker",
    "Task",
    "TeachingResourceProvider",
    "supports",
    "teaching_resource",
    "teaching_resources",
]


class Task(Protocol):
    """What every task provides: a judgement of one outcome of one instance."""

    def score(self, instance: Instance, outcome: Any) -> Score:
        """Score ``outcome``, one attempt's result, against ``instance``.

        Return ``Score.local`` when the task can judge it, ``Score.external``
        when a service reported it, and ``Score.unavailable`` otherwise, for
        example when the reference is hidden. Never guess.
        """
        ...


class GeneratedTask(Task, Protocol):
    """A task that makes instances from task-defined parameters and a seed."""

    def generate(self, *, params: Mapping[str, Any] | None = None, seed: int) -> Instance:
        """Make one instance. The same ``params`` and ``seed`` must give an equal instance."""
        ...


class EnumeratedTask(Task, Protocol):
    """A task with a fixed set of instances, each with an id."""

    def ids(self) -> Sequence[str]:
        """Every instance id, in a stable order."""
        ...

    def get(self, instance_id: str) -> Instance:
        """The instance with that id.

        Raises:
            UnknownInstanceError: No instance has that id.
        """
        ...


class PartialChecker(Task, Protocol):
    """A task that can judge an intermediate state (feature ``"check_partial"``)."""

    def check_partial(self, instance: Instance, state: Any) -> PartialReport:
        """Judge ``state``, a task-defined partial solution of ``instance``."""
        ...


class TeachingResourceProvider(Task, Protocol):
    """A task that offers task-level material for teachers (feature ``"teaching_resources"``)."""

    def teaching_resources(self) -> Mapping[str, Any]:
        """Resources keyed by name, such as a worked example or a solving procedure."""
        ...


FEATURES: Mapping[str, tuple[str, ...]] = {
    "generate": ("generate",),
    "enumerate": ("ids", "get"),
    "check_partial": ("check_partial",),
    "teaching_resources": ("teaching_resources",),
}
"""The optional features of a task, each with the methods it needs."""


def supports(task: object, feature: str) -> bool:
    """Whether ``task`` offers ``feature``: one of ``FEATURES``.

    A task offers a feature when it has every method the feature needs and,
    if it defines a ``supports(feature)`` method of its own, that method
    agrees. So a task declares a feature just by implementing it, and can
    switch off a feature its class implements (for example when a setting
    turns it off) by answering ``False`` from its own ``supports``.

    A task's own ``supports`` only switches features off: it answers
    ``False`` for those and ``True`` for every other feature. It must not
    call this function, because this function calls it.

    Raises:
        ValueError: ``feature`` is not a task feature. That is a mistake in
            the calling code, not something a task can answer.
        TypeError: The task has a ``supports`` attribute that is not a method.
    """
    return has_feature(task, feature, FEATURES, word="feature", words="features")


def teaching_resources(task: object) -> dict[str, Any]:
    """The task's teaching resources by name, or an empty dict if it offers none."""
    if not supports(task, "teaching_resources"):
        return {}
    return dict(cast(TeachingResourceProvider, task).teaching_resources())


def teaching_resource(task: object, name: str) -> Any:
    """One teaching resource of ``task``.

    Raises:
        MissingTeachingResourceError: The task offers no resource named
            ``name``. The error lists the resources it does offer.
    """
    resources = teaching_resources(task)
    if name not in resources:
        raise MissingTeachingResourceError(name, resources)
    return resources[name]
