"""Measurements: one value per instance, taken from its attempt, with a status.

A **measurement** is a function that takes a ``Trial`` (the instance, its
record, the outcome, and the task) and returns a value. It is registered by
name in ``measurements``:

    def steps(trial):
        return len(trial.record.steps)

    measurements.register("steps", steps)

Every value carries a status, so a missing value is never mistaken for
zero: see ``MeasuredValue``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..core import Registry
from ..data import Record, RecordStatus
from ..tasks import Instance, Score

__all__ = [
    "InstanceResult",
    "MeasuredValue",
    "Trial",
    "ValueStatus",
    "measurements",
]

ValueStatus = Literal["available", "external", "unavailable", "failed"]
"""Where a measured value came from, or why there is none.

``available``: computed here. ``external``: reported by a service, such as
an ARC-AGI-3 scorecard, and kept as reported. ``unavailable``: it cannot be
computed, for example because no reference is accessible. ``failed``: the
measurement raised an error, or the attempt could not be completed.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasuredValue:
    """One measurement of one instance, or the reason there is none.

    Attributes:
        status: See ``ValueStatus``.
        value: The value. ``None`` exactly when unavailable or failed.
        detail: Optional. More about the value, or why there is none, such
            as the error.

    Build one with ``MeasuredValue.available(value)``, ``.external(value)``,
    ``.unavailable(reason)``, or ``.failed(reason)``.
    """

    status: ValueStatus
    value: Any = None
    detail: Any = None

    def __post_init__(self) -> None:
        if self.status not in ("available", "external", "unavailable", "failed"):
            raise ValueError(
                "status must be 'available', 'external', 'unavailable', or 'failed', "
                f"got {self.status!r}"
            )
        if (self.value is None) != (self.status in ("unavailable", "failed")):
            raise ValueError(
                f"a value is required when {self.status!r} and not allowed otherwise; "
                "put a reason in detail"
            )

    @classmethod
    def available(cls, value: Any, detail: Any = None) -> "MeasuredValue":
        """A value computed here."""
        return cls(status="available", value=value, detail=detail)

    @classmethod
    def external(cls, value: Any, detail: Any = None) -> "MeasuredValue":
        """A value reported by an external service, kept as reported."""
        return cls(status="external", value=value, detail=detail)

    @classmethod
    def unavailable(cls, reason: Any = None) -> "MeasuredValue":
        """No value can be computed; ``reason`` says why."""
        return cls(status="unavailable", detail=reason)

    @classmethod
    def failed(cls, reason: Any = None) -> "MeasuredValue":
        """The measurement, or the attempt, failed; ``reason`` says how."""
        return cls(status="failed", detail=reason)

    @classmethod
    def of(cls, value: Any) -> "MeasuredValue":
        """What a measurement returned, as a measured value.

        A ``MeasuredValue`` is kept. A task's ``Score`` keeps its status:
        local becomes available. ``None`` is unavailable. Anything else is
        available.
        """
        if isinstance(value, MeasuredValue):
            return value
        if isinstance(value, Score):
            if value.status == "unavailable":
                return cls.unavailable(value.detail)
            if value.status == "external":
                return cls.external(value.value, value.detail)
            return cls.available(value.value, value.detail)
        if value is None:
            return cls.unavailable("the measurement gave no value")
        return cls.available(value)


@dataclass(frozen=True, slots=True)
class Trial:
    """One instance's attempt, as a measurement sees it.

    Attributes:
        instance: The instance attempted, reference included.
        record: The attempt's record: every step, with its observations,
            actions, teacher responses, timing and model diagnostics.
        outcome: What the task scores: by default the last prediction, or
            the final observation (see ``StandardAssessment``).
        task: The task, for its ``score`` and ``check_partial``.
    """

    instance: Instance
    record: Record
    outcome: Any
    task: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceResult:
    """What an assessment found for one instance.

    Attributes:
        instance_id: The instance's id.
        collection: The collection it was drawn from.
        descriptors: The instance's descriptors, for grouping.
        record_id: Its record's id, to read the record back from the store.
        status: How its attempt ended: ``"terminated"``, ``"truncated"``,
            or ``"failed"``.
        values: Its measured values, keyed by measurement name.
    """

    instance_id: str
    collection: str
    descriptors: Mapping[str, Any]
    record_id: str
    status: RecordStatus
    values: Mapping[str, MeasuredValue]


def score(trial: Trial) -> Any:
    """The task's score of the outcome: ``task.score(instance, outcome)``."""
    return trial.task.score(trial.instance, trial.outcome)


measurements: Registry[Callable[[Trial], Any]] = Registry("measurement")
"""Registered measurements. Also available as ``ssapling.registries.measurements``.

A measurement is registered as the function itself, and looked up by name
with ``measurements.get(name)``. ``"score"`` is built in.
"""
measurements.register("score", score)
