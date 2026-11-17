"""The small result types of the Tasks area: an instance, a score, and a partial-check report."""

import numbers
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core._declarations import check_identifier

__all__ = ["Instance", "PartialReport", "Score", "ScoreStatus"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Instance:
    """One concrete problem, such as one ARC puzzle or one addition.

    Attributes:
        id: A stable identifier, so records, scores, and teacher requests can
            refer back to the instance. A generated instance's id should
            follow from its parameters and seed.
        inputs: Everything presented for solving: initial conditions and the
            problem statement. This is the only part a student is shown.
            Required: pass ``inputs=None`` explicitly for an instance with no
            local inputs, as for a remote game that an environment adapter
            resolves by ``id``, so inputs are never left out by accident.
        reference: Optional. An expected output, a reference solution such as
            a sequence of steps, or both. Consumed by teachers and
            evaluators, never shown to the student unless a downstream
            project explicitly chooses so.
        descriptors: Optional task-defined attributes, such as difficulty or
            length, keyed by name. SSAPling passes them through
            uninterpreted; Data samples by them and Evaluate groups by them.
        equivalence_key: Optional. Instances with the same key are the same
            problem as far as the task is concerned (for example the same
            puzzle under a different seed), so Data keeps them in one
            collection. A hashable value, such as a string or a tuple of
            strings and numbers.

    Payloads (``inputs``, ``reference``, descriptor values) are whatever the
    task defines. They are treated as read-only once the instance exists.
    """

    id: str
    inputs: Any
    reference: Any = None
    descriptors: Mapping[str, Any] = field(default_factory=dict)
    equivalence_key: Hashable | None = None

    def __post_init__(self) -> None:
        check_identifier(self.id, "an instance id")
        if not isinstance(self.descriptors, Mapping):
            raise TypeError(
                f"descriptors must be a mapping of names to values, got {self.descriptors!r}"
            )
        if any(not isinstance(key, str) for key in self.descriptors):
            raise TypeError("descriptor names must be strings")
        if self.equivalence_key is not None:
            # Data keeps reserved keys in a set, so check here rather than fail there.
            try:
                hash(self.equivalence_key)
            except TypeError:
                raise TypeError(
                    "an equivalence key must be hashable, such as a string or a tuple of "
                    f"strings and numbers, got {self.equivalence_key!r}"
                ) from None
        object.__setattr__(self, "descriptors", dict(self.descriptors))

    @property
    def problem_key(self) -> Hashable:
        """Which problem this is: the equivalence key, or the id when the task declares none.

        Data uses it to keep equivalent instances in the same collection.
        """
        return self.equivalence_key if self.equivalence_key is not None else self.id

    def __hash__(self) -> int:
        # Equality compares every field; hashing by id keeps instances usable
        # in sets and dicts even when their payloads are unhashable lists.
        return hash(self.id)


ScoreStatus = Literal["local", "external", "unavailable"]
"""Where a score came from, or why there is none.

``local``: computed by the task. ``external``: reported by a service, such as
an ARC-AGI-3 scorecard, and recorded as reported. ``unavailable``: no score
can be given, for example because the reference is hidden or a service has
not reported yet. A score is never invented.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class Score:
    """The task's judgement of one outcome of one instance.

    Attributes:
        status: ``"local"``, ``"external"``, or ``"unavailable"``.
        value: The score, task-defined (often a number in [0, 1], or a bool).
            Always ``None`` when unavailable, never ``None`` otherwise.
        detail: Optional task-defined extra: per-part correctness, the
            service's raw report, or the reason a score is unavailable.

    Build one with ``Score.local(value)``, ``Score.external(value)``, or
    ``Score.unavailable(reason)``.
    """

    status: ScoreStatus
    value: Any = None
    detail: Any = None

    def __post_init__(self) -> None:
        if self.status not in ("local", "external", "unavailable"):
            raise ValueError(
                f"status must be 'local', 'external', or 'unavailable', got {self.status!r}"
            )
        if self.status == "unavailable" and self.value is not None:
            raise ValueError("an unavailable score has no value; put the reason in detail")
        if self.status != "unavailable" and self.value is None:
            raise ValueError(
                f"a {self.status} score needs a value; use Score.unavailable() when there is none"
            )

    @classmethod
    def local(cls, value: Any, detail: Any = None) -> "Score":
        """A score the task computed itself."""
        return cls(status="local", value=value, detail=detail)

    @classmethod
    def external(cls, value: Any, detail: Any = None) -> "Score":
        """A score reported by an external service, recorded as reported."""
        return cls(status="external", value=value, detail=detail)

    @classmethod
    def unavailable(cls, reason: Any = None) -> "Score":
        """No score can be given; ``reason`` says why and is kept in ``detail``."""
        return cls(status="unavailable", detail=reason)

    @property
    def available(self) -> bool:
        """Whether there is a value to use."""
        return self.status != "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class PartialReport:
    """A task's judgement of an intermediate state, before the outcome is final.

    Every field is optional, because tasks differ in what they can judge
    part-way: whether the state is still valid, how far along it is, or where
    the first mistake is.

    Attributes:
        valid: Whether the state so far is consistent with a correct solution.
        progress: How much of the problem is solved, in [0, 1].
        first_error: Task-defined: where or what the first mistake is, or
            ``None`` when there is none (or the task cannot tell).
        detail: Any other task-defined information.
    """

    valid: bool | None = None
    progress: float | None = None
    first_error: Any = None
    detail: Any = None

    def __post_init__(self) -> None:
        if self.progress is None:
            return
        # Any real number, such as a NumPy float32, but not a bool.
        if isinstance(self.progress, bool) or not isinstance(self.progress, numbers.Real):
            raise TypeError(f"progress must be a number between 0 and 1, got {self.progress!r}")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"progress must be between 0 and 1, got {self.progress!r}")
