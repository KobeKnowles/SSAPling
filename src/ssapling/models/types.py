"""Small types of the Models area: the "no output yet" marker, a description mismatch, and a
model's refusal of a description."""

import enum
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["NOTHING", "DescriptionKind", "DescriptionMismatch", "Nothing", "Refusal"]


class Nothing(enum.Enum):
    """The type of ``NOTHING``, which ``poll()`` returns when no output is ready."""

    NOTHING = "NOTHING"

    def __repr__(self) -> str:
        return "NOTHING"


NOTHING = Nothing.NOTHING
"""What ``poll()`` returns when no output is ready.

It is not ``None``, because ``None`` can be a real output. Compare with
``is``: ``if output is NOTHING``. It is an enum member, so it stays the same
object when sent to or from another process.
"""


DescriptionKind = Literal["observation", "action"]
"""Which description: of the observations a model receives, or of the actions it gives."""


@dataclass(frozen=True, slots=True)
class DescriptionMismatch:
    """One description a model cannot handle.

    Attributes:
        kind: ``"observation"`` or ``"action"``.
        given: The environment's description.
        handled: The model's own description, or ``None`` when the model
            decided through ``accepts_description`` instead.
        reason: Why the model refused, when its ``accepts_description``
            answered with a ``Refusal``; otherwise ``None``.
    """

    kind: DescriptionKind
    given: Any
    handled: Any = None
    reason: str | None = None

    def __str__(self) -> str:
        if self.handled is None:
            refused = f"{self.kind}s: the model does not accept {self.given!r}"
            return refused if self.reason is None else f"{refused}: {self.reason}"
        return (
            f"{self.kind}s: the model handles {self.handled!r}, "
            f"the environment gives {self.given!r}"
        )


@dataclass(frozen=True, slots=True)
class Refusal:
    """What ``accepts_description`` may answer instead of ``False``: a
    refusal that says why, such as which field the model reads and the
    environment doesn't give. It is false, so code that reads the answer as
    a bool still reads a refusal; ``check_descriptions`` keeps the reason.

    Attributes:
        reason: Why the model refuses the description.
    """

    reason: str

    def __bool__(self) -> bool:
        return False
