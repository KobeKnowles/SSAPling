"""Errors specific to models."""

from collections.abc import Iterable

from ..core.errors import SSAPlingError
from .types import DescriptionMismatch


class DescriptionMismatchError(SSAPlingError, ValueError):
    """A model cannot handle an environment's observations or actions.

    Attributes:
        model: The model's class name.
        participant: The participant the model was checked for, or ``None``.
        mismatches: Every mismatch found, not just the first.
    """

    def __init__(
        self, model: str, mismatches: Iterable[DescriptionMismatch], *, participant: str | None
    ) -> None:
        self.model = model
        self.participant = participant
        self.mismatches = tuple(mismatches)
        role = f" as participant {participant!r}" if participant else ""
        lines = [f"Model {model} cannot handle the environment's descriptions{role}:"]
        lines += [str(mismatch) for mismatch in self.mismatches]
        super().__init__("\n  - ".join(lines))
