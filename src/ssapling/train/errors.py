"""Errors specific to train."""

from collections.abc import Mapping

from ..core.errors import SSAPlingError


class ProcedureStuckError(SSAPlingError, RuntimeError):
    """Nothing in a fixed procedure can run, and its stop condition does not hold.

    Raised instead of looping forever, after a whole pass through the list
    in which no entry's trigger fired.

    Attributes:
        counters: The run's counters at that point.
    """

    def __init__(self, counters: Mapping[str, int]) -> None:
        self.counters = dict(counters)
        shown = ", ".join(f"{name} {count}" for name, count in self.counters.items())
        super().__init__(
            f"Nothing in the procedure can run: no entry's trigger fires, and the stop "
            f"condition does not hold ({shown}). Check the triggers and the stop condition."
        )
