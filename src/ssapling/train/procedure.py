"""Procedures: which stages a run does, when, and when it stops.

A **procedure** lists entries, each a registered stage with its settings and
an optional trigger, and a stop condition. It is written in Python or read
from a file, and checked the same way::

    procedure:
      - stage: interact
        settings: {episodes: 32}
      - stage: learn_operation
        every: {episodes: 500}
        settings: {operation: consolidate}
    stop: {episodes: 100000}

A **controller** decides what the run does next. The built-in
``FixedController`` follows the list.
"""

from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..core import setting
from .components import ComponentEntry
from .context import COUNTERS, RunContext
from .errors import ProcedureStuckError

__all__ = [
    "Condition",
    "Controller",
    "Decision",
    "Entry",
    "Every",
    "FixedController",
    "Procedure",
    "ReplayController",
]


@dataclass(frozen=True)
class _Counts:
    """The counters a trigger or a condition can name, and the run's time."""

    episodes: int | None = setting(None, ge=1, description="Episodes run.")
    steps: int | None = setting(None, ge=1, description="Environment steps taken.")
    updates: int | None = setting(None, ge=1, description="Calls to a learning operation.")
    stages: int | None = setting(None, ge=1, description="Stages started.")
    seconds: float | None = setting(
        None,
        gt=0,
        description="Seconds the run has been running. Unlike the counters, not reproducible.",
    )

    def named(self) -> dict[str, float]:
        """The counters given, with their numbers."""
        return {
            name: getattr(self, name)
            for name in (*COUNTERS, "seconds")
            if getattr(self, name) is not None
        }


@dataclass(frozen=True)
class Every(_Counts):
    """A trigger: run each time a counter has grown by this much since the entry last ran.

    Name exactly one counter, such as ``{episodes: 500}``. The first time,
    the counter is counted from the start of the run.
    """

    def validate(self) -> None:
        if len(self.named()) != 1:
            raise ValueError("every names exactly one counter, such as {episodes: 500}")


@dataclass(frozen=True)
class Condition(_Counts):
    """Holds when any counter named has reached its number, or the value named is in bounds.

    ``{episodes: 100000}`` holds after 100000 episodes, and ``{value:
    monitoring.accuracy, at_least: 0.95}`` once that published value is at
    least 0.95. Given both, whichever comes first. A value not yet
    published leaves the condition unmet, and so does an undefined one
    (``None``), such as an aggregate over no values.
    """

    value: str | None = setting(
        None,
        min_length=1,
        description=(
            "A published value, by its name or a dotted path into it, such as "
            "'monitoring.accuracy'."
        ),
    )
    at_least: float | None = setting(None, description="The value has reached this.")
    at_most: float | None = setting(None, description="The value has come down to this.")

    def validate(self) -> None:
        bounded = self.at_least is not None or self.at_most is not None
        if self.value is None and bounded:
            raise ValueError("at_least and at_most bound a value; name it in value")
        if self.value is not None and not bounded:
            raise ValueError(f"give the value {self.value!r} a bound: at_least or at_most")
        if self.value is None and not self.named():
            raise ValueError(
                "a condition names a counter, such as {episodes: 1000}, or a value and a bound"
            )

    @property
    def reads(self) -> str | None:
        """The name of the published value the condition reads, if any."""
        return None if self.value is None else self.value.split(".")[0]

    def holds(self, context: RunContext) -> bool:
        """Whether the condition holds in ``context`` now.

        Raises:
            LookupError: The value is published, but the path leads nowhere
                in it (see ``RunContext.read``).
            TypeError: The value is not a number.
        """
        now = _now(context)
        if any(now[name] >= limit for name, limit in self.named().items()):
            return True
        if self.value is None or self.reads not in context.values:
            return False
        value = context.read(self.value)
        if value is None:
            return False
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{self.value!r} is {value!r}, not a number, so it has no bounds")
        return (self.at_least is None or value >= self.at_least) and (
            self.at_most is None or value <= self.at_most
        )


@dataclass(frozen=True)
class Entry:
    """One entry of a procedure: a registered stage, its settings, and when it runs.

    It has at most one trigger. Without one, it runs each time the list
    reaches it.
    """

    stage: str = setting(min_length=1, description="The registered stage.")
    settings: dict[str, Any] = setting(default_factory=dict, description="The stage's settings.")
    once: bool = setting(False, description="Run the first time the list reaches it, only.")
    every: Every | None = setting(
        None, description="Run each time a counter has grown by this much since it last ran."
    )
    until: Condition | None = setting(
        None, description="Run each time the list reaches it, until the condition holds."
    )
    when: Condition | None = setting(
        None,
        description="Run once: the first time the list reaches it while the condition holds.",
    )

    def validate(self) -> None:
        triggers = [name for name in ("every", "until", "when") if getattr(self, name) is not None]
        if self.once:
            triggers.insert(0, "once")
        if len(triggers) > 1:
            raise ValueError(f"an entry has at most one trigger, got {', '.join(triggers)}")


@dataclass(frozen=True)
class Procedure:
    """The stages a run does, in order with their triggers, and when it stops.

    It may also list the components the run builds, so that one file
    describes the whole run.
    """

    procedure: list[Entry] = setting(description="The entries, in order.")
    stop: Condition = setting(
        description=(
            "When the run stops. It is checked before every stage, whatever the controller "
            "decides, so it is also the run's budget."
        )
    )
    components: dict[str, ComponentEntry] = setting(
        default_factory=dict,
        description="The components the run builds, by role, in the order they are built.",
    )


Decision = Entry | Mapping[str, Any] | None
"""What a controller decides: an entry to run, as an ``Entry`` or a mapping of
its settings, or ``None`` to stop."""


class Controller(Protocol):
    """Decides what a run does next: which stage, with which settings, or to stop.

    It may also have ``export_state()`` and ``restore_state(state)``, so its
    state is saved with each checkpoint.
    """

    def decide(
        self, context: RunContext, last_error: Exception | None
    ) -> Decision | Awaitable[Decision]:
        """The entry to run next, or ``None`` to stop. May be ``async def``.

        The entry may be an ``Entry`` or a mapping of its settings, such as
        ``{"stage": "interact", "settings": {"episodes": 8}}``, and is
        checked like an entry of a procedure. ``last_error`` says why the
        previous decision was refused, or is ``None``. A refused decision
        did not run, so the controller can choose again. Only the entry's
        stage and settings are used; its trigger, if any, is the
        controller's own business.
        """
        ...


class FixedController:
    """Follows a procedure's list: goes through it again and again, running each entry that fires.

    The run stops when the stop condition holds. If a whole pass through
    the list finds nothing to run before that, it raises
    ``ProcedureStuckError`` instead of looping forever.
    """

    def __init__(self, procedure: Procedure) -> None:
        self.entries = tuple(procedure.procedure)
        self.position = 0  # the next entry to consider
        # The counters when each entry last ran; None if it has not.
        self.last: list[dict[str, float] | None] = [None] * len(self.entries)

    def decide(self, context: RunContext, last_error: Exception | None) -> Entry:
        """The next entry in the list whose trigger fires.

        Raises:
            ProcedureStuckError: No entry's trigger fires.
            Exception: ``last_error``: the list was checked before the run,
                so a refused entry cannot be chosen differently.
        """
        if last_error is not None:
            raise last_error
        for _ in self.entries:
            index = self.position
            self.position = (index + 1) % len(self.entries)
            if self._fires(index, context):
                self.last[index] = _now(context)
                return self.entries[index]
        raise ProcedureStuckError(context.counters)

    def export_state(self) -> dict[str, Any]:
        """Where it is in the list, and when each entry last ran, for a checkpoint."""
        return {"position": self.position, "last": list(self.last)}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Return to what ``export_state`` saved."""
        self.position, self.last = state["position"], list(state["last"])

    def _fires(self, index: int, context: RunContext) -> bool:
        entry, last = self.entries[index], self.last[index]
        if entry.once:
            return last is None
        if entry.when is not None:
            return last is None and entry.when.holds(context)
        if entry.until is not None:
            return not entry.until.holds(context)
        if entry.every is not None:
            ((counter, amount),) = entry.every.named().items()
            since = 0 if last is None else last[counter]
            return _now(context)[counter] - since >= amount
        return True


class ReplayController:
    """Makes a run's recorded decisions again, in order, then stops: whatever made them.

    Example::

        decisions = RecordStore("runs/llm").load_document("decisions")["decisions"]
        await Runner(procedure, components, seed=7, controller=ReplayController(decisions)).run()

    Args:
        decisions: A run's decision log (see ``RunContext.decisions``).
            The decisions that were refused are left out.
    """

    def __init__(self, decisions: Iterable[Mapping[str, Any]]) -> None:
        self.decisions = [
            {"stage": decision["stage"], "settings": decision["settings"]}
            for decision in decisions
            if decision["refused"] is None
        ]
        self.position = 0

    def decide(self, context: RunContext, last_error: Exception | None) -> Decision:
        """The next recorded decision, or ``None`` after the last.

        Raises:
            Exception: ``last_error``: a recorded decision that cannot run
                now cannot be replayed.
        """
        if last_error is not None:
            raise last_error
        if self.position == len(self.decisions):
            return None
        self.position += 1
        return self.decisions[self.position - 1]

    def export_state(self) -> dict[str, int]:
        """How far it has replayed, for a checkpoint."""
        return {"position": self.position}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Return to what ``export_state`` saved."""
        self.position = state["position"]


def _now(context: RunContext) -> dict[str, float]:
    return {**context.counters, "seconds": context.seconds}
