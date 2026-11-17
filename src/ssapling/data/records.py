"""Records: the log of one attempt at one instance.

SSAPling fixes the structure of a record, so that Data, Train, and Evaluate
can handle any project's records. What the payloads contain (observations,
actions, predictions, teacher contributions, diagnostics) is downstream:
they pass through uninterpreted.

A record has a header, fixed when the attempt starts; steps, one per
environment step or, in static solving, one per presentation; and a
summary, saying how the attempt ended. A step holds the environment's
``StepResult`` and the teachers' ``TeacherResponse`` objects themselves
rather than copying their fields, so each shape is defined once.
``RecordBuilder`` builds a record by appending steps, and a record is valid
before it has a summary, so it can be streamed during interaction and kept
when an episode is interrupted.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core._declarations import check_identifier, check_int, keyed, names
from ..environments.types import ResetResult, StepResult
from ..teachers.types import TeacherResponse
from .provenance import Provenance

__all__ = [
    "RECORD_SCHEMA_VERSION",
    "AgentTiming",
    "Clock",
    "LateOutput",
    "Record",
    "RecordBuilder",
    "RecordHeader",
    "RecordStatus",
    "RecordStep",
    "RecordSummary",
    "TeacherPlacement",
]

RECORD_SCHEMA_VERSION = 1
"""The version of the record structure this SSAPling writes and reads.

It changes when the structure does, so stored records can be recognised and
read back correctly.
"""

Clock = Literal["simulated", "wall"]
"""The clock an attempt was timed with.

``simulated``: advanced only by the coordinating code, so runs are
reproducible and time is counted in model steps. ``wall``: real time, in
seconds, for real-time and remote environments.
"""

TeacherPlacement = Literal["coordinator", "environment", "participant"]
"""Where the teacher sat during an attempt.

``coordinator``: the coordinating code consulted it between the model and
the environment. ``environment``: the environment owned it and reported its
responses in its transition results. ``participant``: it controlled a
participant through the ordinary agent interface.
"""

LateOutput = Literal["discarded", "used"]
"""What happened to an output that arrived after its deadline.

``discarded``: thrown away, and the agent waited for the answer to its
latest observation. ``used``: used as the agent's action at its next turn.
"""

RecordStatus = Literal["terminated", "truncated", "failed"]
"""How an attempt ended.

``terminated``: the episode reached a terminal state (won, lost, solved).
``truncated``: it was cut off before one, for example by a step limit.
``failed``: it could not continue, for example because a remote environment
could not be reached.
"""

_CLOCKS: tuple[Clock, ...] = ("simulated", "wall")
_PLACEMENTS: tuple[TeacherPlacement, ...] = ("coordinator", "environment", "participant")
_LATE: tuple[LateOutput, ...] = ("discarded", "used")
_STATUSES: tuple[RecordStatus, ...] = ("terminated", "truncated", "failed")


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordHeader:
    """What is fixed when an attempt starts.

    Attributes:
        id: Identifies the record, so results and storage can refer to it.
            The coordinating code chooses it, for example the episode id.
        instance_id: The id of the instance attempted. The record refers to
            the instance rather than holding it, so the instance's
            reference never travels with its records.
        participants: The agent ids taking part, including any participant
            a teacher controls.
        reset: The environment's ``ResetResult``, holding each participant's
            initial observation. ``None`` for static solving, which has no
            environment.
        provenance: Optional. Where the instance came from.
        seeds: The seeds the coordinating code used, keyed by name, such as
            ``{"reset": 7, "model": 3}``.
        clock: The clock the attempt was timed with, ``"simulated"`` or
            ``"wall"``. ``None`` when no timing was applied, as in a record
            built by hand or a teacher's demonstration.
        teacher_placement: Where the teacher sat: ``"coordinator"``,
            ``"environment"``, or ``"participant"``. ``None`` when there was
            no teacher.
        metadata: Downstream fields, keyed by name, passed through
            uninterpreted.
        schema_version: The version of the record structure. Defaults to
            the one this SSAPling writes.
    """

    id: str
    instance_id: str
    participants: tuple[str, ...]
    reset: ResetResult | None = None
    provenance: Provenance | None = None
    seeds: Mapping[str, int] = field(default_factory=dict)
    clock: Clock | None = None
    teacher_placement: TeacherPlacement | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        check_identifier(self.id, "a record id")
        check_identifier(self.instance_id, "an instance id")
        participants = _agent_ids(self.participants, "participants")
        if not participants:
            raise ValueError("participants must name at least one agent id")
        set_(self, "participants", participants)

        if self.reset is not None and not isinstance(self.reset, ResetResult):
            raise TypeError(f"reset must be a ResetResult or None, got {self.reset!r}")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise TypeError(f"provenance must be a Provenance or None, got {self.provenance!r}")
        seeds = keyed(self.seeds, "seeds", "names")
        for name, seed in seeds.items():
            check_int(seed, f"seeds[{name!r}]")
        set_(self, "seeds", seeds)
        set_(self, "metadata", keyed(self.metadata, "metadata", "names"))

        _check_choice(self.clock, _CLOCKS, "clock")
        _check_choice(self.teacher_placement, _PLACEMENTS, "teacher_placement")
        if self.static and self.teacher_placement == "environment":
            raise ValueError(
                "teacher_placement is 'environment', but static solving (no reset) has no "
                "environment to hold the teacher"
            )

        check_int(self.schema_version, "schema_version", minimum=1)
        if self.schema_version > RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"record {self.id!r} has schema version {self.schema_version}, but this "
                f"version of SSAPling reads up to {RECORD_SCHEMA_VERSION}; upgrade SSAPling "
                "to read it"
            )

    @property
    def static(self) -> bool:
        """Whether the attempt is static solving: no environment, so no reset."""
        return self.reset is None

    @property
    def initial_observations(self) -> Mapping[str, Any]:
        """Each participant's initial observation, keyed by agent id; empty for static solving."""
        return self.reset.observations if self.reset is not None else {}


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTiming:
    """How long one agent took to act at one step, and what timing changed.

    Attributes:
        elapsed: From perceiving the observation to the output: calls to
            the model's ``advance`` on the simulated clock, seconds on the
            wall clock. The whole time allowed, when no output came in time.
        defaulted: No output came in time. The agent's default action was
            executed or, without one, it was left out of the step, so the
            environment applied its own. In static solving: no prediction.
        late: An output that answered an earlier observation arrived at
            this step: ``"discarded"`` or ``"used"`` (see ``LateOutput``).
            ``None`` when there was none.
    """

    elapsed: float
    defaulted: bool = False
    late: LateOutput | None = None

    def __post_init__(self) -> None:
        elapsed = self.elapsed
        if isinstance(elapsed, bool) or not isinstance(elapsed, int | float) or not elapsed >= 0:
            raise ValueError(f"elapsed must be a number, 0 or more, got {elapsed!r}")
        if not isinstance(self.defaulted, bool):
            raise TypeError(f"defaulted must be True or False, got {self.defaulted!r}")
        _check_choice(self.late, _LATE, "late")


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordStep:
    """One step of an attempt: one environment step, or one presentation in static solving.

    Attributes:
        index: The step's position in the record, counting from 0.
        acting: The agent ids that acted at this step.
        proposals: The actions the acting agents proposed, keyed by agent id.
            Interactive steps only.
        teacher_responses: The teacher's responses made during this step.
        executed_actions: The actions actually executed, keyed by agent id.
            Interactive steps only.
        transition: The environment's ``StepResult``. ``None`` for static
            solving, or when the step did not complete, for example because
            the environment could not be reached.
        presented: What each agent was shown, keyed by agent id. Static
            solving only: there, the presentation is the observation.
        predictions: What each agent predicted, keyed by agent id. Static
            solving only.
        timing: How long each acting agent took, and what timing changed,
            keyed by agent id. Empty when no timing was applied.
        diagnostics: What each acting agent's model reported about its
            decision, such as the tokens or compute it used, keyed by agent
            id and passed through uninterpreted. The environment's own
            diagnostics are in ``transition.info``.
        metadata: Downstream fields, keyed by name, passed through
            uninterpreted.

    ``proposals``, ``presented``, ``predictions``, ``timing`` and
    ``diagnostics`` may only name agents in ``acting``.

    ``executed_actions`` and ``teacher_responses`` are the same whatever the
    teacher's placement, so code that reads records never needs to know
    where the teacher sat. When the environment reports them in its
    transition result, as it does when it owns the teacher, the coordinating
    code passes the environment's report on here unchanged.
    """

    index: int
    acting: tuple[str, ...]
    proposals: Mapping[str, Any] = field(default_factory=dict)
    teacher_responses: Sequence[TeacherResponse] = ()
    executed_actions: Mapping[str, Any] | None = None
    transition: StepResult | None = None
    presented: Mapping[str, Any] | None = None
    predictions: Mapping[str, Any] | None = None
    timing: Mapping[str, AgentTiming] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        check_int(self.index, "index", minimum=0)
        set_(self, "acting", _agent_ids(self.acting, "acting"))
        if self.transition is not None and not isinstance(self.transition, StepResult):
            raise TypeError(f"transition must be a StepResult or None, got {self.transition!r}")

        set_(self, "proposals", self._by_acting(self.proposals, "proposals"))
        for name in ("presented", "predictions"):
            value = getattr(self, name)
            if value is not None:
                set_(self, name, self._by_acting(value, name))
        if self.executed_actions is not None:
            # Not checked against acting: they may be the environment's report.
            set_(self, "executed_actions", _per_agent(self.executed_actions, "executed_actions"))
        set_(self, "teacher_responses", _responses(self.teacher_responses))
        set_(self, "timing", self._by_acting(self.timing, "timing"))
        if any(not isinstance(timing, AgentTiming) for timing in self.timing.values()):
            raise TypeError("timing must hold only AgentTiming objects")
        set_(self, "diagnostics", self._by_acting(self.diagnostics, "diagnostics"))
        set_(self, "metadata", keyed(self.metadata, "metadata", "names"))

    def _by_acting(self, value: object, what: str) -> dict[str, Any]:
        """``value`` keyed by agent id, naming only agents that acted at this step."""
        per_agent = _per_agent(value, what)
        others = [agent for agent in per_agent if agent not in self.acting]
        if others:
            raise ValueError(
                f"step {self.index}: {what} names {_listed(others)}, which did not act "
                f"(acting: {_listed(self.acting) or 'none'})"
            )
        return per_agent


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordSummary:
    """How an attempt ended.

    Attributes:
        status: ``"terminated"``, ``"truncated"``, or ``"failed"`` (see
            ``RecordStatus``).
        detail: Optional. Why: for example which limit truncated the
            attempt, or the error that made it fail.
        metadata: Downstream fields, keyed by name, such as counts kept as a
            shortcut for filtering stored records.

    Build one with ``RecordSummary.terminated()``, ``.truncated()``, or
    ``.failed(reason)``.
    """

    status: RecordStatus
    detail: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            allowed = ", ".join(repr(status) for status in _STATUSES)
            raise ValueError(f"status must be one of {allowed}; got {self.status!r}")
        object.__setattr__(self, "metadata", keyed(self.metadata, "metadata", "names"))

    @classmethod
    def terminated(
        cls, detail: Any = None, *, metadata: Mapping[str, Any] | None = None
    ) -> "RecordSummary":
        """The episode reached a terminal state."""
        return cls(status="terminated", detail=detail, metadata=metadata or {})

    @classmethod
    def truncated(
        cls, detail: Any = None, *, metadata: Mapping[str, Any] | None = None
    ) -> "RecordSummary":
        """The episode was cut off before a terminal state; ``detail`` may say by what."""
        return cls(status="truncated", detail=detail, metadata=metadata or {})

    @classmethod
    def failed(
        cls, reason: Any = None, *, metadata: Mapping[str, Any] | None = None
    ) -> "RecordSummary":
        """The attempt could not continue; ``reason`` says why and is kept in ``detail``."""
        return cls(status="failed", detail=reason, metadata=metadata or {})


@dataclass(frozen=True, slots=True, kw_only=True)
class Record:
    """The log of one attempt at one instance.

    Attributes:
        header: What was fixed when the attempt started.
        steps: The steps so far, in order.
        summary: How the attempt ended; ``None`` while it is under way or
            when it was interrupted. A record is valid without one.

    A record is checked as a whole when it is made, by the same rules
    ``RecordBuilder`` applies as it goes (see ``RecordBuilder.append``).
    """

    header: RecordHeader
    steps: tuple[RecordStep, ...] = ()
    summary: RecordSummary | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.header, RecordHeader):
            raise TypeError(f"header must be a RecordHeader, got {self.header!r}")
        if isinstance(self.steps, Mapping) or not isinstance(self.steps, Sequence):
            raise TypeError(f"steps must be a sequence of RecordStep, got {self.steps!r}")
        steps = tuple(self.steps)
        for position, step in enumerate(steps):
            _check_step(self.header, step, position, steps[position - 1] if position else None)
        object.__setattr__(self, "steps", steps)
        if self.summary is not None and not isinstance(self.summary, RecordSummary):
            raise TypeError(f"summary must be a RecordSummary or None, got {self.summary!r}")

    @property
    def finished(self) -> bool:
        """Whether the record has a summary."""
        return self.summary is not None

    @classmethod
    def _from_checked(
        cls, header: RecordHeader, steps: tuple[RecordStep, ...], summary: RecordSummary | None
    ) -> "Record":
        """A record from parts the builder has already checked, without checking them again."""
        record = object.__new__(cls)
        object.__setattr__(record, "header", header)
        object.__setattr__(record, "steps", steps)
        object.__setattr__(record, "summary", summary)
        return record


class RecordBuilder:
    """Builds a record step by step.

    Example::

        builder = RecordBuilder(header)
        builder.append(RecordStep(index=0, acting=("student",),
                                  proposals={"student": "2"}, transition=result))
        builder.record        # a Record so far: valid, with no summary yet
        builder.finish(RecordSummary.terminated())

    Each step is checked as it is appended, so a mistake is reported where
    it is made. ``record`` can be read at any time, for example to stream
    the attempt or to learn from it online.
    """

    def __init__(self, header: RecordHeader) -> None:
        if not isinstance(header, RecordHeader):
            raise TypeError(f"header must be a RecordHeader, got {header!r}")
        self._header = header
        self._steps: list[RecordStep] = []
        self._summary: RecordSummary | None = None
        self._record: Record | None = None

    @property
    def header(self) -> RecordHeader:
        """The record's header."""
        return self._header

    @property
    def next_index(self) -> int:
        """The index the next appended step must have."""
        return len(self._steps)

    @property
    def finished(self) -> bool:
        """Whether ``finish`` has been called."""
        return self._summary is not None

    @property
    def record(self) -> Record:
        """The record so far. Its ``summary`` is ``None`` until ``finish`` is called."""
        if self._record is None:
            self._record = Record._from_checked(self._header, tuple(self._steps), self._summary)
        return self._record

    def append(self, step: RecordStep) -> None:
        """Add the next step.

        Raises:
            TypeError: ``step`` is not a ``RecordStep``.
            ValueError: The record is finished; the step's index is not
                ``next_index``; it names an agent that is not a participant;
                it does not fit the record's kind (a static record's steps
                have presentations and predictions, an interactive record's
                have proposals, executed actions and transitions); or the
                environment ended the episode at the previous step.
        """
        if self.finished:
            raise ValueError(
                f"record {self._header.id!r} is finished; a finished record takes no more steps"
            )
        _check_step(self._header, step, self.next_index, self._steps[-1] if self._steps else None)
        self._steps.append(step)
        self._record = None

    def finish(self, summary: RecordSummary) -> Record:
        """Say how the attempt ended, and return the finished record.

        Raises:
            TypeError: ``summary`` is not a ``RecordSummary``.
            ValueError: The record is already finished.
        """
        if not isinstance(summary, RecordSummary):
            raise TypeError(f"summary must be a RecordSummary, got {summary!r}")
        if self.finished:
            raise ValueError(f"record {self._header.id!r} is already finished")
        self._summary = summary
        self._record = None
        return self.record


def _check_step(
    header: RecordHeader, step: object, position: int, previous: RecordStep | None
) -> None:
    """Check that ``step`` fits ``header`` as the step at ``position``, after ``previous``."""
    if not isinstance(step, RecordStep):
        raise TypeError(f"a record's steps must be RecordStep objects, got {step!r}")
    if step.index != position:
        raise ValueError(f"step index {step.index} is out of order: the next step is {position}")
    at = f"step {step.index}"
    strangers = [agent for agent in step.acting if agent not in header.participants]
    if strangers:
        raise ValueError(
            f"{at}: {_listed(strangers)} acted but not among the participants "
            f"({_listed(header.participants)})"
        )
    if (
        previous is not None
        and previous.transition is not None
        and previous.transition.episode_over
    ):
        raise ValueError(
            f"{at}: the environment ended the episode at step {previous.index}; a record "
            "holds one attempt, so start a new record for the next episode"
        )

    if header.static:
        mixed = [
            name
            for name, present in (
                ("proposals", bool(step.proposals)),
                ("executed_actions", step.executed_actions is not None),
                ("transition", step.transition is not None),
            )
            if present
        ]
        if mixed:
            raise ValueError(
                f"{at} has {', '.join(mixed)}, but record {header.id!r} is static solving (its "
                "header has no reset): its steps hold presentations and predictions"
            )
    else:
        mixed = [name for name in ("presented", "predictions") if getattr(step, name) is not None]
        if mixed:
            raise ValueError(
                f"{at} has {', '.join(mixed)}, but record {header.id!r} is interactive (its "
                "header has a reset): its steps hold proposals, executed actions and "
                "transitions"
            )


def _per_agent(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{what} must be keyed by agent id, even with one agent, such as "
            f"{{'agent': ...}}, got {value!r}"
        )
    return keyed(value, what, "agent ids")


def _agent_ids(value: object, what: str) -> tuple[str, ...]:
    ids = names(value, what, of="agent ids", example="('student',)")
    repeated = sorted({agent for agent in ids if ids.count(agent) > 1})
    if repeated:
        raise ValueError(f"{what} names {_listed(repeated)} more than once")
    return ids


def _responses(value: object) -> tuple[TeacherResponse, ...]:
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        raise TypeError(f"teacher_responses must be a sequence of TeacherResponse, got {value!r}")
    if any(not isinstance(response, TeacherResponse) for response in value):
        raise TypeError("teacher_responses must contain only TeacherResponse objects")
    return tuple(value)


def _check_choice(value: object, allowed: tuple[str, ...], what: str) -> None:
    if value is not None and value not in allowed:
        choices = ", ".join(repr(choice) for choice in allowed)
        raise ValueError(f"{what} must be one of {choices}, or None; got {value!r}")


def _listed(agents: Sequence[str]) -> str:
    return ", ".join(repr(agent) for agent in agents)
