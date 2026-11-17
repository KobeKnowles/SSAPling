"""The result types of the Teachers area: a request, a response, and the response's status.

Environments and Data may import this module, and only this one, from
``teachers``: a transition result can report teacher responses, and a record
step holds them. So it holds plain data and no teacher behaviour.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core._declarations import check_identifier, check_int, keyed

__all__ = ["ResponseStatus", "TeacherRequest", "TeacherResponse"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TeacherRequest:
    """A request for a teacher's help, such as a review of a proposed action.

    Attributes:
        id: Identifies the request, so that its response can be matched to it.
        operation: What is asked, such as ``"review_action"``. Each teacher
            lists the operations it supports; the names are downstream choices.
        context: What the teacher may use, keyed by name: for example the
            instance, the current observation, or a partial answer. It need
            not match what the student sees; the coordinating code decides
            what goes in.
        proposals: Proposed actions for the teacher to consider, keyed by
            agent id.
        episode: For live interaction, the episode the request belongs to.
        participant: For live interaction, the agent the request concerns.
        step: For live interaction, the environment step, counting from 0.

    The request keeps its own copy of ``context`` and ``proposals``, made when
    it is created, so a teacher that answers late still reads the state it
    was asked about. Their values are treated as read-only: pass values that
    will not change afterwards, such as a copy of a buffer the environment
    reuses.
    """

    id: str
    operation: str
    context: Mapping[str, Any] = field(default_factory=dict)
    proposals: Mapping[str, Any] = field(default_factory=dict)
    episode: str | None = None
    participant: str | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        check_identifier(self.id, "a request id")
        check_identifier(self.operation, "an operation")
        object.__setattr__(self, "context", keyed(self.context, "context", "names"))
        object.__setattr__(self, "proposals", keyed(self.proposals, "proposals", "agent ids"))
        if self.episode is not None:
            check_identifier(self.episode, "an episode id")
        if self.participant is not None:
            check_identifier(self.participant, "a participant id")
        if self.step is not None:
            check_int(self.step, "step", minimum=0)


ResponseStatus = Literal["answered", "abstained", "failed", "cancelled", "stale"]
"""What became of a request.

``answered``: the teacher answered; its contributions say how.
``abstained``: the teacher chose not to contribute.
``failed``: no answer, because the teacher raised an error or gave an invalid
response. ``cancelled``: no answer, because the request was cancelled first,
for example when no answer came in time. ``stale``: an answer that arrived
after the state it was about had moved on. Its contributions are kept for the
record but are never applied automatically.

Only an answered response is applied automatically. Every other status means
the teacher has not approved anything.
"""

_STATUSES: tuple[ResponseStatus, ...] = ("answered", "abstained", "failed", "cancelled", "stale")
_HAVE_CONTRIBUTIONS: tuple[ResponseStatus, ...] = ("answered", "stale")


@dataclass(frozen=True, slots=True, kw_only=True)
class TeacherResponse:
    """A teacher's response to one request.

    Attributes:
        request_id: The id of the request this responds to.
        status: ``"answered"``, ``"abstained"``, ``"failed"``,
            ``"cancelled"``, or ``"stale"`` (see ``ResponseStatus``).
        contributions: What the teacher contributed, keyed by kind and then by
            agent id, such as ``{"feedback": {"student": "Carry the 1."}}``.
            Kinds are open-ended. Common ones are ``supervision`` (a target,
            not a command), ``feedback`` (not a reward), ``intervention`` (a
            change to an executed action, applied only where supported),
            ``demonstration``, ``explanation``, and ``instances`` (proposed
            practice problems). Only answered and stale responses have
            contributions.
        detail: Optional. For an answer, a teacher-defined extra such as its
            cost; for any other status, the reason.

    Build one with ``TeacherResponse.answered(request, contributions)``,
    ``.abstained(request, reason)``, ``.failed(request, reason)``, or
    ``.cancelled(request, reason)``. These copy the request's id, so the
    response always matches its request.
    """

    request_id: str
    status: ResponseStatus
    contributions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    detail: Any = None

    def __post_init__(self) -> None:
        check_identifier(self.request_id, "a request id")
        if self.status not in _STATUSES:
            allowed = ", ".join(repr(status) for status in _STATUSES)
            raise ValueError(f"status must be one of {allowed}; got {self.status!r}")
        contributions = keyed(self.contributions, "contributions", "kinds")
        for kind, per_agent in contributions.items():
            if not isinstance(per_agent, Mapping):
                raise TypeError(
                    f"contributions[{kind!r}] must map agent ids to payloads, such as "
                    f"{{'student': ...}}, got {per_agent!r}"
                )
            contributions[kind] = keyed(per_agent, f"contributions[{kind!r}]", "agent ids")
        if contributions and self.status not in _HAVE_CONTRIBUTIONS:
            raise ValueError(
                f"a response with status {self.status!r} has no contributions; "
                "put the reason in detail"
            )
        object.__setattr__(self, "contributions", contributions)

    @classmethod
    def answered(
        cls,
        request: TeacherRequest,
        contributions: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        detail: Any = None,
    ) -> "TeacherResponse":
        """The teacher's answer to ``request``. ``contributions`` may be empty."""
        return cls(
            request_id=request.id,
            status="answered",
            contributions=contributions or {},
            detail=detail,
        )

    @classmethod
    def abstained(cls, request: TeacherRequest, reason: Any = None) -> "TeacherResponse":
        """The teacher chose not to contribute to ``request``."""
        return cls(request_id=request.id, status="abstained", detail=reason)

    @classmethod
    def failed(cls, request: TeacherRequest, reason: Any = None) -> "TeacherResponse":
        """The teacher could not answer ``request``."""
        return cls(request_id=request.id, status="failed", detail=reason)

    @classmethod
    def cancelled(cls, request: TeacherRequest, reason: Any = None) -> "TeacherResponse":
        """``request`` was cancelled before an answer arrived."""
        return cls(request_id=request.id, status="cancelled", detail=reason)

    def as_stale(self) -> "TeacherResponse":
        """This answer, marked as having arrived too late to apply.

        The coordinating code calls this when an answer arrives after the
        state it was about has moved on. The contributions are kept, so the
        record shows what the teacher said, but they are never applied
        automatically.

        Raises:
            ValueError: The response is not an answer. Other statuses have
                nothing that could be applied, so they are recorded as they are.
        """
        if self.status != "answered":
            raise ValueError(f"only an answered response can be marked stale, not {self.status!r}")
        return dataclasses.replace(self, status="stale")
