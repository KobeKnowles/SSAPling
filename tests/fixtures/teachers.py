"""Toy teachers for tests. They exercise the teacher contract and are not part of SSAPling.

Everything here is a plain module-level class, so the fixtures can be
registered by name and rebuilt in a worker process. ``DigitTutor`` and
``AsyncExplainer`` behave; the rest misbehave in one way each.
"""

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ssapling.core import setting
from ssapling.tasks import Instance, Task, teaching_resource
from ssapling.teachers import TeacherRequest, TeacherResponse


@dataclass(frozen=True)
class DigitTutorSettings:
    intervene: bool = setting(False, description="Also replace a wrong digit with the right one.")


class DigitTutor:
    """Reviews each digit of an addition answer as the student enters it, units first."""

    Settings = DigitTutorSettings
    operations = frozenset({"review_action"})
    required_context: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "review_action": ("instance", "state")
    }
    required_resources = ("procedure",)

    def __init__(self, config: DigitTutorSettings | None = None, *, task: Task) -> None:
        self.config = config or DigitTutorSettings()
        self.procedure = teaching_resource(task, "procedure")
        self.reviews = 0

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        instance: Instance = request.context["instance"]
        state: str = request.context["state"]  # the digits entered so far, units first
        expected = str(instance.reference)[::-1]
        if len(state) >= len(expected):
            return TeacherResponse.abstained(request, "the answer is already complete")
        self.reviews += 1
        right = expected[len(state)]
        if request.proposals.get("student") == right:
            return TeacherResponse.answered(request, {"feedback": {"student": "Correct."}})
        contributions = {
            "supervision": {"student": right},
            "feedback": {"student": f"Not quite. {self.procedure}"},
        }
        if self.config.intervene:
            contributions["intervention"] = {"student": right}
        return TeacherResponse.answered(request, contributions)

    def reset(self) -> None:
        self.reviews = 0

    def export_state(self) -> dict[str, int]:
        return {"reviews": self.reviews}

    def restore_state(self, state: dict[str, int]) -> None:
        self.reviews = state["reviews"]


class AsyncExplainer:
    """Explains how to add, asynchronously, after a delay, like a call to a remote model."""

    operations = frozenset({"explain"})

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    async def respond(self, request: TeacherRequest) -> TeacherResponse:
        await asyncio.sleep(self.delay)
        return TeacherResponse.answered(
            request, {"explanation": {"student": "Add the units first, then the tens."}}
        )


# Teachers that misbehave


class FailingTeacher:
    """Raises the error it was given, as a teacher with a bug or a broken connection would."""

    operations = frozenset({"review_action"})

    def __init__(self, error: Exception) -> None:
        self.error = error

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        raise self.error


class WrongIdTeacher:
    """Answers a different request than the one it was asked."""

    operations = frozenset({"review_action"})

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        return TeacherResponse(
            request_id=f"{request.id}-old",
            status="answered",
            contributions={"feedback": {"student": "Looks fine."}},
        )


class NotAResponseTeacher:
    """Returns a plain dict instead of a TeacherResponse."""

    operations = frozenset({"review_action"})

    def respond(self, request: TeacherRequest) -> Any:
        return {"feedback": {"student": "Looks fine."}}


class SlowTeacher:
    """A synchronous teacher that takes ``delay`` seconds to answer."""

    operations = frozenset({"review_action"})

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        time.sleep(self.delay)
        return TeacherResponse.answered(request, {"feedback": {"student": "Looks fine."}})


class SelfCancellingTeacher:
    """Cancels the request itself, as a human closing the dialog would."""

    operations = frozenset({"review_action"})

    async def respond(self, request: TeacherRequest) -> TeacherResponse:
        raise asyncio.CancelledError


class AsyncTeacher:
    """Makes any synchronous teacher asynchronous, as a remote one would be."""

    def __init__(self, teacher: Any) -> None:
        self.teacher = teacher

    async def respond(self, request: TeacherRequest) -> TeacherResponse:
        response: TeacherResponse = self.teacher.respond(request)
        return response

    def __getattr__(self, name: str) -> Any:
        if name == "teacher" or name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.teacher, name)
