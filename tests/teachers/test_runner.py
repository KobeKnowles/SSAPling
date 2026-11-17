"""The teacher runner: sync and async teachers, its checks, and what becomes of a bad answer."""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from ssapling.teachers import (
    MissingContextError,
    TeacherRequest,
    TeacherResponse,
    TeacherRunner,
    UnsupportedOperationError,
)
from tests.fixtures.tasks import AdditionTask
from tests.fixtures.teachers import (
    AsyncExplainer,
    DigitTutor,
    FailingTeacher,
    NotAResponseTeacher,
    SelfCancellingTeacher,
    SlowTeacher,
    WrongIdTeacher,
)

TASK = AdditionTask()
INSTANCE = TASK.generate(params={"digits": 2}, seed=42)
UNITS = str(INSTANCE.reference)[-1]  # the first digit the student enters


def review(proposal: str, state: str = "", request_id: str = "review-1") -> TeacherRequest:
    return TeacherRequest(
        id=request_id,
        operation="review_action",
        context={"instance": INSTANCE, "state": state},
        proposals={"student": proposal},
        participant="student",
        step=len(state),
    )


EXPLAIN = TeacherRequest(id="explain-1", operation="explain")


def ask(teacher: Any, request: TeacherRequest, runner: TeacherRunner | None = None, **options):
    return asyncio.run((runner or TeacherRunner()).request(teacher, request, **options))


class Spy:
    """Answers every review, and counts how often it is asked."""

    operations = frozenset({"review_action"})
    required_context = {"review_action": ("instance", "state")}  # noqa: RUF012

    def __init__(self) -> None:
        self.calls = 0

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        self.calls += 1
        return TeacherResponse.answered(request)


# Synchronous and asynchronous teachers


@pytest.mark.parametrize("placement", ["direct", "thread"])
def test_a_synchronous_teacher_answers(placement):
    tutor = DigitTutor(task=TASK)
    right = ask(tutor, review(UNITS), TeacherRunner(placement=placement))
    assert right == TeacherResponse.answered(review(UNITS), {"feedback": {"student": "Correct."}})

    wrong = str((int(UNITS) + 1) % 10)
    answer = ask(tutor, review(wrong), TeacherRunner(placement=placement))
    assert answer.status == "answered"
    assert answer.contributions["supervision"] == {"student": UNITS}
    assert answer.contributions["feedback"]["student"].startswith("Not quite. Add the units")
    assert "intervention" not in answer.contributions


@pytest.mark.parametrize("placement", ["direct", "thread"])
def test_an_asynchronous_teacher_answers(placement):
    answer = ask(AsyncExplainer(), EXPLAIN, TeacherRunner(placement=placement))
    assert answer.status == "answered"
    assert answer.request_id == "explain-1"
    assert answer.contributions == {
        "explanation": {"student": "Add the units first, then the tens."}
    }


def test_thread_placement_runs_a_synchronous_teacher_off_the_loop_thread():
    class WhereAmI:
        operations = frozenset({"explain"})

        def respond(self, request: TeacherRequest) -> TeacherResponse:
            return TeacherResponse.answered(request, detail=threading.get_ident())

    async def main(runner: TeacherRunner) -> tuple[Any, int]:
        answer = await runner.request(WhereAmI(), EXPLAIN)
        return answer.detail, threading.get_ident()

    called_in, loop_thread = asyncio.run(main(TeacherRunner()))
    assert called_in == loop_thread
    with ThreadPoolExecutor(max_workers=1) as pool:
        runner = TeacherRunner(placement="thread", executor=pool)
        assert runner.placement == "thread"
        called_in, loop_thread = asyncio.run(main(runner))
    assert called_in != loop_thread


def test_an_abstention_is_passed_on():
    complete = str(INSTANCE.reference)[::-1]
    answer = ask(DigitTutor(task=TASK), review("1", state=complete))
    assert answer == TeacherResponse.abstained(
        review("1", state=complete), "the answer is already complete"
    )


def test_a_delayed_answer_reads_the_state_it_was_asked_about():
    class SlowReader:
        operations = frozenset({"review_action"})

        async def respond(self, request: TeacherRequest) -> TeacherResponse:
            await asyncio.sleep(0.01)
            return TeacherResponse.answered(request, detail=request.context["state"])

    async def main() -> TeacherResponse:
        context = {"instance": INSTANCE, "state": ""}
        request = TeacherRequest(id="review-1", operation="review_action", context=context)
        pending = asyncio.create_task(TeacherRunner().request(SlowReader(), request))
        await asyncio.sleep(0)  # the teacher starts thinking
        context["state"] = "999"  # and the episode moves on
        return await pending

    assert asyncio.run(main()).detail == ""


# Checks before asking


def test_an_unsupported_operation_is_refused_before_the_teacher_is_asked():
    spy = Spy()
    with pytest.raises(UnsupportedOperationError) as caught:
        ask(spy, EXPLAIN)
    assert str(caught.value) == (
        "Teacher Spy does not support operation 'explain'; it supports: 'review_action'."
    )
    assert spy.calls == 0


def test_missing_context_is_refused_before_the_teacher_is_asked():
    spy = Spy()
    request = TeacherRequest(
        id="review-1", operation="review_action", context={"instance": INSTANCE}
    )
    with pytest.raises(MissingContextError) as caught:
        ask(spy, request)
    assert str(caught.value) == (
        "Teacher Spy needs context 'state' for operation 'review_action'; "
        "the request gives: 'instance'."
    )
    assert (caught.value.missing, caught.value.given) == (("state",), ("instance",))
    assert spy.calls == 0


def test_context_beyond_what_is_required_is_allowed():
    request = review(UNITS)
    extended = TeacherRequest(
        id=request.id,
        operation=request.operation,
        context={**request.context, "observation": "..."},
        proposals=request.proposals,
    )
    assert ask(DigitTutor(task=TASK), extended).status == "answered"


def test_required_context_may_only_name_listed_operations():
    class Typo:
        operations = frozenset({"review_action"})
        required_context = {"reveiw_action": ("state",)}  # noqa: RUF012

        def respond(self, request: TeacherRequest) -> TeacherResponse:
            raise NotImplementedError

    with pytest.raises(ValueError, match=r"Typo.required_context names operations missing from"):
        ask(Typo(), review(UNITS))


def test_only_teachers_and_requests_are_accepted():
    with pytest.raises(TypeError, match="request must be a TeacherRequest"):
        ask(Spy(), {"id": "review-1"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AdditionTask has no respond method"):
        ask(TASK, review(UNITS))


# What becomes of a bad answer


def test_an_error_in_the_teacher_becomes_a_failed_response(caplog):
    with caplog.at_level(logging.WARNING, logger="ssapling.teachers"):
        answer = ask(FailingTeacher(RuntimeError("boom")), review(UNITS))
    assert answer == TeacherResponse.failed(review(UNITS), "RuntimeError: boom")
    (record,) = caplog.records
    assert "Teacher FailingTeacher raised an error on request 'review-1'" in record.message
    assert record.exc_info is not None  # the traceback is logged, not lost


def test_a_response_to_another_request_becomes_a_failed_response():
    answer = ask(WrongIdTeacher(), review(UNITS))
    assert answer.status == "failed"
    assert answer.request_id == "review-1"
    assert answer.detail == "the response is for request 'review-1-old', not 'review-1'"


def test_something_other_than_a_response_becomes_a_failed_response():
    answer = ask(NotAResponseTeacher(), review(UNITS))
    assert answer.status == "failed"
    assert answer.detail == "respond returned dict, not a TeacherResponse"


@pytest.mark.parametrize("placement", ["direct", "thread"])
def test_no_answer_in_time_becomes_a_cancelled_response(placement):
    started = time.perf_counter()
    answer = ask(
        AsyncExplainer(delay=1.0), EXPLAIN, TeacherRunner(placement=placement), timeout=0.02
    )
    assert answer == TeacherResponse.cancelled(EXPLAIN, "no response within 0.02 s")
    assert time.perf_counter() - started < 0.5


def test_a_slow_synchronous_teacher_in_a_thread_can_be_timed_out():
    async def main() -> tuple[TeacherResponse, float]:
        started = time.perf_counter()
        runner = TeacherRunner(placement="thread")
        answer = await runner.request(SlowTeacher(delay=0.3), review(UNITS), timeout=0.02)
        return answer, time.perf_counter() - started

    answer, waited = asyncio.run(main())
    assert answer.status == "cancelled"
    assert waited < 0.25  # stopped waiting; the thread finishes on its own


def test_a_timeout_cannot_interrupt_a_synchronous_teacher_run_directly():
    with pytest.raises(ValueError, match=r"cannot interrupt .*\. Use placement='thread'"):
        ask(SlowTeacher(delay=0.0), review(UNITS), timeout=1.0)


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), True])
def test_a_timeout_must_be_a_positive_number(timeout):
    with pytest.raises(ValueError, match="timeout must be a positive number of seconds"):
        ask(AsyncExplainer(), EXPLAIN, timeout=timeout)


def test_the_teachers_own_timeout_is_a_failure_not_a_cancellation():
    error = TimeoutError("the service did not answer")
    answer = ask(
        FailingTeacher(error), review(UNITS), timeout=5.0, runner=TeacherRunner(placement="thread")
    )
    assert answer == TeacherResponse.failed(
        review(UNITS), "TimeoutError: the service did not answer"
    )


def test_a_teacher_cancelling_the_request_becomes_a_cancelled_response():
    answer = ask(SelfCancellingTeacher(), review(UNITS))
    assert answer == TeacherResponse.cancelled(review(UNITS), "the teacher cancelled the request")


@pytest.mark.parametrize(
    "teacher",
    [
        FailingTeacher(RuntimeError("boom")),
        FailingTeacher(TimeoutError("its own timeout")),
        WrongIdTeacher(),
        NotAResponseTeacher(),
        SelfCancellingTeacher(),
        AsyncExplainer(delay=1.0),
    ],
    ids=lambda teacher: type(teacher).__name__,
)
def test_no_failure_is_ever_an_answer(teacher):
    """Failed and cancelled responses carry no contributions, so nothing can be applied."""
    operation = next(iter(teacher.operations))
    request = review(UNITS) if operation == "review_action" else EXPLAIN
    answer = ask(teacher, request, runner=TeacherRunner(placement="thread"), timeout=0.05)
    assert answer.status in ("failed", "cancelled")
    assert answer.contributions == {}
    assert answer.request_id == request.id


# Cancelling the caller


def test_cancelling_the_caller_cancels_the_request_as_usual():
    async def main() -> None:
        pending = asyncio.create_task(TeacherRunner().request(AsyncExplainer(delay=1.0), EXPLAIN))
        await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(main())


def test_the_callers_own_timeout_raises_as_usual():
    async def main() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                TeacherRunner().request(AsyncExplainer(delay=1.0), EXPLAIN), timeout=0.01
            )

    asyncio.run(main())


def test_a_failing_task_group_cancels_a_pending_request():
    asking: list[asyncio.Task[TeacherResponse]] = []

    async def fail_soon() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("another part of the run failed")

    async def run_together() -> None:
        async with asyncio.TaskGroup() as group:
            request = TeacherRunner().request(AsyncExplainer(delay=5.0), EXPLAIN)
            asking.append(group.create_task(request))
            group.create_task(fail_soon())

    async def main() -> None:
        with pytest.raises(ExceptionGroup):
            await run_together()

    started = time.perf_counter()
    asyncio.run(main())
    assert asking[0].cancelled()  # cancelled, not quietly finished with a response
    assert time.perf_counter() - started < 1.0
