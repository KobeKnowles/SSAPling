"""The teacher runner: one awaitable entry point for synchronous and asynchronous teachers."""

import asyncio
import logging
from concurrent.futures import Executor

from ..core._declarations import name_of
from ..core._execution import DeadlineExpiredError, Execution, Placement
from .contracts import Teacher, _operations, _required_context, supports
from .errors import MissingContextError, UnsupportedOperationError
from .types import TeacherRequest, TeacherResponse

__all__ = ["TeacherRunner"]

_log = logging.getLogger("ssapling.teachers")


class TeacherRunner:
    """Sends requests to teachers, whether they respond with ``def`` or ``async def``.

    Example::

        runner = TeacherRunner()                     # sync teachers run in the loop's thread
        runner = TeacherRunner(placement="thread")   # sync teachers run in a worker thread
        response = await runner.request(teacher, request, timeout=2.0)

    The runner checks that a request can be asked of a teacher, and that the
    response answers it. It does not judge whether the advice is correct.

    Args:
        placement: Where a synchronous ``respond`` runs. ``"direct"`` runs it
            in the event loop's thread, blocking the loop until it returns.
            ``"thread"`` runs it in a worker thread, so the loop keeps going
            and a timeout can stop waiting for it. An ``async def`` respond
            always runs in the loop. To run a teacher in a separate process,
            build it with ``ssapling.interaction.in_process``.
        executor: For ``"thread"``: the thread pool to use. Pass a
            ``ThreadPoolExecutor(max_workers=1)`` for a teacher that must
            always be called from the same thread.
    """

    def __init__(
        self, *, placement: Placement = "direct", executor: Executor | None = None
    ) -> None:
        self._execution = Execution(placement, executor)

    @property
    def placement(self) -> Placement:
        """Where a synchronous ``respond`` runs: ``"direct"`` or ``"thread"``."""
        return self._execution.placement

    async def request(
        self, teacher: Teacher, request: TeacherRequest, *, timeout: float | None = None
    ) -> TeacherResponse:
        """Ask ``teacher`` and return its response, or a failed or cancelled one.

        Before asking, the runner checks that the teacher supports the
        request's operation and that the request gives every context name the
        teacher requires for it. A failed check is a mistake in the calling
        code, so it raises and the teacher is not asked.

        Whatever goes wrong with the teacher itself becomes a response, so it
        can be recorded:

        - ``respond`` raises: ``failed``, with the error as the reason. The
          error is also logged, with its traceback, to ``ssapling.teachers``.
        - ``respond`` returns something other than a ``TeacherResponse``, or
          a response to a different request: ``failed``.
        - No response within ``timeout`` seconds: ``cancelled``.
        - The teacher cancels the request itself, for example a human closing
          the dialog: ``cancelled``.

        None of these has contributions, so none can be taken for approval.

        Cancelling the task that awaits this, for example with
        ``asyncio.wait_for`` or a ``TaskGroup``, raises ``CancelledError``
        here as usual in ``asyncio``: the runner does not swallow it. A caller
        that keeps a record writes ``TeacherResponse.cancelled(request,
        reason)`` itself.

        Raises:
            UnsupportedOperationError: The teacher does not support the
                request's operation.
            MissingContextError: The request lacks context the teacher
                requires for the operation.
            TypeError: ``request`` is not a ``TeacherRequest``, or
                ``teacher`` is not a teacher.
            ValueError: ``timeout`` is not a positive number, or it is given
                for a synchronous teacher run directly, which no timeout can
                interrupt.
        """
        if not isinstance(request, TeacherRequest):
            raise TypeError(f"request must be a TeacherRequest, got {request!r}")
        respond = getattr(teacher, "respond", None)
        if not callable(respond):
            raise TypeError(f"{name_of(teacher)} has no respond method, so it is not a teacher")
        if not supports(teacher, request.operation):
            raise UnsupportedOperationError(
                name_of(teacher), request.operation, _operations(teacher)
            )
        required = _required_context(teacher, request.operation)
        missing = [name for name in required if name not in request.context]
        if missing:
            raise MissingContextError(name_of(teacher), request.operation, missing, request.context)
        if timeout is not None:
            self._execution.check_timeout(timeout, respond)

        try:
            response: object = await self._execution.call_with_timeout(timeout, respond, request)
        except DeadlineExpiredError as expired:
            return TeacherResponse.cancelled(request, str(expired))
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is None or task.cancelling():
                raise  # the caller was cancelled, not the request
            return TeacherResponse.cancelled(request, "the teacher cancelled the request")
        except Exception as error:  # including the teacher's own TimeoutError
            return _failed(teacher, request, error)
        return _checked(teacher, request, response)


def _failed(teacher: object, request: TeacherRequest, error: BaseException) -> TeacherResponse:
    _log.warning(
        "Teacher %s raised an error on request %r (operation %r).",
        name_of(teacher),
        request.id,
        request.operation,
        exc_info=error,
    )
    return TeacherResponse.failed(request, f"{type(error).__name__}: {error}")


def _checked(teacher: object, request: TeacherRequest, response: object) -> TeacherResponse:
    """``response`` if it answers ``request``, otherwise a failed response saying why."""
    if not isinstance(response, TeacherResponse):
        reason = f"respond returned {type(response).__name__}, not a TeacherResponse"
    elif response.request_id != request.id:
        reason = f"the response is for request {response.request_id!r}, not {request.id!r}"
    else:
        return response
    _log.warning("Teacher %s gave an invalid response: %s.", name_of(teacher), reason)
    return TeacherResponse.failed(request, reason)
