"""Calling a component's methods the same way whether they are synchronous or asynchronous.

Internal: the teacher runner, the environment runner, and the interaction loop
use this so that they are written once, with ``asyncio``, and still drive
synchronous components. Users choose a placement through those areas' public
settings, not by importing this module.

A synchronous call is not made non-blocking by being awaited. Run directly, it
blocks the event loop until it returns, and a timeout cannot interrupt it; run
in a thread, the loop keeps going while it works. Running a component in a
separate process needs a long-lived worker that keeps the component's state:
that is ``ssapling.interaction.in_process``, whose stand-in is called like
any other component.

Timeouts are handled here too, once, so every caller tells its own deadline
apart from a ``TimeoutError`` the component raises itself in the same way.
"""

import asyncio
import contextvars
import functools
import inspect
import sys
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any, Literal, overload

Placement = Literal["direct", "thread"]


class DeadlineExpiredError(Exception):
    """The caller's timeout passed before the call returned.

    Raised instead of ``TimeoutError``, so a ``TimeoutError`` the component
    raises itself is not mistaken for it.

    Attributes:
        timeout: The timeout, in seconds.
    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"no response within {timeout} s")


@dataclass(frozen=True, slots=True)
class Execution:
    """Where a component's synchronous calls run.

    Attributes:
        placement: ``"direct"`` runs a synchronous call in the event loop's
            thread. ``"thread"`` runs it in a worker thread, so the loop is
            not blocked and a timeout can stop waiting for it.
        executor: For ``"thread"``: the thread pool to use. ``None`` uses
            ``asyncio.to_thread``, which may run successive calls on different
            threads. For components that must always be called from the same
            thread, pass a ``ThreadPoolExecutor(max_workers=1)``.

    Asynchronous calls always run in the event loop: there they already
    yield while they wait, so a thread would add nothing.
    """

    placement: Placement = "direct"
    executor: Executor | None = None

    def __post_init__(self) -> None:
        if self.placement not in ("direct", "thread"):
            raise ValueError(f"placement must be 'direct' or 'thread', not {self.placement!r}")
        if self.executor is not None and self.placement != "thread":
            raise ValueError("an executor applies only to placement='thread'")
        if _is_process_pool(self.executor):
            raise TypeError(
                "a process pool would lose the component's state between calls; "
                "to run a component in a process, use ssapling.interaction.in_process"
            )

    @overload
    async def call[R](
        self, function: Callable[..., Awaitable[R]], /, *args: Any, **kwargs: Any
    ) -> R: ...

    @overload
    async def call[R](self, function: Callable[..., R], /, *args: Any, **kwargs: Any) -> R: ...

    async def call(self, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Call ``function`` with the arguments and return its result.

        An ``async def`` function is awaited in the event loop. A synchronous
        function runs according to ``placement``. If a synchronous function
        returns an awaitable (as a wrapper around an async function might),
        that is awaited too.
        """
        if inspect.iscoroutinefunction(function):
            return await function(*args, **kwargs)
        if self.placement == "direct":
            result = function(*args, **kwargs)
        elif self.executor is None:
            result = await asyncio.to_thread(function, *args, **kwargs)
        else:
            # Like asyncio.to_thread, carry context variables into the thread.
            context = contextvars.copy_context()
            call = functools.partial(context.run, function, *args, **kwargs)
            result = await asyncio.get_running_loop().run_in_executor(self.executor, call)
        if inspect.isawaitable(result):
            return await result
        return result

    def check_timeout(self, timeout: float, function: Callable[..., Any]) -> None:
        """Check that ``timeout`` is usable for calls to ``function``.

        Call it before the call, so a mistake in the timeout is reported as
        the caller's, not mistaken for a failure of the component.

        Raises:
            ValueError: ``timeout`` is not a positive number of seconds, or
                ``function`` is synchronous and runs directly, where no
                timeout can interrupt it.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not timeout > 0:
            raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")
        if self.placement == "direct" and not inspect.iscoroutinefunction(function):
            name = getattr(function, "__qualname__", repr(function))
            raise ValueError(
                f"a timeout cannot interrupt {name}: it is synchronous and runs in the event "
                "loop's thread, blocking the loop until it returns. Use placement='thread'."
            )

    async def call_with_timeout(
        self, timeout: float | None, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """``call``, giving up after ``timeout`` seconds; ``None`` waits as long as it takes.

        Raises:
            DeadlineExpiredError: ``timeout`` passed first. A ``TimeoutError`` the
                function raises itself propagates unchanged.
            ValueError: An unusable timeout (see ``check_timeout``).
        """
        if timeout is not None:
            self.check_timeout(timeout, function)
        try:
            async with asyncio.timeout(timeout) as deadline:
                return await self.call(function, *args, **kwargs)
        except TimeoutError:
            if deadline.expired() and timeout is not None:
                raise DeadlineExpiredError(timeout) from None
            raise


def _is_process_pool(executor: Executor | None) -> bool:
    # Importing ProcessPoolExecutor loads multiprocessing, which importing
    # ssapling should not do. A process pool exists only if its module was
    # imported, so look the class up there instead.
    process = sys.modules.get("concurrent.futures.process")
    return process is not None and isinstance(executor, process.ProcessPoolExecutor)
