"""The environment runner: one awaitable entry point for sync and async environments."""

from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from typing import Any

from ..core._declarations import name_of, takes
from ..core._execution import DeadlineExpiredError, Execution, Placement
from .contracts import supports
from .errors import EnvironmentCommunicationError, UnsupportedCapabilityError
from .types import ResetResult, StepResult

__all__ = ["EnvironmentRunner"]


class EnvironmentRunner:
    """Resets, steps, and closes environments, whether written with ``def`` or ``async def``.

    Example::

        runner = EnvironmentRunner()                    # sync envs run in the loop's thread
        runner = EnvironmentRunner(placement="thread")  # sync envs run in a worker thread
        first = await runner.reset(env, seed=7, options=instance.inputs)
        result = await runner.step(env, {"student": action}, timeout=5.0)

    The runner checks that each call is possible and that each result has the
    right type. It never retries a call: a step that may already have
    executed must not be submitted twice.

    Args:
        placement: Where a synchronous environment's methods run.
            ``"direct"`` runs them in the event loop's thread, blocking the
            loop until they return. ``"thread"`` runs them in a worker thread,
            so the loop keeps going and a timeout can stop waiting. An
            ``async def`` method always runs in the loop. To run an
            environment in a separate process, build it with
            ``ssapling.interaction.in_process``.
        executor: For ``"thread"``: the thread pool to use. Pass a
            ``ThreadPoolExecutor(max_workers=1)`` for an environment that must
            always be called from the same thread.
    """

    def __init__(
        self, *, placement: Placement = "direct", executor: Executor | None = None
    ) -> None:
        self._execution = Execution(placement, executor)

    @property
    def placement(self) -> Placement:
        """Where a synchronous environment's methods run: ``"direct"`` or ``"thread"``."""
        return self._execution.placement

    async def reset(
        self,
        env: object,
        *,
        seed: int | None = None,
        options: Any = None,
        timeout: float | None = None,
    ) -> ResetResult:
        """Start a new episode of ``env``.

        ``seed`` and ``options`` are passed only when given, so an environment
        that takes neither needs neither parameter.

        Raises:
            UnsupportedCapabilityError: A seed was given, but the environment
                does not support seeded reset.
            EnvironmentCommunicationError: The environment did not answer
                within ``timeout`` seconds, or a remote adapter lost its
                connection.
            TypeError: The environment does not take ``options=``, or reset
                did not return a ``ResetResult``.
            ValueError: An invalid ``timeout``.
        """
        reset = _method(env, "reset")
        arguments: dict[str, Any] = {}
        if seed is not None:
            if not supports(env, "seeded_reset"):
                raise UnsupportedCapabilityError(name_of(env), "seeded_reset")
            arguments["seed"] = seed
        if options is not None:
            if not takes(reset, "options"):
                raise TypeError(
                    f"{name_of(env)}.reset does not take options=, so it cannot be set up "
                    "from an instance's inputs; give it an options=None parameter"
                )
            arguments["options"] = options
        result = await self._call(env, "reset", reset, timeout, **arguments)
        if not isinstance(result, ResetResult):
            raise TypeError(_wrong_result(env, "reset", result, "ResetResult"))
        return result

    async def step(
        self, env: object, actions: Mapping[str, Any], *, timeout: float | None = None
    ) -> StepResult:
        """Apply ``actions``, keyed by agent id, to ``env`` and return the result.

        Raises:
            EnvironmentCommunicationError: The environment did not answer
                within ``timeout`` seconds, or a remote adapter lost its
                connection. The step may have executed, so it is not retried.
            TypeError: ``actions`` is not keyed by agent id, or step did not
                return a ``StepResult``.
            ValueError: An invalid ``timeout``.
        """
        if not isinstance(actions, Mapping) or any(not isinstance(a, str) for a in actions):
            raise TypeError(
                f"actions must be keyed by agent id, even with one agent, such as "
                f"{{'agent': action}}, got {actions!r}"
            )
        result = await self._call(env, "step", _method(env, "step"), timeout, actions)
        if not isinstance(result, StepResult):
            raise TypeError(_wrong_result(env, "step", result, "StepResult"))
        return result

    async def close(self, env: object) -> None:
        """Release what ``env`` holds by calling its ``close()``, if it has one."""
        close = getattr(env, "close", None)
        if close is not None:
            await self._execution.call(close)

    async def _call(
        self,
        env: object,
        operation: str,
        method: Callable[..., Any],
        timeout: float | None,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        # The environment's own TimeoutError propagates as it is.
        try:
            return await self._execution.call_with_timeout(timeout, method, *args, **kwargs)
        except DeadlineExpiredError as expired:
            raise EnvironmentCommunicationError(
                name_of(env), operation, str(expired), may_have_executed=True
            ) from None


def _method(env: object, name: str) -> Callable[..., Any]:
    method = getattr(env, name, None)
    if not callable(method):
        raise TypeError(f"{name_of(env)} has no {name} method, so it is not an environment")
    return method  # type: ignore[no-any-return]


def _wrong_result(env: object, operation: str, result: object, expected: str) -> str:
    return (
        f"{name_of(env)}.{operation} returned {type(result).__name__}, not a {expected}; "
        f"an adapter converts the wrapped environment's result into a {expected} "
        "(see docs/environments.md)"
    )
