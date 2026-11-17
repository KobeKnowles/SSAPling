"""Giving an act-only model the buffered interface, so simple models never deal with buffers."""

import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Hashable, Mapping
from typing import Any, cast

from ..core._declarations import name_of, takes
from .contracts import BufferedModel, Model
from .types import NOTHING

__all__ = ["as_buffered"]


def as_buffered(model: Model) -> BufferedModel:
    """``model`` with the buffered interface: ``perceive``, ``poll``, and ``advance``.

    A model that already has ``perceive`` and ``poll`` is returned unchanged.
    An act-only model is wrapped:

    - ``perceive(observations)`` queues the observations.
    - ``advance()`` calls the model's ``act`` on the oldest queued
      observations and queues the output. So an act-only model needs exactly
      one internal step per output.
    - ``poll()`` returns the oldest output not yet read, or ``NOTHING``.
    - ``act(observations)`` still calls the model's ``act`` directly.

    The queues keep every observation and output, in order, so nothing is
    dropped silently: what to do with an output that arrives late is the
    coordinating code's choice. ``advance`` and ``act`` are ``async def``
    exactly when the model's ``act`` is. Each method takes an optional
    ``key``, so one model can serve several participants or lanes, each with
    its own queues (see ``reset_episode``).

    ``export_state()`` gives the model's own state, if it has one, with the
    queues, so a checkpoint taken mid-step keeps the step, and
    ``restore_state(state)`` returns to it. Everything else, such as
    ``learn`` or ``learning_operations``, is the model's own, reached
    through the wrapper.

    Raises:
        TypeError: ``model`` has neither ``act`` nor both ``perceive`` and
            ``poll``.
    """
    perceive, poll = getattr(model, "perceive", None), getattr(model, "poll", None)
    if callable(perceive) and callable(poll):
        return cast(BufferedModel, model)
    if callable(perceive) or callable(poll):
        missing = "poll" if callable(perceive) else "perceive"
        raise TypeError(f"{name_of(model)} has a buffered interface without {missing}")
    act = getattr(model, "act", None)
    if not callable(act):
        raise TypeError(
            f"{name_of(model)} is not a model: it has no act method, and no perceive and poll"
        )
    if inspect.iscoroutinefunction(act):
        return _AsyncActAdapter(model)
    return _ActAdapter(model)


class _Buffers:
    """The parts of the adapter that do not depend on whether ``act`` is async."""

    def __init__(self, model: object) -> None:
        self.model = model
        self._act: Callable[..., Any] = model.act  # type: ignore[attr-defined]
        self._inputs: dict[Hashable | None, deque[Any]] = {}
        self._outputs: dict[Hashable | None, deque[Any]] = {}

    def perceive(self, observations: Any, *, key: Hashable | None = None) -> None:
        """Queue ``observations`` for the model."""
        _check_key(self._act, key, self.model)
        self._inputs.setdefault(key, deque()).append(observations)

    def poll(self, *, key: Hashable | None = None) -> Any:
        """The oldest output not yet read, or ``NOTHING``."""
        outputs = self._outputs.get(key)
        return outputs.popleft() if outputs else NOTHING

    def reset_episode(self, *, key: Hashable | None = None) -> Any:
        """Empty the queues for ``key``, then call the model's ``reset_episode``, if it has one.

        Returns what the model's ``reset_episode`` returns, which callers
        await if the model's method is ``async def``.
        """
        reset = getattr(self.model, "reset_episode", None)
        if reset is not None:
            _check_key(reset, key, self.model)  # before emptying anything
        self._inputs.pop(key, None)
        self._outputs.pop(key, None)
        return None if reset is None else _call(reset, key, self.model)

    def export_state(self) -> Any:
        """The model's state (``None`` if it keeps none) and the queues, as
        ``{"model": ..., "queues": [[key, observations, outputs], ...]}``.

        The queues are copies, one entry per key, in order. Awaitable exactly
        when the model's ``export_state`` is ``async def``.
        """
        queues = [
            [key, list(self._inputs.get(key, ())), list(self._outputs.get(key, ()))]
            for key in dict.fromkeys([*self._inputs, *self._outputs])
        ]
        export = getattr(self.model, "export_state", None)
        state = None if export is None else export()
        if inspect.isawaitable(state):
            return _with_queues(state, queues)
        return {"model": state, "queues": queues}

    def restore_state(self, state: Mapping[str, Any]) -> Any:
        """Return to an exported state: the queues, then the model's own state.

        A key that came back as a list, as JSON gives a tuple, is a tuple
        again. Awaitable exactly when the model's ``restore_state`` is
        ``async def``.
        """
        self._inputs = {_key(key): deque(inputs) for key, inputs, _ in state["queues"]}
        self._outputs = {_key(key): deque(outputs) for key, _, outputs in state["queues"]}
        restore = getattr(self.model, "restore_state", None)
        return None if restore is None else restore(state["model"])

    def _next_input(self, key: Hashable | None) -> tuple[bool, Any]:
        inputs = self._inputs.get(key)
        return (True, inputs.popleft()) if inputs else (False, None)

    def _put_output(self, key: Hashable | None, output: Any) -> None:
        self._outputs.setdefault(key, deque()).append(output)

    def __getattr__(self, name: str) -> Any:
        # Called only for attributes the adapter lacks: those are the model's.
        if name == "model" or name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.model, name)

    def __repr__(self) -> str:
        return f"as_buffered({self.model!r})"


class _ActAdapter(_Buffers):
    """An act-only model with a synchronous ``act``, seen through buffers."""

    def advance(self, *, key: Hashable | None = None) -> None:
        """Act on the oldest queued observations, if any, and queue the output."""
        waiting, observations = self._next_input(key)
        if waiting:
            self._put_output(key, _call(self._act, key, self.model, observations))

    def act(self, observations: Any, *, key: Hashable | None = None) -> Any:
        """The model's own ``act``, called directly."""
        return _call(self._act, key, self.model, observations)


class _AsyncActAdapter(_Buffers):
    """An act-only model with an ``async def act``, seen through buffers."""

    async def advance(self, *, key: Hashable | None = None) -> None:
        """Act on the oldest queued observations, if any, and queue the output."""
        waiting, observations = self._next_input(key)
        if waiting:
            self._put_output(key, await _call(self._act, key, self.model, observations))

    async def act(self, observations: Any, *, key: Hashable | None = None) -> Any:
        """The model's own ``act``, called directly."""
        return await _call(self._act, key, self.model, observations)


async def _with_queues(state: Awaitable[Any], queues: list[Any]) -> dict[str, Any]:
    return {"model": await state, "queues": queues}


def _key(key: Any) -> Hashable | None:
    """A lane key as exported: a list (a tuple, through JSON) becomes a tuple again."""
    return tuple(_key(part) for part in key) if isinstance(key, list) else key


def _call(method: Callable[..., Any], key: Hashable | None, model: object, *args: Any) -> Any:
    """Call ``method``, passing ``key=`` only when there is one."""
    if key is None:
        return method(*args)
    _check_key(method, key, model)
    return method(*args, key=key)


def _check_key(method: Callable[..., Any], key: Hashable | None, model: object) -> None:
    if key is not None and not takes(method, "key"):
        raise TypeError(
            f"{name_of(model)}.{getattr(method, '__name__', 'method')} does not take key=, so it "
            "cannot keep participants or lanes apart; give it a key=None parameter"
        )
