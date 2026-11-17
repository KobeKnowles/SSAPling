"""Running a component in its own process, behind a stand-in that behaves like it.

A component normally runs in the caller's process: directly in the event
loop's thread, or in a worker thread (the runners' ``placement``). A
component that needs a process of its own, for example because it is
CPU-bound, needs its own GPU context, or might crash, is built in a
long-lived worker process by ``in_process``, which returns a **stand-in**.

The stand-in has the component's public methods and attributes, so
everything that uses a component (the runners, ``supports``, the episode
loop) uses it unchanged. Each is looked up in the worker when it is used:

- A method comes back as a stand-in method: ``def`` or ``async def``
  exactly as the component's is, taking the same parameters. A call runs in
  the worker; the caller waits.
- Any other attribute comes back as its value, so it is never out of date.
- An error the component raises arrives as itself.

The worker is started with ``spawn``, which works on every platform, so the
factory, its arguments, and every argument and result must be picklable.
"""

import asyncio
import inspect
import pickle
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any, cast

from .errors import ComponentProcessError

__all__ = ["in_process"]

_STOP_TIMEOUT = 5.0
"""Seconds to wait for a call in progress before stopping a worker forcibly."""


@contextmanager
def in_process[T](factory: Callable[..., T], /, *args: Any, **kwargs: Any) -> Iterator[T]:
    """Build a component in a worker process, and stand in for it until the block ends.

    Example::

        with in_process(DigitPadEnv) as env:
            first = await runner.reset(env, options=instance.inputs)

        with in_process(build_model, "checkpoints/7") as model:
            ...

    The worker builds the component with ``factory(*args, **kwargs)`` and
    keeps it until the block ends. A factory that needs the worker prepared
    first, for example by the project's ``register_components``, is a
    module-level function that does so before building. Then it
    stops, once any call in progress has finished, or after 5 seconds by
    force. Calling the component's own ``close``, if it has one, is up to
    its owner, before the block ends.

    A synchronous method of the stand-in blocks the thread that calls it
    until the worker answers. To keep an event loop running meanwhile, and to
    allow a timeout, run the component in a thread as well
    (``placement="thread"``), as for any synchronous component.

    Args:
        factory: Builds the component: a class, or a function defined at
            module level.
        *args, **kwargs: Passed to ``factory``.

    Yields:
        The stand-in.

    Raises:
        Exception: Whatever ``factory`` raises, as itself, when the block is
            entered.
        ComponentProcessError: From a later call: the worker has exited, or
            a result could not be sent back.
    """
    host = _Host(factory, args, kwargs)
    try:
        yield cast(T, host.stand_in)
    finally:
        host.stop()


class _Host:
    """The caller's side: the worker process, and the pipe to it."""

    def __init__(
        self, factory: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        # Imported here: importing ssapling must not load multiprocessing.
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=_serve, args=(child, factory, args, kwargs), daemon=True
        )
        self._lock = threading.Lock()  # one request at a time, from any thread
        self._stopped = False
        self.name = getattr(factory, "__qualname__", repr(factory))
        try:
            self._process.start()
        finally:
            child.close()
        status, value = self._receive()
        if status == "error":
            self.stop()
            value.add_note(f"While building {self.name} in a worker process.")
            raise value
        self.name = value  # the component's class name
        self.stand_in: _StandIn = object.__new__(type(self.name, (_StandIn,), {}))
        object.__setattr__(self.stand_in, "_host", self)

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def request(
        self,
        kind: str,
        name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Call a method of the component (``"call"``) or read an attribute (``"get"``)."""
        with self._lock:
            if self._stopped:
                raise ComponentProcessError(self.name, "has stopped: its in_process block ended")
            try:
                self._connection.send((kind, name, args, kwargs or {}))
            except OSError:
                raise self._exited() from None
            except Exception as error:
                error.add_note(f"{self.name} runs in another process, so arguments are pickled.")
                raise
            status, value = self._receive()
        if status == "error":
            value.add_note(f"Raised by {self.name}.{name} in its worker process.")
            raise value
        if status == "method":
            return _method(self, name, *value)
        return value

    def stop(self) -> None:
        """Stop the worker once any call in progress has finished, or by force after a while."""
        self._stopped = True
        with suppress(OSError):  # it may have exited already
            self._connection.send(None)  # read once the call in progress, if any, is answered
        self._process.join(_STOP_TIMEOUT)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._connection.close()

    def _receive(self) -> tuple[str, Any]:
        try:
            status, value = self._connection.recv()
        except (EOFError, OSError):
            raise self._exited() from None
        except Exception as error:  # the reply arrived but cannot be rebuilt here
            problem = f"sent back something this process cannot rebuild ({_describe(error)})"
            raise ComponentProcessError(self.name, problem) from None
        if status == "unsent":
            raise ComponentProcessError(self.name, f"could not send back {value}")
        return status, value

    def _exited(self) -> ComponentProcessError:
        self._process.join(timeout=1)
        return ComponentProcessError(self.name, f"exited (exit code {self._process.exitcode})")


class _StandIn:
    """Stands in for a component in another process; see ``in_process``.

    Each hosted component gets a subclass named after its class, so messages
    name the component.
    """

    _host: _Host

    def __getattr__(self, name: str) -> Any:
        # Every public attribute is looked up in the worker.
        if name.startswith("_"):
            raise AttributeError(name)
        return self._host.request("get", name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"cannot set {name!r}: {type(self).__name__} runs in another process, so change it "
            "through one of its methods"
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} in process {self._host.pid}>"


def _method(
    host: _Host, name: str, is_async: bool, signature: inspect.Signature | None
) -> Callable[..., Any]:
    """A stand-in for one method: ``def`` or ``async def`` as the original, with its parameters."""

    def call(*args: Any, **kwargs: Any) -> Any:
        return host.request("call", name, args, kwargs)

    async def call_async(*args: Any, **kwargs: Any) -> Any:
        # The request blocks, so it waits in a thread while the event loop goes on.
        return await asyncio.to_thread(host.request, "call", name, args, kwargs)

    method: Callable[..., Any] = call_async if is_async else call
    method.__name__, method.__qualname__ = name, f"{host.name}.{name}"
    if signature is not None:
        method.__signature__ = signature  # type: ignore[attr-defined]
    return method


# In the worker process


def _serve(
    connection: Any, factory: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """Build the component, then answer requests until told to stop."""
    try:
        component = factory(*args, **kwargs)
        reply: tuple[str, Any] = ("ok", type(component).__name__)
    except Exception as error:
        reply = ("error", error)
    _send(connection, reply)
    if reply[0] != "ok":
        return

    loop: asyncio.AbstractEventLoop | None = None  # for async methods, made when first needed
    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        if request is None:
            break
        kind, name, args, kwargs = request
        try:
            value = getattr(component, name)
            if kind == "call":
                value = value(*args, **kwargs)
                if inspect.isawaitable(value):
                    loop = loop or asyncio.new_event_loop()
                    value = loop.run_until_complete(value)
                reply = ("ok", value)
            elif callable(value) and not isinstance(value, type):  # a method: describe it
                reply = ("method", (inspect.iscoroutinefunction(value), _signature(value)))
            else:
                reply = ("ok", value)
        except (Exception, asyncio.CancelledError) as error:
            reply = ("error", error)
        _send(connection, reply)
    if loop is not None:
        loop.close()


def _signature(method: Callable[..., Any]) -> inspect.Signature | None:
    """``method``'s signature without annotations and default values, which may not pickle."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None  # the stand-in then takes any arguments
    parameters = [
        p.replace(annotation=p.empty, default=p.empty if p.default is p.empty else ...)
        for p in signature.parameters.values()
    ]
    return signature.replace(parameters=parameters, return_annotation=signature.empty)


def _send(connection: Any, reply: tuple[str, Any]) -> None:
    """Send ``reply``, or, if it cannot be sent and rebuilt, say so in words."""
    status, value = reply
    try:
        if status == "error":
            pickle.loads(pickle.dumps(value))  # an error that cannot be rebuilt is sent as text
        connection.send(reply)
    except Exception as problem:
        what = f"its error {_describe(value)}" if status == "error" else f"a {type(value).__name__}"
        connection.send(("unsent", f"{what} ({_describe(problem)})"))


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"
