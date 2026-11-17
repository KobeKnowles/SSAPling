"""A component for testing ``in_process``. It is not part of SSAPling.

It reports which process it runs in, keeps state between calls, and
misbehaves on request. Everything here is defined at module level, so a
spawned worker process can import it.
"""

import asyncio
import os
from collections.abc import Callable
from typing import Any


class Probe:
    """Counts, and does whatever a test needs to see from another process."""

    label = "probe"

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError("start must be 0 or more")
        self.count = start

    @property
    def doubled(self) -> int:
        # Like a property that is valid only after reset: reading it too early fails.
        if self.count == 0:
            raise RuntimeError("nothing counted yet")
        return self.count * 2

    def pid(self) -> int:
        return os.getpid()

    def bump(self, by: int = 1, *, twice: bool = False) -> int:
        self.count += by * 2 if twice else by
        return self.count

    def apply(self, value: Any, how: Callable[[Any], Any] = lambda value: value * 2) -> Any:
        return how(value)  # the default cannot be pickled, so it stays in the worker

    async def later(self, value: Any) -> Any:
        await asyncio.sleep(0)
        return value

    def fail(self, error: Exception) -> None:
        raise error

    def fail_stubbornly(self) -> None:
        raise StubbornError("a", "b")

    def unsendable(self) -> Any:
        return lambda: None

    def unrebuildable(self) -> Any:
        return StubbornError("a", "b")  # pickles here, but cannot be rebuilt by the caller

    def crash(self) -> None:
        os._exit(3)


class StubbornError(Exception):
    """An error pickle cannot rebuild, because its ``__init__`` needs two arguments."""

    def __init__(self, first: str, second: str) -> None:
        super().__init__(f"{first} and {second}")


def configured_probe() -> Probe:
    """A factory that prepares the process it runs in, then builds the component."""
    Probe.label = "configured"
    return Probe()


def given(component: Any) -> Any:
    """A factory for an object built here: it arrives in the worker pickled, as itself."""
    return component
