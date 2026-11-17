"""Toy stages for tests. They exercise the stage contract and are not part of SSAPling.

Each stage notes its runs in the ``log`` component, a list, as
``(name, episodes so far)``, so a test can see what ran and when.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from ssapling.core import setting
from ssapling.train import RunContext, stages


def note(context: RunContext, name: str) -> None:
    context.components["log"].append((name, context.counters["episodes"]))


@dataclass(frozen=True)
class PlaySettings:
    episodes: int = setting(1, ge=1, description="Episodes to pretend to run.")


class Play:
    """Pretends to run episodes: counts them, without an environment."""

    Settings = PlaySettings
    requires = ("log",)

    def __init__(self, config: PlaySettings) -> None:
        self.config = config

    def run(self, context: RunContext) -> None:
        context.counters["episodes"] += self.config.episodes
        note(context, "play")


class Learn:
    """Pretends to learn: counts an update, asynchronously."""

    requires = ("log",)

    async def run(self, context: RunContext) -> None:
        await asyncio.sleep(0)
        context.counters["updates"] += 1
        note(context, "learn")


class Grade:
    """Publishes ``grade``, with a score that grows with the episodes: ``episodes / 4``."""

    requires = ("log",)
    publishes = ("grade",)

    def run(self, context: RunContext) -> None:
        context.publish("grade", {"score": context.counters["episodes"] / 4})
        note(context, "grade")


@dataclass(frozen=True)
class NoteSettings:
    name: str = setting("note", description="What to note in the log.")


class Note:
    """Notes its name in the log."""

    Settings = NoteSettings
    requires = ("log",)

    def __init__(self, config: NoteSettings) -> None:
        self.config = config

    def run(self, context: RunContext) -> None:
        note(context, self.config.name)


@dataclass(frozen=True)
class ReaderSettings:
    value: str = setting(description="The published value to read, by name or path.")


class Reader:
    """Reads a published value and notes it in the log."""

    Settings = ReaderSettings
    requires = ("log",)

    def __init__(self, config: ReaderSettings) -> None:
        self.config = config
        self.reads = (config.value.split(".")[0],)

    def run(self, context: RunContext) -> None:
        context.components["log"].append(("read", context.read(self.config.value)))


class Picky:
    """Needs a model that can learn, and says so before the run."""

    requires = ("model",)

    def check(self, context: RunContext) -> None:
        if not callable(getattr(context.components["model"], "learn", None)):
            raise ValueError("the model has no learn method")

    def run(self, context: RunContext) -> None:
        pass


@dataclass(frozen=True)
class NapSettings:
    seconds: float = setting(0.01, gt=0)


class Nap:
    """Sleeps a little: time passes, nothing else."""

    Settings = NapSettings

    def __init__(self, config: NapSettings) -> None:
        self.config = config

    async def run(self, context: RunContext) -> None:
        await asyncio.sleep(self.config.seconds)


TOYS: dict[str, Any] = {
    "play": Play,
    "learn": Learn,
    "grade": Grade,
    "note": Note,
    "reader": Reader,
    "picky": Picky,
    "nap": Nap,
}


def register_components() -> None:
    for name, stage in TOYS.items():
        stages.register(name, stage)
