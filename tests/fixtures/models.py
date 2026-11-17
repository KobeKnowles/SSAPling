"""Toy models for tests. They exercise the model contract and are not part of SSAPling.

Everything here is a plain module-level class, so the fixtures can be
registered by name and rebuilt in a worker process.
"""

import asyncio
import time
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ssapling.core import setting
from ssapling.models import NOTHING


@dataclass(frozen=True)
class LookupSettings:
    table: dict[str, str] = field(default_factory=dict)  # the output for each observation
    default: str = setting("?", description="The output for an observation not in the table.")


class LookupModel:
    """Acts by looking each observation up in a table. The simplest model: act only."""

    Settings = LookupSettings

    def __init__(self, config: LookupSettings | None = None) -> None:
        self.config = config or LookupSettings()

    def act(self, observations: str) -> str:
        return self.config.table.get(observations, self.config.default)


class SlowThinker:
    """A buffered model that needs ``think_steps`` internal steps per output.

    It remembers how many observations it has seen this episode, separately
    for each participant or lane, and outputs ``"<observation>#<count>"``.
    """

    def __init__(self, think_steps: int = 2) -> None:
        self.think_steps = think_steps
        self._inbox: dict[Hashable | None, deque[str]] = {}
        self._outbox: dict[Hashable | None, deque[str]] = {}
        self._progress: dict[Hashable | None, int] = {}
        self.seen: dict[Hashable | None, int] = {}

    def perceive(self, observations: str, *, key: Hashable | None = None) -> None:
        self._inbox.setdefault(key, deque()).append(observations)

    def advance(self, *, key: Hashable | None = None) -> None:
        inbox = self._inbox.get(key)
        if not inbox:
            return
        self._progress[key] = self._progress.get(key, 0) + 1
        if self._progress[key] < self.think_steps:
            return
        self._progress[key] = 0
        self.seen[key] = self.seen.get(key, 0) + 1
        self._outbox.setdefault(key, deque()).append(f"{inbox.popleft()}#{self.seen[key]}")

    def poll(self, *, key: Hashable | None = None) -> Any:
        outbox = self._outbox.get(key)
        return outbox.popleft() if outbox else NOTHING

    def reset_episode(self, *, key: Hashable | None = None) -> None:
        for state in (self._inbox, self._outbox, self._progress, self.seen):
            state.pop(key, None)


class AsyncEcho:
    """Repeats its observations back, asynchronously, after a delay, like a remote model."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    async def act(self, observations: Any, *, key: Hashable | None = None) -> Any:
        await asyncio.sleep(self.delay)
        return observations


class CountingLearner:
    """An act-only model that learns by counting updates, and declares everything optional."""

    learning_operations = frozenset({"learn_from_batch", "consolidate"})
    signals = frozenset({"reward", "supervision"})
    observation_description = "a string of digits"
    action_description = "one digit"

    def __init__(self) -> None:
        self.updates = 0
        self.examples_seen = 0
        self.evaluating = False
        self.last_signals: dict[str, Any] = {}

    def act(self, observations: str) -> str:
        return str(self.updates % 10)

    def learn(self, operation: str, **payload: Any) -> Mapping[str, Any]:
        if operation == "learn_from_batch":
            self.examples_seen += len(payload["batch"])
            self.last_signals = dict(payload.get("signals", {}))
        self.updates += 1
        return {"updates": self.updates, "examples_seen": self.examples_seen}

    def set_evaluating(self, evaluating: bool) -> None:
        self.evaluating = evaluating

    def export_state(self) -> dict[str, int]:
        return {"updates": self.updates, "examples_seen": self.examples_seen}

    def restore_state(self, state: Mapping[str, int]) -> None:
        self.updates = state["updates"]
        self.examples_seen = state["examples_seen"]


class GridModel:
    """Handles any square grid up to ``size`` cells across, so it decides for itself."""

    def __init__(self, size: int = 30) -> None:
        self.size = size

    def act(self, observations: Any) -> Any:
        return observations

    def accepts_description(self, kind: str, description: Any) -> bool:
        if kind == "action":
            return bool(description == "a grid")
        return isinstance(description, Mapping) and int(description.get("max_size", 0)) <= self.size


class Scripted:
    """Gives the outputs it was given, in order, and remembers what it observed.

    Each participant or lane (``key``) works through its own copy of the
    script, from the start of each episode.
    """

    def __init__(self, outputs: Sequence[Any]) -> None:
        self.outputs = list(outputs)
        self.seen: dict[Hashable | None, list[Any]] = {}

    def act(self, observations: Any, *, key: Hashable | None = None) -> Any:
        seen = self.seen.setdefault(key, [])
        seen.append(observations)
        return self.outputs[len(seen) - 1]

    def reset_episode(self, *, key: Hashable | None = None) -> None:
        self.seen.pop(key, None)


class AsyncAct:
    """Makes any act-only model asynchronous, as a remote one would be."""

    def __init__(self, model: Any) -> None:
        self.model = model

    async def act(self, observations: Any, *, key: Hashable | None = None) -> Any:
        if key is None:
            return self.model.act(observations)
        return self.model.act(observations, key=key)

    def reset_episode(self, *, key: Hashable | None = None) -> None:
        self.model.reset_episode(key=key)


class Sleepy:
    """An act-only model that takes ``delay`` seconds to answer, like a slow remote one."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def act(self, observations: Any) -> Any:
        time.sleep(self.delay)
        return observations


class InstantEcho:
    """A buffered model that works on its own, with no advance: it answers as it perceives."""

    def __init__(self) -> None:
        self._outbox: deque[Any] = deque()

    def perceive(self, observations: Any) -> None:
        self._outbox.append(observations)

    def poll(self) -> Any:
        return self._outbox.popleft() if self._outbox else NOTHING


class Adder:
    """Solves additions from their inputs, ``{"a": .., "b": ..}``, except the pairs in ``wrong``."""

    def __init__(self, wrong: Sequence[tuple[int, int]] = ()) -> None:
        self.wrong = set(wrong)

    def act(self, observations: Mapping[str, int], *, key: Hashable | None = None) -> int:
        a, b = observations["a"], observations["b"]
        return a + b + ((a, b) in self.wrong)


class PadAdder:
    """Enters a + b on the digit pad, units first; a wrong first digit for pairs in ``wrong``."""

    def __init__(self, wrong: Sequence[tuple[int, int]] = ()) -> None:
        self.wrong = set(wrong)

    def act(self, observations: Mapping[str, Any], *, key: Hashable | None = None) -> str:
        a, b, entered = observations["a"], observations["b"], observations["entered"]
        digit = str(a + b)[::-1][len(entered)]
        if (a, b) in self.wrong and not entered:
            return str((int(digit) + 1) % 10)
        return digit


class Apprentice:
    """Adds, statically or on the digit pad, but only once it has learned enough.

    Until it has had ``needs`` updates, each answer is one too many. Any
    learning operation is one update; it keeps what each call was given,
    so a test can check it.
    """

    learning_operations = frozenset({"learn_from_experience", "learn_from_batch", "consolidate"})
    signals = frozenset({"reward", "supervision"})

    def __init__(self, needs: int = 3) -> None:
        self.needs = needs
        self.updates = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def act(self, observations: Mapping[str, Any], *, key: Hashable | None = None) -> Any:
        total = observations["a"] + observations["b"] + (self.updates < self.needs)
        if "entered" not in observations:
            return total  # static solving: the whole answer
        return str(total)[::-1][len(observations["entered"])]  # the pad: the next digit

    def learn(self, operation: str, **payload: Any) -> Mapping[str, Any]:
        self.calls.append((operation, payload))
        self.updates += 1
        return {"updates": self.updates}

    def export_state(self) -> dict[str, int]:
        return {"updates": self.updates}

    def restore_state(self, state: Mapping[str, int]) -> None:
        self.updates = state["updates"]


class Fragile(Apprentice):
    """An apprentice whose machine goes down when it reaches ``fail_at`` updates."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at

    def learn(self, operation: str, **payload: Any) -> Mapping[str, Any]:
        if self.updates == self.fail_at:
            raise RuntimeError("the machine went down")
        return super().learn(operation, **payload)
