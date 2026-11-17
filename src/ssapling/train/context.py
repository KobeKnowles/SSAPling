"""The run context: what the stages of a run share.

A stage never hands anything to another stage directly. It reads from the
run context and publishes values to it: the components built for the run,
the run's counters, its seeds, and a table of published values.
"""

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core._declarations import check_identifier
from ..core._plain import plain_copy
from ..core.errors import Problem

__all__ = ["COUNTERS", "Published", "RunContext"]

COUNTERS = ("episodes", "steps", "updates", "stages")
"""The run's counters.

``episodes`` and ``steps``: episodes run and environment steps taken.
``updates``: calls to a learning operation. ``stages``: stages started.
"""


@dataclass(frozen=True, slots=True)
class Published:
    """A published value, stamped with the counters at which it was published.

    Attributes:
        value: The value: plain data.
        counters: The run's counters when it was published.
    """

    value: Any
    counters: Mapping[str, int]


class RunContext:
    """What the stages of a run share.

    Attributes:
        components: The run's components, keyed by role, such as
            ``"model"`` or ``"environment"``. ``"store"`` is the run's
            store, if it has one.
        counters: The counters in ``COUNTERS``, starting at 0. The stages
            that do the work count it: the interact stage counts episodes
            and steps, a learning stage counts updates.
        values: The published values, keyed by name.
        root_seed: The run's root seed, from which every other seed is
            derived (see ``seed``).
        controller: What decides the run's next stage; a checkpoint saves
            its state.
        session: 0 for a run as first started, and ``k`` once it has been
            resumed ``k`` times (see ``output_name``).
        decisions: Every decision the controller has made, in order, as
            plain data: the stage and settings it chose, what it saw (the
            counters, running time and published values, as
            ``export_state`` gives them), and why it was refused, if it
            was.
    """

    def __init__(
        self, components: Mapping[str, Any], *, seed: int, controller: object = None
    ) -> None:
        self.components = dict(components)
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.values: dict[str, Published] = {}
        self.root_seed = seed
        self.controller = controller
        self.session = 0
        self.decisions: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._before = 0.0  # seconds run before it was last resumed

    @property
    def seconds(self) -> float:
        """Seconds the run has been running, across its resumptions."""
        return self._before + time.monotonic() - self._started

    def seed(self, name: str) -> int:
        """The seed for ``name``, derived from the root seed.

        The rule is fixed: the same root seed and name always give the same
        seed, on every machine, and different names give unrelated seeds.
        """
        return derive_seed(self.root_seed, name)

    def output_name(self, base: str) -> str:
        """The name to write a record or document under: ``base``, or in a resumed run a new one.

        A run resumes from its last checkpoint, and what it wrote after
        that checkpoint is still in its store. So once resumed, it writes
        under ``"resume-<session>/<base>"``, and replaces nothing.
        """
        return base if self.session == 0 else f"resume-{self.session}/{base}"

    def export_state(self) -> dict[str, Any]:
        """The counters, running time and published values, for a checkpoint."""
        return {
            "counters": dict(self.counters),
            "seconds": self.seconds,
            "values": {
                name: {"value": published.value, "counters": dict(published.counters)}
                for name, published in self.values.items()
            },
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Return to what ``export_state`` saved."""
        self.counters = dict(state["counters"])
        self._before, self._started = state["seconds"], time.monotonic()
        self.values = {
            name: Published(saved["value"], saved["counters"])
            for name, saved in state["values"].items()
        }

    def publish(self, name: str, value: Any) -> None:
        """Publish ``value`` under ``name``, stamped with the current counters.

        It replaces what was published under ``name`` before. Values are
        plain data (``None``, booleans, numbers, strings, and lists and
        string-keyed dicts of these), so they can be saved with a checkpoint
        and shown to a controller.

        Raises:
            ValueError: ``name`` is empty or has a dot, which ``read`` uses
                to look inside a value.
            TypeError: Part of ``value`` is not plain.
        """
        check_identifier(name, "a value's name")
        if "." in name:
            raise ValueError(f"a value's name has no dots, which separate a path: {name!r}")
        problems: list[Problem] = []
        plain = plain_copy(value, name, problems)
        if problems:
            raise TypeError("published values are plain data: " + "; ".join(map(str, problems)))
        self.values[name] = Published(plain, dict(self.counters))

    def read(self, path: str) -> Any:
        """The value published under a name, or inside one by a dotted path.

        ``read("monitoring")`` is the whole value the assess stage published
        for the assessment ``monitoring``, and
        ``read("monitoring.accuracy")`` is one aggregate in it.

        Raises:
            LookupError: Nothing is published under the name, or the path
                leads nowhere in its value.
        """
        name, *keys = path.split(".")
        if name not in self.values:
            published = ", ".join(repr(name) for name in self.values) or "none"
            raise LookupError(f"nothing is published as {name!r}; published: {published}")
        value = self.values[name].value
        for depth, key in enumerate(keys):
            if not isinstance(value, Mapping) or key not in value:
                where = ".".join([name, *keys[:depth]])
                has = ", ".join(map(repr, value)) if isinstance(value, Mapping) else "no keys"
                raise LookupError(f"{where!r} has no {key!r}; it has {has}")
            value = value[key]
        return value


def derive_seed(root: int, name: str) -> int:
    """The seed for ``name`` in a run with the root seed ``root``: see ``RunContext.seed``."""
    return random.Random(f"{root}:{name}").getrandbits(32)
