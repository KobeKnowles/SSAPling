"""Toy tasks for tests. They exercise the task contract and are not part of SSAPling.

Everything here is a plain module-level class, so the fixtures can be
registered by name and rebuilt in a worker process.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ssapling.core import setting
from ssapling.tasks import Instance, PartialReport, Score, UnknownInstanceError


@dataclass(frozen=True)
class AdditionSettings:
    max_digits: int = setting(2, ge=1, description="Largest number of digits per operand.")


class AdditionTask:
    """Add two numbers. Generated from a seed, with an optional ``digits`` parameter."""

    Settings = AdditionSettings

    def __init__(self, config: AdditionSettings | None = None) -> None:
        self.config = config or AdditionSettings()

    def generate(self, *, params: Mapping[str, Any] | None = None, seed: int) -> Instance:
        digits = int((params or {}).get("digits", self.config.max_digits))
        if not 1 <= digits <= self.config.max_digits:
            raise ValueError(f"digits must be between 1 and {self.config.max_digits}")
        rng = random.Random(f"addition/{digits}/{seed}")
        a, b = (rng.randrange(10 ** (digits - 1), 10**digits) for _ in range(2))
        return Instance(
            id=f"add-{digits}d-{seed}",
            inputs={"a": a, "b": b},
            reference=a + b,
            descriptors={"digits": digits},
            # Two seeds can produce the same pair; the task considers those the same problem.
            equivalence_key=("add", a, b),
        )

    def score(self, instance: Instance, outcome: Any) -> Score:
        return Score.local(1.0 if outcome == instance.reference else 0.0)

    def check_partial(self, instance: Instance, state: Any) -> PartialReport:
        """``state`` is the digits of the answer entered so far, right to left, as a string."""
        if not isinstance(state, str) or (state and not state.isdigit()):
            return PartialReport(valid=False, detail="state must be a string of digits")
        expected = str(instance.reference)[::-1]  # units digit first, like the state
        pairs = zip(state, expected, strict=False)
        mismatch = next((i for i, (s, e) in enumerate(pairs) if s != e), None)
        if mismatch is None and len(state) > len(expected):
            mismatch = len(expected)  # too many digits
        valid = mismatch is None
        return PartialReport(
            valid=valid,
            progress=len(state) / len(expected) if valid else None,
            first_error=None if valid else {"position_from_right": mismatch},
        )

    def teaching_resources(self) -> Mapping[str, Any]:
        return {"procedure": "Add the units, carry, then add the tens."}


class PuzzleBook:
    """A fixed set of puzzles. Some have no reference, like a hidden test set."""

    PUZZLES: Sequence[Instance] = (
        Instance(
            id="p1",
            inputs={"grid": [[1, 0], [0, 1]]},
            reference={"grid": [[0, 1], [1, 0]]},
            descriptors={"difficulty": "easy"},
            equivalence_key="family-1",
        ),
        Instance(
            id="p1-mirror",
            inputs={"grid": [[0, 1], [1, 0]]},
            reference={"grid": [[1, 0], [0, 1]]},
            descriptors={"difficulty": "easy"},
            equivalence_key="family-1",
        ),
        Instance(
            id="p2",
            inputs={"grid": [[2, 2], [2, 2]]},
            reference={"grid": [[3, 3], [3, 3]]},
            descriptors={"difficulty": "hard"},
        ),
        Instance(
            id="hidden",
            inputs={"grid": [[5]]},
            descriptors={"difficulty": "hard"},
        ),
    )

    def __init__(self) -> None:
        self._by_id = {puzzle.id: puzzle for puzzle in self.PUZZLES}

    def ids(self) -> Sequence[str]:
        return tuple(self._by_id)

    def get(self, instance_id: str) -> Instance:
        try:
            return self._by_id[instance_id]
        except KeyError:
            raise UnknownInstanceError(instance_id, task="puzzle_book") from None

    def score(self, instance: Instance, outcome: Any) -> Score:
        if instance.reference is None:
            return Score.unavailable("the reference output is hidden")
        return Score.local(outcome == instance.reference)


class RemoteGameTask:
    """Instances are game ids resolved by a remote service; scores come from its scorecard."""

    def __init__(self, scorecard: Mapping[str, float] | None = None) -> None:
        self.scorecard = dict(scorecard or {})

    def ids(self) -> Sequence[str]:
        return ("game-a", "game-b")

    def get(self, instance_id: str) -> Instance:
        if instance_id not in self.ids():
            raise UnknownInstanceError(instance_id, task="remote_game")
        return Instance(id=instance_id, inputs=None)  # no local inputs, no reference

    def score(self, instance: Instance, outcome: Any) -> Score:
        if instance.id in self.scorecard:
            return Score.external(self.scorecard[instance.id], detail={"source": "scorecard"})
        return Score.unavailable("the service has not reported a score yet")


class SwitchableChecks(AdditionTask):
    """An addition task whose partial checks can be switched off by a setting."""

    def __init__(self, config: AdditionSettings | None = None, *, checks: bool = True) -> None:
        super().__init__(config)
        self.checks = checks

    def supports(self, feature: str) -> bool:
        return self.checks if feature == "check_partial" else True


class NumberBook:
    """Doubling numbers 0 to ``count - 1``: a listed source with descriptors and shared problems.

    Each instance has a ``size`` descriptor (``"small"`` below 10, else
    ``"large"``) and a ``parity``. Numbers ``2k`` and ``2k + 1`` share the
    equivalence key ``("pair", k)``: the task counts them as one problem.
    """

    def __init__(self, count: int = 30) -> None:
        self.instances = tuple(
            Instance(
                id=f"n{i}",
                inputs=i,
                reference=2 * i,
                descriptors={"size": "small" if i < 10 else "large", "parity": i % 2},
                equivalence_key=("pair", i // 2),
            )
            for i in range(count)
        )
        self._by_id = {instance.id: instance for instance in self.instances}

    def ids(self) -> Sequence[str]:
        return tuple(self._by_id)

    def get(self, instance_id: str) -> Instance:
        try:
            return self._by_id[instance_id]
        except KeyError:
            raise UnknownInstanceError(instance_id, task="number_book") from None

    def score(self, instance: Instance, outcome: Any) -> Score:
        return Score.local(outcome == instance.reference)
