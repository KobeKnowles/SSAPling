"""The project's own components: a task, an environment, a teacher and a model.

A downstream project writes components like these and registers them by
name; SSAPling builds them from the run file and runs everything else.

- ``AdditionTask``: problems ``a + b`` with a given number of digits.
- ``DigitPad``: the answer is entered one digit at a time, units first; a
  wrong digit ends the episode.
- ``ColumnTutor``: reviews each digit, says how the column works out, and
  offers the right digit when the student is wrong.
- ``FactLearner``: knows no arithmetic. It memorises what the tutor tells it
  about each column (two digits and a carry in give a digit to write and a
  carry out), and answers from what it has memorised.
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ssapling import registries as r
from ssapling.core import setting
from ssapling.data import Record
from ssapling.environments import ResetResult, StepResult
from ssapling.tasks import Instance, Score
from ssapling.teachers import TeacherRequest, TeacherResponse

STUDENT = "student"  # the one agent on the digit pad


def digit(number: int, place: int) -> int:
    """The digit of ``number`` at ``place``: 0 is the units, 1 the tens, and so on."""
    return int(number // 10**place % 10)


# The task


@dataclass(frozen=True)
class AdditionSettings:
    max_digits: int = setting(3, ge=1, description="The most digits an operand may have.")


class AdditionTask:
    """Add two whole numbers. Generated from a number of digits and a seed."""

    Settings = AdditionSettings

    def __init__(self, config: AdditionSettings) -> None:
        self.config = config

    def generate(self, *, params: Mapping[str, Any], seed: int) -> Instance:
        digits = params.get("digits", self.config.max_digits)
        if not 1 <= digits <= self.config.max_digits:
            raise ValueError(f"digits must be from 1 to {self.config.max_digits}, got {digits}")
        rng = random.Random(f"addition:{digits}:{seed}")
        a, b = (rng.randrange(10 ** (digits - 1) if digits > 1 else 0, 10**digits) for _ in "ab")
        return Instance(
            id=f"{a}+{b}",
            inputs={"a": a, "b": b},  # all the student ever sees
            reference=a + b,
            descriptors={"digits": digits},
            # 12 + 34 and 34 + 12 are the same problem, so never in two collections.
            equivalence_key=("+", min(a, b), max(a, b)),
        )

    def score(self, instance: Instance, outcome: Any) -> Score:
        return Score.local(1.0 if outcome == instance.reference else 0.0)


def entered_number(record: Record) -> int | None:
    """What a record's student entered on the pad, as a number: the outcome the task scores."""
    last = record.steps[-1].transition if record.steps else None
    entered: str = last.observations[STUDENT]["entered"] if last else ""
    return int(entered[::-1]) if entered else None


# The environment


class DigitPad:
    """Enter ``a + b`` one digit at a time, units first. A wrong digit ends the episode.

    Observation: ``{"a": .., "b": .., "entered": "..."}``, the digits entered
    so far, units first. Action: one digit, as a string. No rewards.
    """

    def __init__(self) -> None:
        self.a = self.b = 0
        self.answer = self.entered = ""

    def reset(self, *, options: Mapping[str, int] | None = None) -> ResetResult:
        if options is None:
            raise ValueError("the pad is set up from an instance's inputs, given as options")
        self.a, self.b = options["a"], options["b"]
        self.answer, self.entered = str(self.a + self.b)[::-1], ""
        return ResetResult(observations={STUDENT: self._observation()})

    def step(self, actions: Mapping[str, str]) -> StepResult:
        entry = actions[STUDENT]
        correct = entry == self.answer[len(self.entered)]
        self.entered += entry
        done = not correct or self.entered == self.answer
        return StepResult(
            observations={STUDENT: self._observation()},
            terminated={STUDENT: done},
            info={"correct": correct},
        )

    def _observation(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "entered": self.entered}


# The teacher


class ColumnTutor:
    """Reviews each digit the student proposes, column by column, as taught at school.

    Every answer supervises: it says which column it was, and the digit and
    carry it gives. When the student is wrong, it also offers the right
    digit as an intervention, which the episode applies only when its
    settings say ``intervene``.
    """

    operations = frozenset({"review_action"})
    required_context: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "review_action": ("instance", "observations")
    }

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        inputs = request.context["instance"].inputs
        a, b = inputs["a"], inputs["b"]
        place = len(request.context["observations"][STUDENT]["entered"])
        x, y = digit(a, place), digit(b, place)
        carry = (a % 10**place + b % 10**place) // 10**place  # the carry into this column
        write, carry_out = (x + y + carry) % 10, (x + y + carry) // 10
        lesson = {"x": x, "y": y, "carry": carry, "write": write, "carry_out": carry_out}
        contributions: dict[str, dict[str, Any]] = {"supervision": {STUDENT: lesson}}
        if request.proposals.get(STUDENT) != str(write):
            contributions["intervention"] = {STUDENT: str(write)}
        return TeacherResponse.answered(request, contributions)


# The model


class FactLearner:
    """Answers from the column facts it has memorised, and memorises those the tutor gives.

    A fact maps a column's two digits and the carry in to the digit to write
    and the carry out, such as ``"7+5+1" -> [3, 1]``. To answer, it works
    through the columns from the units, using its facts for the carries.
    Where it has no fact, it guesses 0 and no carry.
    """

    learning_operations = frozenset({"learn_from_experience"})
    signals = frozenset({"supervision"})  # the only learning signal it takes

    def __init__(self) -> None:
        self.facts: dict[str, list[int]] = {}

    def act(self, observations: Mapping[str, Any]) -> str:
        a, b, place = observations["a"], observations["b"], len(observations["entered"])
        carry = write = 0
        for p in range(place + 1):
            x, y = digit(a, p), digit(b, p)
            write, carry = self.facts.get(f"{x}+{y}+{carry}", [0, 0])
        return str(write)

    def learn(self, operation: str, **payload: Any) -> dict[str, int]:
        lesson = payload["signals"].get("supervision", {}).get(STUDENT)
        if lesson is not None:
            fact = f"{lesson['x']}+{lesson['y']}+{lesson['carry']}"
            self.facts[fact] = [lesson["write"], lesson["carry_out"]]
        return {"facts": len(self.facts)}

    def export_state(self) -> dict[str, Any]:
        return {"facts": dict(self.facts)}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.facts = dict(state["facts"])


def register_components() -> None:
    """Register this project's components by name, so a run file can use them."""
    r.tasks.register("addition", AdditionTask)
    r.environments.register("digit_pad", DigitPad)
    r.teachers.register("column_tutor", ColumnTutor)
    r.models.register("fact_learner", FactLearner)
