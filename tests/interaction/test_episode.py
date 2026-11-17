"""run_episode: the episode loop, turn-based."""

import asyncio
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from ssapling.data import Record, RecordStep, RecordStore, RecordSummary
from ssapling.environments import ResetResult
from ssapling.interaction import in_process, run_episode, solve
from ssapling.models import NOTHING
from ssapling.tasks import Instance
from ssapling.teachers import TeacherRequest, TeacherResponse
from tests.fixtures.environments import (
    ArcEnv,
    AsyncProxy,
    DigitPadEnv,
    FakeArcService,
    NimEnv,
    PenniesEnv,
)
from tests.fixtures.hosting import given
from tests.fixtures.models import AsyncAct, LookupModel, Scripted, SlowThinker
from tests.fixtures.tasks import AdditionTask
from tests.fixtures.teachers import AsyncTeacher, DigitTutor, DigitTutorSettings, SlowTeacher

SUM = Instance(id="add-45-57", inputs={"a": 45, "b": 57}, reference=102)  # entered as 2, 0, 1
NO_INPUTS = Instance(id="game-1", inputs=None)


def run(*args: Any, **kwargs: Any) -> Record:
    return asyncio.run(run_episode(*args, **kwargs))


def tutor(*, intervene: bool = True) -> DigitTutor:
    return DigitTutor(DigitTutorSettings(intervene=intervene), task=AdditionTask())


def state(instance: Instance, seen: Mapping[str, Any]) -> dict[str, Any]:
    """The context DigitTutor needs: the instance, and the digits entered so far."""
    return {"instance": instance, "state": seen["student"]["entered"]}


def column(record: Record, field: str) -> list[Any]:
    return [getattr(step, field)["student"] for step in record.steps]


# The loop


def test_an_episode_is_recorded_step_by_step():
    model = Scripted(["2", "0", "1"])
    record = run(DigitPadEnv(), model, SUM, record_id="ep-1", seed=7)
    assert column(record, "proposals") == column(record, "executed_actions") == ["2", "0", "1"]
    assert record.header.instance_id == "add-45-57"
    assert record.header.seeds == {"reset": 7}
    assert record.header.teacher_placement is None
    assert record.summary == RecordSummary.terminated()
    entered = [observation["entered"] for observation in model.seen[None]]
    assert entered == ["", "2", "20"]  # each step, the latest observation


class SpyPad(DigitPadEnv):
    """Remembers what it was set up from."""

    def reset(self, *, seed: int | None = None, options: Any = None) -> ResetResult:
        self.options = options
        return super().reset(seed=seed, options=options)


def test_only_the_instance_inputs_reach_the_environment():
    env = SpyPad()
    run(env, Scripted(["2", "0", "1"]), SUM, record_id="ep-1")
    assert env.options == {"a": 45, "b": 57}  # never the reference


def test_agents_act_in_turn_and_a_shared_model_keeps_them_apart():
    model = Scripted([1, 1])
    record = run(NimEnv(stones=3), model, NO_INPUTS, record_id="nim-1")
    assert [step.acting for step in record.steps] == [("player_1",), ("player_2",), ("player_1",)]
    assert model.seen == {"player_1": [3, 1], "player_2": [2]}  # its key is the agent id
    assert record.summary == RecordSummary.terminated()


def test_each_participant_can_have_its_own_model():
    first, second = Scripted([1, 1]), Scripted([1])
    run(NimEnv(stones=3), {"player_1": first, "player_2": second}, NO_INPUTS, record_id="nim-1")
    assert first.seen == {None: [3, 1]}  # one participant each, so no key
    assert second.seen == {None: [2]}


def test_simultaneous_agents_and_a_model_that_takes_time_to_think():
    thinker = SlowThinker(think_steps=3)  # needs three advance calls per output
    record = run(PenniesEnv(rounds=2), thinker, NO_INPUTS, record_id="pennies-1")
    assert [step.acting for step in record.steps] == [("player_1", "player_2")] * 2
    assert record.steps[0].proposals == {"player_1": "None#1", "player_2": "None#1"}
    assert record.steps[1].proposals == {"player_1": "None#1#2", "player_2": "None#1#2"}
    assert record.summary == RecordSummary.truncated()  # the environment's round limit


class QuietStart(NimEnv):
    """Gives no observation at reset, so the first player moves blind."""

    def reset(self, *, seed: int | None = None) -> ResetResult:
        super().reset(seed=seed)
        return ResetResult(observations={})


class Stranger(NimEnv):
    """Names an agent to act that is not one of its participants."""

    next_agents = ("ghost",)


def test_an_agent_with_no_observation_yet_perceives_none():
    model = Scripted([1, 1])
    run(QuietStart(stones=3), model, NO_INPUTS, record_id="nim-1")
    assert model.seen["player_1"] == [None, 1]


def test_an_agent_named_to_act_must_be_a_participant():
    with pytest.raises(ValueError, match=r"names \['ghost'\] to act: not participants"):
        run(Stranger(), Scripted([1]), NO_INPUTS, record_id="nim-1")


def test_max_steps_truncates_the_episode():
    model = Scripted(["2", "0", "1"])
    record = run(DigitPadEnv(), model, SUM, record_id="ep-1", settings={"max_steps": 2})
    assert len(record.steps) == 2
    assert record.summary == RecordSummary.truncated("step limit")


def test_episode_state_is_reset_at_the_start_of_each_episode():
    env, model = DigitPadEnv(), Scripted(["2", "0", "1"])  # the script restarts per episode
    first = run(env, model, SUM, record_id="ep-1")
    second = run(env, model, SUM, record_id="ep-2")
    assert column(first, "proposals") == column(second, "proposals") == ["2", "0", "1"]


def test_the_record_is_written_to_the_store(tmp_path: Path):
    store = RecordStore(tmp_path)
    record = run(DigitPadEnv(), Scripted(["2", "0", "1"]), SUM, record_id="ep-1", store=store)
    assert store.read("ep-1") == record


def test_each_step_is_passed_on_once_it_is_recorded(tmp_path: Path):
    store = RecordStore(tmp_path)
    passed: list[RecordStep] = []
    stored: list[int] = []

    async def learn(step: RecordStep) -> None:  # async, as a learning operation may be
        passed.append(step)
        stored.append(len(store.read("ep-1").steps))

    model = Scripted(["2", "0", "1"])
    record = run(DigitPadEnv(), model, SUM, record_id="ep-1", store=store, on_step=learn)
    assert passed == list(record.steps)
    assert stored == [1, 2, 3]  # each step once recorded, and before the next


def test_static_solving_passes_on_each_attempt():
    passed: list[RecordStep] = []
    record = asyncio.run(
        solve(Scripted([3, 4]), SUM, record_id="p-1", attempts=2, on_step=passed.append)
    )
    assert passed == list(record.steps)


def test_an_error_passing_on_a_step_keeps_the_record_as_failed(tmp_path: Path):
    def refuse(step: RecordStep) -> None:
        raise RuntimeError("cannot learn from this")

    store, model = RecordStore(tmp_path), Scripted(["2", "0", "1"])
    with pytest.raises(RuntimeError, match="cannot learn from this"):
        run(DigitPadEnv(), model, SUM, record_id="ep-1", store=store, on_step=refuse)
    kept = store.read("ep-1")
    assert len(kept.steps) == 1
    assert kept.summary is not None
    assert kept.summary.status == "failed"


# Teachers


def responses(record: Record) -> list[list[tuple[str, Any]]]:
    return [[(r.status, r.contributions) for r in step.teacher_responses] for step in record.steps]


def test_a_teacher_gives_the_same_record_wherever_it_sits():
    wrong_first = ["9", "0", "1"]  # the tutor puts the first digit right
    coordinator = run(
        DigitPadEnv(),
        Scripted(wrong_first),
        SUM,
        record_id="c",
        teacher=tutor(),
        teacher_context=state,
        settings={"intervene": True},
    )
    environment = run(
        DigitPadEnv(teacher=tutor()),
        Scripted(wrong_first),
        SUM,
        record_id="e",
        teacher_placement="environment",
    )
    for record in (coordinator, environment):
        assert column(record, "proposals") == ["9", "0", "1"]
        assert column(record, "executed_actions") == ["2", "0", "1"]
        assert record.summary == RecordSummary.terminated()
    assert responses(coordinator) == responses(environment)
    assert coordinator.header.teacher_placement == "coordinator"
    assert environment.header.teacher_placement == "environment"

    # A teacher controlling the participant acts through the model interface.
    participant = run(
        DigitPadEnv(),
        Scripted(["2", "0", "1"]),
        SUM,
        record_id="p",
        teacher_placement="participant",
    )
    assert column(participant, "executed_actions") == ["2", "0", "1"]
    assert participant.header.teacher_placement == "participant"


def test_an_observing_teacher_is_recorded_but_never_applied():
    record = run(
        DigitPadEnv(),
        Scripted(["9"]),
        SUM,
        record_id="ep-1",
        teacher=tutor(),
        teacher_context=state,
    )
    (step,) = record.steps  # a wrong digit ends the game
    assert step.executed_actions == {"student": "9"}
    assert step.teacher_responses[0].contributions["intervention"] == {"student": "2"}


class Watcher:
    """A teacher that remembers what it was asked."""

    operations = frozenset({"review_action"})

    def __init__(self) -> None:
        self.requests: list[TeacherRequest] = []
        self.resets = 0

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        self.requests.append(request)
        return TeacherResponse.answered(request, {"feedback": {"student": "Fine."}})

    def reset(self) -> None:
        self.resets += 1


def test_the_teacher_is_asked_each_step_with_the_instance_and_the_observations():
    watcher = Watcher()
    run(DigitPadEnv(), Scripted(["2", "0", "1"]), SUM, record_id="ep-1", teacher=watcher)
    assert watcher.resets == 1
    assert [request.id for request in watcher.requests] == ["ep-1:0", "ep-1:1", "ep-1:2"]
    first = watcher.requests[0]
    assert first.context == {
        "instance": SUM,
        "observations": {"student": {"a": 45, "b": 57, "entered": ""}},
    }
    assert first.proposals == {"student": "2"}
    assert (first.episode, first.step) == ("ep-1", 0)


class LateTutor(Watcher):
    """Its answers arrive stale: about a state that has moved on."""

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        answer = TeacherResponse.answered(request, {"intervention": {"student": "7"}})
        return answer.as_stale()


def test_only_an_answered_response_intervenes():
    model = Scripted(["2", "0", "1"])
    record = run(
        DigitPadEnv(),
        model,
        SUM,
        record_id="ep-1",
        teacher=LateTutor(),
        settings={"intervene": True},
    )
    assert column(record, "executed_actions") == ["2", "0", "1"]
    assert record.steps[0].teacher_responses[0].status == "stale"


def test_a_teacher_in_a_thread_that_is_too_slow_gives_cancelled_responses():
    record = run(
        DigitPadEnv(),
        Scripted(["2", "0", "1"]),
        SUM,
        record_id="ep-1",
        teacher=SlowTeacher(0.5),
        settings={"threads": ["teacher"], "teacher_timeout": 0.05},
    )
    assert [step.teacher_responses[0].status for step in record.steps] == ["cancelled"] * 3
    assert record.summary == RecordSummary.terminated()


# Every component, sync or async, wherever it runs


@pytest.mark.parametrize("where", ["direct", "thread", "process"])
@pytest.mark.parametrize("async_teacher", [False, True], ids=["sync teacher", "async teacher"])
@pytest.mark.parametrize("async_model", [False, True], ids=["sync model", "async model"])
@pytest.mark.parametrize("async_env", [False, True], ids=["sync env", "async env"])
def test_every_mix_of_sync_and_async_components_wherever_they_run(
    async_env: bool, async_model: bool, async_teacher: bool, where: str
):
    env: Any = AsyncProxy(DigitPadEnv()) if async_env else DigitPadEnv()
    model: Any = AsyncAct(Scripted(["9", "0", "1"])) if async_model else Scripted(["9", "0", "1"])
    teacher: Any = AsyncTeacher(tutor()) if async_teacher else tutor()
    threads = [] if where == "direct" else ["environment", "model", "teacher"]
    with ExitStack() as stack:
        if where == "process":
            env, model, teacher = (
                stack.enter_context(in_process(given, each)) for each in (env, model, teacher)
            )
        record = run(
            env,
            model,
            SUM,
            record_id="ep-1",
            teacher=teacher,
            teacher_context=state,
            settings={"intervene": True, "threads": threads},
        )
    assert column(record, "executed_actions") == ["2", "0", "1"]
    assert record.summary == RecordSummary.terminated()


# Failures


def test_a_lost_connection_ends_the_record_as_failed():
    service = FakeArcService()
    service.drop_next = True
    click = {"action": "ACTION6", "x": 1, "y": 0}
    record = run(ArcEnv(service), Scripted([click]), NO_INPUTS, record_id="arc-1")
    (step,) = record.steps
    assert step.transition is None
    assert step.proposals == {"agent": click}
    assert record.summary is not None
    assert record.summary.status == "failed"
    assert "connection reset by peer" in record.summary.detail


def test_any_other_error_is_raised_once_the_record_is_kept_as_failed(tmp_path: Path):
    store = RecordStore(tmp_path)
    with pytest.raises(IndexError):  # the script runs out at the second step
        run(DigitPadEnv(), Scripted(["2"]), SUM, record_id="ep-1", store=store)
    kept = store.read("ep-1")
    assert len(kept.steps) == 1
    assert kept.summary is not None
    assert kept.summary.status == "failed"
    assert kept.summary.detail.startswith("IndexError")


class Background:
    """A buffered model that works on its own, with no advance for the loop to call."""

    def perceive(self, observations: Any) -> None: ...

    def poll(self) -> Any:
        return NOTHING


def test_mistakes_are_refused_before_the_episode_starts():
    env = DigitPadEnv()
    with pytest.raises(ValueError, match="cannot interrupt"):
        run(env, Scripted([]), SUM, record_id="ep-1", settings={"environment_timeout": 1.0})
    with pytest.raises(ValueError, match="cannot interrupt"):
        run(
            env,
            Scripted([]),
            SUM,
            record_id="ep-1",
            teacher=tutor(),
            settings={"teacher_timeout": 1},
        )
    with pytest.raises(ValueError, match="only as 'coordinator'"):
        run(
            env,
            Scripted([]),
            SUM,
            record_id="ep-1",
            teacher=tutor(),
            teacher_placement="environment",
        )
    assert env.native.a == 0  # never reset
    with pytest.raises(ValueError, match="no model controls 'player_2'"):
        run(NimEnv(), {"player_1": Scripted([1])}, NO_INPUTS, record_id="nim-1")
    with pytest.raises(TypeError, match="no advance"):
        run(DigitPadEnv(), Background(), SUM, record_id="ep-1")


def test_a_shared_model_must_take_a_key():
    lookup = LookupModel()  # act(observations), with no key
    with pytest.raises(TypeError, match="give it a key=None parameter"):
        run(NimEnv(), lookup, NO_INPUTS, record_id="nim-1")
