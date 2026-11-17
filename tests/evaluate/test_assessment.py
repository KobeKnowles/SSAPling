"""The standard assessment: attempts, measurements, aggregates, and stored results."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ssapling.core import ConfigError, UnknownComponentError
from ssapling.data import (
    Collections,
    GeneratedSampler,
    GeneratedSettings,
    Record,
    RecordStore,
    UniformSampler,
)
from ssapling.environments import StepResult
from ssapling.evaluate import (
    AssessmentResults,
    Trial,
    aggregate,
    assessments,
    load_results,
    measurements,
    save_results,
)
from tests.fixtures.environments import ArcEnv, AsyncProxy, DigitPadEnv, FakeArcService
from tests.fixtures.models import Adder, CountingLearner, PadAdder, Scripted
from tests.fixtures.tasks import AdditionTask, RemoteGameTask
from tests.fixtures.teachers import DigitTutor, DigitTutorSettings

ACCURACY = {"accuracy": {"measurement": "score", "aggregator": "mean"}}


def addition() -> tuple[AdditionTask, Collections]:
    task = AdditionTask()
    collections = Collections(task, source_name="addition")
    sampler = GeneratedSampler(GeneratedSettings(vary="digits", values=[1, 2]), source=task)
    collections.build("validation", sampler=sampler, size=6, seed=1)
    collections.build("test", sampler=sampler, size=3, seed=2, held_out=True)
    return task, collections


def run(assessment: Any, model: Any, **kwargs: Any) -> AssessmentResults:
    kwargs = {"run_id": "a1", "seed": 7, **kwargs}
    return asyncio.run(assessment.run(model, **kwargs))  # type: ignore[no-any-return]


def entered_number(record: Record) -> int:
    """What the digit pad shows at the end, as a number: the outcome the task scores."""
    entered: str = record.steps[-1].transition.observations["student"]["entered"]  # type: ignore[union-attr]
    return int(entered[::-1])


def scores(results: AssessmentResults) -> list[Any]:
    return [r.values["score"].value for r in results.instances]


def first_pair(collections: Collections, name: str = "validation") -> tuple[int, int]:
    inputs = collections.get(name)[0].instance.inputs
    return inputs["a"], inputs["b"]


# Running


def test_a_static_assessment_solves_measures_and_aggregates(tmp_path: Path):
    task, collections = addition()
    store = RecordStore(tmp_path)
    assessment = assessments.create(
        "standard",
        {"collections": ["validation"], "aggregations": ACCURACY},
        collections=collections,
        task=task,
        store=store,
    )
    results = run(assessment, Adder(wrong=[first_pair(collections)]))
    drawn = collections.get("validation")
    assert [r.instance_id for r in results.instances] == [d.instance.id for d in drawn]
    assert scores(results) == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert results.aggregates["accuracy"].value == pytest.approx(5 / 6)
    first = results.instances[0]
    assert first.record_id == f"a1:{drawn[0].instance.id}"
    assert first.collection == "validation"
    assert first.descriptors == drawn[0].instance.descriptors
    record = store.read(first.record_id)
    assert record.header.static
    assert record.header.provenance == drawn[0].provenance


def test_static_and_interactive_assessments_of_the_same_instances_agree():
    task, collections = addition()
    wrong = [first_pair(collections)]
    settings = {"collections": ["validation"], "aggregations": ACCURACY}
    static = assessments.create("standard", settings, collections=collections, task=task)
    interactive = assessments.create(
        "standard",
        settings,
        collections=collections,
        task=task,
        env=DigitPadEnv(),
        outcome=entered_number,
    )
    assert scores(run(static, Adder(wrong))) == scores(run(interactive, PadAdder(wrong)))


class Remote(AsyncProxy):
    """A digit pad behind a network: each step waits ``delay`` turns of the event loop."""

    def __init__(self, delay: int = 1) -> None:
        super().__init__(DigitPadEnv())
        self.delay = delay
        self.steps = 0

    async def step(self, actions: Any) -> StepResult:
        for _ in range(self.delay):
            await asyncio.sleep(0)
        self.steps += 1
        return await super().step(actions)


def test_lanes_give_the_same_results_as_one_environment():
    task, collections = addition()
    settings = {"collections": ["validation"], "aggregations": ACCURACY}
    model = PadAdder(wrong=[first_pair(collections)])

    def assess(env: Any) -> AssessmentResults:
        assessment = assessments.create(
            "standard",
            settings,
            collections=collections,
            task=task,
            env=env,
            outcome=entered_number,
        )
        return run(assessment, model)

    lanes = [Remote(delay=50), Remote()]  # the first lane is slow, so lanes finish out of order
    one, two = assess(Remote()), assess(lanes)
    assert all(lane.steps for lane in lanes)  # both lanes did work
    drawn = [item.instance.id for item in collections.get("validation")]
    assert [r.instance_id for r in two.instances] == drawn  # results in collection order
    assert one.instances == two.instances
    assert one.aggregates == two.aggregates
    assert two.provenance["lanes"] == 2


def test_each_instance_gets_its_own_seed_whatever_the_order(tmp_path: Path):
    task, collections = addition()

    def seeds(order: list[str]) -> dict[str, int]:
        store = RecordStore(tmp_path / "-".join(order))
        assessment = assessments.create(
            "standard",
            {"collections": order},
            collections=collections,
            task=task,
            env=DigitPadEnv(),
            outcome=entered_number,
            store=store,
        )
        results = run(assessment, PadAdder())
        return {
            r.instance_id: store.read(r.record_id).header.seeds["reset"] for r in results.instances
        }

    forwards, backwards = seeds(["validation", "test"]), seeds(["test", "validation"])
    assert forwards == backwards
    assert len(set(forwards.values())) == len(forwards)  # different instances, different seeds


def test_held_out_collections_are_assessed_and_limits_apply():
    task, collections = addition()
    settings = {"collections": ["test", "validation"], "max_instances": 4}
    assessment = assessments.create("standard", settings, collections=collections, task=task)
    results = run(assessment, Adder())
    assert [r.collection for r in results.instances] == ["test"] * 3 + ["validation"]


def test_a_step_limit_truncates_and_the_task_scores_what_was_reached():
    task, collections = addition()
    settings = {
        "collections": ["validation"],
        "episode": {"max_steps": 1},
        "aggregations": {
            "all": {"measurement": "score", "aggregator": "mean"},
            "finished": {"measurement": "score", "aggregator": "mean", "exclude_truncated": True},
        },
    }
    assessment = assessments.create(
        "standard",
        settings,
        collections=collections,
        task=task,
        env=DigitPadEnv(),
        outcome=entered_number,
    )
    results = run(assessment, PadAdder())
    truncated = [r for r in results.instances if r.status == "truncated"]
    assert truncated  # sums of two or more digits need more than one step
    assert all(r.values["score"].value == 0.0 for r in truncated)  # scored as reached
    assert results.aggregates["finished"].excluded == {"truncated": len(truncated)}


def test_a_failed_attempt_is_recorded_and_the_assessment_goes_on():
    task = RemoteGameTask(scorecard={"game-a": 0.5, "game-b": 0.75})
    collections = Collections(task, source_name="remote")
    collections.build("games", sampler=UniformSampler(source=task), size=2, seed=1)
    service = FakeArcService()
    service.drop_next = True  # the first attempt loses its connection
    clicks = [{"action": "ACTION6", "x": 1, "y": 0}, {"action": "ACTION6", "x": 0, "y": 1}]
    assessment = assessments.create(
        "standard",
        {"collections": ["games"], "aggregations": ACCURACY},
        collections=collections,
        task=task,
        env=ArcEnv(service),
    )
    first, second = run(assessment, Scripted(clicks)).instances
    assert first.status == "failed"
    assert first.values["score"].status == "failed"
    assert second.values["score"].status == "external"  # from the scorecard, as reported


# Measurements


def unreliable(trial: Trial) -> Any:
    """Has no value for one-digit problems, and fails on two-digit ones."""
    if trial.instance.descriptors["digits"] == 1:
        return None
    raise RuntimeError("cannot measure this")


measurements.register("unreliable", unreliable)


def test_measurements_that_give_nothing_or_fail_are_recorded_as_such():
    task, collections = addition()
    settings = {
        "collections": ["validation"],
        "measurements": ["score", "unreliable"],
        "aggregations": {"u": {"measurement": "unreliable", "aggregator": "mean"}},
    }
    assessment = assessments.create("standard", settings, collections=collections, task=task)
    results = run(assessment, Adder())
    statuses = {r.descriptors["digits"]: r.values["unreliable"].status for r in results.instances}
    assert statuses == {1: "unavailable", 2: "failed"}
    summary = results.aggregates["u"]
    assert summary.value is None
    assert sum(summary.excluded.values()) == len(results.instances)

    stopping = assessments.create(
        "standard", {**settings, "stop_on_failure": True}, collections=collections, task=task
    )
    with pytest.raises(RuntimeError, match="cannot measure this"):
        run(stopping, Adder())


class Talker(Adder):
    """Reports the tokens it used for each answer."""

    def diagnostics(self, *, key: Any = None) -> dict[str, int]:
        return {"tokens": 12}


def tokens(trial: Trial) -> int:
    return sum(step.diagnostics["solver"]["tokens"] for step in trial.record.steps)


measurements.register("tokens", tokens)


def test_a_measurement_can_read_what_the_model_reported_and_results_keep_model_info():
    task, collections = addition()
    settings = {
        "collections": ["validation"],
        "measurements": ["score", "tokens"],
        "aggregations": {"tokens": {"measurement": "tokens", "aggregator": "mean"}},
    }
    assessment = assessments.create("standard", settings, collections=collections, task=task)
    results = run(assessment, Talker(), model_info={"checkpoint": "ckpt-12", "episodes": 41200})
    assert results.aggregates["tokens"].value == 12
    assert results.provenance["model"] == {"checkpoint": "ckpt-12", "episodes": 41200}


def seen_outcome(trial: Trial) -> Any:
    """The outcome the assessment worked out, as the task would be given it."""
    return trial.outcome


measurements.register("seen_outcome", seen_outcome)


def test_the_default_outcome_is_the_last_prediction_or_the_final_observation():
    task, collections = addition()
    settings = {"collections": ["validation"], "measurements": ["seen_outcome"], "max_instances": 1}
    static = assessments.create("standard", settings, collections=collections, task=task)
    on_the_pad = assessments.create(
        "standard", settings, collections=collections, task=task, env=DigitPadEnv()
    )
    a, b = first_pair(collections)
    (solved,) = run(static, Adder()).instances
    (entered,) = run(on_the_pad, PadAdder()).instances
    assert solved.values["seen_outcome"].value == a + b
    assert entered.values["seen_outcome"].value == {"a": a, "b": b, "entered": str(a + b)[::-1]}


# The model during evaluation


class Watched(CountingLearner):
    """Remembers what it was told about evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.told: list[bool] = []

    def set_evaluating(self, evaluating: bool) -> None:
        self.told.append(evaluating)


def test_the_model_is_evaluated_without_learning_unless_learning_is_on():
    task, collections = addition()
    learner = Watched()
    before = learner.export_state()
    settings: dict[str, Any] = {"collections": ["validation"]}
    run(assessments.create("standard", settings, collections=collections, task=task), learner)
    assert learner.export_state() == before  # persistent state unchanged
    assert learner.told == [True, False]

    learning = {**settings, "learning_operation": "consolidate"}
    run(assessments.create("standard", learning, collections=collections, task=task), learner)
    assert learner.updates == len(collections.get("validation"))
    with pytest.raises(ValueError, match="does not support 'learn_from_experience'"):
        run(
            assessments.create(
                "standard",
                {**settings, "learning_operation": "learn_from_experience"},
                collections=collections,
                task=task,
            ),
            learner,
        )


def test_an_observing_teacher_is_recorded_but_never_applied(tmp_path: Path):
    task, collections = addition()
    store = RecordStore(tmp_path)
    tutor = DigitTutor(DigitTutorSettings(intervene=True), task=task)
    assessment = assessments.create(
        "standard",
        {"collections": ["validation"]},
        collections=collections,
        task=task,
        env=DigitPadEnv(),
        teacher=tutor,
        teacher_context=lambda instance, seen: {
            "instance": instance,
            "state": seen["student"]["entered"],
        },
        outcome=entered_number,
        store=store,
    )
    results = run(assessment, PadAdder(wrong=[first_pair(collections)]))
    assert results.provenance["teacher"] == "observe"
    first = store.read(results.instances[0].record_id).steps[0]
    assert first.teacher_responses[0].contributions["intervention"]  # said, but not done
    assert first.executed_actions == first.proposals
    assert results.instances[0].values["score"].value == 0.0


# Results


def test_results_are_stored_and_aggregates_recomputed_without_the_model(tmp_path: Path):
    task, collections = addition()
    settings = {"collections": ["validation"], "aggregations": ACCURACY}
    assessment = assessments.create("standard", settings, collections=collections, task=task)
    results = run(assessment, Adder(wrong=[first_pair(collections)]))
    store = RecordStore(tmp_path)
    save_results(store, results)
    loaded = load_results(store, "a1")
    assert loaded == results
    by_digits = {"by_digits": {"measurement": "score", "aggregator": "mean", "group_by": "digits"}}
    assert aggregate(loaded.instances, by_digits) == aggregate(results.instances, by_digits)


def test_mistakes_in_the_settings_are_refused_when_the_assessment_is_made():
    task, collections = addition()
    with pytest.raises(UnknownComponentError, match="measurement"):
        assessments.create(
            "standard",
            {"collections": ["validation"], "measurements": ["sore"]},
            collections=collections,
            task=task,
        )
    with pytest.raises(UnknownComponentError, match="aggregator"):
        assessments.create(
            "standard",
            {
                "collections": ["validation"],
                "aggregations": {"m": {"measurement": "score", "aggregator": "median"}},
            },
            collections=collections,
            task=task,
        )
    with pytest.raises(ConfigError, match="name measurements that are not taken"):
        assessments.create(
            "standard",
            {
                "collections": ["validation"],
                "aggregations": {"t": {"measurement": "tokens", "aggregator": "mean"}},
            },
            collections=collections,
            task=task,
        )
