"""The built-in stages: interact, learn_from_data, learn_operation, update_sampler, assess."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ssapling.core import ConfigError
from ssapling.data import (
    Collections,
    GeneratedSampler,
    GeneratedSettings,
    Record,
    RecordStore,
    UniformSampler,
)
from ssapling.evaluate import assessments, load_results
from ssapling.tasks import Instance
from ssapling.train import Published, RunContext, Runner
from tests.fixtures.environments import DigitPadEnv, PenniesEnv
from tests.fixtures.models import Apprentice
from tests.fixtures.stages import register_components
from tests.fixtures.tasks import AdditionTask, NumberBook, RemoteGameTask
from tests.fixtures.teachers import AsyncExplainer, DigitTutor

register_components()

MONITORING = {
    "collections": ["validation"],
    "aggregations": {"accuracy": {"measurement": "score", "aggregator": "mean"}},
}


def addition(tmp_path: Path | None = None, *, weights: Any = None, **extra: Any) -> dict[str, Any]:
    """A training run's components: additions entered on the digit pad, assessed statically.

    The model is an ``Apprentice``, right once it has had three updates.
    ``weights`` weigh the training sampler's digits, ``{"1": 1}`` for
    one-digit problems only. The validation collection is reserved.
    """
    task = AdditionTask()
    digits = GeneratedSettings(vary="digits", values=[1, 2], weights=weights or {})
    collections = Collections(task, source_name="addition")
    validation = GeneratedSampler(GeneratedSettings(vary="digits", values=[1, 2]), source=task)
    collections.build("validation", sampler=validation, size=4, seed=1)
    parts: dict[str, Any] = {
        "model": Apprentice(),
        "environment": DigitPadEnv(),
        "sampler": GeneratedSampler(digits, source=task),
        "collections": collections,
        "monitoring": assessments.create(
            "standard", MONITORING, collections=collections, task=task
        ),
    }
    if tmp_path is not None:
        parts["store"] = RecordStore(tmp_path)
    return {**parts, **extra}


def train(procedure: Mapping[str, Any], parts: Mapping[str, Any], seed: int = 7) -> RunContext:
    """Run ``procedure`` with ``parts``; their ``store``, if any, is the run's output."""
    given = dict(parts)
    store = given.pop("store", None)
    return asyncio.run(Runner(procedure, given, seed=seed, output=store).run())


def episodes(store: RecordStore, count: int) -> list[Record]:
    return [store.read(f"episode-{number}") for number in range(count)]


def instance(record: Record) -> Instance:
    """The instance a training record attempted, made again from its provenance."""
    provenance = record.header.provenance
    assert provenance is not None
    assert provenance.seed is not None
    return AdditionTask().generate(params=provenance.params, seed=provenance.seed)


def digits_entered(instance: Instance, seen: Mapping[str, Any]) -> dict[str, Any]:
    """The context the digit tutor needs: the instance, and the digits entered so far."""
    return {"instance": instance, "state": seen["student"]["entered"]}


# Interacting


def test_interact_runs_episodes_on_instances_outside_every_collection(tmp_path: Path):
    parts = addition(tmp_path, model=Apprentice(needs=0))  # right from the start
    procedure = {"procedure": [{"stage": "interact", "settings": {"episodes": 5}}]}
    context = train({**procedure, "stop": {"episodes": 5}}, parts)
    records = episodes(parts["store"], 5)
    assert context.counters["steps"] == sum(len(record.steps) for record in records) > 5
    reserved = {drawn.instance.problem_key for drawn in parts["collections"].get("validation")}
    for number, record in enumerate(records):
        assert record.header.provenance is not None
        assert record.header.provenance.collection is None  # from the remainder
        assert instance(record).problem_key not in reserved
        assert record.header.seeds == {"reset": context.seed(f"episode-{number}")}
    assert len({record.header.instance_id for record in records}) == 5


class Doubler:
    def act(self, number: int) -> int:
        return 2 * number


def test_training_never_sees_a_reserved_problem(tmp_path: Path):
    book = NumberBook(count=8)  # four problems: n and n + 1 are the same one, for even n
    sampler = UniformSampler(source=book)
    collections = Collections(book)
    collections.build("test", sampler=sampler, size=2, seed=1, held_out=True)
    reserved = {
        drawn.instance.problem_key for drawn in collections.get("test", purpose="evaluation")
    }

    def trained_on(name: str, **settings: Any) -> set[Any]:
        store = RecordStore(tmp_path / name)
        parts = {"model": Doubler(), "sampler": sampler, "collections": collections, "store": store}
        settings = {"episodes": 12, "environment": None, **settings}
        stage = {"stage": "interact", "settings": settings}
        train({"procedure": [stage], "stop": {"stages": 1}}, parts)
        return {book.get(r.header.instance_id).problem_key for r in episodes(store, 12)}

    every_problem = {("pair", k) for k in range(4)}
    assert trained_on("reserving") == every_problem - reserved
    assert trained_on("not reserving", collections=None) == every_problem


def test_interact_with_a_teacher_and_without_one(tmp_path: Path):
    parts = addition(tmp_path, tutor=DigitTutor(task=AdditionTask()), context=digits_entered)
    taught = {"episodes": 2, "teacher": "tutor", "teacher_context": "context"}
    procedure = [
        {"stage": "interact", "settings": taught, "once": True},
        {"stage": "interact", "settings": {"episodes": 2}},
    ]
    train({"procedure": procedure, "stop": {"episodes": 4}}, parts)
    with_tutor, alone = episodes(parts["store"], 2), episodes(parts["store"], 4)[2:]
    for record in with_tutor:
        assert record.header.teacher_placement == "coordinator"
        assert all(step.teacher_responses for step in record.steps)
    for record in alone:
        assert record.header.teacher_placement is None
        assert not any(step.teacher_responses for step in record.steps)


def test_interact_without_an_environment_solves_each_instance_statically(tmp_path: Path):
    parts = addition(tmp_path)
    procedure = [{"stage": "interact", "settings": {"episodes": 2, "environment": None}}]
    context = train({"procedure": procedure, "stop": {"episodes": 2}}, parts)
    (record, _) = episodes(parts["store"], 2)
    assert record.header.static
    assert record.steps[0].presented == {"solver": instance(record).inputs}
    assert context.counters["steps"] == 2  # one attempt each


# Learning


def test_learning_after_each_step_gets_the_step_and_only_the_signals_the_model_declares():
    parts = addition(tutor=DigitTutor(task=AdditionTask()), context=digits_entered)
    settings = {
        "episodes": 2,
        "teacher": "tutor",
        "teacher_context": "context",
        "learn": {"operation": "learn_from_experience", "after": "step"},
    }
    context = train(
        {"procedure": [{"stage": "interact", "settings": settings}], "stop": {"episodes": 2}}, parts
    )
    model = parts["model"]
    assert context.counters["updates"] == context.counters["steps"] == len(model.calls) == 2
    operation, payload = model.calls[0]
    assert operation == "learn_from_experience"
    step = payload["step"]
    shown = step.transition.observations["student"]
    right = str(shown["a"] + shown["b"])[::-1][0]
    # The tutor gave feedback and supervision; the model declares supervision only, and the
    # digit pad gives no rewards.
    assert step.teacher_responses[0].contributions.keys() == {"supervision", "feedback"}
    assert payload["signals"] == {"supervision": {"student": right}}
    assert context.read("learn_from_experience") == {"updates": 2}


class Matcher:
    """Shows heads every round, and keeps the learning signals it is given."""

    learning_operations = frozenset({"learn_from_experience"})
    signals = frozenset({"reward"})

    def __init__(self) -> None:
        self.given: list[Any] = []

    def act(self, observations: Any, *, key: Any = None) -> str:
        return "heads"

    def learn(self, operation: str, **payload: Any) -> Mapping[str, Any]:
        self.given.append(payload["signals"])
        return {}


def test_rewards_reach_a_model_that_declares_them_as_a_learning_signal():
    games = RemoteGameTask()  # instances with no inputs: matching pennies needs none
    model = Matcher()
    parts = {
        "model": model,
        "environment": PenniesEnv(rounds=2),
        "sampler": UniformSampler(source=games),
    }
    settings = {"collections": None, "learn": {"after": "step"}}
    train(
        {"procedure": [{"stage": "interact", "settings": settings}], "stop": {"episodes": 1}}, parts
    )
    # Both players show heads, so they match: player_1 wins each round.
    assert model.given == [{"reward": {"player_1": 1, "player_2": -1}}] * 2


class LateTutorPad(DigitPadEnv):
    """A digit pad that owns its tutor, whose answers come too late to use: they are stale."""

    def _review(self, proposed: str) -> Any:
        responses, executed = super()._review(proposed)
        stale = tuple(r.as_stale() if r.status == "answered" else r for r in responses or ())
        return stale, executed


def test_a_stale_answer_is_no_learning_signal():
    late = LateTutorPad(teacher=DigitTutor(task=AdditionTask()))
    parts = addition(environment=late)
    settings = {"learn": {"after": "step"}, "episode": {"teacher_operation": "review_action"}}
    train(
        {"procedure": [{"stage": "interact", "settings": settings}], "stop": {"episodes": 1}}, parts
    )
    ((_, payload),) = parts["model"].calls
    assert payload["step"].teacher_responses[0].status == "stale"
    assert payload["signals"] == {}


def test_learning_after_each_episode_gets_its_record():
    parts = addition()
    settings = {"episodes": 3, "learn": {"after": "episode"}}
    context = train(
        {"procedure": [{"stage": "interact", "settings": settings}], "stop": {"episodes": 3}}, parts
    )
    calls = parts["model"].calls
    assert [operation for operation, _ in calls] == ["learn_from_experience"] * 3
    assert [payload["record"].header.id for _, payload in calls] == [
        "episode-0",
        "episode-1",
        "episode-2",
    ]
    assert context.counters["updates"] == 3


def test_learning_from_stored_records_leaves_out_attempts_at_collection_instances(tmp_path: Path):
    store = RecordStore(tmp_path)
    parts = addition(store=store, ids=lambda items: [record.header.id for record in items])
    parts["monitoring"] = assessments.create(  # this one keeps its records in the run's store
        "standard", MONITORING, collections=parts["collections"], task=AdditionTask(), store=store
    )
    learning = {"source": "store", "batches": 3, "batch_size": 2, "collator": "ids"}
    procedure = [
        {"stage": "interact", "settings": {"episodes": 3}, "once": True},
        {"stage": "assess", "settings": {"assessment": "monitoring"}, "once": True},
        {"stage": "learn_from_data", "settings": learning},
    ]
    context = train({"procedure": procedure, "stop": {"updates": 3}}, parts)
    batches = [payload["batch"] for _, payload in parts["model"].calls]
    assert len(batches) == 3
    assert all(len(batch) == 2 for batch in batches)
    assert {record_id for batch in batches for record_id in batch} <= {
        "episode-0",
        "episode-1",
        "episode-2",
    }
    assert any(record_id.startswith("monitoring-2:") for record_id in store.record_ids())
    assert context.read("learn_from_batch") == {"updates": 3}


def test_batches_are_drawn_reproducibly_from_the_root_seed():
    def batches(seed: int) -> list[Any]:
        model = Apprentice()
        settings = {"source": "numbers", "batches": 2, "batch_size": 3}
        procedure = {"procedure": [{"stage": "learn_from_data", "settings": settings}]}
        train({**procedure, "stop": {"updates": 2}}, {"model": model, "numbers": range(50)}, seed)
        return [payload["batch"] for _, payload in model.calls]

    assert batches(7) == batches(7)
    assert batches(7) != batches(8)


def test_a_source_with_nothing_to_learn_from_is_an_error():
    procedure = {"procedure": [{"stage": "learn_from_data", "settings": {"source": "nothing"}}]}
    with pytest.raises(ValueError, match="'nothing' has no items to learn from"):
        train({**procedure, "stop": {"updates": 1}}, {"model": Apprentice(), "nothing": []})


def test_any_declared_learning_operation_can_be_called_and_what_it_returns_is_published():
    model = Apprentice()
    settings = {"operation": "consolidate", "payload": {"limits": {"seconds": 600}}}
    context = train(
        {
            "procedure": [{"stage": "learn_operation", "settings": settings}],
            "stop": {"value": "consolidate.updates", "at_least": 3},
        },
        {"model": model},
    )
    assert model.calls == [("consolidate", {"limits": {"seconds": 600}})] * 3
    assert context.values["consolidate"] == Published(
        {"updates": 3}, {"episodes": 0, "steps": 0, "updates": 3, "stages": 3}
    )


@pytest.mark.parametrize(
    ("stage", "settings", "problem"),
    [
        ("learn_operation", {"operation": "dream"}, "does not support 'dream'"),
        ("learn_from_data", {"source": "model"}, "neither a store nor a collection of items"),
        (
            "learn_from_data",
            {"source": "records", "operation": "dream"},
            "does not support 'dream'",
        ),
        ("interact", {"learn": {"operation": "dream"}}, "does not support 'dream'"),
        ("interact", {"teacher": "explainer"}, "the teacher does not support 'review_action'"),
        ("update_sampler", {"weights": {"1": 1}, "sampler": "model"}, "has no set_weights"),
        ("interact", {"environment": "pad"}, "requires 'pad', which the run does not have"),
        ("interact", {"sampler": "curriculum"}, "requires 'curriculum'"),
    ],
)
def test_what_a_stage_needs_is_checked_before_the_run(stage: str, settings: Any, problem: str):
    parts = addition(explainer=AsyncExplainer(), records=[])
    with pytest.raises(ConfigError, match=problem):
        train({"procedure": [{"stage": stage, "settings": settings}], "stop": {"stages": 1}}, parts)
    assert parts["model"].calls == []


# A curriculum, and assessing


def test_a_curriculum_moves_on_when_the_assessment_says_so(tmp_path: Path):
    parts = addition(tmp_path, weights={"1": 1})  # one-digit problems only, to begin with
    procedure = [
        {"stage": "interact", "settings": {"learn": {"after": "episode"}}},
        {"stage": "assess", "settings": {"assessment": "monitoring"}},
        {
            "stage": "update_sampler",
            "settings": {"weights": {"2": 1}},
            "when": {"value": "monitoring.accuracy", "at_least": 1},
        },
    ]
    context = train({"procedure": procedure, "stop": {"episodes": 6}}, parts)
    digits = [record.header.provenance.params["digits"] for record in episodes(parts["store"], 6)]  # type: ignore[union-attr]
    assert digits == [1, 1, 1, 2, 2, 2]  # the apprentice is right after its third update
    assert context.read("monitoring.accuracy") == 1.0


def test_assessing_saves_the_results_with_the_run_counters_and_publishes_the_aggregates(
    tmp_path: Path,
):
    parts = addition(tmp_path)
    procedure = [
        {"stage": "interact", "settings": {"episodes": 2}},
        {"stage": "assess", "settings": {"assessment": "monitoring"}},
    ]
    context = train({"procedure": procedure, "stop": {"stages": 2}}, parts)
    results = load_results(parts["store"], "monitoring-2")  # stage 2
    assert results.provenance["model"] == context.counters  # how much training it had had
    assert context.values["monitoring"] == Published(
        {"accuracy": results.aggregates["accuracy"].value}, context.counters
    )

    # The same assessment, standalone, of the same model, with the same seed, agrees.
    standalone = asyncio.run(
        assessments.create(
            "standard", MONITORING, collections=parts["collections"], task=AdditionTask()
        ).run(
            parts["model"],
            run_id="monitoring-2",
            seed=context.seed("assessment-monitoring"),
            model_info=context.counters,
        )
    )
    assert standalone == results


def test_built_in_stages_and_a_projects_own_work_together():
    parts = addition(log=[])
    procedure = [
        {"stage": "interact", "settings": {"episodes": 2, "learn": {"after": "episode"}}},
        {"stage": "assess", "settings": {"assessment": "monitoring"}},
        {"stage": "reader", "settings": {"value": "monitoring.accuracy"}},  # the project's own
    ]
    train({"procedure": procedure, "stop": {"stages": 6}}, parts)
    assert parts["log"] == [("read", 0.0), ("read", 1.0)]  # right after its third update
