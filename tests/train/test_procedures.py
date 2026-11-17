"""Procedures: entries and triggers, the stop condition, checks before the run, the context."""

import asyncio
import random
import textwrap
from pathlib import Path
from typing import Any

import pytest

from ssapling.core import ConfigError, load_config
from ssapling.train import (
    Condition,
    Entry,
    Every,
    Procedure,
    ProcedureStuckError,
    Published,
    RunContext,
    Runner,
)
from tests.fixtures.stages import register_components

register_components()


def run(procedure: Any, **components: Any) -> tuple[RunContext, list[Any]]:
    """Run ``procedure`` with a fresh log; return the context and the log."""
    log: list[Any] = []
    runner = Runner(procedure, {"log": log, **components}, seed=7)
    return asyncio.run(runner.run()), log


# Running


def test_a_procedure_goes_through_its_list_in_passes_until_the_stop_condition_holds():
    context, log = run(
        {
            "procedure": [{"stage": "play", "settings": {"episodes": 2}}, {"stage": "learn"}],
            "stop": {"episodes": 6},
        }
    )
    # The stop condition is checked before every stage: the last learn never starts.
    assert log == [("play", 2), ("learn", 2), ("play", 4), ("learn", 4), ("play", 6)]
    assert context.counters == {"episodes": 6, "steps": 0, "updates": 2, "stages": 5}


def test_triggers_decide_when_each_entry_runs():
    procedure = {
        "procedure": [
            {"stage": "play"},
            {"stage": "note", "settings": {"name": "first"}, "once": True},
            {"stage": "learn", "every": {"episodes": 2}},
            {"stage": "grade"},  # publishes grade.score = episodes / 4
            {
                "stage": "note",
                "settings": {"name": "practise"},
                "until": {"value": "grade.score", "at_least": 0.5},
            },
            {
                "stage": "note",
                "settings": {"name": "graduate"},
                "when": {"value": "grade.score", "at_least": 0.75},
            },
        ],
        "stop": {"episodes": 5},
    }
    _, log = run(procedure)
    assert log == [
        ("play", 1),
        ("first", 1),  # once
        ("grade", 1),
        ("practise", 1),  # until: the score is 0.25
        ("play", 2),
        ("learn", 2),  # every 2 episodes, counted from the start
        ("grade", 2),  # practise no more: the score is 0.5
        ("play", 3),
        ("grade", 3),
        ("graduate", 3),  # when: the first time the score is 0.75
        ("play", 4),
        ("learn", 4),
        ("grade", 4),  # graduate no more: when runs once
        ("play", 5),
    ]


def test_the_same_procedure_from_python_and_from_a_file(tmp_path: Path):
    file = tmp_path / "procedure.yaml"
    file.write_text(
        textwrap.dedent(
            """
            procedure:
              - stage: play
                settings: {episodes: 2}
              - stage: learn
                every: {episodes: 4}
              - stage: grade
            stop: {value: grade.score, at_least: 2}
            """
        )
    )
    in_python = Procedure(
        [
            Entry("play", {"episodes": 2}),
            Entry("learn", every=Every(episodes=4)),
            Entry("grade"),
        ],
        stop=Condition(value="grade.score", at_least=2),
    )
    from_file, file_log = run(load_config(file))
    from_python, python_log = run(in_python)
    assert file_log == python_log
    assert (
        from_file.counters
        == from_python.counters
        == {
            "episodes": 8,
            "steps": 0,
            "updates": 2,
            "stages": 10,
        }
    )


def test_any_controller_can_drive_the_run():
    class Scripted:
        def __init__(self, entries: list[Entry | None]) -> None:
            self.entries = entries

        async def decide(self, context: RunContext, last_error: Exception | None) -> Any:
            return self.entries.pop(0)

    log: list[Any] = []
    script = [
        Entry("play"),
        Entry("learn"),
        Entry("play"),
        Entry("grade"),
        Entry("reader", {"value": "grade.score"}),  # published by now, so it can be read
        None,
    ]
    runner = Runner(
        {"procedure": [], "stop": {"stages": 10}}, {"log": log}, seed=1, controller=Scripted(script)
    )
    context = asyncio.run(runner.run())
    assert log == [("play", 1), ("learn", 1), ("play", 2), ("grade", 2), ("read", 0.5)]
    assert context.counters["stages"] == 5  # the controller stopped the run


def test_time_can_stop_a_run():
    context, _ = run({"procedure": [{"stage": "nap"}], "stop": {"seconds": 0.03}})
    assert context.seconds >= 0.03  # how many naps that took depends on the system's clock


def test_a_procedure_that_cannot_reach_its_stop_condition_fails_instead_of_looping():
    with pytest.raises(ProcedureStuckError, match="no entry's trigger fires") as caught:
        run({"procedure": [{"stage": "play", "once": True}], "stop": {"episodes": 2}})
    assert caught.value.counters == {"episodes": 1, "steps": 0, "updates": 0, "stages": 1}


# Checks before the run

NOBODY = "which no stage in the procedure publishes"


def test_every_problem_is_reported_before_anything_runs():
    procedure = {
        "procedure": [
            {"stage": "plya"},
            {"stage": "play", "settings": {"episodes": 0}},
            {"stage": "picky"},  # requires a model, which the run does not have
            {"stage": "reader", "settings": {"value": "loss.mean"}},
            {"stage": "note", "until": {"value": "monitoring.accuracy", "at_least": 0.9}},
        ],
        "stop": {"value": "grade.score", "at_least": 1},
    }
    log: list[Any] = []
    with pytest.raises(ConfigError) as caught:
        asyncio.run(Runner(procedure, {"log": log}, seed=1).run())
    assert [(p.path, p.message.split(";")[0].split(".")[0]) for p in caught.value.problems] == [
        ("procedure[0].stage", "Unknown stage 'plya'"),
        ("procedure[1].settings.episodes", "must be >= 1, got 0"),
        (
            "procedure[2]",
            "stage 'picky' requires 'model', which the run does not have (its components: 'log')",
        ),
        ("procedure[3]", f"reads 'loss', {NOBODY}"),
        ("procedure[4].until.value", f"reads 'monitoring', {NOBODY}"),
        ("stop.value", f"reads 'grade', {NOBODY}"),
    ]
    assert log == []


def test_a_when_condition_that_reads_a_value_no_stage_publishes_is_refused_before_the_run():
    procedure = {
        "procedure": [{"stage": "play", "when": {"value": "grade.score", "at_least": 1}}],
        "stop": {"episodes": 2},
    }
    with pytest.raises(ConfigError) as caught:
        run(procedure)
    assert [(p.path, p.message) for p in caught.value.problems] == [
        ("procedure[0].when.value", f"reads 'grade', {NOBODY}")
    ]


def test_a_stage_can_check_what_it_needs_before_the_run():
    class Frozen:
        pass

    procedure = {"procedure": [{"stage": "picky"}], "stop": {"stages": 1}}
    with pytest.raises(ConfigError, match="stage 'picky': the model has no learn method"):
        run(procedure, model=Frozen())


def test_procedure_settings_are_checked_when_the_runner_is_made():
    with pytest.raises(ConfigError) as caught:
        Runner(
            {
                "procedure": [
                    {"stage": "play", "once": True, "every": {"episodes": 2}},
                    {"stage": "learn", "every": {"episodes": 2, "updates": 1}},
                ],
                "stop": {"value": "grade.score"},
            },
            seed=1,
        )
    assert [str(p) for p in caught.value.problems] == [
        "procedure[0]: an entry has at most one trigger, got once, every",
        "procedure[1].every: every names exactly one counter, such as {episodes: 500}",
        "stop: give the value 'grade.score' a bound: at_least or at_most",
    ]


@pytest.mark.parametrize(
    ("decision", "problem"),
    [
        (Entry("plya"), "stage: Unknown stage 'plya'"),
        ({"stage": "play", "settings": {"episodes": 0}}, "settings.episodes: must be >= 1"),
        ({"stage": "reader", "settings": {"value": "grade.score"}}, "reads 'grade', which is not"),
        ({"stag": "play"}, "stag: unknown setting; did you mean 'stage'?"),
    ],
)
def test_a_decision_that_cannot_run_is_refused(decision: Any, problem: str):
    class Careless:
        def decide(self, context: RunContext, last_error: Exception | None) -> Any:
            return decision

    runner = Runner(
        {"procedure": [], "stop": {"stages": 1}}, {"log": []}, seed=1, controller=Careless()
    )
    with pytest.raises(ConfigError) as caught:
        asyncio.run(runner.run())
    assert str(caught.value.problems[0]).startswith(problem)


# Conditions


def test_a_condition_holds_when_any_of_its_parts_does():
    context = RunContext({}, seed=1)
    either = Condition(episodes=10, value="grade.score", at_least=0.5)
    assert not either.holds(context)  # the value is not published yet: unmet
    context.publish("grade", {"score": None})
    assert not either.holds(context)  # an undefined value: unmet
    context.publish("grade", {"score": 0.5})
    assert either.holds(context)
    context.publish("grade", {"score": 0.1})
    context.counters["episodes"] = 10
    assert either.holds(context)
    assert Condition(value="grade.score", at_least=0, at_most=0.2).holds(context)
    assert Condition(value="grade.score", at_most=0.1).holds(context)  # bounds are inclusive
    assert not Condition(value="grade.score", at_most=0.05).holds(context)


def test_a_condition_on_something_that_is_not_a_number_is_an_error():
    context = RunContext({}, seed=1)
    context.publish("grade", {"score": "good", "passed": True})
    with pytest.raises(TypeError, match="not a number"):
        Condition(value="grade.score", at_least=1).holds(context)
    with pytest.raises(TypeError, match="not a number"):
        Condition(value="grade.passed", at_least=1).holds(context)
    with pytest.raises(LookupError, match="'grade' has no 'scroe'; it has 'score', 'passed'"):
        Condition(value="grade.scroe", at_least=1).holds(context)


def test_a_condition_names_a_counter_or_a_bounded_value():
    from ssapling.core import validate_settings

    with pytest.raises(ConfigError, match="names a counter"):
        validate_settings(Condition, {})
    with pytest.raises(ConfigError, match="bound a value; name it"):
        validate_settings(Condition, {"at_least": 1})
    with pytest.raises(ConfigError, match="must be >= 1"):
        validate_settings(Condition, {"updates": 0})


# The run context


def test_published_values_are_plain_stamped_with_the_counters_and_read_by_path():
    context = RunContext({}, seed=1)
    context.counters["updates"] = 3
    context.publish("monitoring", {"accuracy": 0.8, "groups": (1, 2)})
    assert context.values["monitoring"] == Published(
        {"accuracy": 0.8, "groups": [1, 2]},
        {"episodes": 0, "steps": 0, "updates": 3, "stages": 0},
    )
    assert context.read("monitoring.accuracy") == 0.8
    with pytest.raises(LookupError, match="nothing is published as 'loss'; published: 'monitor"):
        context.read("loss")
    with pytest.raises(TypeError, match="plain data"):
        context.publish("model", {"weights": object()})
    with pytest.raises(ValueError, match="no dots"):
        context.publish("monitoring.accuracy", 0.8)


def test_seeds_are_derived_from_the_root_seed_by_a_fixed_rule():
    context = RunContext({}, seed=7)
    assert context.seed("model") == RunContext({}, seed=7).seed("model")
    assert context.seed("model") == random.Random("7:model").getrandbits(32)
    assert context.seed("model") != context.seed("sampler")
    assert context.seed("model") != RunContext({}, seed=8).seed("model")
