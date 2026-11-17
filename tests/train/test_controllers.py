"""Controllers that decide as the run goes: refusals, budgets, decision logs, replay, catalogue."""

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ssapling.core import ConfigError, settings_schema
from ssapling.data import RecordStore
from ssapling.train import Entry, ReplayController, RunContext, Runner, stages
from tests.fixtures.models import Fragile
from tests.fixtures.stages import register_components
from tests.fixtures.training import addition

register_components()

INTERACT = {"stage": "interact", "settings": {"learn": {"after": "episode"}}}
ASSESS = {"stage": "assess", "settings": {"assessment": "monitoring"}}
CHECKPOINT = {"stage": "checkpoint", "settings": {}}


def run(runner: Runner) -> RunContext:
    return asyncio.run(runner.run())


def chosen(context: RunContext) -> list[Any]:
    """The decisions a run made, without what each saw."""
    return [(d["stage"], d["settings"], d["refused"] is not None) for d in context.decisions]


class Tutor:
    """Interacts, then assesses, in turn; stops once the assessment is perfect."""

    def __init__(self) -> None:
        self.turns = 0

    def decide(self, context: RunContext, last_error: Exception | None) -> Any:
        if "monitoring" in context.values and context.read("monitoring.accuracy") == 1.0:
            return None
        self.turns += 1
        return INTERACT if self.turns % 2 else ASSESS

    def export_state(self) -> dict[str, int]:
        return {"turns": self.turns}

    def restore_state(self, state: dict[str, int]) -> None:
        self.turns = state["turns"]


def test_a_controller_decides_as_the_run_goes_from_what_it_sees():
    context = run(
        Runner({"procedure": [], "stop": {"stages": 100}}, addition(), seed=7, controller=Tutor())
    )
    # The apprentice is right after three updates, so the third assessment is perfect.
    assert [stage for stage, _, _ in chosen(context)] == ["interact", "assess"] * 3
    assert context.counters["episodes"] == 3
    third = context.decisions[2]  # what the controller saw when it chose the second interact
    assert third["seen"]["counters"]["stages"] == 2
    assert third["seen"]["values"]["monitoring"]["value"] == {"accuracy": 0.0}


def test_a_refused_decision_goes_back_to_the_controller_which_chooses_again():
    class Hasty:
        def __init__(self) -> None:
            self.told: list[Exception | None] = []

        def decide(self, context: RunContext, last_error: Exception | None) -> Any:
            self.told.append(last_error)
            script = [{"stage": "interac"}, {"stag": "interact"}, INTERACT]
            return script[len(self.told) - 1] if len(self.told) <= len(script) else None

    hasty = Hasty()
    context = run(
        Runner({"procedure": [], "stop": {"stages": 10}}, addition(), seed=7, controller=hasty)
    )
    assert hasty.told[0] is None
    assert "Did you mean 'interact'?" in str(hasty.told[1])
    assert "stag: unknown setting; did you mean 'stage'?" in str(hasty.told[2])
    assert hasty.told[3] is None  # the third decision ran
    assert chosen(context) == [
        ("interac", {}, True),
        (None, None, True),  # not even an entry
        ("interact", INTERACT["settings"], False),
    ]
    assert context.counters["stages"] == 1  # refused decisions run nothing


def test_three_refusals_in_a_row_end_the_run():
    asked: list[Any] = []

    def stubborn(context: RunContext, last_error: Exception | None) -> Any:
        asked.append(last_error)
        return {"stage": "interact", "settings": {"episodes": 0}}

    class Stubborn:
        decide = staticmethod(stubborn)

    runner = Runner(
        {"procedure": [], "stop": {"stages": 10}}, addition(), seed=7, controller=Stubborn()
    )
    with pytest.raises(ConfigError, match="episodes: must be >= 1, got 0"):
        run(runner)
    assert len(asked) == 3


def test_a_decision_that_runs_resets_the_count_of_refusals_in_a_row():
    refused = {"stage": "interact", "settings": {"episodes": 0}}
    script = [refused, refused, INTERACT, refused, refused, INTERACT]

    class Patchy:
        def decide(self, context: RunContext, last_error: Exception | None) -> Any:
            return script.pop(0) if script else None

    runner = Runner(
        {"procedure": [], "stop": {"stages": 10}}, addition(), seed=7, controller=Patchy()
    )
    context = run(runner)  # never three refusals in a row, so the run goes on
    assert [was_refused for _, _, was_refused in chosen(context)] == [True, True, False] * 2
    assert context.counters["episodes"] == 2


class Greedy:
    """Always wants more of the same."""

    def __init__(self, decision: Any) -> None:
        self.decision = decision

    def decide(self, context: RunContext, last_error: Exception | None) -> Any:
        return self.decision


@pytest.mark.parametrize(
    ("stop", "decision", "counter", "count"),
    [
        (
            {"updates": 3},
            {"stage": "learn_operation", "settings": {"operation": "consolidate"}},
            "updates",
            3,
        ),
        ({"stages": 4}, INTERACT, "stages", 4),
    ],
)
def test_the_stop_condition_is_a_budget_whatever_the_controller_decides(
    stop: Any, decision: Any, counter: str, count: int
):
    runner = Runner(
        {"procedure": [], "stop": stop}, addition(), seed=7, controller=Greedy(decision)
    )
    assert run(runner).counters[counter] == count


def test_time_is_a_budget_too():
    runner = Runner(
        {"procedure": [], "stop": {"seconds": 0.05}}, seed=7, controller=Greedy({"stage": "nap"})
    )
    assert run(runner).seconds >= 0.05


class Whimsical:
    """Chooses at random, unseeded: two runs choose differently."""

    def __init__(self) -> None:
        self.random = random.Random()

    def decide(self, context: RunContext, last_error: Exception | None) -> Any:
        episodes = self.random.randint(1, 3)
        return self.random.choice(
            [
                {"stage": "interact", "settings": {"episodes": episodes, "learn": {}}},
                ASSESS,
                {"stage": "learn_operation", "settings": {"operation": "consolidate"}},
            ]
        )


def test_a_run_is_replayed_exactly_from_its_decisions_whatever_made_them(tmp_path: Path):
    procedure = {"procedure": [], "stop": {"stages": 12}}
    first = run(
        Runner(procedure, addition(), seed=7, output=tmp_path / "first", controller=Whimsical())
    )
    log = RecordStore(tmp_path / "first").load_document("decisions")["decisions"]
    assert log == first.decisions
    again = run(
        Runner(
            procedure,
            addition(),
            seed=7,
            output=tmp_path / "again",
            controller=ReplayController(log),
        )
    )
    assert chosen(again) == chosen(first)
    assert again.counters == first.counters
    assert again.values == first.values
    assert again.components["model"].calls == first.components["model"].calls
    stores = RecordStore(tmp_path / "first"), RecordStore(tmp_path / "again")
    assert stores[0].record_ids() == stores[1].record_ids()
    for record_id in stores[0].record_ids():
        assert stores[0].read(record_id) == stores[1].read(record_id)


def test_a_replay_leaves_out_refused_decisions_and_stops_where_the_log_ends():
    log: list[dict[str, Any]] = [
        {"stage": "interac", "settings": {}, "refused": "Unknown stage 'interac'"},
        {**INTERACT, "refused": None},
        {**ASSESS, "refused": None},
    ]
    procedure = {"procedure": [], "stop": {"stages": 100}}
    replayed = run(Runner(procedure, addition(), seed=7, controller=ReplayController(log)))
    assert chosen(replayed) == [
        ("interact", INTERACT["settings"], False),
        ("assess", ASSESS["settings"], False),
    ]
    assert replayed.counters["stages"] == 2


def test_a_recorded_decision_that_cannot_run_now_cannot_be_replayed():
    log = [{**INTERACT, "refused": None}]
    given = addition()
    del given["environment"]  # the run it is replayed in has no environment
    runner = Runner(
        {"procedure": [], "stop": {"stages": 9}}, given, seed=7, controller=ReplayController(log)
    )
    with pytest.raises(ConfigError, match="requires 'environment'"):
        run(runner)


def test_a_controllers_state_is_checkpointed_and_restored_on_resuming(tmp_path: Path):
    script = [INTERACT, CHECKPOINT, INTERACT, INTERACT, CHECKPOINT, INTERACT, ASSESS]
    decisions = [{"stage": d["stage"], "settings": d["settings"], "refused": None} for d in script]
    procedure = {"procedure": [], "stop": {"stages": 20}}
    whole = run(
        Runner(
            procedure,
            addition(),
            seed=7,
            output=tmp_path / "whole",
            controller=ReplayController(decisions),
        )
    )

    broken = tmp_path / "broken"
    with pytest.raises(RuntimeError, match="the machine went down"):
        run(
            Runner(
                procedure,
                addition(Fragile(fail_at=3)),
                seed=7,
                output=broken,
                controller=ReplayController(decisions),
            )
        )
    failed_log = RecordStore(broken).load_document("decisions")["decisions"]
    assert len(failed_log) == 6  # saved when the run failed: up to the stage that failed
    with pytest.raises(ValueError, match="give the run its controller again, a ReplayController"):
        Runner.resume(broken, components=addition())
    replay = ReplayController(decisions)
    resumed = run(Runner.resume(broken, components=addition(), controller=replay))
    assert replay.position == len(decisions)  # restored to 5, then two more
    assert chosen(resumed) == chosen(whole)  # the log was restored too
    assert resumed.counters == whole.counters
    assert resumed.values == whole.values


def test_the_stage_catalogue_is_json_schema_ready_for_a_controller():
    catalogue = {entry["name"]: entry for entry in json.loads(json.dumps(stages.catalogue()))}
    assert {
        "interact",
        "learn_from_data",
        "learn_operation",
        "update_sampler",
        "assess",
        "checkpoint",
    } <= set(catalogue)
    for entry in catalogue.values():
        jsonschema.Draft202012Validator.check_schema(entry["settings"])
    interact = jsonschema.Draft202012Validator(catalogue["interact"]["settings"])
    interact.validate({"episodes": 8, "teacher": None, "learn": {"after": "episode"}})
    assert not interact.is_valid({"episodes": 0})
    assert catalogue["assess"]["description"].startswith("Runs an assessment of the model")

    decision = jsonschema.Draft202012Validator(settings_schema(Entry))
    decision.validate({"stage": "interact", "settings": {"episodes": 8}})
    assert not decision.is_valid({"stage": "interact", "every": {"episodes": "often"}})


# A controller that offers its entries; the error that ended a run


class Offering:
    """A controller that offers its entries (procedure()) and decides the first once."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        self.decided = 0

    def procedure(self) -> list[dict[str, Any]]:
        return list(self.entries)

    def decide(self, context: RunContext, last_error: Exception | None) -> Any:
        if last_error is not None:
            raise last_error
        if self.decided:
            return None
        self.decided += 1
        return self.entries[0]


def test_a_runner_takes_its_entries_from_a_controller_that_offers_them():
    entries = [{"stage": "interact", "settings": {"episodes": 1}}]
    runner = Runner({"stop": {"stages": 5}}, addition(), seed=1, controller=Offering(entries))
    assert asyncio.run(runner.run()).counters["episodes"] == 1


def test_the_entries_a_controller_offers_are_checked_before_the_run():
    """Given a controller offering an entry of an unknown stage, when the run starts, then it is
    refused before the controller decides anything.
    Catches: a custom controller whose decisions are never checked before the run."""
    offering = Offering([{"stage": "dance"}])
    runner = Runner({"stop": {"stages": 5}}, addition(), seed=1, controller=offering)
    with pytest.raises(ConfigError, match="dance"):
        asyncio.run(runner.run())
    assert offering.decided == 0


def test_a_run_with_no_entries_and_no_controller_offering_them_is_refused():
    with pytest.raises(ConfigError, match="procedure"):
        Runner({"stop": {"stages": 5}}, addition(), seed=1)


LEARNING = {
    "procedure": [{"stage": "interact", "settings": {"learn": {"after": "episode"}}}],
    "stop": {"episodes": 3},
}


def test_the_error_that_ended_a_run_is_saved_with_its_decisions(tmp_path: Path):
    """Given a model whose machine goes down at its second update, when the run stops with that
    error, then the decision log in the run's store holds the error's type and message.
    Catches: a stopped run whose reason exists only as an exception that has gone."""
    runner = Runner(LEARNING, addition(Fragile(fail_at=1)), seed=7, output=tmp_path)
    with pytest.raises(RuntimeError, match="the machine went down"):
        asyncio.run(runner.run())
    saved = RecordStore(tmp_path).load_document("decisions")
    assert saved["error"] == {"type": "RuntimeError", "message": "the machine went down"}


def test_a_run_that_ends_well_saves_no_error(tmp_path: Path):
    asyncio.run(Runner(LEARNING, addition(), seed=7, output=tmp_path).run())
    assert "error" not in RecordStore(tmp_path).load_document("decisions")
