"""Timing modes, clocks, late outputs, default actions, lanes, and static solving."""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from ssapling.core import ConfigError, validate_settings
from ssapling.data import AgentTiming, Record, RecordStore, RecordSummary
from ssapling.interaction import EpisodeSettings, run_episode, solve
from ssapling.tasks import Instance
from tests.fixtures.environments import DigitPadEnv, NimEnv, Ticker
from tests.fixtures.models import InstantEcho, Scripted, Sleepy, SlowThinker

TICKS = Instance(id="ticks", inputs=None)
SUM = Instance(id="add-45-57", inputs={"a": 45, "b": 57}, reference=102)
PUZZLE = Instance(id="p1", inputs={"grid": [[1, 0], [0, 1]]}, reference={"grid": [[0, 1], [1, 0]]})
DEADLINE = {"timing": "deadline", "budget": 2, "default_actions": {"agent": "pass"}}


def run(*args: Any, **kwargs: Any) -> Record:
    return asyncio.run(run_episode(*args, **kwargs))


def timings(record: Record, agent: str = "agent") -> list[AgentTiming]:
    return [step.timing[agent] for step in record.steps]


# The simulated clock: time is counted in calls to advance


def test_turn_based_waits_as_long_as_the_model_needs():
    record = run(Ticker(ticks=2), SlowThinker(think_steps=3), TICKS, record_id="t")
    assert record.header.clock == "simulated"
    assert [step.proposals for step in record.steps] == [{"agent": "t0#1"}, {"agent": "t1#2"}]
    assert timings(record) == [AgentTiming(elapsed=3)] * 2


def test_an_output_that_came_in_time_is_recorded_as_not_defaulted():
    turn_based = run(Ticker(ticks=2), SlowThinker(think_steps=3), TICKS, record_id="t")
    in_time = run(
        Ticker(ticks=2), SlowThinker(think_steps=1), TICKS, record_id="d", settings=DEADLINE
    )
    for record in (turn_based, in_time):
        assert [(t.defaulted, t.late) for t in timings(record)] == [(False, None), (False, None)]


def test_a_deadline_applies_the_default_action_and_marks_it():
    env = Ticker(ticks=3)
    record = run(env, SlowThinker(think_steps=3), TICKS, record_id="t", settings=DEADLINE)
    assert env.received == ["pass", "pass", "pass"]
    assert [step.proposals for step in record.steps] == [{}, {}, {}]
    assert timings(record) == [
        AgentTiming(elapsed=2, defaulted=True),
        # Each later answer is to the previous tick's observation: late, so discarded.
        AgentTiming(elapsed=2, defaulted=True, late="discarded"),
        AgentTiming(elapsed=2, defaulted=True, late="discarded"),
    ]


def test_a_late_output_can_be_used_at_the_next_turn_instead():
    env = Ticker(ticks=3)
    settings = {**DEADLINE, "late_outputs": "use"}
    record = run(env, SlowThinker(think_steps=3), TICKS, record_id="t", settings=settings)
    assert env.received == ["pass", "t0#1", "pass"]
    assert timings(record) == [
        AgentTiming(elapsed=2, defaulted=True),
        AgentTiming(elapsed=1, late="used"),
        AgentTiming(elapsed=2, defaulted=True),
    ]


def test_without_a_default_action_the_environment_applies_its_own():
    env = Ticker(ticks=1)
    settings = {"timing": "deadline", "budget": 1}
    record = run(env, SlowThinker(think_steps=3), TICKS, record_id="t", settings=settings)
    assert env.received == ["wait"]
    assert record.steps[0].executed_actions == {}


def test_a_deadline_acts_as_soon_as_it_can_and_real_time_at_the_end_of_the_tick():
    for timing, elapsed in (("deadline", 3), ("real_time", 5)):
        settings = {"timing": timing, "budget": 5}
        record = run(
            Ticker(ticks=2), SlowThinker(think_steps=3), TICKS, record_id="t", settings=settings
        )
        assert [step.proposals for step in record.steps] == [{"agent": "t0#1"}, {"agent": "t1#2"}]
        assert timings(record) == [AgentTiming(elapsed=elapsed)] * 2


def test_runs_on_the_simulated_clock_are_identical():
    settings = {**DEADLINE, "late_outputs": "use"}
    first, second = (
        run(Ticker(ticks=5), SlowThinker(think_steps=3), TICKS, record_id="t", settings=settings)
        for _ in range(2)
    )
    assert first == second


def test_timing_is_stored_with_the_record(tmp_path: Path):
    store = RecordStore(tmp_path)
    settings = {**DEADLINE, "late_outputs": "use"}
    record = run(
        Ticker(), SlowThinker(think_steps=3), TICKS, record_id="t", store=store, settings=settings
    )
    assert store.read("t") == record


# The wall clock: time is counted in seconds


def test_a_model_that_works_on_its_own_needs_the_wall_clock():
    with pytest.raises(TypeError, match="use the wall clock"):
        run(Ticker(), InstantEcho(), TICKS, record_id="t")
    record = run(Ticker(ticks=2), InstantEcho(), TICKS, record_id="t", settings={"clock": "wall"})
    assert record.header.clock == "wall"
    assert [step.proposals for step in record.steps] == [{"agent": "t0"}, {"agent": "t1"}]


def test_real_time_on_the_wall_clock_acts_at_the_end_of_each_tick():
    settings = {"clock": "wall", "timing": "real_time", "budget": 0.02}
    record = run(Ticker(ticks=2), InstantEcho(), TICKS, record_id="t", settings=settings)
    assert [step.proposals for step in record.steps] == [{"agent": "t0"}, {"agent": "t1"}]
    assert all(timing.elapsed >= 0.02 for timing in timings(record))


def test_a_wall_clock_deadline_needs_a_synchronous_model_in_a_thread():
    settings = {**DEADLINE, "clock": "wall", "budget": 0.05}
    with pytest.raises(ValueError, match="cannot interrupt") as caught:
        run(Ticker(), Sleepy(0.5), TICKS, record_id="t", settings=settings)
    assert "In the episode loop, list 'model' in settings.threads." in caught.value.__notes__
    env = Ticker(ticks=2)
    settings = {**settings, "threads": ["model"]}
    start = time.monotonic()
    record = run(env, Sleepy(1.0), TICKS, record_id="t", settings=settings)
    assert time.monotonic() - start < 0.6  # the loop never waits for the busy model
    assert env.received == ["pass", "pass"]  # the model was still thinking about the first tick
    assert all(timing.defaulted and timing.elapsed >= 0.05 for timing in timings(record))


class Broken:
    """An async act-only model that fails, as a remote model whose service is down would."""

    async def act(self, observations: Any) -> Any:
        await asyncio.sleep(0.01)
        raise RuntimeError("the service is down")


def test_on_the_wall_clock_an_error_raised_while_a_model_is_thinking_reaches_the_caller():
    settings = {"clock": "wall", "timing": "deadline", "budget": 0.5}
    with pytest.raises(RuntimeError, match="the service is down"):
        run(Ticker(ticks=3), Broken(), TICKS, record_id="t", settings=settings)


class Ponderer:
    """An async act-only model that takes its time, and notes how many of its calls overlap."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.running = self.most = 0

    async def act(self, observations: Any) -> Any:
        self.running += 1
        self.most = max(self.most, self.running)
        await asyncio.sleep(self.delay)
        self.running -= 1
        return observations


def test_on_the_wall_clock_a_slow_model_works_on_one_observation_at_a_time():
    model = Ponderer(0.2)
    settings = {"clock": "wall", "timing": "deadline", "budget": 0.03}
    record = run(Ticker(ticks=3), model, TICKS, record_id="t", settings=settings)
    assert all(timing.defaulted for timing in timings(record))
    assert model.most == 1  # its first call was continued, never started again alongside


# Lanes


def test_lanes_keep_episodes_that_share_a_model_apart():
    model = Scripted(["2", "0", "1"])

    async def two_lanes() -> list[Record]:
        return await asyncio.gather(
            *(
                run_episode(DigitPadEnv(), model, SUM, record_id=f"ep-{lane}", lane=lane)
                for lane in ("a", "b")
            )
        )

    for record in asyncio.run(two_lanes()):
        assert [step.executed_actions for step in record.steps] == [
            {"student": "2"},
            {"student": "0"},
            {"student": "1"},
        ]
    assert set(model.seen) == {"a", "b"}

    players = Scripted([1, 1])
    run(NimEnv(stones=3), players, TICKS, record_id="nim", lane="c")
    assert set(players.seen) == {("c", "player_1"), ("c", "player_2")}


# What models report


class Talker(Scripted):
    """Reports how many tokens it used for each output."""

    def diagnostics(self, *, key: Any = None) -> dict[str, int]:
        return {"tokens": 10 * len(self.seen.get(key, []))}


def test_what_a_model_reports_about_each_decision_is_recorded_and_stored(tmp_path: Path):
    store = RecordStore(tmp_path)
    record = run(Ticker(ticks=2), Talker(["go", "go"]), TICKS, record_id="t", store=store)
    assert [step.diagnostics for step in record.steps] == [
        {"agent": {"tokens": 10}},
        {"agent": {"tokens": 20}},
    ]
    assert store.read("t") == record
    solved = asyncio.run(solve(Talker(["answer"]), PUZZLE, record_id="p"))
    assert solved.steps[0].diagnostics == {"solver": {"tokens": 10}}


# Static solving


def test_static_solving_presents_the_inputs_and_records_the_predictions():
    model = Scripted(["first", "second"])
    record = asyncio.run(solve(model, PUZZLE, record_id="p-1", attempts=2))
    assert record.header.static
    assert record.header.participants == ("solver",)
    assert [step.presented for step in record.steps] == [{"solver": PUZZLE.inputs}] * 2
    assert [step.predictions for step in record.steps] == [
        {"solver": "first"},
        {"solver": "second"},
    ]
    assert model.seen == {None: [PUZZLE.inputs, PUZZLE.inputs]}  # never the reference
    assert record.summary == RecordSummary.terminated()


def test_a_prediction_that_does_not_come_in_time_is_left_out():
    puzzle = Instance(id="p2", inputs="a grid")
    settings = {"timing": "deadline", "budget": 2}
    record = asyncio.run(
        solve(
            SlowThinker(think_steps=3),
            puzzle,
            record_id="p-2",
            participant="student",
            settings=settings,
        )
    )
    (step,) = record.steps
    assert step.predictions == {}
    assert step.timing == {"student": AgentTiming(elapsed=2, defaulted=True)}


def test_solving_needs_at_least_one_attempt():
    with pytest.raises(ValueError, match="attempts must be 1 or more, got 0"):
        asyncio.run(solve(SlowThinker(), PUZZLE, record_id="p-0", attempts=0))


# Settings


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        ({"timing": "deadline"}, "need a budget"),
        ({"budget": 3}, "turn_based has none"),
        ({"timing": "deadline", "budget": 2.5}, "whole number"),
    ],
)
def test_timing_settings_that_do_not_fit_together_are_refused(
    settings: dict[str, Any], problem: str
):
    with pytest.raises(ConfigError, match=problem):
        validate_settings(EpisodeSettings, settings)
    validate_settings(
        EpisodeSettings, {**settings, "budget": 2.5, "clock": "wall", "timing": "deadline"}
    )
