"""Native and adapted behaviour compared: action order, rewards, final observations, completion.

Each test plays the same script on a native environment directly and through
its adapter, driven by the runner the way the episode loop (Stage 7) will
drive it: asking ``next_agents`` who acts, then stepping. Every adapter runs
both synchronously and asynchronously.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from ssapling.environments import EnvironmentRunner, StepResult, next_agents
from tests.fixtures.environments import (
    ArcEnv,
    AsyncProxy,
    DigitPadEnv,
    FakeArcService,
    LaneBatch,
    NativeDigitPad,
    NativeNim,
    NativePennies,
    NimEnv,
    PenniesEnv,
)
from tests.fixtures.tasks import AdditionTask
from tests.fixtures.teachers import DigitTutor

SYNC_AND_ASYNC = pytest.mark.parametrize(
    "wrap", [lambda env: env, AsyncProxy], ids=["sync", "async"]
)


def play(env: Any, script: Sequence[Mapping[str, Any]], **reset: Any) -> list[StepResult]:
    """Reset ``env``, then step it through ``script``, checking whose turn each step is."""

    async def main() -> list[StepResult]:
        runner = EnvironmentRunner()
        last: Any = await runner.reset(env, **reset)
        results = []
        for actions in script:
            assert set(actions) == set(next_agents(env, last)), "actions out of turn"
            last = await runner.step(env, actions)
            results.append(last)
        return results

    return asyncio.run(main())


# One agent


@SYNC_AND_ASYNC
def test_single_agent_adapter_matches_the_native_environment(wrap):
    native = NativeDigitPad()
    native.reset(problem=(45, 57))
    native_steps = [native.step(digit) for digit in "201"]

    adapter = DigitPadEnv()
    results = play(wrap(adapter), [{"student": d} for d in "201"], options={"a": 45, "b": 57})

    assert adapter.native.log == native.log == ["2", "0", "1"]  # the same actions, in order
    assert [r.observations["student"] for r in results] == [obs for obs, _, _ in native_steps]
    assert [r.terminated for r in results] == [{"student": done} for _, done, _ in native_steps]
    assert [r.info for r in results] == [info for _, _, info in native_steps]
    assert all(r.rewards is None for r in results)  # no rewards: not invented as zeros
    assert [r.episode_terminated for r in results] == [False, False, True]


@SYNC_AND_ASYNC
def test_a_wrong_digit_ends_the_episode_in_both(wrap):
    native = NativeDigitPad()
    native.reset(problem=(45, 57))
    _, native_done, _ = native.step("9")
    (result,) = play(wrap(DigitPadEnv()), [{"student": "9"}], options={"a": 45, "b": 57})
    assert native_done is True
    assert result.episode_terminated
    assert not result.episode_truncated


def test_the_episode_is_set_up_from_an_instances_inputs():
    instance = AdditionTask().generate(params={"digits": 2}, seed=7)  # 45 + 57
    env = DigitPadEnv()
    first = asyncio.run(EnvironmentRunner().reset(env, options=instance.inputs))
    assert first.observations["student"] == {"a": 45, "b": 57, "entered": ""}


# Two agents taking turns


@SYNC_AND_ASYNC
def test_sequential_adapter_preserves_turn_order_rewards_and_completion(wrap):
    native = NativeNim(stones=5)
    for n in (2, 1, 2):
        native.take(n)

    adapter = NimEnv(stones=5)
    script = [{"player_1": 2}, {"player_2": 1}, {"player_1": 2}]
    results = play(wrap(adapter), script)

    assert adapter.native.log == native.log == [("player_1", 2), ("player_2", 1), ("player_1", 2)]
    assert results[-1].rewards == native.scores() == {"player_1": 1, "player_2": -1}
    assert [r.rewards for r in results[:-1]] == [{"player_1": 0, "player_2": 0}] * 2
    assert results[-1].observations == {"player_1": 0, "player_2": 0}  # final observations
    assert results[-1].terminated == {"player_1": True, "player_2": True}
    assert results[-1].episode_terminated
    assert not any(r.episode_over for r in results[:-1])


def test_sequential_adapter_refuses_an_action_out_of_turn():
    env = NimEnv()
    env.reset()
    with pytest.raises(ValueError, match="it is player_1's turn"):
        env.step({"player_1": 1, "player_2": 1})


# Two agents acting at once


@SYNC_AND_ASYNC
def test_simultaneous_adapter_matches_rewards_and_truncates_at_the_round_limit(wrap):
    choices = [("heads", "heads"), ("heads", "tails"), ("tails", "tails")]
    native = NativePennies(rounds=3)
    native_rewards = [native.play(*pair) for pair in choices]

    adapter = PenniesEnv(rounds=3)
    script = [{"player_1": one, "player_2": two} for one, two in choices]
    results = play(wrap(adapter), script)  # both agents act in every step

    assert adapter.native.log == native.log == choices
    assert [(r.rewards or {})["player_1"] for r in results] == [one for one, _ in native_rewards]
    assert [(r.rewards or {})["player_2"] for r in results] == [two for _, two in native_rewards]
    assert results[-1].observations == {"player_1": "tails", "player_2": "tails"}
    assert results[-1].episode_truncated
    assert not results[-1].episode_terminated
    assert not any(r.episode_over for r in results[:-1])


# An interactive ARC-AGI-3-like game


ARC_SCRIPT: list[dict[str, Any]] = [
    {"agent": {"action": "ACTION6", "x": 0, "y": 0}},  # a miss
    {"agent": {"action": "ACTION6", "x": 1, "y": 0}},  # completes level 1
    {"agent": {"action": "ACTION1"}},  # a parameterless action
    {"agent": {"action": "ACTION6", "x": 0, "y": 1}},  # completes level 2: a win
]


def test_arc_adapter_preserves_frames_actions_and_game_status():
    native = FakeArcService()
    native.reset("game-a")
    native_responses = [
        native.act(
            step["agent"]["action"],
            {k: v for k, v in step["agent"].items() if k != "action"} or None,
        )
        for step in ARC_SCRIPT
    ]

    service = FakeArcService()
    results = play(ArcEnv(service), ARC_SCRIPT, options={"game_id": "game-a"})

    assert service.calls == native.calls  # the same parameterised actions, in order
    assert [r.observations["agent"]["frames"] for r in results] == [
        response["frame"] for response in native_responses
    ]
    assert len(results[1].observations["agent"]["frames"]) == 2  # a multi-frame observation
    assert [r.info["state"] for r in results] == ["NOT_FINISHED"] * 3 + ["WIN"]
    assert [r.action_restrictions for r in results] == [
        {"agent": response["available_actions"]} for response in native_responses
    ]
    assert all(r.rewards is None for r in results)  # benchmark scoring is downstream


def test_completing_a_level_is_progress_not_the_end_of_the_episode():
    results = play(ArcEnv(), ARC_SCRIPT[:2])
    assert results[1].info["levels_completed"] == 1
    assert not results[1].episode_over
    final = play(ArcEnv(), ARC_SCRIPT)[-1]
    assert final.episode_terminated
    assert final.action_restrictions == {"agent": []}


# Lanes: independent copies stepped together


def test_lanes_are_independent_and_an_ended_lane_waits_to_be_reset():
    batch = LaneBatch(DigitPadEnv, lanes=2)
    first = batch.reset_lanes([0, 1], seeds={0: 1, 1: 2})
    assert first[0] != first[1]  # different seeds, different problems

    answers = {
        lane: str(sum(r.observations["student"][k] for k in "ab"))[::-1]
        for lane, r in first.items()
    }
    ended = batch.step_lanes({0: {"student": "x"}, 1: {"student": answers[1][0]}})
    assert ended[0].episode_over  # a wrong digit ends lane 0 only
    assert not ended[1].episode_over

    with pytest.raises(ValueError, match="lane 0 has ended; reset it first"):
        batch.step_lanes({0: {"student": "1"}})
    again = batch.reset_lanes([0], seeds={0: 1})
    assert ended[0].observations["student"]["entered"] == "x"  # the final observation...
    assert again[0].observations["student"]["entered"] == ""  # ...is not the next first one


# A teacher inside the environment


def test_an_environment_that_owns_a_teacher_reports_its_responses_and_executed_actions():
    task = AdditionTask()
    env = DigitPadEnv(teacher=DigitTutor(task=task, config=DigitTutor.Settings(intervene=True)))
    (corrected, right) = play(env, [{"student": "9"}, {"student": "0"}], options={"a": 45, "b": 57})

    assert corrected.executed_actions == {"student": "2"}  # the intervention replaced the 9
    assert env.native.log == ["2", "0"]
    assert corrected.teacher_responses is not None
    (response,) = corrected.teacher_responses
    assert response.contributions["intervention"] == {"student": "2"}
    assert not corrected.episode_over  # the corrected digit kept the episode going
    assert right.executed_actions == {"student": "0"}
    assert right.teacher_responses is not None
    assert right.teacher_responses[0].contributions == {"feedback": {"student": "Correct."}}


def test_the_environment_that_owns_a_teacher_resets_it():
    tutor = DigitTutor(task=AdditionTask())
    env = DigitPadEnv(teacher=tutor)
    play(env, [{"student": "9"}], options={"a": 45, "b": 57})
    assert tutor.reviews == 1
    env.reset(options={"a": 45, "b": 57})  # a new episode: the owner resets its teacher
    assert tutor.reviews == 0
