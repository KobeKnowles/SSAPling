"""Acting: act-only and buffered models, as_buffered, NOTHING, and participants or lanes."""

import asyncio
import inspect
import pickle
from typing import Any

import pytest

from ssapling.data import JsonCodec
from ssapling.models import NOTHING, as_buffered
from tests.fixtures.models import (
    Adder,
    AsyncEcho,
    CountingLearner,
    LookupModel,
    LookupSettings,
    SlowThinker,
)


def lookup() -> LookupModel:
    return LookupModel(LookupSettings(table={"2+2": "4", "3+4": "7"}))


# NOTHING


def test_nothing_is_not_none_and_survives_pickling():
    assert NOTHING is not None
    assert repr(NOTHING) == "NOTHING"
    assert pickle.loads(pickle.dumps(NOTHING)) is NOTHING  # e.g. from a worker process


# An act-only model


def test_an_act_only_model_just_acts():
    assert lookup().act("2+2") == "4"
    assert lookup().act("9+9") == "?"


def test_an_act_only_model_is_wrapped_with_buffers():
    model: Any = as_buffered(lookup())
    model.perceive("2+2")
    assert model.poll() is NOTHING  # perceived, not yet processed
    model.advance()
    assert model.poll() == "4"
    assert model.poll() is NOTHING  # each output is read once


def test_the_wrapper_keeps_every_observation_and_output_in_order():
    """Nothing is dropped: what to do with a late output is the coordinating code's choice."""
    model: Any = as_buffered(lookup())
    model.perceive("2+2")
    model.perceive("3+4")
    model.advance()
    model.advance()
    model.advance()  # nothing left to process: no-op
    assert [model.poll(), model.poll(), model.poll()] == ["4", "7", NOTHING]


def test_the_wrapper_still_offers_act_directly():
    model: Any = as_buffered(lookup())
    assert model.act("3+4") == "7"


def test_an_async_act_only_model_gets_async_methods():
    model: Any = as_buffered(AsyncEcho())
    assert inspect.iscoroutinefunction(model.advance)
    assert inspect.iscoroutinefunction(model.act)
    simple: Any = as_buffered(lookup())
    assert not inspect.iscoroutinefunction(simple.advance)

    async def main() -> list[object]:
        model.perceive("hello")
        before = model.poll()
        await model.advance()
        return [before, model.poll(), await model.act("direct")]

    assert asyncio.run(main()) == [NOTHING, "hello", "direct"]


# A buffered model


def test_a_buffered_model_is_used_as_it_is():
    thinker = SlowThinker()
    assert as_buffered(thinker) is thinker


def test_a_buffered_model_takes_as_many_steps_as_it_needs():
    thinker = SlowThinker(think_steps=3)
    thinker.perceive("frame-1")
    outputs = []
    for _ in range(3):
        thinker.advance()
        outputs.append(thinker.poll())
    assert outputs == [NOTHING, NOTHING, "frame-1#1"]


def test_wrapping_twice_changes_nothing():
    once: Any = as_buffered(lookup())
    assert as_buffered(once) is once


def test_a_half_buffered_model_is_rejected():
    class PerceiveOnly:
        def perceive(self, observations: object) -> None:
            pass

    with pytest.raises(TypeError, match="PerceiveOnly has a buffered interface without poll"):
        as_buffered(PerceiveOnly())  # type: ignore[arg-type]


def test_an_object_that_cannot_act_is_not_a_model():
    with pytest.raises(TypeError, match="object is not a model: it has no act method"):
        as_buffered(object())  # type: ignore[arg-type]


# Participants and lanes


def test_each_key_has_its_own_queues():
    model: Any = as_buffered(AsyncEcho())

    async def main() -> list[object]:
        model.perceive("for player 1", key="player_1")
        model.perceive("for player 2", key="player_2")
        await model.advance(key="player_2")
        return [model.poll(key="player_1"), model.poll(key="player_2"), model.poll()]

    assert asyncio.run(main()) == [NOTHING, "for player 2", NOTHING]


def test_episode_state_is_reset_per_participant_or_lane():
    thinker = SlowThinker(think_steps=1)
    for key in ("lane-0", "lane-1"):
        thinker.perceive("x", key=key)
        thinker.advance(key=key)
        thinker.poll(key=key)
    thinker.reset_episode(key="lane-0")
    thinker.perceive("y", key="lane-0")
    thinker.perceive("y", key="lane-1")
    thinker.advance(key="lane-0")
    thinker.advance(key="lane-1")
    assert thinker.poll(key="lane-0") == "y#1"  # reset: counting again from 1
    assert thinker.poll(key="lane-1") == "y#2"  # untouched


def test_resetting_the_wrapper_empties_that_keys_queues_only():
    model: Any = as_buffered(AsyncEcho())

    async def main() -> list[object]:
        for key in ("a", "b"):
            model.perceive(f"left over from {key}", key=key)
            model.perceive(f"unprocessed for {key}", key=key)
            await model.advance(key=key)
        model.reset_episode(key="a")
        await model.advance(key="a")  # nothing queued any more
        return [model.poll(key="a"), model.poll(key="b"), model.poll(key="b")]

    assert asyncio.run(main()) == [NOTHING, "left over from b", NOTHING]


def test_resetting_the_wrapper_resets_the_model_too():
    class Remembers:
        def __init__(self) -> None:
            self.resets: list[object] = []

        def act(self, observations: object, *, key: object = None) -> object:
            return observations

        def reset_episode(self, *, key: object = None) -> None:
            self.resets.append(key)

    inner = Remembers()
    model: Any = as_buffered(inner)
    model.reset_episode()
    model.reset_episode(key="lane-3")
    assert inner.resets == [None, "lane-3"]


def test_a_key_is_passed_only_when_there_is_one():
    """A simple model with one participant never needs a key parameter."""
    model: Any = as_buffered(lookup())
    model.perceive("2+2")
    model.advance()
    assert model.poll() == "4"


def test_a_model_given_a_key_must_take_one():
    model: Any = as_buffered(lookup())
    with pytest.raises(TypeError, match=r"LookupModel.act does not take key=.*key=None parameter"):
        model.perceive("2+2", key="player_1")


# Everything else is the model's own


def test_the_wrapper_reaches_the_models_other_members():
    learner = CountingLearner()
    model: Any = as_buffered(learner)
    assert model.model is learner
    assert model.learning_operations == learner.learning_operations
    assert model.learn("consolidate") == {"updates": 1, "examples_seen": 0}
    assert model.export_state() == {"model": {"updates": 1, "examples_seen": 0}, "queues": []}
    with pytest.raises(AttributeError):
        model.not_a_member  # noqa: B018


# State: the queues of a wrapped act-only model are part of its state


class AsyncState:
    """An act-only model whose state methods are async."""

    def __init__(self) -> None:
        self.count = 0

    def act(self, observations: Any) -> Any:
        return observations

    async def export_state(self) -> dict[str, int]:
        return {"count": self.count}

    async def restore_state(self, state: dict[str, int]) -> None:
        self.count = state["count"]


def test_a_wrapped_models_queues_are_in_its_exported_state():
    """Given a wrapped learner holding one output not yet polled and one observation not yet
    advanced on, when its state is exported and restored into a fresh wrapped learner, then the
    fresh one polls the held output, then advances on the queued observation, as the original
    would, and the learner's own state came back too.
    Catches: a restore mid-step that loses the step, which fails a model-state check."""
    first: Any = as_buffered(CountingLearner())
    first.learn("consolidate")  # one update: it now answers "1"
    first.perceive("2+2")
    first.advance()  # "1" held, not polled
    first.perceive("3+4")  # queued, not advanced
    again: Any = as_buffered(CountingLearner())
    again.restore_state(first.export_state())
    assert again.updates == 1
    assert again.poll() == "1"
    again.advance()
    assert again.poll() == "1"
    assert again.poll() is NOTHING


def test_each_lanes_queues_are_kept_apart_and_the_export_is_a_copy():
    """Given a wrapped model with no state of its own, observations queued in two lanes, one
    keyed by a tuple, when its state goes through the JSON codec into a fresh wrapped model,
    then each lane advances on its own observation; and an observation queued after the export
    is not in it.
    Catches: lanes merged in the state; an export that shares the live queues."""
    model: Any = as_buffered(Adder())
    model.perceive({"a": 1, "b": 2}, key="x")
    model.perceive({"a": 5, "b": 5}, key=("lane", 2))
    state = model.export_state()
    model.perceive({"a": 9, "b": 9}, key="x")
    codec = JsonCodec()
    again: Any = as_buffered(Adder())
    again.restore_state(codec.decode(codec.encode(state)))
    again.advance(key="x")
    again.advance(key=("lane", 2))
    assert again.poll(key="x") == 3
    assert again.poll(key=("lane", 2)) == 10
    again.advance(key="x")
    assert again.poll(key="x") is NOTHING


def test_a_models_async_state_methods_stay_async():
    model: Any = as_buffered(AsyncState())
    model.model.count = 4
    model.perceive("x")
    state = asyncio.run(model.export_state())
    again: Any = as_buffered(AsyncState())
    asyncio.run(again.restore_state(state))
    assert again.model.count == 4
    again.advance()
    assert again.poll() == "x"
