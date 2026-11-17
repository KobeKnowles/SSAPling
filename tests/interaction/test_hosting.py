"""in_process: a component built in a worker process and used through a stand-in."""

import asyncio
import inspect
import multiprocessing
import os
import pickle
from contextlib import ExitStack
from typing import Any

import pytest

from ssapling import environments
from ssapling.data import UnknownRecordError
from ssapling.environments import EnvironmentRunner, UnsupportedCapabilityError
from ssapling.interaction import ComponentProcessError, in_process
from ssapling.models import as_buffered
from ssapling.teachers import TeacherRequest, TeacherRunner
from tests.fixtures.environments import AsyncProxy, DigitPadEnv, NimEnv
from tests.fixtures.hosting import Probe, configured_probe
from tests.fixtures.models import AsyncEcho, LookupModel, LookupSettings
from tests.fixtures.teachers import AsyncExplainer, SelfCancellingTeacher, SlowTeacher

REVIEW = TeacherRequest(id="r1", operation="review_action")

# The stand-in


def test_a_component_runs_in_its_own_process_and_keeps_its_state():
    with in_process(Probe, 5) as probe:
        assert probe.pid() != os.getpid()
        assert probe.bump() == 6
        assert probe.bump(2, twice=True) == 10
        assert probe.count == 10  # attributes are read from the worker...
        assert probe.doubled == 20  # ...each time, properties included
        assert repr(probe) == f"<Probe in process {probe.pid()}>"


def test_methods_keep_their_kind_and_parameters():
    with in_process(Probe) as probe:
        assert type(probe).__name__ == "Probe"
        assert list(inspect.signature(probe.bump).parameters) == ["by", "twice"]
        assert list(inspect.signature(probe.apply).parameters) == ["value", "how"]
        assert probe.apply(3) == 6
        assert not inspect.iscoroutinefunction(probe.bump)
        assert inspect.iscoroutinefunction(probe.later)
        assert asyncio.run(probe.later("x")) == "x"


def test_a_method_a_wrapper_forwards_runs_on_the_component_itself():
    with in_process(AsyncProxy, DigitPadEnv()) as env:  # AsyncProxy forwards close and closed
        env.close()
        assert env.closed


def test_a_property_is_read_only_when_asked_for():
    with in_process(Probe) as probe:  # Probe.doubled fails before anything is counted
        probe.bump()
        assert probe.doubled == 2


def test_a_missing_attribute_is_an_attribute_error_and_attributes_cannot_be_set():
    with in_process(Probe) as probe:
        assert getattr(probe, "snapshot", None) is None
        with pytest.raises(AttributeError):
            _ = probe.nothing_here  # type: ignore[attr-defined]
        with pytest.raises(AttributeError, match="runs in another process"):
            probe.count = 3


def test_a_factory_function_runs_in_the_worker():
    with in_process(configured_probe) as probe:
        assert probe.label == "configured"
    assert Probe.label == "probe"  # this process is untouched


# Components of each area


def test_an_environment_in_a_process_is_used_like_any_other():
    runner = EnvironmentRunner()
    with in_process(NimEnv, stones=3) as env:
        assert not environments.supports(env, "seeded_reset")  # its own supports, in the worker
        with pytest.raises(UnsupportedCapabilityError, match="Environment NimEnv does not"):
            asyncio.run(runner.reset(env, seed=1))
        first = asyncio.run(runner.reset(env))
        assert environments.next_agents(env, first) == ("player_1",)
        result = asyncio.run(runner.step(env, {"player_1": 1}))
        assert environments.next_agents(env, result) == ("player_2",)
    with in_process(DigitPadEnv) as pad:
        assert environments.supports(pad, "snapshot")
        first = asyncio.run(runner.reset(pad, seed=1, options={"a": 45, "b": 57}))
        assert first.observations["student"]["entered"] == ""


def test_a_teacher_and_a_model_in_a_process_are_used_like_any_other():
    explain = TeacherRequest(id="r2", operation="explain")
    lookup = LookupSettings(table={"2+2": "4"})
    with in_process(AsyncExplainer) as teacher, in_process(LookupModel, lookup) as model:
        assert asyncio.run(TeacherRunner().request(teacher, explain)).status == "answered"
        buffered: Any = as_buffered(model)  # an act-only model, wrapped here around its stand-in
        buffered.perceive("2+2")
        buffered.advance()
        assert buffered.poll() == "4"
    with in_process(AsyncEcho) as echo:
        wrapped: Any = as_buffered(echo)
        assert asyncio.run(wrapped.act("hello")) == "hello"


def test_a_timeout_needs_a_thread_as_for_any_synchronous_component():
    with in_process(SlowTeacher, 1.0) as teacher:
        with pytest.raises(ValueError, match="cannot interrupt"):
            asyncio.run(TeacherRunner().request(teacher, REVIEW, timeout=0.1))
        threaded = TeacherRunner(placement="thread")
        response = asyncio.run(threaded.request(teacher, REVIEW, timeout=0.1))
        assert response.status == "cancelled"
        assert asyncio.run(threaded.request(teacher, REVIEW)).status == "answered"


def test_a_teacher_cancelling_in_its_process_gives_a_cancelled_response():
    with in_process(SelfCancellingTeacher) as teacher:
        response = asyncio.run(TeacherRunner().request(teacher, REVIEW))
    assert response.status == "cancelled"


# Errors


def test_an_error_raised_in_the_worker_arrives_as_itself():
    with in_process(Probe) as probe:
        with pytest.raises(ValueError, match="bad digit") as caught:
            probe.fail(ValueError("bad digit"))
        assert "Raised by Probe.fail in its worker process." in caught.value.__notes__
        with pytest.raises(UnknownRecordError) as missing:
            probe.fail(UnknownRecordError("ep-1", "runs/a"))
        assert missing.value.record_id == "ep-1"
        assert probe.bump() == 1  # the worker carries on


def test_what_cannot_be_sent_back_is_reported_and_the_worker_carries_on():
    with in_process(Probe) as probe:
        with pytest.raises(ComponentProcessError, match="could not send back a function"):
            probe.unsendable()
        with pytest.raises(
            ComponentProcessError, match="could not send back its error StubbornError"
        ):
            probe.fail_stubbornly()
        assert probe.bump() == 1


def test_an_argument_that_cannot_be_pickled_is_reported_and_the_worker_carries_on():
    with in_process(Probe) as probe:
        # Which error pickle raises for a local function differs between Python versions.
        with pytest.raises((AttributeError, pickle.PicklingError)) as caught:
            probe.apply(1, how=lambda value: value)
        assert "Probe runs in another process, so arguments are pickled." in caught.value.__notes__
        assert probe.bump() == 1


def test_a_reply_this_process_cannot_rebuild_is_reported_and_the_worker_carries_on():
    with in_process(Probe) as probe:
        with pytest.raises(ComponentProcessError, match="sent back something this process cannot"):
            probe.unrebuildable()
        assert probe.bump() == 1


def test_a_failure_while_building_is_raised_when_the_block_is_entered():
    with ExitStack() as stack, pytest.raises(ValueError, match="start must be 0") as caught:
        stack.enter_context(in_process(Probe, -1))
    assert "While building Probe in a worker process." in caught.value.__notes__


def test_a_worker_that_exits_is_reported():
    with in_process(Probe) as probe:
        with pytest.raises(ComponentProcessError, match=r"Probe exited \(exit code 3\)"):
            probe.crash()
        with pytest.raises(ComponentProcessError, match="exited"):
            probe.bump()


def test_the_worker_stops_when_the_block_ends():
    with in_process(Probe) as probe:
        pid = probe.pid()
    assert pid not in [child.pid for child in multiprocessing.active_children()]
    with pytest.raises(ComponentProcessError, match="its in_process block ended"):
        probe.bump()
