"""The environment runner: sync and async environments, seeds and options, failures, timeouts."""

import asyncio
import threading
import time
from collections.abc import Mapping
from typing import Any

import pytest

from ssapling.environments import (
    EnvironmentCommunicationError,
    EnvironmentRunner,
    ResetResult,
    StepResult,
    UnsupportedCapabilityError,
)
from tests.fixtures.environments import (
    ArcEnv,
    AsyncProxy,
    DigitPadEnv,
    FakeArcService,
    NimEnv,
    PenniesEnv,
)

PROBLEM = {"a": 45, "b": 57}  # 102: entered units first as "2", "0", "1"


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# Synchronous and asynchronous environments


@pytest.mark.parametrize("placement", ["direct", "thread"])
@pytest.mark.parametrize("wrap", [lambda env: env, AsyncProxy], ids=["sync", "async"])
def test_sync_and_async_environments_are_driven_the_same_way(placement, wrap):
    env = wrap(DigitPadEnv())
    runner = EnvironmentRunner(placement=placement)

    async def main() -> list[Any]:
        first = await runner.reset(env, options=PROBLEM)
        results = [await runner.step(env, {"student": d}) for d in "201"]
        return [first, *results]

    first, *results = run(main())
    assert first.observations == {"student": {"a": 45, "b": 57, "entered": ""}}
    assert [r.observations["student"]["entered"] for r in results] == ["2", "20", "201"]
    assert [r.episode_over for r in results] == [False, False, True]


def test_thread_placement_runs_a_synchronous_environment_off_the_loop_thread():
    class WhereAmI:
        def reset(self) -> ResetResult:
            return ResetResult(observations={"agent": threading.get_ident()})

        def step(self, actions: Mapping[str, Any]) -> StepResult:
            raise NotImplementedError

    async def main(runner: EnvironmentRunner) -> tuple[int, int]:
        first = await runner.reset(WhereAmI())
        return first.observations["agent"], threading.get_ident()

    called_in, loop_thread = run(main(EnvironmentRunner()))
    assert called_in == loop_thread
    called_in, loop_thread = run(main(EnvironmentRunner(placement="thread")))
    assert called_in != loop_thread


def test_close_is_called_if_the_environment_has_one():
    env = DigitPadEnv()
    run(EnvironmentRunner().close(env))
    assert env.closed
    run(EnvironmentRunner().close(PenniesEnv()))  # no close: nothing to do


# Seeds and options


def test_a_seed_is_passed_only_when_given():
    first = run(EnvironmentRunner().reset(DigitPadEnv(), seed=7))
    again = run(EnvironmentRunner().reset(DigitPadEnv(), seed=7))
    assert first == again  # seeded reset is reproducible
    run(EnvironmentRunner().reset(PenniesEnv()))  # reset() takes no seed, and gets none


def test_a_seed_for_an_environment_without_seeded_reset_is_refused():
    with pytest.raises(UnsupportedCapabilityError) as caught:
        run(EnvironmentRunner().reset(NimEnv(), seed=7))  # switched off by the adapter
    assert str(caught.value) == (
        "Environment NimEnv does not support 'seeded_reset'; "
        "check environments.supports(env, 'seeded_reset') before using it."
    )
    with pytest.raises(UnsupportedCapabilityError):
        run(EnvironmentRunner().reset(PenniesEnv(), seed=7))


def test_options_need_a_reset_that_takes_them():
    with pytest.raises(TypeError, match=r"PenniesEnv.reset does not take options="):
        run(EnvironmentRunner().reset(PenniesEnv(), options={"rounds": 3}))


# Checking what comes back


def test_a_step_that_returns_the_native_result_is_reported_clearly():
    class Unadapted:
        def reset(self) -> ResetResult:
            return ResetResult(observations={"agent": 0})

        def step(self, actions: Mapping[str, Any]) -> Any:
            return 0, True, {}  # the wrapped environment's own result, not converted

    with pytest.raises(TypeError, match=r"Unadapted.step returned tuple, not a StepResult; an "):
        run(EnvironmentRunner().step(Unadapted(), {"agent": 1}))


def test_a_reset_that_returns_the_native_result_is_reported_clearly():
    class Unadapted:
        def reset(self) -> Any:
            return {"agent": 0}, {}  # the wrapped environment's own result, not converted

        def step(self, actions: Mapping[str, Any]) -> StepResult:
            raise AssertionError("never reached")

    with pytest.raises(TypeError, match=r"Unadapted.reset returned tuple, not a ResetResult; an "):
        run(EnvironmentRunner().reset(Unadapted()))


def test_actions_are_keyed_by_agent():
    with pytest.raises(TypeError, match="actions must be keyed by agent id, even with one agent"):
        run(EnvironmentRunner().step(DigitPadEnv(), "2"))  # type: ignore[arg-type]


def test_an_object_without_reset_or_step_is_not_an_environment():
    with pytest.raises(TypeError, match="object has no reset method"):
        run(EnvironmentRunner().reset(object()))


# Communication failures are not the end of an episode


def test_a_lost_connection_is_reported_separately_and_not_retried():
    service = FakeArcService()
    env = ArcEnv(service)
    runner = EnvironmentRunner()
    run(runner.reset(env))
    service.drop_next = True
    with pytest.raises(EnvironmentCommunicationError) as caught:
        run(runner.step(env, {"agent": {"action": "ACTION6", "x": 1, "y": 0}}))
    assert caught.value.may_have_executed
    assert str(caught.value) == (
        "ArcEnv.step failed to communicate: connection reset by peer; "
        "it may have taken effect, so it was not retried."
    )
    assert isinstance(caught.value, ConnectionError)
    assert [call for call, _ in service.calls] == ["RESET", "ACTION6"]  # sent exactly once


def test_no_answer_in_time_is_a_communication_failure_and_is_not_retried():
    class SlowRemote:
        def __init__(self) -> None:
            self.steps = 0

        async def reset(self) -> ResetResult:
            return ResetResult(observations={"agent": 0})

        async def step(self, actions: Mapping[str, Any]) -> StepResult:
            self.steps += 1
            await asyncio.sleep(1.0)
            raise AssertionError("never reached")

    env = SlowRemote()
    started = time.perf_counter()
    with pytest.raises(
        EnvironmentCommunicationError, match=r"no response within 0\.02 s"
    ) as caught:
        run(EnvironmentRunner().step(env, {"agent": 1}, timeout=0.02))
    assert time.perf_counter() - started < 0.5
    assert env.steps == 1
    assert caught.value.may_have_executed  # the step may have run before time ran out


def test_a_slow_synchronous_environment_in_a_thread_can_be_timed_out():
    class Slow:
        def reset(self) -> ResetResult:
            time.sleep(0.3)
            return ResetResult(observations={"agent": 0})

        def step(self, actions: Mapping[str, Any]) -> StepResult:
            raise NotImplementedError

    runner = EnvironmentRunner(placement="thread")
    with pytest.raises(EnvironmentCommunicationError) as caught:
        run(runner.reset(Slow(), timeout=0.02))
    assert caught.value.operation == "reset"


def test_a_timeout_cannot_interrupt_a_synchronous_environment_run_directly():
    with pytest.raises(ValueError, match=r"cannot interrupt .*\. Use placement='thread'"):
        run(EnvironmentRunner().step(DigitPadEnv(), {"student": "2"}, timeout=1.0))


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), True])
def test_a_timeout_must_be_a_positive_number(timeout):
    with pytest.raises(ValueError, match="timeout must be a positive number of seconds"):
        run(EnvironmentRunner().reset(ArcEnv(), timeout=timeout))


def test_the_environments_own_errors_are_raised_as_they_are():
    with pytest.raises(ValueError, match="it is player_1's turn"):
        run(EnvironmentRunner().step(NimEnv(), {"player_2": 1}))
