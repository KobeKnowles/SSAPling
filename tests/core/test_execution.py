"""The internal execution helper: sync and async calls through one awaitable entry point."""

import asyncio
import contextvars
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from ssapling.core._execution import DeadlineExpiredError, Execution

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="none")


def add(a: int, b: int) -> int:
    return a + b


async def add_async(a: int, b: int) -> int:
    await asyncio.sleep(0)
    return a + b


def thread_id() -> int:
    return threading.get_ident()


async def thread_id_async() -> int:
    return threading.get_ident()


def fail() -> None:
    raise RuntimeError("boom")


def calls_from(execution: Execution, function, times: int = 1) -> tuple[list[int], int]:
    """Run ``function`` ``times`` times and return the threads it ran on and the loop's thread."""

    async def main() -> tuple[list[int], int]:
        ids = await asyncio.gather(*(execution.call(function) for _ in range(times)))
        return list(ids), threading.get_ident()

    return asyncio.run(main())


# Synchronous and asynchronous functions alike


@pytest.mark.parametrize("execution", [Execution(), Execution("thread")])
@pytest.mark.parametrize("function", [add, add_async])
def test_sync_and_async_functions_are_called_the_same_way(execution, function):
    assert asyncio.run(execution.call(function, 2, b=3)) == 5


def test_sync_function_returning_an_awaitable_is_awaited():
    assert asyncio.run(Execution().call(lambda: add_async(1, 2))) == 3


@pytest.mark.parametrize("execution", [Execution(), Execution("thread")])
def test_exceptions_propagate(execution):
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(execution.call(fail))


# Where calls run


def test_direct_placement_runs_sync_calls_in_the_loop_thread():
    called_in, loop_thread = calls_from(Execution(), thread_id)
    assert called_in == [loop_thread]


def test_thread_placement_runs_sync_calls_off_the_loop_thread():
    called_in, loop_thread = calls_from(Execution("thread"), thread_id)
    assert called_in[0] != loop_thread


def test_a_single_thread_executor_keeps_every_call_on_one_thread():
    with ThreadPoolExecutor(max_workers=1) as pool:
        called_in, loop_thread = calls_from(Execution("thread", pool), thread_id, times=5)
    assert len(set(called_in)) == 1
    assert called_in[0] != loop_thread


@pytest.mark.parametrize("execution", [Execution(), Execution("thread")])
def test_async_functions_always_run_in_the_loop_thread(execution):
    called_in, loop_thread = calls_from(execution, thread_id_async)
    assert called_in == [loop_thread]


def test_a_timeout_stops_waiting_for_a_call_in_a_thread():
    async def main() -> float:
        start = time.perf_counter()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(Execution("thread").call(time.sleep, 0.3), timeout=0.02)
        return time.perf_counter() - start

    # The loop stopped waiting long before the 0.3 s call finished.
    assert asyncio.run(main()) < 0.25


# Timeouts


def own_timeout() -> None:
    raise TimeoutError("the component's own")


async def slow_async() -> None:
    await asyncio.sleep(1.0)


@pytest.mark.parametrize(
    ("execution", "function"),
    [(Execution("thread"), lambda: time.sleep(0.3)), (Execution(), slow_async)],
    ids=["sync in a thread", "async"],
)
def test_the_callers_deadline_is_told_apart_from_the_components_own_timeout(execution, function):
    with pytest.raises(DeadlineExpiredError, match=r"no response within 0\.02 s") as caught:
        asyncio.run(execution.call_with_timeout(0.02, function))
    assert caught.value.timeout == 0.02
    with pytest.raises(TimeoutError, match="the component's own"):
        asyncio.run(Execution("thread").call_with_timeout(5.0, own_timeout))


def test_no_timeout_waits_as_long_as_it_takes():
    assert asyncio.run(Execution().call_with_timeout(None, add, 2, 3)) == 5


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), True, "1"])
def test_a_timeout_must_be_a_positive_number(timeout):
    with pytest.raises(ValueError, match="timeout must be a positive number of seconds"):
        Execution("thread").check_timeout(timeout, add)


def test_a_timeout_cannot_interrupt_a_synchronous_call_run_directly():
    with pytest.raises(ValueError, match=r"cannot interrupt add: .* Use placement='thread'"):
        asyncio.run(Execution().call_with_timeout(1.0, add, 2, 3))
    Execution().check_timeout(1.0, add_async)  # an async call yields, so a timeout can stop it


@pytest.mark.parametrize("pool", [None, "single"])
def test_context_variables_reach_the_thread(pool):
    def read() -> str:
        return request_id.get()

    async def main(execution: Execution) -> str:
        request_id.set("review-12")
        return await execution.call(read)

    if pool is None:
        assert asyncio.run(main(Execution("thread"))) == "review-12"
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert asyncio.run(main(Execution("thread", executor))) == "review-12"


# Invalid settings


def test_unknown_placement_is_rejected():
    with pytest.raises(ValueError, match="placement must be"):
        Execution("process")  # type: ignore[arg-type]


def test_an_executor_needs_thread_placement():
    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(ValueError, match="thread"):
        Execution("direct", pool)


def test_a_process_pool_is_rejected():
    with ProcessPoolExecutor(max_workers=1) as pool, pytest.raises(TypeError, match="state"):
        Execution("thread", pool)
