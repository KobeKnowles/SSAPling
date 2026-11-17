"""`learn`: a learning operation called from a stage, the update counted and the statistics
published, for the stages of downstream projects as for the built-in ones."""

import asyncio
from typing import Any

from ssapling.train import RunContext, learn
from tests.fixtures.models import CountingLearner


class AsyncLearner:
    """An act-only model whose learn is async and returns no statistics."""

    learning_operations = frozenset({"consolidate"})

    def act(self, observations: Any) -> Any:
        return observations

    async def learn(self, operation: str, **payload: Any) -> None:
        return None


def test_learn_calls_the_operation_counts_the_update_and_publishes_its_statistics():
    context = RunContext({"model": CountingLearner()}, seed=1)
    stats = asyncio.run(learn(context, "learn_from_batch", batch=[1, 2, 3]))
    assert stats == {"updates": 1, "examples_seen": 3}
    assert context.counters["updates"] == 1
    assert context.read("learn_from_batch") == stats


def test_learn_awaits_an_async_learn_and_publishes_nothing_when_it_returns_nothing():
    context = RunContext({"model": AsyncLearner()}, seed=1)
    assert asyncio.run(learn(context, "consolidate")) is None
    assert context.counters["updates"] == 1
    assert "consolidate" not in context.values
