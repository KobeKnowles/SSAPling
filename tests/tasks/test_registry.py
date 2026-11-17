"""Tasks as components, and a static (no-environment) workflow end to end."""

from typing import TYPE_CHECKING, Any, assert_type

import pytest

from ssapling import registries
from ssapling.core import ConfigError, Registry
from ssapling.tasks import Score, Task, registry
from tests.fixtures.tasks import AdditionTask, PuzzleBook

# The task registry builds Tasks. mypy checks this; it never runs, so the
# shared registry is left empty.
if TYPE_CHECKING:
    assert_type(registry.create("addition"), Task)


@pytest.fixture
def tasks() -> Registry[Any]:
    local: Registry[Any] = Registry("task")
    local.register("addition", AdditionTask)
    local.register("puzzle_book", PuzzleBook)
    return local


def test_tasks_are_built_through_core(tasks):
    task = tasks.create("addition", {"max_digits": 3})
    assert isinstance(task, AdditionTask)
    assert task.config.max_digits == 3
    with pytest.raises(ConfigError, match="max_digits: must be >= 1"):
        tasks.create("addition", {"max_digits": 0})


def test_the_task_registry_is_collected_in_ssapling_registries():
    assert registries.tasks is registry
    assert registry.kind == "task"


def test_catalogue_describes_a_task_and_its_settings(tasks):
    entries = {entry["name"]: entry for entry in tasks.catalogue()}
    assert entries["addition"]["description"] == (
        "Add two numbers. Generated from a seed, with an optional ``digits`` parameter."
    )
    assert entries["addition"]["settings"]["properties"]["max_digits"]["minimum"] == 1
    assert entries["puzzle_book"]["settings"]["properties"] == {}


# A static workflow: no environment, as for ARC-AGI-2. That a student is shown
# the inputs alone is tested where SSAPling presents them: the episode loop and
# static solving (Stage 7).


def toy_solver(inputs: dict[str, int]) -> int:
    """Downstream solving code; SSAPling's own static solving arrives in Stage 7."""
    return inputs["a"] + inputs["b"]


def test_static_workflow_scores_a_prediction_made_from_the_inputs():
    task = AdditionTask()
    instance = task.generate(params={"digits": 2}, seed=42)
    prediction = toy_solver(instance.inputs)
    score = task.score(instance, prediction)
    assert score == Score.local(1.0)


def test_static_workflow_over_an_enumerated_set_leaves_hidden_scores_unavailable():
    book = PuzzleBook()
    results = {}
    for instance_id in book.ids():
        instance = book.get(instance_id)
        prediction = {"grid": instance.inputs["grid"]}  # a solver that copies the input
        results[instance_id] = book.score(instance, prediction)
    assert {k: v.status for k, v in results.items()} == {
        "p1": "local",
        "p1-mirror": "local",
        "p2": "local",
        "hidden": "unavailable",
    }
    assert all(v.value is False for k, v in results.items() if v.available)
