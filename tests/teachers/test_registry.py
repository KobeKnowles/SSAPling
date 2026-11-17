"""Teachers as components, and a task's teaching resources reaching a teacher end to end."""

import asyncio
from typing import TYPE_CHECKING, Any, assert_type

import pytest

from ssapling import registries, tasks, teachers
from ssapling.core import ConfigError, Registry
from ssapling.teachers import (
    MissingResourcesError,
    Teacher,
    TeacherRequest,
    TeacherRunner,
    registry,
)
from tests.fixtures.tasks import AdditionTask, PuzzleBook
from tests.fixtures.teachers import AsyncExplainer, DigitTutor

# The teacher registry builds Teachers. mypy checks this; it never runs, so the
# shared registry is left empty.
if TYPE_CHECKING:
    assert_type(registry.create("digit_tutor", task=AdditionTask()), Teacher)


@pytest.fixture
def local() -> Registry[Any]:
    local: Registry[Any] = Registry("teacher")
    local.register("digit_tutor", DigitTutor)
    local.register("explainer", AsyncExplainer)
    return local


def test_teachers_are_built_through_core_with_the_task_as_a_live_object(local):
    tutor = local.create("digit_tutor", {"intervene": True}, task=AdditionTask())
    assert isinstance(tutor, DigitTutor)
    assert tutor.config.intervene is True
    assert tutor.procedure == "Add the units, carry, then add the tens."
    with pytest.raises(ConfigError, match="intervene"):
        local.create("digit_tutor", {"intervene": "yes"}, task=AdditionTask())


def test_the_teacher_registry_is_collected_in_ssapling_registries():
    assert registries.teachers is registry
    assert registry.kind == "teacher"


def test_catalogue_describes_a_teacher_and_its_settings(local):
    entries = {entry["name"]: entry for entry in local.catalogue()}
    assert entries["digit_tutor"]["description"] == (
        "Reviews each digit of an addition answer as the student enters it, units first."
    )
    assert entries["digit_tutor"]["settings"]["properties"]["intervene"]["default"] is False
    assert entries["explainer"]["settings"]["properties"] == {}


# A task, a teacher, and one review, as the coordinating code (Stage 7) will do it.


def test_a_teacher_uses_the_tasks_teaching_resources_end_to_end(local):
    task = AdditionTask()
    teachers.check_resources(local.get("digit_tutor"), tasks.teaching_resources(task))
    tutor = local.create("digit_tutor", {"intervene": True}, task=task)

    instance = task.generate(params={"digits": 2}, seed=7)
    units = str(instance.reference)[-1]
    wrong = str((int(units) + 1) % 10)
    request = TeacherRequest(
        id="review-1",
        operation="review_action",
        context={"instance": instance, "state": ""},
        proposals={"student": wrong},
        episode="ep-1",
        participant="student",
        step=0,
    )
    assert teachers.supports(tutor, "review_action")
    response = asyncio.run(TeacherRunner().request(tutor, request))

    assert response.status == "answered"
    assert response.contributions["intervention"] == {"student": units}
    assert response.contributions["feedback"] == {
        "student": "Not quite. Add the units, carry, then add the tens."
    }


def test_a_task_without_the_resource_is_caught_before_the_run(local):
    with pytest.raises(MissingResourcesError, match="'procedure'; none are available"):
        teachers.check_resources(local.get("digit_tutor"), tasks.teaching_resources(PuzzleBook()))
    # Building it anyway fails too, where the teacher fetches the resource.
    with pytest.raises(tasks.MissingTeachingResourceError):
        local.create("digit_tutor", task=PuzzleBook())
